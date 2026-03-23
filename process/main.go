package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/qdrant/go-client/qdrant"
	"go.ytsaurus.tech/yt/go/ypath"
	"go.ytsaurus.tech/yt/go/yt"
	"go.ytsaurus.tech/yt/go/yt/ythttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type SearchItem struct {
	Id     int64     `yson:"id"`
	Vector []float32 `yson:"data"`
}

type ResultItem struct {
	Id    int64 `yson:"id" json:"id"`
	RepId int64 `yson:"rep_id" json:"rep_id"`
}

type Pair struct {
	ItemID int64
	RepID  int64
}

type Config struct {
	InputTable      string
	OutputTable     string
	QdrantHost      string
	QdrantPort      int
	CollectionName  string
	SearchLimit     int
	SearchThreshold float32
	QdrantBatchSize int
	YtReadBatchSize int
	ProcSize        int
}

func main() {
	config := &Config{
		InputTable:      getEnvOrDefault("INPUT_TABLE", "//home/mlinfra/dups_research/zelda_dataset_16_02-16_03"),
		OutputTable:     getEnvOrDefault("OUTPUT_TABLE", "//home/mlinfra/dups_research/zelda_dataset_16_02-16_03.out_v2"),
		QdrantHost:      getEnvOrDefault("QDRANT_HOST", "qdrant.dedup.clouds.idzn.ru"),
		QdrantPort:      getEnvIntOrDefault("QDRANT_PORT", 6334),
		CollectionName:  getEnvOrDefault("COLLECTION_NAME", "VkVideoZeldaVideoFusedVkV2"),
		SearchLimit:     getEnvIntOrDefault("SEARCH_LIMIT", 64),
		SearchThreshold: getEnvFloat32OrDefault("SEARCH_THRESHOLD", 0.97),
		QdrantBatchSize: getEnvIntOrDefault("Q_BATCH_SIZE", 1000),
		YtReadBatchSize: getEnvIntOrDefault("YT_BATCH_SIZE", 1000),
		ProcSize:        getEnvIntOrDefault("PROC_SIZE", 1000000000000),
	}

	if config.InputTable == "" || config.OutputTable == "" {
		log.Fatal("INPUT_TABLE and OUTPUT_TABLE environment variables must be set")
	}

	if err := process(config); err != nil {
		log.Fatalf("Error: %v", err)
	}
}

func process(config *Config) error {
	log.Printf("Starting processing with config: %+v", config)

	// Create YT client
	yc, err := ythttp.NewClient(&yt.Config{
		Token: os.Getenv("YT_TOKEN"),
		Proxy: os.Getenv("YT_PROXY"),
	})
	if err != nil {
		return err
	}

	conn, err := grpc.NewClient(config.QdrantHost, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect to Qdrant: %w", err)
	}

	pointsClient := qdrant.NewPointsClient(conn)

	ctx := context.Background()

	// Read input table
	log.Printf("Reading input table: %s", config.InputTable)

	inputPath := ypath.Path(config.InputTable)
	reader, err := yc.ReadTable(ctx, inputPath, nil)
	if err != nil {
		return err
	}
	defer reader.Close()

	// Create UnionFind
	uf := NewUnionFind()

	allIDS := make([]int64, 0)
	itemCount := 0
	duplicatesFound := 0
	batchCount := 0

	for {
		// Read batch of 10000 items
		batch := make([]*SearchItem, 0, config.YtReadBatchSize)
		for len(batch) < config.YtReadBatchSize && reader.Next() {
			var item SearchItem
			if err := reader.Scan(&item); err != nil {
				log.Printf("Error scanning row: %v", err)
				continue
			}
			batch = append(batch, &item)
		}

		if len(batch) == 0 {
			break // No more items
		}

		batchCount++
		log.Printf("📦 Read batch %d with %d items", batchCount, len(batch))

		// Initialize UnionFind for all items in batch
		if len(allIDS) > config.ProcSize {
			break
		}
		for _, item := range batch {
			allIDS = append(allIDS, item.Id)
			uf.MakeSet(item.Id)
		}

		// Process batch in chunks of 1000 items (10 requests per batch)
		for i := 0; i < len(batch); i += config.QdrantBatchSize {
			end := i + config.QdrantBatchSize
			if end > len(batch) {
				end = len(batch)
			}
			chunk := batch[i:end]

			log.Printf("🔍 Processing chunk %d-%d (%d items)", i, end-1, len(chunk))

			// Prepare batch search request
			searchRequests := make([]*qdrant.SearchPoints, len(chunk))
			for j, item := range chunk {
				// Convert int64 vector to float32 for Qdrant
				vector := make([]float32, len(item.Vector))
				for k, v := range item.Vector {
					vector[k] = float32(v)
				}

				searchRequests[j] = &qdrant.SearchPoints{
					CollectionName: config.CollectionName,
					Vector:         vector,
					Limit:          uint64(config.SearchLimit),
					ScoreThreshold: &config.SearchThreshold,
					WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: false}},
				}
			}

			// Perform batch search
			var batchResp *qdrant.SearchBatchResponse
			var err error

			batchReq := &qdrant.SearchBatchPoints{
				CollectionName: config.CollectionName,
				SearchPoints:   searchRequests,
			}

			for attempt := 0; attempt <= 3; attempt++ {
				batchResp, err = pointsClient.SearchBatch(ctx, batchReq)
				if err == nil {
					break
				}
				if attempt == 3 {
					log.Printf("❌ Failed to search batch %d after %d retries: %v", i/config.QdrantBatchSize+1, 3, err)
					return fmt.Errorf("batch search failed after %d retries: %w", 3, err)
				}
				log.Printf("⚠️ Batch search attempt %d failed, retrying...", attempt+1)
			}

			// Process results
			chunkDuplicates := 0
			for j, result := range batchResp.GetResult() {
				itemId := chunk[j].Id
				itemDuplicates := 0

				for _, scored := range result.GetResult() {
					neighborId := int64(scored.Id.GetNum())
					if neighborId != itemId { // Don't count self-matches
						uf.Union(itemId, neighborId)
						itemDuplicates++
					}
				}

				// Count actual duplicate items (not items with duplicates)
				chunkDuplicates += itemDuplicates
			}

			duplicatesFound += chunkDuplicates
			itemCount += len(chunk)

			log.Printf("✅ Chunk %d-%d completed: %d duplicates found", i, end-1, chunkDuplicates)
		}

		log.Printf("📊 Batch %d completed: processed %d items, total duplicates: %d (%.1f%%)",
			batchCount, itemCount, duplicatesFound, float64(duplicatesFound)/float64(itemCount)*100)
	}

	if err := reader.Err(); err != nil {
		return err
	}

	log.Printf("✅ Completed processing %d items", itemCount)

	// Generate results and collect statistics
	log.Printf("📊 Generating results and collecting statistics...")

	// Generate results for each item ensuring we have exactly the same count as input
	results := make([]*ResultItem, 0, itemCount)

	// Statistics collection
	repToMembers := make(map[int64][]int64)

	for _, id := range allIDS {
		repId := uf.Find(id)
		// If no duplicates found, repId should be equal to id
		if repId == id {
			results = append(results, &ResultItem{
				Id:    id,
				RepId: id, // No duplicates found, use item itself as representative
			})
		} else {
			results = append(results, &ResultItem{
				Id:    id,
				RepId: repId, // Use the root representative
			})
		}

		// Collect statistics
		if _, exists := repToMembers[repId]; !exists {
			repToMembers[repId] = []int64{}
		}
		repToMembers[repId] = append(repToMembers[repId], id)
	}

	// Verify that we have the same number of results as input items
	if len(results) != itemCount {
		log.Printf("⚠️ Warning: processed %d items but generated %d results", itemCount, len(results))
	}

	// Log statistics
	totalGroups := len(repToMembers)
	singletons := 0
	duplicateGroups := 0
	totalDuplicates := 0
	maxGroupSize := 0

	for _, members := range repToMembers {
		groupSize := len(members)
		if groupSize > maxGroupSize {
			maxGroupSize = groupSize
		}

		if groupSize == 1 {
			singletons++
		} else {
			duplicateGroups++
			totalDuplicates += groupSize - 1 // exclude the representative itself
		}
	}

	log.Printf("📈 Duplicate Analysis:")
	log.Printf("   Total items processed: %d", itemCount)
	log.Printf("   Total groups found: %d", totalGroups)
	log.Printf("   Singletons (no duplicates): %d (%.1f%%)", singletons, float64(singletons)/float64(itemCount)*100)
	log.Printf("   Duplicate groups: %d", duplicateGroups)
	log.Printf("   Total duplicate items: %d", totalDuplicates)
	log.Printf("   Largest duplicate group size: %d", maxGroupSize)
	log.Printf("   Duplication rate: %.2f%%", float64(totalDuplicates)/float64(itemCount)*100)

	// Create output table if it doesn't exist
	log.Printf("Creating output table: %s", config.OutputTable)
	if err := createOutputTable(ctx, yc, config.OutputTable); err != nil {
		return fmt.Errorf("failed to create output table: %w", err)
	}

	// Write results to output table
	log.Printf("Writing %d results to output table: %s", len(results), config.OutputTable)
	// outputPath := ypath.Path(config.OutputTable)

	// writer, err := yt.WriteTable(ctx, yc, outputPath, nil)
	// if err != nil {
	// 	return err
	// }

	// for _, result := range results {
	// 	if err := writer.Write(result); err != nil {
	// 		return err
	// 	}
	// }
	// writer.Commit()

	clusterFile := "output_clusters.jsonl"
	writeToJSONL(clusterFile, results)

	log.Printf("Successfully completed processing")
	return nil
}

func createOutputTable(ctx context.Context, yc yt.Client, tablePath string) error {
	// Check if table already exists
	exists, err := yc.NodeExists(ctx, ypath.Path(tablePath), nil)
	if err != nil {
		return fmt.Errorf("failed to check if table exists: %w", err)
	}

	if exists {
		log.Printf("Table %s already exists", tablePath)
		return nil
	}

	// Create table with schema using Node command with attributes
	attributes := map[string]interface{}{
		"dynamic": false,
		"schema": []map[string]interface{}{
			{
				"name": "id",
				"type": "int64",
			},
			{
				"name": "rep_id",
				"type": "int64",
			},
		},
	}

	_, err = yc.CreateNode(ctx, ypath.Path(tablePath), yt.NodeTable, &yt.CreateNodeOptions{
		Attributes: attributes,
	})
	if err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}

	log.Printf("Successfully created table %s", tablePath)
	return nil
}

// Utility functions
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := parseInt(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvFloat32OrDefault(key string, defaultValue float32) float32 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := parseFloat32(value); err == nil {
			return floatValue
		}
	}
	return defaultValue
}

func parseInt(s string) (int, error) {
	var result int
	_, err := fmt.Sscanf(s, "%d", &result)
	return result, err
}

func parseFloat32(s string) (float32, error) {
	var result float32
	_, err := fmt.Sscanf(s, "%f", &result)
	return result, err
}

// UnionFind implementation
type UnionFind struct {
	parent map[int64]int64
	rank   map[int64]int
}

func NewUnionFind() *UnionFind {
	return &UnionFind{
		parent: make(map[int64]int64),
		rank:   make(map[int64]int),
	}
}

func (uf *UnionFind) MakeSet(x int64) {
	if _, exists := uf.parent[x]; !exists {
		uf.parent[x] = x
		uf.rank[x] = 0
	}
}

func (uf *UnionFind) Find(x int64) int64 {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int64) {
	uf.MakeSet(x)
	uf.MakeSet(y)

	rootX := uf.Find(x)
	rootY := uf.Find(y)

	if rootX == rootY {
		return
	}

	if uf.rank[rootX] < uf.rank[rootY] {
		uf.parent[rootX] = rootY
	} else if uf.rank[rootX] > uf.rank[rootY] {
		uf.parent[rootY] = rootX
	} else {
		uf.parent[rootY] = rootX
		uf.rank[rootX]++
	}
}

func (uf *UnionFind) Size() int {
	return len(uf.parent)
}

func writeToJSONL(filename string, results []*ResultItem) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)

	for _, item := range results {
		if err := encoder.Encode(item); err != nil {
			return err
		}
	}

	return nil
}
