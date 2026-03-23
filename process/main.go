package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/qdrant/go-client/qdrant"
	hdf5 "github.com/scigolib/hdf5"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type SearchItem struct {
	Id     int64
	Vector []float32
}

type ResultItem struct {
	Id    int64 `json:"id"`
	RepId int64 `json:"rep_id"`
}

type Config struct {
	HDF5Path        string
	DatasetName     string
	OutputFile      string
	QdrantHost      string
	QdrantPort      int
	CollectionName  string
	SearchLimit     int
	SearchThreshold float32
	QdrantBatchSize int
	HDF5ReadBatch   int
	ProcSize        int
	StartID         int64
}

func main() {
	config := &Config{
		HDF5Path:        getEnvOrDefault("HDF5_PATH", "/home/alex/RustroverProjects/vector-db-benchmark_mine/datasets/deep-image-96-angular/deep-image-96-angular.hdf5"),
		DatasetName:     getEnvOrDefault("HDF5_DATASET", "train"),
		OutputFile:      getEnvOrDefault("OUTPUT_FILE", "output_clusters.jsonl"),
		QdrantHost:      getEnvOrDefault("QDRANT_HOST", "qdrant.dedup.clouds.idzn.ru"),
		QdrantPort:      getEnvIntOrDefault("QDRANT_PORT", 6334),
		CollectionName:  getEnvOrDefault("COLLECTION_NAME", "VkVideoZeldaVideoFusedVkV2"),
		SearchLimit:     getEnvIntOrDefault("SEARCH_LIMIT", 64),
		SearchThreshold: getEnvFloat32OrDefault("SEARCH_THRESHOLD", 0.97),
		QdrantBatchSize: getEnvIntOrDefault("Q_BATCH_SIZE", 1000),
		HDF5ReadBatch:   getEnvIntOrDefault("HDF5_BATCH_SIZE", 1000),
		ProcSize:        getEnvIntOrDefault("PROC_SIZE", 1000000000000),
		StartID:         getEnvInt64OrDefault("START_ID", 0),
	}

	if config.HDF5Path == "" || config.DatasetName == "" {
		log.Fatal("HDF5_PATH and HDF5_DATASET environment variables must be set")
	}
	if config.HDF5ReadBatch <= 0 {
		log.Fatal("HDF5_BATCH_SIZE must be > 0")
	}
	if config.QdrantBatchSize <= 0 {
		log.Fatal("Q_BATCH_SIZE must be > 0")
	}
	if config.ProcSize <= 0 {
		log.Fatal("PROC_SIZE must be > 0")
	}

	if err := process(config); err != nil {
		log.Fatalf("Error: %v", err)
	}
}

func process(config *Config) error {
	log.Printf("Starting processing with config: %+v", config)

	file, err := hdf5.Open(config.HDF5Path)
	if err != nil {
		return fmt.Errorf("open hdf5 file: %w", err)
	}
	defer file.Close()

	ds, foundPath, err := findDataset(file, config.DatasetName)
	if err != nil {
		return err
	}
	totalRows, dim, err := inferDatasetShape(ds)
	if err != nil {
		return err
	}
	log.Printf("Dataset found: path=%s rows=%d dim=%d", foundPath, totalRows, dim)

	conn, err := grpc.NewClient(config.QdrantHost, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect to Qdrant: %w", err)
	}

	pointsClient := qdrant.NewPointsClient(conn)

	ctx := context.Background()

	// Create UnionFind
	uf := NewUnionFind()

	allIDS := make([]int64, 0)
	itemCount := 0
	duplicatesFound := 0
	batchCount := 0

	for offset := 0; offset < totalRows; offset += config.HDF5ReadBatch {
		if len(allIDS) >= config.ProcSize {
			break
		}

		nRows := config.HDF5ReadBatch
		if remain := totalRows - offset; remain < nRows {
			nRows = remain
		}
		if allow := config.ProcSize - len(allIDS); allow < nRows {
			nRows = allow
		}
		if nRows <= 0 {
			break
		}

		raw, err := ds.ReadSlice([]uint64{uint64(offset), 0}, []uint64{uint64(nRows), uint64(dim)})
		if err != nil {
			return fmt.Errorf("read slice offset=%d count=%d: %w", offset, nRows, err)
		}
		buf, err := toFloat32Slice(raw, nRows*dim)
		if err != nil {
			return fmt.Errorf("convert batch offset=%d count=%d: %w", offset, nRows, err)
		}

		batch := make([]*SearchItem, 0, nRows)
		for i := 0; i < nRows; i++ {
			start := i * dim
			end := start + dim
			vector := make([]float32, dim)
			copy(vector, buf[start:end])
			batch = append(batch, &SearchItem{
				Id:     config.StartID + int64(offset+i),
				Vector: vector,
			})
		}

		batchCount++
		log.Printf("📦 Read batch %d with %d items", batchCount, len(batch))

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
				searchRequests[j] = &qdrant.SearchPoints{
					CollectionName: config.CollectionName,
					Vector:         item.Vector,
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

	log.Printf("Writing %d results to output file: %s", len(results), config.OutputFile)
	if err := writeToJSONL(config.OutputFile, results); err != nil {
		return fmt.Errorf("failed to write output file: %w", err)
	}

	log.Printf("Successfully completed processing")
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

func getEnvInt64OrDefault(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if intValue, err := parseInt64(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func parseInt(s string) (int, error) {
	var result int
	_, err := fmt.Sscanf(s, "%d", &result)
	return result, err
}

func parseInt64(s string) (int64, error) {
	var result int64
	_, err := fmt.Sscanf(s, "%d", &result)
	return result, err
}

func parseFloat32(s string) (float32, error) {
	var result float32
	_, err := fmt.Sscanf(s, "%f", &result)
	return result, err
}

func findDataset(file *hdf5.File, datasetRef string) (*hdf5.Dataset, string, error) {
	target := strings.TrimSpace(datasetRef)
	targetNoSlash := strings.TrimPrefix(target, "/")
	var (
		match     *hdf5.Dataset
		matchPath string
	)

	file.Walk(func(path string, obj hdf5.Object) {
		if match != nil {
			return
		}
		ds, ok := obj.(*hdf5.Dataset)
		if !ok {
			return
		}
		pathNoSlash := strings.TrimPrefix(path, "/")
		if path == target || pathNoSlash == targetNoSlash || ds.Name() == targetNoSlash || ds.Name() == target {
			match = ds
			matchPath = path
		}
	})

	if match == nil {
		return nil, "", fmt.Errorf("dataset %q not found in file", datasetRef)
	}
	return match, matchPath, nil
}

func inferDatasetShape(ds *hdf5.Dataset) (int, int, error) {
	iter, err := ds.ChunkIterator()
	if err == nil {
		dims := iter.DatasetDims()
		if len(dims) == 2 && dims[0] > 0 && dims[1] > 0 {
			return int(dims[0]), int(dims[1]), nil
		}
	}

	dim, err := inferUpperBoundAxis(ds, 1)
	if err != nil {
		return 0, 0, fmt.Errorf("infer dim: %w", err)
	}
	rows, err := inferUpperBoundAxis(ds, 0)
	if err != nil {
		return 0, 0, fmt.Errorf("infer rows: %w", err)
	}
	return rows, dim, nil
}

func inferUpperBoundAxis(ds *hdf5.Dataset, axis int) (int, error) {
	canReadIndex := func(idx int) bool {
		if idx < 0 {
			return false
		}
		start := []uint64{0, 0}
		count := []uint64{1, 1}
		if axis == 0 {
			start[0] = uint64(idx)
		} else {
			start[1] = uint64(idx)
		}
		_, err := ds.ReadSlice(start, count)
		return err == nil
	}

	if !canReadIndex(0) {
		return 0, fmt.Errorf("axis %d has invalid size", axis)
	}

	lo, hi := 0, 1
	for canReadIndex(hi) {
		lo = hi
		if hi > (1 << 30) {
			break
		}
		hi *= 2
	}

	for lo+1 < hi {
		mid := lo + (hi-lo)/2
		if canReadIndex(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}

	return lo + 1, nil
}

func toFloat32Slice(v any, expected int) ([]float32, error) {
	switch arr := v.(type) {
	case []float64:
		if len(arr) != expected {
			return nil, fmt.Errorf("unexpected slice size: got=%d expected=%d", len(arr), expected)
		}
		out := make([]float32, len(arr))
		for i := range arr {
			out[i] = float32(arr[i])
		}
		return out, nil
	case []float32:
		if len(arr) != expected {
			return nil, fmt.Errorf("unexpected slice size: got=%d expected=%d", len(arr), expected)
		}
		out := make([]float32, len(arr))
		copy(out, arr)
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported data type from ReadSlice: %T", v)
	}
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
