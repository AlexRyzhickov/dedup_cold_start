package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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

type NeighborsResult struct {
	ItemID      int64
	NeighborIDs []int64
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
	SearchWorkers   int
}

func main() {
	defaultWorkers := runtime.NumCPU() * 2
	if defaultWorkers < 4 {
		defaultWorkers = 4
	}

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
		SearchWorkers:   getEnvIntOrDefault("SEARCH_WORKERS", defaultWorkers),
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
	if config.SearchWorkers <= 0 {
		log.Fatal("SEARCH_WORKERS must be > 0")
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
	defer conn.Close()

	ctx := context.Background()

	readStart := time.Now()
	items, err := readItemsFromHDF5(ds, totalRows, dim, config)
	if err != nil {
		return err
	}
	readElapsed := time.Since(readStart)
	log.Printf("⏱️ Phase 1/3 (read vectors): items=%d elapsed=%s", len(items), readElapsed.Round(time.Millisecond))

	searchStart := time.Now()
	neighborsByID, ioDuplicates, err := collectNeighborsForItems(ctx, items, config, pointsClient)
	if err != nil {
		return err
	}
	searchElapsed := time.Since(searchStart)
	log.Printf("⏱️ Phase 2/3 (nearest neighbors): items=%d edges=%d elapsed=%s", len(items), ioDuplicates, searchElapsed.Round(time.Millisecond))

	// Create UnionFind
	unionStart := time.Now()
	uf := NewUnionFind()
	allIDS := make([]int64, 0, len(items))
	for _, item := range items {
		allIDS = append(allIDS, item.Id)
	}
	for _, id := range allIDS {
		uf.MakeSet(id)
	}

	unionCount := 0
	for _, id := range allIDS {
		for _, neighborID := range neighborsByID[id] {
			uf.Union(id, neighborID)
			unionCount++
		}
	}
	itemCount := len(allIDS)
	unionElapsed := time.Since(unionStart)

	log.Printf("⏱️ Phase 3/3 (UnionFind): items=%d unions=%d elapsed=%s", itemCount, unionCount, unionElapsed.Round(time.Millisecond))
	log.Printf("⏱️ Total elapsed (3 phases): %s", (readElapsed + searchElapsed + unionElapsed).Round(time.Millisecond))

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
	singletonPct := 0.0
	duplicationRate := 0.0
	if itemCount > 0 {
		singletonPct = float64(singletons) / float64(itemCount) * 100
		duplicationRate = float64(totalDuplicates) / float64(itemCount) * 100
	}
	log.Printf("   Singletons (no duplicates): %d (%.1f%%)", singletons, singletonPct)
	log.Printf("   Duplicate groups: %d", duplicateGroups)
	log.Printf("   Total duplicate items: %d", totalDuplicates)
	log.Printf("   Largest duplicate group size: %d", maxGroupSize)
	log.Printf("   Duplication rate: %.2f%%", duplicationRate)

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

func readItemsFromHDF5(ds *hdf5.Dataset, totalRows, dim int, config *Config) ([]*SearchItem, error) {
	items := make([]*SearchItem, 0, min(totalRows, config.ProcSize))
	batchCount := 0

	for offset := 0; offset < totalRows && len(items) < config.ProcSize; offset += config.HDF5ReadBatch {
		nRows := config.HDF5ReadBatch
		if remain := totalRows - offset; remain < nRows {
			nRows = remain
		}
		if remain := config.ProcSize - len(items); remain < nRows {
			nRows = remain
		}
		if nRows <= 0 {
			break
		}

		raw, err := ds.ReadSlice([]uint64{uint64(offset), 0}, []uint64{uint64(nRows), uint64(dim)})
		if err != nil {
			return nil, fmt.Errorf("read slice offset=%d count=%d: %w", offset, nRows, err)
		}
		buf, err := toFloat32Slice(raw, nRows*dim)
		if err != nil {
			return nil, fmt.Errorf("convert batch offset=%d count=%d: %w", offset, nRows, err)
		}

		batch := make([]*SearchItem, 0, nRows)
		for i := 0; i < nRows; i++ {
			start := i * dim
			end := start + dim
			vector := make([]float32, dim)
			copy(vector, buf[start:end])

			id := config.StartID + int64(offset+i)
			batch = append(batch, &SearchItem{
				Id:     id,
				Vector: vector,
			})
		}
		items = append(items, batch...)
		batchCount++
		log.Printf("📦 Read batch %d with %d items", batchCount, len(batch))
	}

	return items, nil
}

func collectNeighborsForItems(
	ctx context.Context,
	items []*SearchItem,
	config *Config,
	pointsClient qdrant.PointsClient,
) (map[int64][]int64, int, error) {
	type searchChunk struct {
		Items []*SearchItem
	}

	neighborsByID := make(map[int64][]int64)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	workCh := make(chan searchChunk, config.SearchWorkers*2)
	resultCh := make(chan []NeighborsResult, config.SearchWorkers*2)
	errCh := make(chan error, 1)

	sendErr := func(err error) {
		select {
		case errCh <- err:
		default:
		}
		cancel()
	}

	var duplicateEdges atomic.Int64

	var workersWG sync.WaitGroup
	for w := 0; w < config.SearchWorkers; w++ {
		workersWG.Add(1)
		go func(workerID int) {
			defer workersWG.Done()

			for chunk := range workCh {
				select {
				case <-ctx.Done():
					return
				default:
				}

				results, edges, err := searchChunkNeighbors(ctx, pointsClient, config, chunk.Items)
				if err != nil {
					sendErr(fmt.Errorf("worker=%d search chunk failed: %w", workerID, err))
					return
				}
				duplicateEdges.Add(int64(edges))

				select {
				case <-ctx.Done():
					return
				case resultCh <- results:
				}
			}
		}(w)
	}

	collectorDone := make(chan struct{})
	go func() {
		defer close(collectorDone)
		for chunkResults := range resultCh {
			for _, r := range chunkResults {
				neighborsByID[r.ItemID] = r.NeighborIDs
			}
		}
	}()

	for i := 0; i < len(items); i += config.QdrantBatchSize {
		end := i + config.QdrantBatchSize
		if end > len(items) {
			end = len(items)
		}
		chunk := searchChunk{Items: items[i:end]}

		select {
		case <-ctx.Done():
			close(workCh)
			workersWG.Wait()
			close(resultCh)
			<-collectorDone
			select {
			case err := <-errCh:
				return nil, 0, err
			default:
				return nil, 0, ctx.Err()
			}
		case workCh <- chunk:
		}
	}

	close(workCh)
	workersWG.Wait()
	close(resultCh)
	<-collectorDone

	select {
	case err := <-errCh:
		return nil, 0, err
	default:
	}

	return neighborsByID, int(duplicateEdges.Load()), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func searchChunkNeighbors(
	ctx context.Context,
	pointsClient qdrant.PointsClient,
	config *Config,
	items []*SearchItem,
) ([]NeighborsResult, int, error) {
	searchRequests := make([]*qdrant.SearchPoints, len(items))
	for i, item := range items {
		searchRequests[i] = &qdrant.SearchPoints{
			CollectionName: config.CollectionName,
			Vector:         item.Vector,
			Limit:          uint64(config.SearchLimit),
			ScoreThreshold: &config.SearchThreshold,
			WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: false}},
		}
	}

	batchReq := &qdrant.SearchBatchPoints{
		CollectionName: config.CollectionName,
		SearchPoints:   searchRequests,
	}

	var (
		batchResp *qdrant.SearchBatchResponse
		err       error
	)
	for attempt := 0; attempt <= 3; attempt++ {
		batchResp, err = pointsClient.SearchBatch(ctx, batchReq)
		if err == nil {
			break
		}
		if attempt == 3 {
			return nil, 0, fmt.Errorf("batch search failed after %d retries: %w", 3, err)
		}
		log.Printf("⚠️ Batch search attempt %d failed, retrying...", attempt+1)
	}

	results := make([]NeighborsResult, 0, len(items))
	edges := 0

	respResults := batchResp.GetResult()
	for i := range items {
		itemID := items[i].Id
		neighborIDs := make([]int64, 0)

		if i < len(respResults) {
			for _, scored := range respResults[i].GetResult() {
				neighborID := int64(scored.Id.GetNum())
				if neighborID != itemID {
					neighborIDs = append(neighborIDs, neighborID)
					edges++
				}
			}
		}

		results = append(results, NeighborsResult{
			ItemID:      itemID,
			NeighborIDs: neighborIDs,
		})
	}

	return results, edges, nil
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
