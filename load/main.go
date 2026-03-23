package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/url"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/qdrant/go-client/qdrant"
	hdf5 "github.com/scigolib/hdf5"
)

type config struct {
	HDF5Path     string
	DatasetName  string
	QdrantURL    string
	APIKey       string
	Collection   string
	BatchSize    int
	Workers      int
	StartID      uint64
	RequestTO    time.Duration
	PollInterval time.Duration

	IndexingThreshold int
}

type batchData struct {
	StartID uint64
	Rows    int
	Dim     int
	Data    []float32
}

type qdrantClient struct {
	client    *qdrant.Client
	requestTO time.Duration
}

func main() {
	cfg := parseFlags()
	validateConfig(cfg)

	log.Printf("Start: dataset=%s file=%s collection=%s", cfg.DatasetName, cfg.HDF5Path, cfg.Collection)
	log.Printf("Upload params: batch=%d workers=%d start_id=%d", cfg.BatchSize, cfg.Workers, cfg.StartID)

	client := newQdrantClient(cfg)
	defer client.client.Close()
	ctx := context.Background()

	start := time.Now()
	totalRows, dim, err := uploadAllVectors(ctx, cfg, client)
	if err != nil {
		log.Fatalf("upload failed: %v", err)
	}
	log.Printf("Upload complete: rows=%d dim=%d elapsed=%s", totalRows, dim, time.Since(start).Round(time.Second))

	if err := enableIndexing(ctx, cfg, client); err != nil {
		log.Fatalf("failed to enable indexing: %v", err)
	}
	log.Printf("Indexing enabled. Waiting for collection %q to become green...", cfg.Collection)

	if err := waitUntilGreen(ctx, cfg, client); err != nil {
		log.Fatalf("indexing wait failed: %v", err)
	}

	log.Printf("Done. Collection %q is green.", cfg.Collection)
}

func parseFlags() config {
	defaultWorkers := runtime.NumCPU() * 2
	if defaultWorkers < 4 {
		defaultWorkers = 4
	}

	cfg := config{}
	flag.StringVar(&cfg.HDF5Path, "hdf5-path", "/home/alex/RustroverProjects/vector-db-benchmark_mine/datasets/deep-image-96-angular/deep-image-96-angular.hdf5", "Path to HDF5 file")
	flag.StringVar(&cfg.DatasetName, "dataset", "train", "Dataset name inside HDF5 file")
	flag.StringVar(&cfg.QdrantURL, "qdrant-url", "127.0.0.1:6334", "Qdrant endpoint (grpc host:port or URL)")
	flag.StringVar(&cfg.APIKey, "api-key", "", "Qdrant API key (optional)")
	flag.StringVar(&cfg.Collection, "collection", "", "Target Qdrant collection name")
	flag.IntVar(&cfg.BatchSize, "batch-size", 2048, "Vectors per upsert request")
	flag.IntVar(&cfg.Workers, "workers", defaultWorkers, "Parallel upload workers")
	flag.Uint64Var(&cfg.StartID, "start-id", 0, "First point ID")
	flag.DurationVar(&cfg.RequestTO, "request-timeout", 90*time.Second, "gRPC request timeout")
	flag.DurationVar(&cfg.PollInterval, "poll-interval", 10*time.Second, "Collection status polling interval")
	flag.IntVar(&cfg.IndexingThreshold, "indexing-threshold", 20000, "Indexing threshold to enable after upload")
	flag.Parse()

	return cfg
}

func validateConfig(cfg config) {
	if strings.TrimSpace(cfg.Collection) == "" {
		log.Fatal("required flag is missing: -collection")
	}
	if cfg.BatchSize <= 0 {
		log.Fatal("-batch-size must be > 0")
	}
	if cfg.Workers <= 0 {
		log.Fatal("-workers must be > 0")
	}
	if cfg.IndexingThreshold <= 0 {
		log.Fatal("-indexing-threshold must be > 0")
	}
}

func newQdrantClient(cfg config) *qdrantClient {
	host, port, useTLS := parseQdrantEndpoint(cfg.QdrantURL)

	poolSize := uint(cfg.Workers)
	if poolSize == 0 {
		poolSize = 1
	}
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:     host,
		Port:     port,
		APIKey:   cfg.APIKey,
		UseTLS:   useTLS,
		PoolSize: poolSize,
	})
	if err != nil {
		log.Fatalf("create qdrant client: %v", err)
	}

	return &qdrantClient{
		client:    client,
		requestTO: cfg.RequestTO,
	}
}

func parseQdrantEndpoint(raw string) (host string, port int, useTLS bool) {
	const defaultPort = 6334

	s := strings.TrimSpace(raw)
	if s == "" {
		return "127.0.0.1", defaultPort, false
	}

	if !strings.Contains(s, "://") {
		// host:port or host
		host = s
		if strings.Contains(host, ":") {
			parts := strings.Split(host, ":")
			if len(parts) >= 2 {
				p, err := strconv.Atoi(parts[len(parts)-1])
				if err == nil {
					port = p
					host = strings.Join(parts[:len(parts)-1], ":")
				}
			}
		}
		if host == "" {
			host = "127.0.0.1"
		}
		if port == 0 {
			port = defaultPort
		}
		if port == 6333 {
			// User often passes REST port by habit. gRPC default is 6334.
			port = 6334
		}
		return host, port, false
	}

	u, err := url.Parse(s)
	if err != nil {
		log.Fatalf("invalid -qdrant-url %q: %v", raw, err)
	}

	host = u.Hostname()
	if host == "" {
		host = "127.0.0.1"
	}
	useTLS = strings.EqualFold(u.Scheme, "https") || strings.EqualFold(u.Scheme, "grpcs")
	port = defaultPort
	if p := u.Port(); p != "" {
		pp, err := strconv.Atoi(p)
		if err != nil {
			log.Fatalf("invalid qdrant port in %q: %v", raw, err)
		}
		port = pp
	} else if strings.EqualFold(u.Scheme, "http") && port == defaultPort {
		// Preserve explicit HTTP scheme intention when port is omitted.
		port = 6333
	}
	if port == 6333 {
		port = 6334
	}
	return host, port, useTLS
}

func uploadAllVectors(ctx context.Context, cfg config, client *qdrantClient) (int, int, error) {
	file, err := hdf5.Open(cfg.HDF5Path)
	if err != nil {
		return 0, 0, fmt.Errorf("open hdf5 file: %w", err)
	}
	defer file.Close()

	ds, foundPath, err := findDataset(file, cfg.DatasetName)
	if err != nil {
		return 0, 0, err
	}

	totalRows, dim, err := inferDatasetShape(ds)
	if err != nil {
		return 0, 0, err
	}

	log.Printf("Dataset found: path=%s rows=%d dim=%d", foundPath, totalRows, dim)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	batchCh := make(chan batchData, cfg.Workers*2)
	errCh := make(chan error, cfg.Workers+1)

	var uploaded atomic.Uint64
	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for b := range batchCh {
				if err := uploadBatch(ctx, client, cfg.Collection, b); err != nil {
					select {
					case errCh <- fmt.Errorf("worker=%d upsert failed: %w", workerID, err):
					default:
					}
					cancel()
					return
				}

				current := uploaded.Add(uint64(b.Rows))
				if current%uint64(cfg.BatchSize*10) == 0 || current == uint64(totalRows) {
					pct := float64(current) / float64(totalRows) * 100
					log.Printf("Uploaded %d/%d vectors (%.2f%%)", current, totalRows, pct)
				}
			}
		}(i)
	}

readLoop:
	for offset := 0; offset < totalRows; offset += cfg.BatchSize {
		select {
		case <-ctx.Done():
			break readLoop
		default:
		}

		nRows := cfg.BatchSize
		if remain := totalRows - offset; remain < nRows {
			nRows = remain
		}

		raw, err := ds.ReadSlice([]uint64{uint64(offset), 0}, []uint64{uint64(nRows), uint64(dim)})
		if err != nil {
			cancel()
			wg.Wait()
			return totalRows, dim, fmt.Errorf("read slice offset=%d count=%d: %w", offset, nRows, err)
		}

		buf, err := toFloat32Slice(raw, nRows*dim)
		if err != nil {
			cancel()
			wg.Wait()
			return totalRows, dim, fmt.Errorf("convert batch offset=%d count=%d: %w", offset, nRows, err)
		}

		b := batchData{
			StartID: cfg.StartID + uint64(offset),
			Rows:    nRows,
			Dim:     dim,
			Data:    buf,
		}

		select {
		case <-ctx.Done():
			break readLoop
		case batchCh <- b:
		}
	}

	close(batchCh)
	wg.Wait()

	select {
	case err := <-errCh:
		return totalRows, dim, err
	default:
	}

	return totalRows, dim, nil
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

	// Fallback for non-chunked layouts:
	// derive shape via bounds probing with tiny ReadSlice calls.
	dim, err := inferUpperBoundAxis(ds, 1, 1)
	if err != nil {
		return 0, 0, fmt.Errorf("infer dim: %w", err)
	}
	rows, err := inferUpperBoundAxis(ds, 0, dim)
	if err != nil {
		return 0, 0, fmt.Errorf("infer rows: %w", err)
	}
	return rows, dim, nil
}

func inferUpperBoundAxis(ds *hdf5.Dataset, axis int, fixedOtherAxis int) (int, error) {
	canReadIndex := func(idx int) bool {
		if idx < 0 {
			return false
		}
		// Read exactly one element at the tested index.
		// This keeps probing O(1) in memory/time per check.
		start := []uint64{0, 0}
		count := []uint64{1, 1}
		if axis == 0 {
			start[0] = uint64(idx)
			start[1] = 0
		} else {
			start[0] = 0
			start[1] = uint64(idx)
		}
		_, err := ds.ReadSlice(start, count)
		return err == nil
	}

	// Validate axis has at least one element.
	if fixedOtherAxis <= 0 {
		fixedOtherAxis = 1
	}
	if axis == 0 {
		_, err := ds.ReadSlice([]uint64{0, 0}, []uint64{1, uint64(fixedOtherAxis)})
		if err != nil {
			return 0, fmt.Errorf("axis %d has invalid size", axis)
		}
	} else {
		_, err := ds.ReadSlice([]uint64{0, 0}, []uint64{1, 1})
		if err != nil {
			return 0, fmt.Errorf("axis %d has invalid size", axis)
		}
	}

	if !canReadIndex(0) {
		return 0, fmt.Errorf("axis %d has invalid size", axis)
	}

	// Search over index domain, then size = last_valid_index + 1.
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

func uploadBatch(ctx context.Context, client *qdrantClient, collection string, b batchData) error {
	points := make([]*qdrant.PointStruct, 0, b.Rows)

	for i := 0; i < b.Rows; i++ {
		start := i * b.Dim
		end := start + b.Dim
		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDNum(b.StartID + uint64(i)),
			Vectors: qdrant.NewVectors(b.Data[start:end]...),
		})
	}

	wait := false
	callCtx, cancel := context.WithTimeout(ctx, client.requestTO)
	defer cancel()

	_, err := client.client.Upsert(callCtx, &qdrant.UpsertPoints{
		CollectionName: collection,
		Wait:           &wait,
		Points:         points,
	})
	if err != nil {
		return fmt.Errorf("qdrant upsert: %w", err)
	}
	return nil
}

func enableIndexing(ctx context.Context, cfg config, client *qdrantClient) error {
	indexingThreshold := uint64(cfg.IndexingThreshold)
	req := &qdrant.UpdateCollection{
		CollectionName: cfg.Collection,
		OptimizersConfig: &qdrant.OptimizersConfigDiff{
			IndexingThreshold: &indexingThreshold,
		},
	}

	callCtx, cancel := context.WithTimeout(ctx, client.requestTO)
	defer cancel()

	if err := client.client.UpdateCollection(callCtx, req); err != nil {
		return fmt.Errorf("qdrant update collection: %w", err)
	}
	return nil
}

func waitUntilGreen(ctx context.Context, cfg config, client *qdrantClient) error {
	ticker := time.NewTicker(cfg.PollInterval)
	defer ticker.Stop()

	for {
		callCtx, cancel := context.WithTimeout(ctx, client.requestTO)
		info, err := client.client.GetCollectionInfo(callCtx, cfg.Collection)
		cancel()
		if err != nil {
			return fmt.Errorf("get collection info: %w", err)
		}

		status := info.GetStatus()
		optimizer := info.GetOptimizerStatus()
		optimizerOK := optimizer == nil || optimizer.GetOk()

		optState := "nil"
		if optimizer != nil {
			if optimizer.GetOk() {
				optState = "ok"
			} else {
				optState = optimizer.GetError()
			}
		}
		log.Printf(
			"Collection status=%s optimizer=%s vectors=%d indexed=%d",
			status.String(),
			optState,
			info.GetPointsCount(),
			info.GetIndexedVectorsCount(),
		)

		if status == qdrant.CollectionStatus_Green && optimizerOK {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
