// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

import (
	"fmt"
	"math/big"
	"math/rand"
	"testing"

	"github.com/luxfi/evmgpu/core/state"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/core/vm"
	ethparams "github.com/luxfi/geth/params"
	"github.com/luxfi/ids"
)

// mockApplyFn produces deterministic receipts without requiring real state.
func mockApplyFn(gasUsed uint64) func(
	*ethparams.ChainConfig,
	*types.Header,
	*types.Transaction,
	*state.StateDB,
	vm.Config,
	int,
) (*types.Receipt, error) {
	return func(
		_ *ethparams.ChainConfig,
		_ *types.Header,
		tx *types.Transaction,
		_ *state.StateDB,
		_ vm.Config,
		_ int,
	) (*types.Receipt, error) {
		return &types.Receipt{
			Type:    tx.Type(),
			Status:  types.ReceiptStatusSuccessful,
			TxHash:  tx.Hash(),
			GasUsed: gasUsed,
		}, nil
	}
}

func testHeader() *types.Header {
	return &types.Header{
		Number:   big.NewInt(100),
		GasLimit: 30_000_000,
		Time:     1700000000,
		Coinbase: common.HexToAddress("0x1111111111111111111111111111111111111111"),
		BaseFee:  big.NewInt(1_000_000_000),
	}
}

// --- DAGExecutor tests ---

func TestDAGExecutorExecuteBlockEmpty(t *testing.T) {
	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 4,
	})

	receipts, err := exec.ExecuteBlock(nil, testHeader(), nil, nil, vm.Config{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if receipts != nil {
		t.Fatal("expected nil receipts for empty block")
	}
}

func TestDAGExecutorExecuteBlockLinear(t *testing.T) {
	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 4,
	})

	txs := types.Transactions{
		testTx(0, common.HexToAddress("0xaaaa")),
		testTx(1, common.HexToAddress("0xbbbb")),
		testTx(2, common.HexToAddress("0xcccc")),
	}

	receipts, err := exec.ExecuteBlock(nil, testHeader(), txs, nil, vm.Config{})
	if err != nil {
		t.Fatalf("ExecuteBlock failed: %v", err)
	}
	if len(receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(receipts))
	}
	for i, r := range receipts {
		if r.Status != types.ReceiptStatusSuccessful {
			t.Fatalf("receipt %d: expected success", i)
		}
	}
}

func TestDAGExecutorExecuteAntichain(t *testing.T) {
	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 4,
	})

	// Create 3 non-conflicting vertices.
	vertices := make([]*EVMVertex, 3)
	for i := 0; i < 3; i++ {
		txs := []*types.Transaction{
			testTx(uint64(i*10), common.HexToAddress(fmt.Sprintf("0x%040x", i+1))),
			testTx(uint64(i*10+1), common.HexToAddress(fmt.Sprintf("0x%040x", i+100))),
		}
		rs := &StorageKeySet{}
		ws := &StorageKeySet{}
		vertices[i] = NewEVMVertex(uint64(i), 0, []ids.ID{ids.Empty}, txs, rs, ws)
	}

	receipts, err := exec.ExecuteAntichain(nil, testHeader(), vertices, nil, vm.Config{})
	if err != nil {
		t.Fatalf("ExecuteAntichain failed: %v", err)
	}
	if len(receipts) != 6 { // 3 vertices * 2 txs each
		t.Fatalf("expected 6 receipts, got %d", len(receipts))
	}
}

func TestDAGExecutorTopologicalCut(t *testing.T) {
	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 4,
	})

	// Create a small DAG:
	//   v0 (root)
	//   |  \
	//   v1  v2  (both depend on v0)
	//   |
	//   v3     (depends on v1)

	txs := func(n uint64) []*types.Transaction {
		return []*types.Transaction{testTx(n, common.HexToAddress(fmt.Sprintf("0x%040x", n+1)))}
	}

	v0 := NewEVMVertex(0, 0, nil, txs(0), &StorageKeySet{}, &StorageKeySet{})
	v1 := NewEVMVertex(1, 0, []ids.ID{v0.ID()}, txs(1), &StorageKeySet{}, &StorageKeySet{})
	v2 := NewEVMVertex(1, 0, []ids.ID{v0.ID()}, txs(2), &StorageKeySet{}, &StorageKeySet{})
	v3 := NewEVMVertex(2, 0, []ids.ID{v1.ID()}, txs(3), &StorageKeySet{}, &StorageKeySet{})

	vertices := []*EVMVertex{v3, v1, v0, v2} // intentionally unordered

	receipts, err := exec.ExecuteTopologicalCut(nil, testHeader(), vertices, nil, vm.Config{})
	if err != nil {
		t.Fatalf("ExecuteTopologicalCut failed: %v", err)
	}
	if len(receipts) != 4 {
		t.Fatalf("expected 4 receipts, got %d", len(receipts))
	}
}

func TestDAGExecutorBuildVertex(t *testing.T) {
	builder := NewBuilder(BuilderConfig{
		Workers:         2,
		MaxTxsPerVertex: 100,
		FrontierFn:      func() []ids.ID { return []ids.ID{{0x01}} },
		HeightFn:        func() uint64 { return 1 },
	})

	exec := NewDAGExecutor(DAGExecutorConfig{
		Builder: builder,
		ApplyFn: mockApplyFn(21000),
	})

	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	v := exec.BuildVertex(txs)
	if v == nil {
		t.Fatal("BuildVertex returned nil")
	}
}

func TestDAGExecutorMetrics(t *testing.T) {
	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 2,
	})

	txs := types.Transactions{
		testTx(0, common.HexToAddress("0xaaaa")),
		testTx(1, common.HexToAddress("0xbbbb")),
	}

	_, err := exec.ExecuteBlock(nil, testHeader(), txs, nil, vm.Config{})
	if err != nil {
		t.Fatalf("ExecuteBlock failed: %v", err)
	}

	m := exec.Metrics()
	if m["txs_processed"] != 2 {
		t.Fatalf("expected 2 txs processed, got %d", m["txs_processed"])
	}
}

// --- VertexStore tests ---

func TestVertexStore(t *testing.T) {
	store := NewVertexStore()

	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	v := NewEVMVertex(1, 0, nil, txs, &StorageKeySet{}, &StorageKeySet{})

	store.Add(v)
	if store.Len() != 1 {
		t.Fatalf("expected 1 vertex, got %d", store.Len())
	}

	got, ok := store.Get(v.ID())
	if !ok {
		t.Fatal("vertex not found in store")
	}
	if got.ID() != v.ID() {
		t.Fatal("wrong vertex returned")
	}

	frontier := store.Frontier()
	if len(frontier) != 1 {
		t.Fatalf("expected 1 frontier vertex, got %d", len(frontier))
	}
}

// --- Benchmarks ---

// generateRandomTxBatch creates n transactions with approximately conflictRate
// fraction sharing addresses with other transactions in the batch.
func generateRandomTxBatch(n int, conflictRate float64, rng *rand.Rand) []*types.Transaction {
	txs := make([]*types.Transaction, n)
	numShared := int(float64(n) * conflictRate)
	if numShared < 1 {
		numShared = 1
	}

	// Generate shared addresses (these create conflicts).
	sharedAddrs := make([]common.Address, numShared)
	for i := range sharedAddrs {
		var addr common.Address
		rng.Read(addr[:])
		sharedAddrs[i] = addr
	}

	for i := 0; i < n; i++ {
		var to common.Address
		if rng.Float64() < conflictRate && len(sharedAddrs) > 0 {
			to = sharedAddrs[rng.Intn(len(sharedAddrs))]
		} else {
			rng.Read(to[:])
		}
		txs[i] = types.NewTransaction(
			uint64(i),
			to,
			big.NewInt(1000),
			21000,
			big.NewInt(1_000_000_000),
			nil,
		)
	}
	return txs
}

func benchmarkDAGExecute(b *testing.B, numTxs int) {
	rng := rand.New(rand.NewSource(42))
	txs := generateRandomTxBatch(numTxs, 0.10, rng)

	exec := NewDAGExecutor(DAGExecutorConfig{
		ApplyFn: mockApplyFn(21000),
		Workers: 8,
	})

	header := testHeader()
	typedTxs := make(types.Transactions, len(txs))
	for i, tx := range txs {
		typedTxs[i] = tx
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		receipts, err := exec.ExecuteBlock(nil, header, typedTxs, nil, vm.Config{})
		if err != nil {
			b.Fatalf("execution failed: %v", err)
		}
		if len(receipts) != numTxs {
			b.Fatalf("expected %d receipts, got %d", numTxs, len(receipts))
		}
	}
}

func BenchmarkDAGEVMExecute10K(b *testing.B)  { benchmarkDAGExecute(b, 10_000) }
func BenchmarkDAGEVMExecute50K(b *testing.B)  { benchmarkDAGExecute(b, 50_000) }
func BenchmarkDAGEVMExecute100K(b *testing.B) { benchmarkDAGExecute(b, 100_000) }

func benchmarkConflictDetection(b *testing.B, numVertices int) {
	rng := rand.New(rand.NewSource(42))

	vertices := make([]*EVMVertex, numVertices)
	for i := 0; i < numVertices; i++ {
		rs := &StorageKeySet{}
		ws := &StorageKeySet{}
		// Each vertex touches 5-15 storage keys.
		numKeys := 5 + rng.Intn(11)
		for j := 0; j < numKeys; j++ {
			var key common.Hash
			rng.Read(key[:])
			rs.Add(key)
			if rng.Float64() < 0.5 {
				ws.Add(key)
			}
		}
		txs := []*types.Transaction{testTx(uint64(i), common.HexToAddress(fmt.Sprintf("0x%040x", i+1)))}
		vertices[i] = NewEVMVertex(uint64(i), 0, nil, txs, rs, ws)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		BatchConflicts(vertices)
	}
}

func BenchmarkConflictDetection100(b *testing.B)  { benchmarkConflictDetection(b, 100) }
func BenchmarkConflictDetection500(b *testing.B)  { benchmarkConflictDetection(b, 500) }
func BenchmarkConflictDetection1000(b *testing.B) { benchmarkConflictDetection(b, 1000) }

func benchmarkVertexBuild(b *testing.B, numTxs int) {
	rng := rand.New(rand.NewSource(42))
	txs := generateRandomTxBatch(numTxs, 0.10, rng)

	builder := NewBuilder(BuilderConfig{
		Workers:         8,
		MaxTxsPerVertex: numTxs,
		FrontierFn:      func() []ids.ID { return []ids.ID{{0x01}} },
		HeightFn:        func() uint64 { return 1 },
	})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		v := builder.BuildVertex(txs)
		if v == nil {
			b.Fatal("BuildVertex returned nil")
		}
	}
}

func BenchmarkVertexBuild10K(b *testing.B)  { benchmarkVertexBuild(b, 10_000) }
func BenchmarkVertexBuild50K(b *testing.B)  { benchmarkVertexBuild(b, 50_000) }
func BenchmarkVertexBuild100K(b *testing.B) { benchmarkVertexBuild(b, 100_000) }
