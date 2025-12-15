// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

import (
	"fmt"
	"sync/atomic"

	"github.com/luxfi/evmgpu/core/state"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/core/vm"
	ethparams "github.com/luxfi/geth/params"
	"github.com/luxfi/ids"
	log "github.com/luxfi/log"
)

// TxApplyFunc executes a single transaction against the given state and
// returns the receipt. Injection point that lets callers plug in the EVM
// backend (Go-EVM / revm / cevm / GPU-EVM) without circular imports.
type TxApplyFunc func(
	config *ethparams.ChainConfig,
	header *types.Header,
	tx *types.Transaction,
	statedb *state.StateDB,
	vmCfg vm.Config,
	txIndex int,
) (*types.Receipt, error)

// DAGExecutor implements parallel.BlockExecutor but operates on DAG vertices
// instead of linear blocks. It receives finalized antichain cuts from the
// nebula DAG engine, topologically sorts the transactions across all vertices
// in the cut, and applies them using Block-STM parallel execution.
//
// For backward compatibility during bootstrap, it also accepts linear blocks
// via ExecuteBlock and processes them sequentially.
type DAGExecutor struct {
	builder *Builder
	applyFn TxApplyFunc
	workers int

	// Metrics
	verticesProcessed atomic.Int64
	txsProcessed      atomic.Int64
	conflictsDetected atomic.Int64
}

// DAGExecutorConfig configures the DAG executor.
type DAGExecutorConfig struct {
	Builder *Builder
	ApplyFn TxApplyFunc
	Workers int
}

// NewDAGExecutor creates a DAG executor.
func NewDAGExecutor(cfg DAGExecutorConfig) *DAGExecutor {
	workers := cfg.Workers
	if workers <= 0 {
		workers = 8
	}
	return &DAGExecutor{
		builder: cfg.Builder,
		applyFn: cfg.ApplyFn,
		workers: workers,
	}
}

// ExecuteBlock implements parallel.BlockExecutor for backward compatibility.
// During bootstrap, the C-Chain still receives linear blocks. This method
// wraps them in a single-vertex DAG cut and executes normally.
func (e *DAGExecutor) ExecuteBlock(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs types.Transactions,
	statedb *state.StateDB,
	vmCfg vm.Config,
) ([]*types.Receipt, error) {
	n := len(txs)
	if n == 0 || e.applyFn == nil {
		return nil, nil
	}

	// Wrap the linear block as a single vertex for unified processing.
	sliceTxs := make([]*types.Transaction, n)
	for i, tx := range txs {
		sliceTxs[i] = tx
	}

	return e.executeTransactions(config, header, sliceTxs, statedb, vmCfg)
}

// ExecuteAntichain processes a set of non-conflicting vertices (an antichain
// from the DAG) in parallel. Transactions from all vertices are merged into
// a single execution batch with conflict-aware ordering.
//
// Precondition: all vertices in the antichain have been verified as non-conflicting
// by the nebula engine (no write-read, read-write, or write-write overlaps).
func (e *DAGExecutor) ExecuteAntichain(
	config *ethparams.ChainConfig,
	header *types.Header,
	vertices []*EVMVertex,
	statedb *state.StateDB,
	vmCfg vm.Config,
) ([]*types.Receipt, error) {
	if len(vertices) == 0 || e.applyFn == nil {
		return nil, nil
	}

	// Collect all transactions from the antichain in deterministic order.
	// Vertices are ordered by ID (deterministic), and txs within each vertex
	// preserve their original order.
	var allTxs []*types.Transaction
	for _, v := range vertices {
		allTxs = append(allTxs, v.Transactions()...)
	}

	if len(allTxs) == 0 {
		return nil, nil
	}

	log.Debug("DAG antichain execution",
		"vertices", len(vertices),
		"txs", len(allTxs),
	)

	e.verticesProcessed.Add(int64(len(vertices)))
	return e.executeTransactions(config, header, allTxs, statedb, vmCfg)
}

// ExecuteTopologicalCut processes vertices from a finalized DAG cut in
// topological order. Vertices that are independent (no parent-child relationship
// within the cut) are executed as an antichain. Dependent vertices are
// executed sequentially respecting causal order.
func (e *DAGExecutor) ExecuteTopologicalCut(
	config *ethparams.ChainConfig,
	header *types.Header,
	vertices []*EVMVertex,
	statedb *state.StateDB,
	vmCfg vm.Config,
) ([]*types.Receipt, error) {
	if len(vertices) == 0 {
		return nil, nil
	}

	// Build a dependency graph among the vertices.
	idToIdx := make(map[ids.ID]int, len(vertices))
	for i, v := range vertices {
		idToIdx[v.ID()] = i
	}

	// inDegree tracks how many parent vertices each vertex has within this cut.
	inDegree := make([]int, len(vertices))
	children := make([][]int, len(vertices))
	for i := range children {
		children[i] = nil
	}

	for i, v := range vertices {
		for _, parentID := range v.Parents() {
			if parentIdx, ok := idToIdx[parentID]; ok {
				inDegree[i]++
				children[parentIdx] = append(children[parentIdx], i)
			}
		}
	}

	// Kahn's algorithm: process vertices with inDegree=0 as antichains.
	var allReceipts []*types.Receipt
	queue := make([]int, 0)
	for i, deg := range inDegree {
		if deg == 0 {
			queue = append(queue, i)
		}
	}

	for len(queue) > 0 {
		// Current antichain: all vertices with zero in-degree.
		antichain := make([]*EVMVertex, len(queue))
		for i, idx := range queue {
			antichain[i] = vertices[idx]
		}

		receipts, err := e.ExecuteAntichain(config, header, antichain, statedb, vmCfg)
		if err != nil {
			return nil, fmt.Errorf("dag executor antichain: %w", err)
		}
		allReceipts = append(allReceipts, receipts...)

		// Advance: decrement in-degree for children.
		var nextQueue []int
		for _, idx := range queue {
			for _, childIdx := range children[idx] {
				inDegree[childIdx]--
				if inDegree[childIdx] == 0 {
					nextQueue = append(nextQueue, childIdx)
				}
			}
		}
		queue = nextQueue
	}

	return allReceipts, nil
}

// executeTransactions runs a batch of transactions using Block-STM parallel
// execution. This is the core execution path shared by both linear blocks
// and DAG antichains.
func (e *DAGExecutor) executeTransactions(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs []*types.Transaction,
	statedb *state.StateDB,
	vmCfg vm.Config,
) ([]*types.Receipt, error) {
	// The DAG executor drives sequential application within an antichain cut
	// (the non-conflict invariant makes order irrelevant by theorem
	// `antichain_order_free` in lux/formal/lean/Consensus/DAGEVM.lean).
	// Parallel speculative execution of the batch is the responsibility of
	// the Block-STM-speculative pass that runs *before* vertex assembly in
	// the builder; by the time we reach ExecuteAntichain the r/w-sets are
	// already disjoint and apply order is free.
	return e.executeSequential(config, header, txs, statedb, vmCfg)
}

// executeSequential processes transactions one at a time.
func (e *DAGExecutor) executeSequential(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs []*types.Transaction,
	statedb *state.StateDB,
	vmCfg vm.Config,
) ([]*types.Receipt, error) {
	receipts := make([]*types.Receipt, 0, len(txs))
	var cumulativeGas uint64

	for i, tx := range txs {
		receipt, err := e.applyFn(config, header, tx, statedb, vmCfg, i)
		if err != nil {
			return nil, fmt.Errorf("sequential exec tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}
		cumulativeGas += receipt.GasUsed
		receipt.CumulativeGasUsed = cumulativeGas
		receipts = append(receipts, receipt)
	}

	e.txsProcessed.Add(int64(len(txs)))
	return receipts, nil
}

// BuildVertex creates a DAG vertex from pending transactions.
func (e *DAGExecutor) BuildVertex(txs []*types.Transaction) *EVMVertex {
	if e.builder == nil {
		return nil
	}
	return e.builder.BuildVertex(txs)
}

// --- Metrics ---

// Metrics returns DAG executor statistics.
func (e *DAGExecutor) Metrics() map[string]int64 {
	return map[string]int64{
		"vertices_processed": e.verticesProcessed.Load(),
		"txs_processed":      e.txsProcessed.Load(),
		"conflicts_detected": e.conflictsDetected.Load(),
	}
}

// VertexStore is an in-memory store of accepted vertices for the DAG.
type VertexStore struct {
	vertices map[ids.ID]*EVMVertex
	frontier map[ids.ID]bool
}

// NewVertexStore creates a new vertex store.
func NewVertexStore() *VertexStore {
	return &VertexStore{
		vertices: make(map[ids.ID]*EVMVertex),
		frontier: make(map[ids.ID]bool),
	}
}

// Add inserts a vertex into the store.
func (s *VertexStore) Add(v *EVMVertex) {
	s.vertices[v.ID()] = v
	s.frontier[v.ID()] = true
	// Remove parents from frontier.
	for _, pid := range v.Parents() {
		delete(s.frontier, pid)
	}
}

// Get retrieves a vertex by ID.
func (s *VertexStore) Get(id ids.ID) (*EVMVertex, bool) {
	v, ok := s.vertices[id]
	return v, ok
}

// Frontier returns the current DAG tips (vertices with no children).
func (s *VertexStore) Frontier() []ids.ID {
	result := make([]ids.ID, 0, len(s.frontier))
	for id := range s.frontier {
		result = append(result, id)
	}
	return result
}

// Height returns max height + 1 across all vertices.
func (s *VertexStore) Height() uint64 {
	var maxH uint64
	for _, v := range s.vertices {
		if v.Height() > maxH {
			maxH = v.Height()
		}
	}
	return maxH + 1
}

// Len returns total number of vertices.
func (s *VertexStore) Len() int { return len(s.vertices) }

// unused but required for common.Hash reference
var _ = common.Hash{}
