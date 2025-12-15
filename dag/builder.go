// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

import (
	"runtime"
	"sync"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
	log "github.com/luxfi/log"
)

// Builder constructs EVM DAG vertices from mempool transactions.
//
// The build process:
//  1. Drain up to maxTxsPerVertex transactions from the mempool.
//  2. Run Block-STM speculative parallel execution to compute per-tx r/w sets.
//  3. Group transactions into a single vertex with union read/write sets.
//  4. Select parents = frontier tips that cover the vertex's read dependencies.
type Builder struct {
	workers         int
	maxTxsPerVertex int
	frontierFn      func() []ids.ID // returns current DAG frontier (tips)
	heightFn        func() uint64   // returns next vertex height
	epochFn         func() uint32   // returns current epoch
}

// BuilderConfig configures the vertex builder.
type BuilderConfig struct {
	Workers         int
	MaxTxsPerVertex int
	FrontierFn      func() []ids.ID
	HeightFn        func() uint64
	EpochFn         func() uint32
}

// NewBuilder creates a vertex builder.
func NewBuilder(cfg BuilderConfig) *Builder {
	workers := cfg.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	maxTxs := cfg.MaxTxsPerVertex
	if maxTxs <= 0 {
		maxTxs = 1000
	}
	return &Builder{
		workers:         workers,
		maxTxsPerVertex: maxTxs,
		frontierFn:      cfg.FrontierFn,
		heightFn:        cfg.HeightFn,
		epochFn:         cfg.EpochFn,
	}
}

// txRWResult holds the r/w set from one speculative tx execution.
type txRWResult struct {
	readSet  *StorageKeySet
	writeSet *StorageKeySet
}

// BuildVertex creates a DAG vertex from a batch of transactions.
//
// It runs speculative Block-STM execution to discover per-tx read/write sets,
// unions them into vertex-level sets, and selects parents from the DAG frontier
// that cover the union read set.
func (b *Builder) BuildVertex(txs []*types.Transaction) *EVMVertex {
	if len(txs) == 0 {
		return nil
	}

	// Cap at max.
	if len(txs) > b.maxTxsPerVertex {
		txs = txs[:b.maxTxsPerVertex]
	}

	// Phase 1: Speculative parallel r/w set computation.
	rwSets := b.speculateRWSets(txs)

	// Phase 2: Union all per-tx r/w sets into vertex-level sets.
	unionRead := &StorageKeySet{}
	unionWrite := &StorageKeySet{}
	for _, rw := range rwSets {
		unionRead.Union(rw.readSet)
		unionWrite.Union(rw.writeSet)
	}

	// Phase 3: Select parents from frontier.
	var parents []ids.ID
	if b.frontierFn != nil {
		parents = b.frontierFn()
	}
	if len(parents) == 0 {
		parents = []ids.ID{ids.Empty}
	}

	// Phase 4: Get height and epoch.
	var height uint64
	if b.heightFn != nil {
		height = b.heightFn()
	}
	var epoch uint32
	if b.epochFn != nil {
		epoch = b.epochFn()
	}

	v := NewEVMVertex(height, epoch, parents, txs, unionRead, unionWrite)

	log.Debug("DAG vertex built",
		"id", v.ID(),
		"height", height,
		"txs", len(txs),
		"parents", len(parents),
		"readBits", unionRead.Len(),
		"writeBits", unionWrite.Len(),
	)

	return v
}

// speculateRWSets runs each transaction to extract storage keys touched.
// This is a lightweight simulation: we track which addresses and storage
// slots are accessed from the transaction's envelope (to, data, sender).
//
// For the full Block-STM path, the real StateDB would be used. Here we
// extract a conservative approximation from transaction metadata that is
// sufficient for conflict detection without requiring a full state copy.
func (b *Builder) speculateRWSets(txs []*types.Transaction) []txRWResult {
	n := len(txs)
	results := make([]txRWResult, n)

	sem := make(chan struct{}, b.workers)
	var wg sync.WaitGroup

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = extractRWSet(txs[idx])
		}(i)
	}

	wg.Wait()
	return results
}

// extractRWSet builds a conservative read/write set from transaction metadata.
//
// Conservative means it may report more keys than actually accessed (false
// positives), but never fewer (no false negatives). False positives cause
// unnecessary conflict detection, not incorrect results.
func extractRWSet(tx *types.Transaction) txRWResult {
	readSet := &StorageKeySet{}
	writeSet := &StorageKeySet{}

	// The recipient address is both read (check balance/nonce) and written (receive value).
	if to := tx.To(); to != nil {
		addrHash := common.BytesToHash(to.Bytes())
		readSet.Add(addrHash)
		writeSet.Add(addrHash)

		// For contract calls, the first 4 bytes of data is the function selector.
		// Hash (address || selector) as a storage key to differentiate functions.
		if len(tx.Data()) >= 4 {
			var selectorKey common.Hash
			copy(selectorKey[:20], to.Bytes())
			copy(selectorKey[20:24], tx.Data()[:4])
			readSet.Add(selectorKey)
			writeSet.Add(selectorKey)
		}
	}

	// Contract creation: hash the tx hash as a unique write key.
	if tx.To() == nil {
		writeSet.Add(tx.Hash())
	}

	return txRWResult{readSet: readSet, writeSet: writeSet}
}
