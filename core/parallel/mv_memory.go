// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"sync"
)

// MvMemory is the multi-version memory structure for Block-STM.
//
// Each memory location maps to a version chain: an ordered set of entries
// indexed by TxIdx. A read for transaction N finds the highest entry with
// TxIdx < N. This is the core data structure enabling optimistic parallel
// execution.
//
// GPU Design: The version chains are stored as fixed-size arrays indexed by
// TxIdx (max block_size entries per location). This maps directly to GPU
// shared memory / global memory with atomic CAS for concurrent access.
// On unified memory (Metal/Apple Silicon), the CPU accesses these arrays
// directly in GPU VRAM via buffer_get_host_ptr().
type MvMemory struct {
	blockSize uint32

	// Location data: full MemoryLocation -> version chain.
	// Using the full key (not LocationHash) prevents hash-collision attacks
	// where an attacker mines two (address, slot) pairs with the same hash.
	// GPU kernel equivalent: global memory array[num_locations][block_size]
	mu   sync.RWMutex
	data map[MemoryLocation]*versionChain

	// Per-transaction read/write sets
	txSets []TxReadWriteSet

	// Lazy evaluation addresses (beneficiary, raw transfer targets)
	lazyMu    sync.Mutex
	lazyAddrs map[MemoryLocation]bool
}

// versionChain holds all versions of a value at one memory location.
// Fixed-size array indexed by TxIdx — GPU-friendly (no pointer chasing).
type versionChain struct {
	mu      sync.RWMutex
	entries []MvEntry // len == blockSize, indexed by TxIdx
}

// NewMvMemory creates a new multi-version memory for a block.
func NewMvMemory(blockSize uint32) *MvMemory {
	return &MvMemory{
		blockSize: blockSize,
		data:      make(map[MemoryLocation]*versionChain),
		txSets:    make([]TxReadWriteSet, blockSize),
		lazyAddrs: make(map[MemoryLocation]bool),
	}
}

// getOrCreateChain returns the version chain for a location, creating if needed.
func (m *MvMemory) getOrCreateChain(loc MemoryLocation) *versionChain {
	m.mu.RLock()
	chain, ok := m.data[loc]
	m.mu.RUnlock()
	if ok {
		return chain
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Double-check after acquiring write lock
	if chain, ok = m.data[loc]; ok {
		return chain
	}
	chain = &versionChain{
		entries: make([]MvEntry, m.blockSize),
	}
	m.data[loc] = chain
	return chain
}

// Read finds the closest version written by a transaction with index < txIdx.
// Returns the entry and whether it was found.
// If an ESTIMATE marker is found, returns (entry, true) with IsEstimate=true,
// signaling the caller to block on that transaction.
func (m *MvMemory) Read(loc MemoryLocation, txIdx TxIdx) (MvEntry, bool) {
	m.mu.RLock()
	chain, ok := m.data[loc]
	m.mu.RUnlock()
	if !ok {
		return MvEntry{}, false
	}

	chain.mu.RLock()
	defer chain.mu.RUnlock()

	// Walk backwards from txIdx-1 to find the closest writer
	for i := int(txIdx) - 1; i >= 0; i-- {
		entry := &chain.entries[i]
		if entry.TxIdx != 0 { // TxIdx stored as N+1, 0 = empty
			return *entry, true
		}
	}
	return MvEntry{}, false
}

// Write records a value written by a transaction.
func (m *MvMemory) Write(loc MemoryLocation, txIdx TxIdx, incarnation uint32, value MemoryValue) {
	chain := m.getOrCreateChain(loc)

	chain.mu.Lock()
	defer chain.mu.Unlock()

	chain.entries[txIdx] = MvEntry{
		TxIdx:       uint32(txIdx) + 1, // Store as N+1 (0 = empty)
		Incarnation: incarnation,
		IsEstimate:  false,
		Value:       value,
	}
}

// WriteEstimate marks a location with an ESTIMATE placeholder.
// This is called when a transaction is aborted — it signals to higher
// transactions that this value is being recomputed.
func (m *MvMemory) WriteEstimate(loc MemoryLocation, txIdx TxIdx) {
	chain := m.getOrCreateChain(loc)

	chain.mu.Lock()
	defer chain.mu.Unlock()

	chain.entries[txIdx] = MvEntry{
		TxIdx:      uint32(txIdx) + 1,
		IsEstimate: true,
	}
}

// ConvertWritesToEstimates replaces all writes by a transaction with ESTIMATE markers.
// Called when a transaction fails validation — this cascades aborts to dependent txs.
func (m *MvMemory) ConvertWritesToEstimates(txIdx TxIdx) {
	txSet := &m.txSets[txIdx]
	txSet.mu.Lock()
	writes := make([]WriteEntry, len(txSet.WriteSet))
	copy(writes, txSet.WriteSet)
	txSet.mu.Unlock()

	for _, w := range writes {
		m.WriteEstimate(w.Location, txIdx)
	}
}

// ValidateReadSet re-checks all reads made by a transaction.
// Returns true if all reads are still valid (same writer, same incarnation).
func (m *MvMemory) ValidateReadSet(txIdx TxIdx) bool {
	txSet := &m.txSets[txIdx]
	txSet.mu.Lock()
	reads := make([]ReadEntry, len(txSet.ReadSet))
	copy(reads, txSet.ReadSet)
	txSet.mu.Unlock()

	for _, r := range reads {
		if r.Origin.FromMvMemory {
			// Verify the same version is still the closest writer
			entry, found := m.Read(r.Location, txIdx)
			if !found {
				// Was reading from MvMemory but now no writer exists — stale
				return false
			}
			if entry.IsEstimate {
				// Writer was aborted — our read is stale
				return false
			}
			if TxIdx(entry.TxIdx-1) != r.Origin.Version.TxIdx ||
				entry.Incarnation != r.Origin.Version.Incarnation {
				// Different writer or different incarnation — stale
				return false
			}
		} else {
			// Was reading from storage — verify no MvMemory entry appeared
			_, found := m.Read(r.Location, txIdx)
			if found {
				// A transaction below us wrote to this location since our read
				return false
			}
		}
	}
	return true
}

// RecordReadSet stores the read set for a transaction (called after execution).
func (m *MvMemory) RecordReadSet(txIdx TxIdx, reads []ReadEntry) {
	txSet := &m.txSets[txIdx]
	txSet.mu.Lock()
	txSet.ReadSet = reads
	txSet.mu.Unlock()
}

// RecordWriteSet stores the write set and applies writes to MvMemory.
func (m *MvMemory) RecordWriteSet(txIdx TxIdx, incarnation uint32, writes []WriteEntry) {
	txSet := &m.txSets[txIdx]
	txSet.mu.Lock()
	txSet.WriteSet = writes
	txSet.mu.Unlock()

	for _, w := range writes {
		m.Write(w.Location, txIdx, incarnation, w.Value)
	}
}

// ClearTxSets clears the read/write sets for a transaction (before re-execution).
func (m *MvMemory) ClearTxSets(txIdx TxIdx) {
	txSet := &m.txSets[txIdx]
	txSet.mu.Lock()
	txSet.ReadSet = txSet.ReadSet[:0]
	txSet.WriteSet = txSet.WriteSet[:0]
	txSet.mu.Unlock()
}

// MarkLazy marks a location as needing lazy post-evaluation.
func (m *MvMemory) MarkLazy(loc MemoryLocation) {
	m.lazyMu.Lock()
	m.lazyAddrs[loc] = true
	m.lazyMu.Unlock()
}

// LazyLocations returns all locations marked for lazy evaluation.
func (m *MvMemory) LazyLocations() []MemoryLocation {
	m.lazyMu.Lock()
	defer m.lazyMu.Unlock()
	locs := make([]MemoryLocation, 0, len(m.lazyAddrs))
	for loc := range m.lazyAddrs {
		locs = append(locs, loc)
	}
	return locs
}

// PreAllocateEstimates pre-populates a location with ESTIMATE markers for all
// transaction indices. Used for the beneficiary address — since every tx pays
// gas to coinbase, this prevents false conflicts.
func (m *MvMemory) PreAllocateEstimates(loc MemoryLocation) {
	chain := m.getOrCreateChain(loc)
	chain.mu.Lock()
	defer chain.mu.Unlock()
	for i := uint32(0); i < m.blockSize; i++ {
		chain.entries[i] = MvEntry{
			TxIdx:      i + 1,
			IsEstimate: true,
		}
	}
}
