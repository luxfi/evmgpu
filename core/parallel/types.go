// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package parallel implements Block-STM parallel EVM execution.
//
// The algorithm speculatively executes transactions in parallel, tracks
// read/write sets, validates for conflicts, and re-executes on abort.
// Designed with GPU-compatible data structures for future kernel offload.
//
// Based on Block-STM (https://arxiv.org/abs/2203.06871) with EVM-specific
// optimizations: lazy beneficiary evaluation, lazy raw transfers, and
// GPU-native multi-version memory layout.
package parallel

import (
	"sync"
	"sync/atomic"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
)

// =============================================================================
// Memory Location — what a transaction reads or writes
// =============================================================================

// LocationType identifies the kind of state access.
// Balance and nonce are separate types to prevent conflation — writing one
// must never clobber the other in MvMemory.
type LocationType uint8

const (
	LocationBalance  LocationType = 0 // Account balance only
	LocationNonce    LocationType = 1 // Account nonce only
	LocationCodeHash LocationType = 2 // Contract code hash
	LocationStorage  LocationType = 3 // Storage slot
)

// MemoryLocation uniquely identifies a piece of EVM state.
// Comparable in Go — used directly as map key in MvMemory to prevent
// hash-collision attacks (see F02). LocationHash is retained for GPU
// kernel sharding hints only.
type MemoryLocation struct {
	Address common.Address // 20 bytes
	Type    LocationType   // 1 byte
	Slot    common.Hash    // 32 bytes (only for LocationStorage)
}

// LocationHash is a pre-computed hash of a MemoryLocation for fast lookups.
// Uses FxHash-style fast hashing — the last 8 bytes of the address.
type LocationHash uint64

// Hash computes the LocationHash for fast map lookups.
func (loc *MemoryLocation) Hash() LocationHash {
	// FxHash: use last 8 bytes of address XOR'd with slot for storage,
	// or just last 8 bytes of address for balance/nonce/codehash.
	var h uint64
	// Read last 8 bytes of address as uint64
	for i := 12; i < 20; i++ {
		h = (h << 8) | uint64(loc.Address[i])
	}
	h ^= uint64(loc.Type) << 56

	if loc.Type == LocationStorage {
		// XOR with last 8 bytes of slot
		for i := 24; i < 32; i++ {
			h ^= uint64(loc.Slot[i]) << ((31 - uint64(i)) * 8)
		}
	}
	return LocationHash(h)
}

// =============================================================================
// Memory Value — what's stored at a location
// =============================================================================

// ValueType identifies the kind of value.
type ValueType uint8

const (
	ValueAbsolute     ValueType = 0 // Full value (balance, nonce, storage)
	ValueLazyCredit   ValueType = 1 // Delta: +balance (beneficiary, transfer recipient)
	ValueLazyDebit    ValueType = 2 // Delta: -balance, +1 nonce (transfer sender)
	ValueSelfDestruct ValueType = 3 // Account destroyed
)

// MemoryValue is the value stored in the multi-version data structure.
// Each LocationType uses only the relevant field:
//   - LocationBalance  → Balance
//   - LocationNonce    → Nonce
//   - LocationCodeHash → Storage (code hash stored as hash)
//   - LocationStorage  → Storage
type MemoryValue struct {
	Type    ValueType
	Balance common.Hash // 32 bytes: absolute or delta (LocationBalance only)
	Nonce   uint64      // LocationNonce only
	Storage common.Hash // 32 bytes: for LocationStorage and LocationCodeHash
}

// =============================================================================
// Read/Write Sets — per-transaction tracking
// =============================================================================

// TxIdx is a transaction index within the block.
type TxIdx uint32

// TxVersion is (TxIdx, Incarnation) — identifies a specific execution attempt.
type TxVersion struct {
	TxIdx       TxIdx
	Incarnation uint32
}

// ReadOrigin records where a value was read from.
type ReadOrigin struct {
	FromMvMemory bool      // true = read from MvMemory, false = read from storage
	Version      TxVersion // if FromMvMemory, the version that wrote the value
}

// ReadEntry is one entry in a transaction's read set.
type ReadEntry struct {
	Location MemoryLocation
	Origin   ReadOrigin
}

// WriteEntry is one entry in a transaction's write set.
type WriteEntry struct {
	Location MemoryLocation
	Value    MemoryValue
}

// =============================================================================
// Task — what a worker thread should do next
// =============================================================================

// TaskType identifies the kind of work.
type TaskType uint8

const (
	TaskNone       TaskType = 0
	TaskExecution  TaskType = 1
	TaskValidation TaskType = 2
)

// Task is a unit of work for a parallel worker.
type Task struct {
	Type    TaskType
	Version TxVersion
}

// =============================================================================
// Transaction Status — state machine per transaction
// =============================================================================

// TxStatus tracks the execution state of a transaction.
type TxStatus uint8

const (
	StatusReadyToExecute TxStatus = 0
	StatusExecuting      TxStatus = 1
	StatusExecuted       TxStatus = 2
	StatusValidated      TxStatus = 3
	StatusAborting       TxStatus = 4
)

// TxState holds the mutable state of a transaction in the scheduler.
type TxState struct {
	Status      TxStatus
	Incarnation uint32
}

// =============================================================================
// Execution Result — output of one transaction execution
// =============================================================================

// ExecResult holds the result of executing a single transaction.
type ExecResult struct {
	Receipt  *types.Receipt
	ReadSet  []ReadEntry
	WriteSet []WriteEntry
	GasUsed  uint64
	Err      error

	// Blocking: if non-nil, this execution is blocked on another tx
	BlockedBy *TxIdx
}

// =============================================================================
// Block-STM Finish Flags
// =============================================================================

// FinishFlags are set when a transaction finishes execution.
type FinishFlags uint8

const (
	FlagWroteNewLocation FinishFlags = 1 << 0 // Wrote to a location not in previous incarnation
	FlagReadDependency   FinishFlags = 1 << 1 // Read from another transaction's output
)

// =============================================================================
// GPU-Compatible Memory Entry
//
// This struct is designed to match the GPU kernel slot layout.
// On unified memory (Metal/Apple Silicon), the MvMemory arrays live in
// GPU VRAM and are accessible to both CPU and GPU kernels.
// =============================================================================

// MvEntry is one version of a value in the multi-version memory.
// 0 = empty, non-zero TxIdx+1 = written by that tx.
type MvEntry struct {
	TxIdx       uint32 // 0 = empty, N+1 = written by tx N
	Incarnation uint32
	IsEstimate  bool // true = placeholder after abort
	Value       MemoryValue
}

// =============================================================================
// Per-Transaction Storage (GPU-aligned)
// =============================================================================

// TxReadWriteSet holds the read and write sets for one transaction.
type TxReadWriteSet struct {
	mu       sync.Mutex
	ReadSet  []ReadEntry
	WriteSet []WriteEntry
}

// =============================================================================
// Stats
// =============================================================================

// Stats tracks parallel execution statistics.
// Not safe to copy — use StatsSnapshot for value semantics.
type Stats struct {
	TotalTxs      uint64
	Executions    atomic.Uint64 // Total execution attempts (including re-executions)
	Validations   atomic.Uint64
	Aborts        atomic.Uint64 // Validation failures causing re-execution
	LazyAddresses atomic.Uint64 // Addresses evaluated lazily
	FellBack      bool          // True if fell back to sequential

	// GPU opcode dispatch metrics
	GPUEligible atomic.Uint64 // Transactions eligible for GPU dispatch
	GPUExecuted atomic.Uint64 // Transactions actually dispatched to GPU
	GPUFallback atomic.Uint64 // Transactions that fell back from GPU to Go EVM
}

// StatsSnapshot is a copy-safe snapshot of Stats.
type StatsSnapshot struct {
	TotalTxs    uint64
	Executions  uint64
	Validations uint64
	Aborts      uint64
	FellBack    bool

	// GPU opcode dispatch metrics
	GPUEligible uint64
	GPUExecuted uint64
	GPUFallback uint64
}

// =============================================================================
// GPU EVM Dispatch Types
// =============================================================================

// GPUEVMResult holds the result of GPU EVM execution for one transaction.
type GPUEVMResult struct {
	GasUsed uint64
	Success bool
}

// GPUDispatcher is the interface for GPU EVM opcode dispatch.
// Implemented by GPUEVMDispatcher (gpu build tag) or nil (CPU-only).
type GPUDispatcher interface {
	Available() bool
	Backend() string
	ExecuteBlock(signer types.Signer, txs []*types.Transaction, senders []common.Address) ([]GPUEVMResult, error)
}

// IsGPUEligible returns true if a transaction can be dispatched to the
// GPU EVM kernel. GPU-eligible transactions are simple value transfers:
// no contract creation, no calldata (which implies no CALL/CREATE/
// DELEGATECALL in the execution trace).
func IsGPUEligible(tx *types.Transaction) bool {
	if tx.To() == nil {
		return false
	}
	if len(tx.Data()) > 0 {
		return false
	}
	return true
}
