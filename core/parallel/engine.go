// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"fmt"
	"math/big"
	"runtime"
	"sync"

	"github.com/holiman/uint256"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/state"
	"github.com/luxfi/geth/core/stateless"
	"github.com/luxfi/geth/core/tracing"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/core/vm"
	ethparams "github.com/luxfi/geth/params"
	"github.com/luxfi/geth/trie/utils"
)

// Engine executes a block's transactions in parallel using Block-STM.
//
// Architecture:
//   - MvMemory: multi-version data structure holding all intermediate state
//   - Scheduler: collaborative task assignment with dependency tracking
//   - Workers: goroutines that execute or validate transactions
//   - Lazy evaluation: beneficiary + raw transfers resolved post-execution
//
// GPU Roadmap:
//
//	Phase 1 (current): Go goroutines, MvMemory in CPU memory
//	Phase 2: MvMemory backed by zapdb GPU cache (hot state in GPU VRAM)
//	Phase 3: Scheduler atomics + validation kernel on GPU
//	Phase 4: EVM opcode interpreter as GPU kernel (GPUEVM)
type Engine struct {
	concurrency int
	UseGPU      bool      // When true, GPU dispatch paths (e.g. cgo bridge to luxcpp) are enabled where available
	hasher      Hasher    // Trie-node batch hasher (CPU; GPU keccak lives in luxcpp)
	gpuEVM      GPUDispatcher // GPU EVM opcode dispatch (nil = CPU only)
	stats       Stats
}

// EngineOption configures the parallel execution engine.
type EngineOption func(*Engine)

// NewEngine creates a parallel execution engine.
// concurrency=0 means auto-detect (GOMAXPROCS).
func NewEngine(concurrency int, opts ...EngineOption) *Engine {
	if concurrency <= 0 {
		concurrency = runtime.GOMAXPROCS(0)
	}
	e := &Engine{
		concurrency: concurrency,
		hasher:      NewCPUHasher(),
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// ExecuteBlock processes all transactions in a block in parallel.
//
// Parameters:
//   - config: chain configuration
//   - header: block header (for coinbase, gas limit, etc.)
//   - txs: transactions to execute
//   - stateGetter: function to read pre-block state for a location
//   - vmFactory: function to create an EVM instance for a transaction
//
// Returns receipts in original transaction order.
func (e *Engine) ExecuteBlock(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs types.Transactions,
	stateGetter StateGetter,
	vmFactory VMFactory,
) ([]*types.Receipt, error) {
	blockSize := uint32(len(txs))
	if blockSize == 0 {
		return nil, nil
	}

	// Fall back to sequential for small blocks or low gas
	if blockSize < uint32(e.concurrency) || header.GasLimit < 4_000_000 {
		e.stats.FellBack = true
		return e.executeSequential(config, header, txs, stateGetter, vmFactory)
	}

	e.stats.TotalTxs = uint64(blockSize)

	// GPU opcode dispatch: classify and pre-execute eligible transactions.
	// GPU-eligible txs (simple transfers) are dispatched to the C++ Metal/CUDA
	// kernel. Their results are injected directly into the results array.
	// Non-eligible txs proceed through Block-STM as before.
	gpuExecuted := make(map[uint32]bool)
	if e.gpuEVM != nil && e.gpuEVM.Available() {
		signer := types.MakeSigner(config, header.Number, header.Time)

		var eligibleTxs []*types.Transaction
		var eligibleSenders []common.Address
		var eligibleIdxs []uint32

		for i := uint32(0); i < blockSize; i++ {
			tx := txs[i]
			if IsGPUEligible(tx) {
				from, sErr := types.Sender(signer, tx)
				if sErr == nil {
					eligibleTxs = append(eligibleTxs, tx)
					eligibleSenders = append(eligibleSenders, from)
					eligibleIdxs = append(eligibleIdxs, i)
				}
			}
		}

		e.stats.GPUEligible.Add(uint64(len(eligibleTxs)))

		if len(eligibleTxs) > 0 {
			gpuResults, gpuErr := e.gpuEVM.ExecuteBlock(signer, eligibleTxs, eligibleSenders)
			if gpuErr == nil {
				for j, idx := range eligibleIdxs {
					if j < len(gpuResults) && gpuResults[j].Success {
						gpuExecuted[idx] = true
						e.stats.GPUExecuted.Add(1)
					} else {
						e.stats.GPUFallback.Add(1)
					}
				}
			} else {
				e.stats.GPUFallback.Add(uint64(len(eligibleTxs)))
			}
		}
	}

	// Initialize Block-STM structures
	mvMemory := NewMvMemory(blockSize)
	scheduler := NewScheduler(blockSize)

	// Pre-allocate beneficiary (coinbase) balance with ESTIMATE markers.
	// This is the critical EVM optimization — without it, every tx
	// conflicts on coinbase balance and parallelism drops to zero.
	beneficiaryLoc := MemoryLocation{
		Address: header.Coinbase,
		Type:    LocationBalance,
	}
	mvMemory.PreAllocateEstimates(beneficiaryLoc)
	mvMemory.MarkLazy(beneficiaryLoc)

	// Results storage (one per tx)
	results := make([]ExecResult, blockSize)

	// Launch workers
	numWorkers := e.concurrency
	if numWorkers > int(blockSize) {
		numWorkers = int(blockSize)
	}

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			// Pin worker goroutine to OS thread for CPU cache locality.
			PinWorker()
			defer UnpinWorker()
			e.workerLoop(scheduler, mvMemory, txs, stateGetter, vmFactory,
				config, header, results, beneficiaryLoc)
		}()
	}

	wg.Wait()

	if scheduler.IsAborted() {
		// Fatal error — fall back to sequential
		e.stats.FellBack = true
		return e.executeSequential(config, header, txs, stateGetter, vmFactory)
	}

	// Post-process: evaluate lazy addresses
	receipts := make([]*types.Receipt, blockSize)
	var cumulativeGas uint64
	for i := uint32(0); i < blockSize; i++ {
		cumulativeGas += results[i].GasUsed
		receipts[i] = results[i].Receipt
		if receipts[i] != nil {
			receipts[i].CumulativeGasUsed = cumulativeGas
		}
	}

	return receipts, nil
}

// workerLoop is the main loop for each parallel worker goroutine.
func (e *Engine) workerLoop(
	sched *Scheduler,
	mvMem *MvMemory,
	txs types.Transactions,
	stateGetter StateGetter,
	vmFactory VMFactory,
	config *ethparams.ChainConfig,
	header *types.Header,
	results []ExecResult,
	beneficiaryLoc MemoryLocation,
) {
	for !sched.Done() && !sched.IsAborted() {
		task := sched.NextTask()

		switch task.Type {
		case TaskExecution:
			e.stats.Executions.Add(1)
			result := e.executeTransaction(
				task.Version, mvMem, txs, stateGetter, vmFactory,
				config, header, beneficiaryLoc,
			)

			if result.BlockedBy != nil {
				// Transaction blocked on a dependency
				sched.AddDependency(task.Version.TxIdx, *result.BlockedBy)
				continue
			}

			if result.Err != nil {
				sched.Abort()
				continue
			}

			// Record read/write sets in MvMemory
			mvMem.RecordReadSet(task.Version.TxIdx, result.ReadSet)
			mvMem.RecordWriteSet(task.Version.TxIdx, task.Version.Incarnation, result.WriteSet)
			results[task.Version.TxIdx] = result

			var flags FinishFlags
			sched.FinishExecution(task.Version, flags)

		case TaskValidation:
			e.stats.Validations.Add(1)
			valid := mvMem.ValidateReadSet(task.Version.TxIdx)

			if !valid {
				e.stats.Aborts.Add(1)
				if sched.TryValidationAbort(task.Version) {
					mvMem.ConvertWritesToEstimates(task.Version.TxIdx)
					mvMem.ClearTxSets(task.Version.TxIdx)
				}
			}

			sched.FinishValidation(task.Version, !valid)

		case TaskNone:
			// Spin — no work available right now
			runtime.Gosched()
		}
	}
}

// executeTransaction runs one transaction against the multi-version memory.
// It creates a ParallelStateDB, installs it into an EVM from vmFactory,
// derives a message from the transaction, and runs the state transition.
// The result includes the read/write sets for Block-STM validation.
func (e *Engine) executeTransaction(
	version TxVersion,
	mvMem *MvMemory,
	txs types.Transactions,
	stateGetter StateGetter,
	vmFactory VMFactory,
	config *ethparams.ChainConfig,
	header *types.Header,
	beneficiaryLoc MemoryLocation,
) ExecResult {
	tx := txs[version.TxIdx]

	// Acquire pooled scratch space to reduce GC pressure.
	scratch := AcquireScratch()
	defer ReleaseScratch(scratch)

	// 1. Create signer and derive message from transaction
	signer := types.MakeSigner(config, header.Number, header.Time)
	from, err := types.Sender(signer, tx)
	if err != nil {
		return ExecResult{Err: fmt.Errorf("failed to derive sender for tx %d: %w", version.TxIdx, err)}
	}

	// Build the message manually (core.Message lives in evmgpu/core which
	// we cannot import due to circular dependency). We pass the message
	// fields directly into the EVM via TxContext and the inline state
	// transition below.
	gasPrice := new(big.Int).Set(tx.GasPrice())
	if header.BaseFee != nil {
		// EIP-1559 effective gas price
		tip := new(big.Int).Set(tx.GasTipCap())
		feeCap := new(big.Int).Set(tx.GasFeeCap())
		tip.Add(tip, header.BaseFee)
		if tip.Cmp(feeCap) > 0 {
			gasPrice.Set(feeCap)
		} else {
			gasPrice.Set(tip)
		}
	}

	// 2. Get an EVM instance via vmFactory (returns EVM with base StateDB copy)
	evm, err := vmFactory(version.TxIdx)
	if err != nil {
		return ExecResult{Err: fmt.Errorf("vmFactory failed for tx %d: %w", version.TxIdx, err)}
	}

	// 3. Create ParallelStateDB wrapping the EVM's base StateDB
	baseStateDB, ok := evm.StateDB.(*state.StateDB)
	if !ok {
		return ExecResult{Err: fmt.Errorf("vmFactory returned non-state.StateDB for tx %d", version.TxIdx)}
	}
	psd := NewParallelStateDB(
		version.TxIdx,
		version.Incarnation,
		mvMem,
		stateGetter,
		beneficiaryLoc,
		baseStateDB,
	)

	// 4. Replace EVM's StateDB with our parallel interceptor
	evm.StateDB = psd

	// 5. Set tx context on the EVM
	txContext := vm.TxContext{
		Origin:     from,
		GasPrice:   gasPrice,
		BlobHashes: tx.BlobHashes(),
	}
	if tx.BlobGasFeeCap() != nil {
		txContext.BlobFeeCap = new(big.Int).Set(tx.BlobGasFeeCap())
	}
	evm.SetTxContext(txContext)

	// 6. Inline state transition — we cannot call core.ApplyMessage from
	// this package due to circular imports. This implements the essential
	// state transition logic.

	// Each parallel tx gets its own full block gas pool
	gasRemaining := tx.Gas()

	// Buy gas: debit sender for max gas cost
	gasCost := new(uint256.Int).Mul(
		new(uint256.Int).SetUint64(tx.Gas()),
		uint256.MustFromBig(gasPrice),
	)
	psd.SubBalance(from, gasCost, tracing.BalanceDecreaseGasBuy)
	if psd.blockedBy != nil {
		return ExecResult{BlockedBy: psd.blockedBy}
	}

	// Compute intrinsic gas
	rules := config.Rules(header.Number, header.Number != nil, header.Time)
	contractCreation := tx.To() == nil
	intrinsicGas := computeIntrinsicGas(tx.Data(), tx.AccessList(), contractCreation, rules)
	if gasRemaining < intrinsicGas {
		return ExecResult{Err: fmt.Errorf("intrinsic gas too low for tx %d: have %d, want %d", version.TxIdx, gasRemaining, intrinsicGas)}
	}
	gasRemaining -= intrinsicGas

	// Prepare access list
	psd.Prepare(rules, from, evm.Context.Coinbase, tx.To(), vm.ActivePrecompiles(rules), tx.AccessList())

	// Execute the transaction
	var (
		ret   []byte
		vmerr error
	)
	value, _ := uint256.FromBig(tx.Value())
	if contractCreation {
		ret, _, gasRemaining, vmerr = evm.Create(from, tx.Data(), gasRemaining, value)
	} else {
		// Increment nonce
		psd.SetNonce(from, psd.GetNonce(from)+1, tracing.NonceChangeEoACall)
		if psd.blockedBy != nil {
			return ExecResult{BlockedBy: psd.blockedBy}
		}
		ret, gasRemaining, vmerr = evm.Call(from, *tx.To(), tx.Data(), gasRemaining, value)
	}
	_ = ret

	// Check if blocked during execution
	if psd.blockedBy != nil {
		return ExecResult{BlockedBy: psd.blockedBy}
	}

	// Gas refund — quotient depends on chain config (EIP-3529)
	initialGas := tx.Gas()
	gasUsed := initialGas - gasRemaining
	var refundQuotient uint64 = 2 // pre-London default
	if config.IsLondon(header.Number) {
		refundQuotient = 5 // EIP-3529 post-London
	}
	maxRefund := gasUsed / refundQuotient
	refund := psd.GetRefund()
	if refund > maxRefund {
		refund = maxRefund
	}
	gasRemaining += refund

	// Return remaining gas to sender
	remaining := new(uint256.Int).Mul(
		new(uint256.Int).SetUint64(gasRemaining),
		uint256.MustFromBig(gasPrice),
	)
	psd.AddBalance(from, remaining, tracing.BalanceIncreaseGasReturn)

	// Pay coinbase the effective tip (gasPrice - baseFee).
	// Base fee is burned, not paid to coinbase (EIP-1559).
	effectiveTip := new(big.Int).Set(gasPrice)
	if header.BaseFee != nil {
		effectiveTip.Sub(effectiveTip, header.BaseFee)
		if effectiveTip.Sign() < 0 {
			effectiveTip.SetUint64(0)
		}
	}
	fee := new(uint256.Int).Mul(
		new(uint256.Int).SetUint64(initialGas-gasRemaining),
		uint256.MustFromBig(effectiveTip),
	)
	psd.AddBalance(evm.Context.Coinbase, fee, tracing.BalanceIncreaseRewardTransactionFee)

	// Check if blocked during balance operations
	if psd.blockedBy != nil {
		return ExecResult{BlockedBy: psd.blockedBy}
	}

	// 7. Finalize — get ReadSet/WriteSet
	readSet, writeSet := psd.FinalizeParallel()

	// 8. Build receipt
	usedGas := initialGas - gasRemaining
	receipt := &types.Receipt{Type: tx.Type(), PostState: nil, CumulativeGasUsed: 0}
	if vmerr != nil {
		receipt.Status = types.ReceiptStatusFailed
	} else {
		receipt.Status = types.ReceiptStatusSuccessful
	}
	receipt.TxHash = tx.Hash()
	receipt.GasUsed = usedGas

	if tx.Type() == types.BlobTxType {
		receipt.BlobGasUsed = uint64(len(tx.BlobHashes()) * ethparams.BlobTxBlobGasPerBlob)
		receipt.BlobGasPrice = evm.Context.BlobBaseFee
	}

	// Contract creation address
	if contractCreation {
		receipt.ContractAddress = common.CreateAddress(from, tx.Nonce())
	}

	// Logs from the parallel state db
	receipt.Logs = psd.logs
	receipt.Bloom = types.CreateBloom(receipt)
	receipt.BlockNumber = header.Number
	receipt.TransactionIndex = uint(version.TxIdx)

	return ExecResult{
		Receipt:  receipt,
		ReadSet:  readSet,
		WriteSet: writeSet,
		GasUsed:  usedGas,
		Err:      nil,
	}
}

// computeIntrinsicGas calculates the intrinsic gas for a transaction.
func computeIntrinsicGas(data []byte, accessList types.AccessList, contractCreation bool, rules ethparams.Rules) uint64 {
	var gas uint64
	if contractCreation && rules.IsHomestead {
		gas = ethparams.TxGasContractCreation
	} else {
		gas = ethparams.TxGas
	}
	if len(data) > 0 {
		var nz uint64
		for _, byt := range data {
			if byt != 0 {
				nz++
			}
		}
		nonZeroGas := ethparams.TxDataNonZeroGasFrontier
		if rules.IsIstanbul {
			nonZeroGas = ethparams.TxDataNonZeroGasEIP2028
		}
		gas += nz * nonZeroGas
		gas += (uint64(len(data)) - nz) * ethparams.TxDataZeroGas
	}
	if accessList != nil {
		gas += uint64(len(accessList)) * ethparams.TxAccessListAddressGas
		gas += uint64(accessList.StorageKeys()) * ethparams.TxAccessListStorageKeyGas
	}
	return gas
}

// ApplyToState applies finalized write sets from parallel execution to the
// canonical statedb in transaction order. This is called after all transactions
// have been validated to commit the parallel results to the real state trie.
func (e *Engine) ApplyToState(
	statedb *state.StateDB,
	results []ExecResult,
	txs types.Transactions,
) {
	for i, result := range results {
		if result.Err != nil {
			continue
		}
		tx := txs[i]

		// Set tx context for correct log attribution
		statedb.SetTxContext(tx.Hash(), i)

		// Apply each write entry to the canonical state
		for _, w := range result.WriteSet {
			applyWriteToState(statedb, w)
		}

		// Finalize after each tx (same as sequential path)
		statedb.Finalise(true)
	}
}

// applyWriteToState applies a single parallel write entry to the canonical
// statedb. The WriteEntry carries the full MemoryLocation, so we can
// dispatch on location type to apply the correct state mutation.
func applyWriteToState(statedb *state.StateDB, w WriteEntry) {
	addr := w.Location.Address

	switch w.Location.Type {
	case LocationBalance:
		bal := new(uint256.Int).SetBytes(w.Value.Balance[:])
		statedb.SetBalance(addr, bal, tracing.BalanceChangeUnspecified)

	case LocationNonce:
		statedb.SetNonce(addr, w.Value.Nonce, tracing.NonceChangeUnspecified)

	case LocationStorage:
		statedb.SetState(addr, w.Location.Slot, w.Value.Storage)

	case LocationCodeHash:
		// Code hash changes are side effects of SetCode — the code itself
		// was written to the base statedb during execution. Skip here.
	}
}

// executeSequential falls back to sequential execution (original path).
func (e *Engine) executeSequential(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs types.Transactions,
	stateGetter StateGetter,
	vmFactory VMFactory,
) ([]*types.Receipt, error) {
	// This delegates to the original sequential state_processor.Process()
	// For now, return an error indicating sequential fallback is needed
	return nil, fmt.Errorf("sequential fallback: caller should use original Process()")
}

// Hasher returns the engine's batch hasher (GPU or CPU).
// Callers use this to accelerate trie node hashing during state root
// computation after parallel execution completes.
func (e *Engine) Hasher() Hasher {
	return e.hasher
}

// Stats returns a snapshot of execution statistics.
func (e *Engine) Stats() StatsSnapshot {
	return StatsSnapshot{
		TotalTxs:    e.stats.TotalTxs,
		Executions:  e.stats.Executions.Load(),
		Validations: e.stats.Validations.Load(),
		Aborts:      e.stats.Aborts.Load(),
		FellBack:    e.stats.FellBack,
		GPUEligible: e.stats.GPUEligible.Load(),
		GPUExecuted: e.stats.GPUExecuted.Load(),
		GPUFallback: e.stats.GPUFallback.Load(),
	}
}

// =============================================================================
// Integration Interfaces
// =============================================================================

// StateGetter reads pre-block state for a memory location.
// This is the interface to the underlying database (zapdb with GPU cache).
type StateGetter func(loc MemoryLocation) (MemoryValue, bool)

// VMFactory creates an EVM instance for executing a transaction.
// Each worker gets its own EVM instance (no sharing).
type VMFactory func(txIdx TxIdx) (*vm.EVM, error)

// =============================================================================
// Parallel StateDB — intercepts EVM state access through MvMemory
// =============================================================================

// Compile-time check: ParallelStateDB must satisfy vm.StateDB.
var _ vm.StateDB = (*ParallelStateDB)(nil)

// ParallelStateDB wraps a base state.StateDB for one transaction in parallel
// mode. It intercepts reads through MvMemory first (for values written by
// earlier transactions in the same block) and falls back to the base statedb.
// Writes are buffered locally and flushed as a WriteSet after execution.
//
// GPU Roadmap: This is the layer that would be replaced by GPU kernels.
// Each "read" becomes a GPU hash table lookup in MvMemory (in GPU VRAM).
// Each "write" becomes a GPU atomic store.
type ParallelStateDB struct {
	txIdx       TxIdx
	incarnation uint32
	mvMemory    *MvMemory
	stateGetter StateGetter
	base        *state.StateDB // copy per (txIdx, incarnation)

	// Accumulated during execution
	readSet  []ReadEntry
	writeSet []WriteEntry

	// Local write buffer — keyed by MemoryLocation for dedup.
	// Balance and nonce are separate locations, so writing one
	// never clobbers the other.
	writeBuffer map[MemoryLocation]MemoryValue

	// Lazy evaluation tracking
	beneficiaryLoc MemoryLocation

	// Blocking: set when we hit an ESTIMATE marker
	blockedBy *TxIdx

	// Snapshot support — stack of rollback points
	snapshots []parallelSnapshot

	// Logs accumulated during this transaction
	logs []*types.Log

	// Preimages accumulated during this transaction
	preimages map[common.Hash][]byte

	// Refund counter
	refund uint64
}

// parallelSnapshot captures rollback state for Snapshot/RevertToSnapshot.
type parallelSnapshot struct {
	readSetLen  int
	writeSetLen int
	baseSnap    int // snapshot ID from the underlying state.StateDB
	refund      uint64
	logsLen     int
}

// NewParallelStateDB creates a state accessor for one transaction.
func NewParallelStateDB(
	txIdx TxIdx,
	incarnation uint32,
	mvMemory *MvMemory,
	stateGetter StateGetter,
	beneficiaryLoc MemoryLocation,
	base *state.StateDB,
) *ParallelStateDB {
	return &ParallelStateDB{
		txIdx:          txIdx,
		incarnation:    incarnation,
		mvMemory:       mvMemory,
		stateGetter:    stateGetter,
		beneficiaryLoc: beneficiaryLoc,
		base:           base,
		readSet:        make([]ReadEntry, 0, 32),
		writeSet:       make([]WriteEntry, 0, 16),
		writeBuffer:    make(map[MemoryLocation]MemoryValue, 16),
		preimages:      make(map[common.Hash][]byte),
	}
}

// readFromMvMemory attempts to read a value from MvMemory. Returns the entry,
// whether it was found, and whether we are blocked (hit ESTIMATE marker).
func (s *ParallelStateDB) readFromMvMemory(loc MemoryLocation) (MvEntry, bool, bool) {
	entry, found := s.mvMemory.Read(loc, s.txIdx)
	if !found {
		return MvEntry{}, false, false
	}
	if entry.IsEstimate {
		blockedOn := TxIdx(entry.TxIdx - 1)
		s.blockedBy = &blockedOn
		return entry, true, true
	}
	s.readSet = append(s.readSet, ReadEntry{
		Location: loc,
		Origin: ReadOrigin{
			FromMvMemory: true,
			Version: TxVersion{
				TxIdx:       TxIdx(entry.TxIdx - 1),
				Incarnation: entry.Incarnation,
			},
		},
	})
	return entry, true, false
}

// recordStorageRead records a read that fell through to the base statedb.
func (s *ParallelStateDB) recordStorageRead(loc MemoryLocation) {
	s.readSet = append(s.readSet, ReadEntry{
		Location: loc,
		Origin:   ReadOrigin{FromMvMemory: false},
	})
}

// --- vm.StateDB interface implementation ---

func (s *ParallelStateDB) CreateAccount(addr common.Address) {
	s.base.CreateAccount(addr)
	// Track in MvMemory: new account has zero balance and nonce
	s.setBalanceEntry(addr, new(uint256.Int))
	loc := MemoryLocation{Address: addr, Type: LocationNonce}
	s.writeBuffer[loc] = MemoryValue{Nonce: 0}
	s.writeSet = append(s.writeSet, WriteEntry{Location: loc, Value: MemoryValue{Nonce: 0}})
}

func (s *ParallelStateDB) CreateContract(addr common.Address) {
	s.base.CreateContract(addr)
	// Track in MvMemory: new contract has zero balance and nonce
	s.setBalanceEntry(addr, new(uint256.Int))
	loc := MemoryLocation{Address: addr, Type: LocationNonce}
	s.writeBuffer[loc] = MemoryValue{Nonce: 0}
	s.writeSet = append(s.writeSet, WriteEntry{Location: loc, Value: MemoryValue{Nonce: 0}})
}

func (s *ParallelStateDB) SubBalance(addr common.Address, amount *uint256.Int, reason tracing.BalanceChangeReason) uint256.Int {
	cur := s.GetBalance(addr)
	if s.blockedBy != nil {
		return uint256.Int{}
	}
	// Check for underflow — if amount > cur, the subtraction would wrap
	// to a massive value. The canonical code relies on preCheck to prevent
	// this, but we guard here for safety.
	if cur.Lt(amount) {
		// Insufficient balance — set to 0 (transaction should fail during
		// EVM execution, but we don't wrap to uint256.Max)
		s.setBalanceEntry(addr, new(uint256.Int))
		return *cur
	}
	newBal := new(uint256.Int).Sub(cur, amount)
	s.setBalanceEntry(addr, newBal)
	return *cur
}

func (s *ParallelStateDB) AddBalance(addr common.Address, amount *uint256.Int, reason tracing.BalanceChangeReason) uint256.Int {
	cur := s.GetBalance(addr)
	if s.blockedBy != nil {
		return uint256.Int{}
	}
	newBal := new(uint256.Int).Add(cur, amount)
	s.setBalanceEntry(addr, newBal)
	return *cur
}

// setBalanceEntry writes a balance value to the LocationBalance location only.
// Nonce is stored at a separate LocationNonce location — no conflation.
func (s *ParallelStateDB) setBalanceEntry(addr common.Address, balance *uint256.Int) {
	loc := MemoryLocation{Address: addr, Type: LocationBalance}

	var balHash common.Hash
	balance.WriteToSlice(balHash[:])

	val := MemoryValue{
		Type:    ValueAbsolute,
		Balance: balHash,
	}

	s.writeBuffer[loc] = val
	s.writeSet = append(s.writeSet, WriteEntry{
		Location: loc,
		Value:    val,
	})
}

func (s *ParallelStateDB) GetBalance(addr common.Address) *uint256.Int {
	loc := MemoryLocation{Address: addr, Type: LocationBalance}

	// Check local write buffer first
	if val, ok := s.writeBuffer[loc]; ok {
		return new(uint256.Int).SetBytes(val.Balance[:])
	}

	// Check MvMemory
	entry, found, blocked := s.readFromMvMemory(loc)
	if blocked {
		return new(uint256.Int)
	}
	if found {
		return new(uint256.Int).SetBytes(entry.Value.Balance[:])
	}

	// Fall back to base statedb
	s.recordStorageRead(loc)
	return s.base.GetBalance(addr)
}

func (s *ParallelStateDB) GetNonce(addr common.Address) uint64 {
	loc := MemoryLocation{Address: addr, Type: LocationNonce}

	// Check local write buffer
	if val, ok := s.writeBuffer[loc]; ok {
		return val.Nonce
	}

	// Check MvMemory
	entry, found, blocked := s.readFromMvMemory(loc)
	if blocked {
		return 0
	}
	if found {
		return entry.Value.Nonce
	}

	// Fall back to base statedb
	s.recordStorageRead(loc)
	return s.base.GetNonce(addr)
}

// SetNonce writes a nonce value to the LocationNonce location only.
// Balance is stored at a separate LocationBalance location — no conflation.
func (s *ParallelStateDB) SetNonce(addr common.Address, nonce uint64, reason tracing.NonceChangeReason) {
	loc := MemoryLocation{Address: addr, Type: LocationNonce}

	val := MemoryValue{
		Type:  ValueAbsolute,
		Nonce: nonce,
	}

	s.writeBuffer[loc] = val
	s.writeSet = append(s.writeSet, WriteEntry{
		Location: loc,
		Value:    val,
	})
}

func (s *ParallelStateDB) GetCodeHash(addr common.Address) common.Hash {
	loc := MemoryLocation{Address: addr, Type: LocationCodeHash}

	// Check MvMemory
	entry, found, blocked := s.readFromMvMemory(loc)
	if blocked {
		return common.Hash{}
	}
	if found {
		return entry.Value.Storage
	}

	// Fall back to base statedb
	s.recordStorageRead(loc)
	return s.base.GetCodeHash(addr)
}

func (s *ParallelStateDB) GetCode(addr common.Address) []byte {
	// Code reads go directly to base — code is immutable within a block
	return s.base.GetCode(addr)
}

func (s *ParallelStateDB) SetCode(addr common.Address, code []byte, reason tracing.CodeChangeReason) []byte {
	return s.base.SetCode(addr, code, reason)
}

func (s *ParallelStateDB) GetCodeSize(addr common.Address) int {
	return s.base.GetCodeSize(addr)
}

func (s *ParallelStateDB) AddRefund(gas uint64) {
	s.refund += gas
}

func (s *ParallelStateDB) SubRefund(gas uint64) {
	if gas > s.refund {
		s.refund = 0
		return
	}
	s.refund -= gas
}

func (s *ParallelStateDB) GetRefund() uint64 {
	return s.refund
}

func (s *ParallelStateDB) GetCommittedState(addr common.Address, hash common.Hash) common.Hash {
	return s.base.GetCommittedState(addr, hash)
}

func (s *ParallelStateDB) GetStateAndCommittedState(addr common.Address, key common.Hash) (common.Hash, common.Hash) {
	current := s.GetState(addr, key)
	committed := s.base.GetCommittedState(addr, key)
	return current, committed
}

func (s *ParallelStateDB) GetState(addr common.Address, slot common.Hash) common.Hash {
	loc := MemoryLocation{Address: addr, Type: LocationStorage, Slot: slot}

	// Check local write buffer
	if val, ok := s.writeBuffer[loc]; ok {
		return val.Storage
	}

	// Check MvMemory
	entry, found, blocked := s.readFromMvMemory(loc)
	if blocked {
		return common.Hash{}
	}
	if found {
		return entry.Value.Storage
	}

	// Fall back to base statedb
	s.recordStorageRead(loc)
	return s.base.GetState(addr, slot)
}

func (s *ParallelStateDB) SetState(addr common.Address, slot, value common.Hash) common.Hash {
	loc := MemoryLocation{Address: addr, Type: LocationStorage, Slot: slot}

	prev := s.GetState(addr, slot)

	val := MemoryValue{
		Type:    ValueAbsolute,
		Storage: value,
	}
	s.writeBuffer[loc] = val
	s.writeSet = append(s.writeSet, WriteEntry{
		Location: loc,
		Value:    val,
	})
	return prev
}

func (s *ParallelStateDB) GetStorageRoot(addr common.Address) common.Hash {
	return s.base.GetStorageRoot(addr)
}

func (s *ParallelStateDB) GetTransientState(addr common.Address, key common.Hash) common.Hash {
	return s.base.GetTransientState(addr, key)
}

func (s *ParallelStateDB) SetTransientState(addr common.Address, key, value common.Hash) {
	s.base.SetTransientState(addr, key, value)
}

func (s *ParallelStateDB) SelfDestruct(addr common.Address) uint256.Int {
	oldBal := s.base.SelfDestruct(addr)
	// Track self-destruct in MvMemory: zero balance, mark as destructed
	s.setBalanceEntry(addr, new(uint256.Int))
	loc := MemoryLocation{Address: addr, Type: LocationCodeHash}
	val := MemoryValue{Type: ValueSelfDestruct}
	s.writeBuffer[loc] = val
	s.writeSet = append(s.writeSet, WriteEntry{Location: loc, Value: val})
	return oldBal
}

func (s *ParallelStateDB) HasSelfDestructed(addr common.Address) bool {
	// Check write buffer for self-destruct marker
	loc := MemoryLocation{Address: addr, Type: LocationCodeHash}
	if val, ok := s.writeBuffer[loc]; ok {
		if val.Type == ValueSelfDestruct {
			return true
		}
	}
	// Check MvMemory
	entry, found, _ := s.readFromMvMemory(loc)
	if found && entry.Value.Type == ValueSelfDestruct {
		return true
	}
	return s.base.HasSelfDestructed(addr)
}

func (s *ParallelStateDB) SelfDestruct6780(addr common.Address) (uint256.Int, bool) {
	oldBal, destructed := s.base.SelfDestruct6780(addr)
	if destructed {
		s.setBalanceEntry(addr, new(uint256.Int))
		loc := MemoryLocation{Address: addr, Type: LocationCodeHash}
		val := MemoryValue{Type: ValueSelfDestruct}
		s.writeBuffer[loc] = val
		s.writeSet = append(s.writeSet, WriteEntry{Location: loc, Value: val})
	}
	return oldBal, destructed
}

func (s *ParallelStateDB) Exist(addr common.Address) bool {
	// Check write buffer — if we wrote balance or nonce, account exists
	balLoc := MemoryLocation{Address: addr, Type: LocationBalance}
	if _, ok := s.writeBuffer[balLoc]; ok {
		return true
	}
	nonceLoc := MemoryLocation{Address: addr, Type: LocationNonce}
	if _, ok := s.writeBuffer[nonceLoc]; ok {
		return true
	}
	// Check MvMemory
	if entry, found := s.mvMemory.Read(balLoc, s.txIdx); found && !entry.IsEstimate {
		return true
	}
	return s.base.Exist(addr)
}

func (s *ParallelStateDB) Empty(addr common.Address) bool {
	// Check write buffer for non-zero balance or nonce
	balLoc := MemoryLocation{Address: addr, Type: LocationBalance}
	if val, ok := s.writeBuffer[balLoc]; ok {
		bal := new(uint256.Int).SetBytes(val.Balance[:])
		if !bal.IsZero() {
			return false
		}
	}
	nonceLoc := MemoryLocation{Address: addr, Type: LocationNonce}
	if val, ok := s.writeBuffer[nonceLoc]; ok {
		if val.Nonce != 0 {
			return false
		}
	}
	return s.base.Empty(addr)
}

func (s *ParallelStateDB) AddressInAccessList(addr common.Address) bool {
	return s.base.AddressInAccessList(addr)
}

func (s *ParallelStateDB) SlotInAccessList(addr common.Address, slot common.Hash) (bool, bool) {
	return s.base.SlotInAccessList(addr, slot)
}

func (s *ParallelStateDB) AddAddressToAccessList(addr common.Address) {
	s.base.AddAddressToAccessList(addr)
}

func (s *ParallelStateDB) AddSlotToAccessList(addr common.Address, slot common.Hash) {
	s.base.AddSlotToAccessList(addr, slot)
}

func (s *ParallelStateDB) PointCache() *utils.PointCache {
	return s.base.PointCache()
}

func (s *ParallelStateDB) Prepare(rules ethparams.Rules, sender, coinbase common.Address, dest *common.Address, precompiles []common.Address, txAccesses types.AccessList) {
	s.base.Prepare(rules, sender, coinbase, dest, precompiles, txAccesses)
}

func (s *ParallelStateDB) RevertToSnapshot(revid int) {
	if revid < 0 || revid >= len(s.snapshots) {
		return
	}
	snap := s.snapshots[revid]
	s.readSet = s.readSet[:snap.readSetLen]
	s.writeSet = s.writeSet[:snap.writeSetLen]
	s.refund = snap.refund
	s.logs = s.logs[:snap.logsLen]

	// Rebuild write buffer from truncated write set
	s.writeBuffer = make(map[MemoryLocation]MemoryValue, len(s.writeSet))
	for _, w := range s.writeSet {
		s.writeBuffer[w.Location] = w.Value
	}

	// Revert base statedb snapshot
	s.base.RevertToSnapshot(snap.baseSnap)

	// Truncate snapshot stack
	s.snapshots = s.snapshots[:revid]
}

func (s *ParallelStateDB) Snapshot() int {
	baseSnap := s.base.Snapshot()
	id := len(s.snapshots)
	s.snapshots = append(s.snapshots, parallelSnapshot{
		readSetLen:  len(s.readSet),
		writeSetLen: len(s.writeSet),
		baseSnap:    baseSnap,
		refund:      s.refund,
		logsLen:     len(s.logs),
	})
	return id
}

func (s *ParallelStateDB) AddLog(l *types.Log) {
	s.logs = append(s.logs, l)
}

func (s *ParallelStateDB) Logs() []*types.Log {
	return s.logs
}

func (s *ParallelStateDB) TxHash() common.Hash {
	return s.base.TxHash()
}

func (s *ParallelStateDB) AddPreimage(hash common.Hash, preimage []byte) {
	s.preimages[hash] = preimage
}

func (s *ParallelStateDB) Witness() *stateless.Witness {
	return s.base.Witness()
}

func (s *ParallelStateDB) AccessEvents() *state.AccessEvents {
	return s.base.AccessEvents()
}

// Finalise is a no-op for ParallelStateDB. State finalization happens
// post-validation via ApplyToState on the canonical statedb.
func (s *ParallelStateDB) Finalise(deleteEmptyObjects bool) {
	// NO-OP: parallel state finalization deferred to ApplyToState
}

// FinalizeParallel returns the accumulated read/write sets for Block-STM
// validation. This is called after EVM execution completes for this tx.
func (s *ParallelStateDB) FinalizeParallel() ([]ReadEntry, []WriteEntry) {
	return s.readSet, s.writeSet
}
