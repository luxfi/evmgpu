// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"sync"
	"sync/atomic"
)

// Scheduler implements collaborative task assignment for Block-STM.
//
// Workers compete for tasks by atomically incrementing execution_idx and
// validation_idx. The scheduler prioritizes validation over execution to
// minimize wasted re-executions. Dependencies between transactions are
// tracked: when a blocked transaction's dependency completes, it is
// automatically re-scheduled.
//
// GPU Design: execution_idx and validation_idx map directly to GPU atomics.
// The status array (one per tx) maps to GPU shared memory with atomic CAS.
// Dependency tracking uses fixed-size arrays indexed by TxIdx.
type Scheduler struct {
	blockSize uint32

	// Per-transaction state (protected by individual mutexes)
	txStates []struct {
		mu    sync.Mutex
		state TxState
	}

	// Per-transaction dependents: who is waiting on this tx
	dependents []struct {
		mu   sync.Mutex
		deps []TxIdx
	}

	// Atomic counters for task selection
	executionIdx    atomic.Uint32 // Next tx to execute
	validationIdx   atomic.Uint32 // Next tx to validate
	numValidated    atomic.Uint32 // Termination counter
	aborted         atomic.Bool   // Fatal error flag
}

// NewScheduler creates a scheduler for a block with the given number of transactions.
func NewScheduler(blockSize uint32) *Scheduler {
	s := &Scheduler{
		blockSize: blockSize,
		txStates:  make([]struct {
			mu    sync.Mutex
			state TxState
		}, blockSize),
		dependents: make([]struct {
			mu   sync.Mutex
			deps []TxIdx
		}, blockSize),
	}

	// All transactions start as ReadyToExecute
	for i := uint32(0); i < blockSize; i++ {
		s.txStates[i].state = TxState{
			Status:      StatusReadyToExecute,
			Incarnation: 0,
		}
	}

	return s
}

// NextTask returns the next task for a worker to execute.
// Returns TaskNone when all transactions are validated (termination).
func (s *Scheduler) NextTask() Task {
	if s.aborted.Load() {
		return Task{Type: TaskNone}
	}

	// Check termination
	if s.numValidated.Load() >= s.blockSize {
		return Task{Type: TaskNone}
	}

	// Prioritize validation over execution
	valIdx := s.validationIdx.Load()
	execIdx := s.executionIdx.Load()

	if valIdx < execIdx && valIdx < s.blockSize {
		// Try to claim a validation task
		idx := s.validationIdx.Add(1) - 1
		if idx < s.blockSize {
			ts := &s.txStates[idx]
			ts.mu.Lock()
			if ts.state.Status == StatusExecuted || ts.state.Status == StatusValidated {
				task := Task{
					Type: TaskValidation,
					Version: TxVersion{
						TxIdx:       TxIdx(idx),
						Incarnation: ts.state.Incarnation,
					},
				}
				ts.mu.Unlock()
				return task
			}
			ts.mu.Unlock()
		}
	}

	// Try to claim an execution task
	if execIdx < s.blockSize {
		idx := s.executionIdx.Add(1) - 1
		if idx < s.blockSize {
			return s.tryIncarnate(TxIdx(idx))
		}
	}

	// No tasks available right now — spin
	return Task{Type: TaskNone}
}

// tryIncarnate attempts to start executing a transaction.
func (s *Scheduler) tryIncarnate(txIdx TxIdx) Task {
	ts := &s.txStates[txIdx]
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if ts.state.Status != StatusReadyToExecute {
		return Task{Type: TaskNone}
	}

	ts.state.Status = StatusExecuting
	return Task{
		Type: TaskExecution,
		Version: TxVersion{
			TxIdx:       txIdx,
			Incarnation: ts.state.Incarnation,
		},
	}
}

// FinishExecution marks a transaction as executed and schedules validation.
func (s *Scheduler) FinishExecution(version TxVersion, flags FinishFlags) {
	ts := &s.txStates[version.TxIdx]
	ts.mu.Lock()

	if ts.state.Status != StatusExecuting || ts.state.Incarnation != version.Incarnation {
		ts.mu.Unlock()
		return
	}

	ts.state.Status = StatusExecuted
	ts.mu.Unlock()

	// Resume any transactions that were blocking on us
	s.resumeDependents(version.TxIdx)

	// Schedule validation: lower validationIdx if needed (CAS retry loop)
	for {
		curVal := s.validationIdx.Load()
		if uint32(version.TxIdx) >= curVal {
			break // Already low enough
		}
		if s.validationIdx.CompareAndSwap(curVal, uint32(version.TxIdx)) {
			break
		}
		// CAS failed — another thread moved it; retry
	}
}

// FinishValidation marks a transaction as validated.
// If aborted is true, the transaction will be re-executed.
func (s *Scheduler) FinishValidation(version TxVersion, aborted bool) {
	if aborted {
		return // Already handled by TryValidationAbort
	}

	ts := &s.txStates[version.TxIdx]
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if ts.state.Incarnation != version.Incarnation {
		return
	}

	ts.state.Status = StatusValidated
	s.numValidated.Add(1)
}

// TryValidationAbort attempts to abort a transaction that failed validation.
// Returns true if the abort was successful.
func (s *Scheduler) TryValidationAbort(version TxVersion) bool {
	ts := &s.txStates[version.TxIdx]
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if ts.state.Incarnation != version.Incarnation {
		return false
	}

	if ts.state.Status != StatusExecuted && ts.state.Status != StatusValidated {
		return false
	}

	if ts.state.Status == StatusValidated {
		s.numValidated.Add(^uint32(0)) // -1
	}

	ts.state.Status = StatusAborting
	// Increment incarnation for next attempt
	ts.state.Incarnation++
	ts.state.Status = StatusReadyToExecute

	// Lower execution_idx to re-schedule this tx (CAS retry loop)
	for {
		curExec := s.executionIdx.Load()
		if uint32(version.TxIdx) >= curExec {
			break
		}
		if s.executionIdx.CompareAndSwap(curExec, uint32(version.TxIdx)) {
			break
		}
	}

	return true
}

// AddDependency records that txIdx is blocked on blockingTx.
// When blockingTx completes, txIdx will be re-scheduled.
//
// CRITICAL: We must register the dependency BEFORE setting status to Aborting.
// Otherwise resumeDependents could drain the list before we add to it,
// leaving the tx stuck in Aborting forever. After registration, we check
// if blockingTx already completed — if so, we self-resume immediately.
func (s *Scheduler) AddDependency(txIdx TxIdx, blockingTx TxIdx) {
	// Step 1: Register dependency under the blocking tx's lock.
	// Hold this lock while checking blockingTx's status to close the race window.
	dep := &s.dependents[blockingTx]
	dep.mu.Lock()

	// Check if blockingTx already completed (Executed or Validated).
	// If so, no one will call resumeDependents for it again — we must self-resume.
	blockingState := &s.txStates[blockingTx]
	blockingState.mu.Lock()
	alreadyDone := blockingState.state.Status == StatusExecuted ||
		blockingState.state.Status == StatusValidated
	blockingState.mu.Unlock()

	if !alreadyDone {
		// blockingTx still in progress — register and wait for resume
		dep.deps = append(dep.deps, txIdx)
	}
	dep.mu.Unlock()

	// Step 2: Set our status to Aborting and bump incarnation
	ts := &s.txStates[txIdx]
	ts.mu.Lock()
	ts.state.Status = StatusAborting
	ts.state.Incarnation++

	if alreadyDone {
		// blockingTx already finished — self-resume immediately
		ts.state.Status = StatusReadyToExecute
	}
	ts.mu.Unlock()

	// Step 3: If self-resumed, lower execution_idx
	if alreadyDone {
		for {
			curExec := s.executionIdx.Load()
			if uint32(txIdx) >= curExec {
				break
			}
			if s.executionIdx.CompareAndSwap(curExec, uint32(txIdx)) {
				break
			}
		}
	}
}

// resumeDependents wakes up all transactions blocked on the given tx.
func (s *Scheduler) resumeDependents(txIdx TxIdx) {
	dep := &s.dependents[txIdx]
	dep.mu.Lock()
	deps := dep.deps
	dep.deps = nil
	dep.mu.Unlock()

	for _, blocked := range deps {
		ts := &s.txStates[blocked]
		ts.mu.Lock()
		if ts.state.Status == StatusAborting {
			ts.state.Status = StatusReadyToExecute
			ts.mu.Unlock()
			// Lower execution_idx to pick up the unblocked tx (CAS retry loop)
			for {
				curExec := s.executionIdx.Load()
				if uint32(blocked) >= curExec {
					break
				}
				if s.executionIdx.CompareAndSwap(curExec, uint32(blocked)) {
					break
				}
			}
		} else {
			ts.mu.Unlock()
		}
	}
}

// Abort signals a fatal error — all workers should stop.
func (s *Scheduler) Abort() {
	s.aborted.Store(true)
}

// IsAborted returns true if a fatal error occurred.
func (s *Scheduler) IsAborted() bool {
	return s.aborted.Load()
}

// Done returns true when all transactions are validated.
func (s *Scheduler) Done() bool {
	return s.numValidated.Load() >= s.blockSize
}
