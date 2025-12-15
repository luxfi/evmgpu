// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestSchedulerBasicFlow verifies the full lifecycle: execute all txs, validate
// all txs, and confirm Done() returns true.
func TestSchedulerBasicFlow(t *testing.T) {
	const n = 5
	s := NewScheduler(n)

	if s.Done() {
		t.Fatal("scheduler should not be done before any work")
	}

	// Grab all execution tasks
	versions := make([]TxVersion, n)
	for i := 0; i < n; i++ {
		task := s.NextTask()
		if task.Type != TaskExecution {
			t.Fatalf("expected TaskExecution, got %d (iter %d)", task.Type, i)
		}
		if task.Version.Incarnation != 0 {
			t.Fatalf("expected incarnation 0, got %d", task.Version.Incarnation)
		}
		versions[i] = task.Version
	}

	// No more execution tasks
	task := s.NextTask()
	if task.Type != TaskNone {
		t.Fatalf("expected TaskNone after all executions claimed, got %d", task.Type)
	}

	// Finish all executions
	for _, v := range versions {
		s.FinishExecution(v, 0)
	}

	// Now validation tasks should appear
	validated := 0
	for validated < n {
		task := s.NextTask()
		if task.Type == TaskNone {
			runtime.Gosched()
			continue
		}
		if task.Type != TaskValidation {
			t.Fatalf("expected TaskValidation, got %d", task.Type)
		}
		s.FinishValidation(task.Version, false)
		validated++
	}

	if !s.Done() {
		t.Fatal("scheduler should be done after all txs validated")
	}
}

// TestSchedulerValidationPriority verifies that validation tasks are returned
// before execution tasks when both are available. We execute and finish tx 0,
// then check that the next task is a validation (for tx 0) rather than an
// execution (for tx 1).
func TestSchedulerValidationPriority(t *testing.T) {
	const n = 5
	s := NewScheduler(n)

	// Execute tx 0
	task0 := s.NextTask()
	if task0.Type != TaskExecution {
		t.Fatalf("expected TaskExecution, got %d", task0.Type)
	}
	s.FinishExecution(task0.Version, 0)

	// Now both validation (tx 0) and execution (tx 1) are available.
	// Scheduler should prefer validation.
	task := s.NextTask()
	if task.Type != TaskValidation {
		t.Fatalf("expected TaskValidation (priority), got %d", task.Type)
	}
	if task.Version.TxIdx != 0 {
		t.Fatalf("expected validation of tx 0, got tx %d", task.Version.TxIdx)
	}
}

// TestSchedulerAbortAndReexecute verifies that aborting a tx via
// TryValidationAbort increments its incarnation and re-schedules it.
func TestSchedulerAbortAndReexecute(t *testing.T) {
	s := NewScheduler(3)

	// Execute all 3, abort tx 1 during its first validation,
	// then drive to completion and verify incarnation 1 was re-executed.
	aborted := false
	foundReexec := false
	deadline := time.Now().Add(5 * time.Second)

	for !s.Done() && time.Now().Before(deadline) {
		task := s.NextTask()
		switch task.Type {
		case TaskExecution:
			if task.Version.TxIdx == 1 && task.Version.Incarnation == 1 {
				foundReexec = true
			}
			s.FinishExecution(task.Version, 0)
		case TaskValidation:
			if task.Version.TxIdx == 1 && !aborted {
				// Abort tx 1 on its first validation
				if s.TryValidationAbort(task.Version) {
					aborted = true
				}
			} else {
				s.FinishValidation(task.Version, false)
			}
		case TaskNone:
			runtime.Gosched()
		}
	}

	if !aborted {
		t.Fatal("tx 1 was never aborted")
	}
	if !foundReexec {
		t.Fatal("tx 1 was not re-executed with incarnation 1")
	}
	if !s.Done() {
		t.Fatal("scheduler should be done after re-execution")
	}
}

// TestSchedulerDependency verifies that AddDependency blocks a tx until its
// dependency finishes.
func TestSchedulerDependency(t *testing.T) {
	const n = 8
	s := NewScheduler(n)

	// Execute tx 0..4
	versions := make(map[TxIdx]TxVersion)
	for i := 0; i < 5; i++ {
		task := s.NextTask()
		if task.Type != TaskExecution {
			t.Fatalf("expected TaskExecution, got %d", task.Type)
		}
		versions[task.Version.TxIdx] = task.Version
	}

	// Finish tx 0, 1, 2 (but NOT tx 3, 4)
	s.FinishExecution(versions[0], 0)
	s.FinishExecution(versions[1], 0)
	s.FinishExecution(versions[2], 0)

	// Tx 5 depends on tx 3 (which has not finished yet)
	// First grab tx 5's execution task
	task5 := s.NextTask()
	if task5.Type != TaskExecution && task5.Type != TaskValidation {
		// Might get validation first due to priority; drain validations
		for task5.Type == TaskValidation {
			s.FinishValidation(task5.Version, false)
			task5 = s.NextTask()
		}
	}

	// Now add dependency: tx 5 blocked on tx 3
	s.AddDependency(5, 3)

	// Tx 5 should NOT be returned by NextTask until tx 3 finishes
	for i := 0; i < 20; i++ {
		task := s.NextTask()
		if task.Type == TaskExecution && task.Version.TxIdx == 5 {
			t.Fatal("tx 5 should be blocked on tx 3")
		}
		if task.Type != TaskNone {
			// Process other tasks
			if task.Type == TaskExecution {
				s.FinishExecution(task.Version, 0)
			} else if task.Type == TaskValidation {
				s.FinishValidation(task.Version, false)
			}
		}
		runtime.Gosched()
	}

	// Finish tx 3 — should unblock tx 5
	s.FinishExecution(versions[3], 0)

	// Tx 5 should now be available
	found := false
	for i := 0; i < 100; i++ {
		task := s.NextTask()
		if task.Type == TaskExecution && task.Version.TxIdx == 5 {
			found = true
			s.FinishExecution(task.Version, 0)
			break
		}
		if task.Type == TaskValidation {
			s.FinishValidation(task.Version, false)
		}
		runtime.Gosched()
	}
	if !found {
		t.Fatal("tx 5 should have been unblocked after tx 3 finished")
	}
}

// TestSchedulerDependencyAlreadyDone verifies the race fix: if the blocking tx
// has already finished when AddDependency is called, the dependent tx
// self-resumes immediately.
func TestSchedulerDependencyAlreadyDone(t *testing.T) {
	const n = 8
	s := NewScheduler(n)

	// Execute and finish first 5 txs, draining any validation tasks too
	for i := 0; i < 5; i++ {
		for {
			task := s.NextTask()
			if task.Type == TaskExecution {
				s.FinishExecution(task.Version, 0)
				break
			}
			if task.Type == TaskValidation {
				s.FinishValidation(task.Version, false)
			}
			if task.Type == TaskNone {
				runtime.Gosched()
			}
		}
	}
	// Drain remaining validations for executed txs
	for {
		task := s.NextTask()
		if task.Type == TaskValidation {
			s.FinishValidation(task.Version, false)
		} else {
			break
		}
	}

	// Tx 3 is already Executed. Now add dependency: tx 5 blocked on tx 3.
	s.AddDependency(5, 3)

	// Tx 5 should self-resume immediately and be available
	found := false
	for i := 0; i < 100; i++ {
		task := s.NextTask()
		if task.Type == TaskExecution && task.Version.TxIdx == 5 {
			found = true
			s.FinishExecution(task.Version, 0)
			break
		}
		if task.Type == TaskValidation {
			s.FinishValidation(task.Version, false)
		}
		runtime.Gosched()
	}
	if !found {
		t.Fatal("tx 5 should self-resume when blocking tx 3 is already done")
	}
}

// TestSchedulerConcurrent launches multiple goroutines executing and validating
// a 20-tx block. Verifies all txs eventually validate with no panics or stuck
// workers.
func TestSchedulerConcurrent(t *testing.T) {
	const blockSize = 20
	const numWorkers = 4
	s := NewScheduler(blockSize)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	deadline := time.After(10 * time.Second)
	done := make(chan struct{})

	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			for !s.Done() && !s.IsAborted() {
				select {
				case <-done:
					return
				default:
				}

				task := s.NextTask()
				switch task.Type {
				case TaskExecution:
					s.FinishExecution(task.Version, 0)
				case TaskValidation:
					s.FinishValidation(task.Version, false)
				case TaskNone:
					runtime.Gosched()
				}
			}
		}()
	}

	// Wait for completion or timeout
	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()

	select {
	case <-finished:
		// success
	case <-deadline:
		close(done)
		t.Fatal("timed out waiting for concurrent scheduler to complete")
	}

	if !s.Done() {
		t.Fatal("scheduler should be done after concurrent execution")
	}
	if s.IsAborted() {
		t.Fatal("scheduler should not be aborted")
	}
}

// TestSchedulerAbortFlag verifies that Abort() stops task dispatch.
func TestSchedulerAbortFlag(t *testing.T) {
	s := NewScheduler(5)

	if s.IsAborted() {
		t.Fatal("should not be aborted initially")
	}

	s.Abort()

	if !s.IsAborted() {
		t.Fatal("should be aborted after Abort()")
	}

	task := s.NextTask()
	if task.Type != TaskNone {
		t.Fatalf("expected TaskNone after Abort(), got %d", task.Type)
	}
}

// TestSchedulerEmptyBlock verifies that a scheduler with 0 txs is immediately
// done.
func TestSchedulerEmptyBlock(t *testing.T) {
	s := NewScheduler(0)

	if !s.Done() {
		t.Fatal("empty block scheduler should be done immediately")
	}

	task := s.NextTask()
	if task.Type != TaskNone {
		t.Fatalf("expected TaskNone for empty block, got %d", task.Type)
	}
}
