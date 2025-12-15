// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"runtime"
	"sync"

	"github.com/luxfi/geth/common"
)

// Arena provides pre-allocated, reusable memory for EVM parallel execution.
// Reduces GC pressure by pooling frequently allocated objects across
// Block-STM worker goroutines. Each worker pins to an OS thread via
// runtime.LockOSThread() to improve cache locality.
//
// This is the Go approximation of "native state (no GC)" -- we cannot
// eliminate the GC, but we can avoid allocating in the hot path.

// evmStack is a 1024-element EVM operand stack backed by a flat array.
// Pooled to avoid per-transaction heap allocation.
type evmStack struct {
	data [1024]common.Hash
	top  int
}

func (s *evmStack) reset() {
	s.top = 0
}

// evmMemory is a growable byte buffer for EVM memory operations.
// Pre-allocated to 4KB and pooled.
type evmMemory struct {
	data []byte
}

func (m *evmMemory) reset() {
	m.data = m.data[:0]
}

// workerScratch holds per-worker scratch space for one transaction execution.
// Pooled to avoid allocating read/write set slices and write buffers on
// every execution attempt.
type workerScratch struct {
	readSet     []ReadEntry
	writeSet    []WriteEntry
	writeBuffer map[MemoryLocation]MemoryValue
}

func (w *workerScratch) reset() {
	w.readSet = w.readSet[:0]
	w.writeSet = w.writeSet[:0]
	for k := range w.writeBuffer {
		delete(w.writeBuffer, k)
	}
}

// Arena pools for the three hot-path allocations during Block-STM execution.
var (
	stackPool = sync.Pool{
		New: func() any {
			return &evmStack{}
		},
	}

	memoryPool = sync.Pool{
		New: func() any {
			return &evmMemory{
				data: make([]byte, 0, 4096),
			}
		},
	}

	scratchPool = sync.Pool{
		New: func() any {
			return &workerScratch{
				readSet:     make([]ReadEntry, 0, 32),
				writeSet:    make([]WriteEntry, 0, 16),
				writeBuffer: make(map[MemoryLocation]MemoryValue, 16),
			}
		},
	}
)

// AcquireStack returns a pooled EVM stack, reset to empty.
func AcquireStack() *evmStack {
	s := stackPool.Get().(*evmStack)
	s.reset()
	return s
}

// ReleaseStack returns an EVM stack to the pool.
func ReleaseStack(s *evmStack) {
	if s != nil {
		stackPool.Put(s)
	}
}

// AcquireMemory returns a pooled EVM memory buffer, reset to empty.
func AcquireMemory() *evmMemory {
	m := memoryPool.Get().(*evmMemory)
	m.reset()
	return m
}

// ReleaseMemory returns an EVM memory buffer to the pool.
func ReleaseMemory(m *evmMemory) {
	if m != nil {
		memoryPool.Put(m)
	}
}

// AcquireScratch returns a pooled worker scratch space.
func AcquireScratch() *workerScratch {
	w := scratchPool.Get().(*workerScratch)
	w.reset()
	return w
}

// ReleaseScratch returns worker scratch space to the pool.
func ReleaseScratch(w *workerScratch) {
	if w != nil {
		scratchPool.Put(w)
	}
}

// PinWorker locks the calling goroutine to its current OS thread.
// This improves CPU cache locality for Block-STM workers by preventing
// the Go scheduler from migrating them between cores mid-execution.
//
// Must be paired with UnpinWorker before the goroutine exits.
func PinWorker() {
	runtime.LockOSThread()
}

// UnpinWorker releases the OS thread lock from PinWorker.
func UnpinWorker() {
	runtime.UnlockOSThread()
}
