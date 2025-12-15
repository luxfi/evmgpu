// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"sync"
	"testing"

	"github.com/luxfi/geth/common"
	"github.com/stretchr/testify/require"
)

func TestAcquireReleaseStack(t *testing.T) {
	s := AcquireStack()
	require.NotNil(t, s)
	require.Equal(t, 0, s.top)

	// Use it.
	s.data[0] = common.HexToHash("0x01")
	s.top = 1

	ReleaseStack(s)

	// Re-acquire -- should be reset.
	s2 := AcquireStack()
	require.NotNil(t, s2)
	require.Equal(t, 0, s2.top)
	ReleaseStack(s2)
}

func TestAcquireReleaseMemory(t *testing.T) {
	m := AcquireMemory()
	require.NotNil(t, m)
	require.Len(t, m.data, 0)
	require.GreaterOrEqual(t, cap(m.data), 4096)

	// Write some data.
	m.data = append(m.data, 0x01, 0x02, 0x03)
	ReleaseMemory(m)

	// Re-acquire -- should be reset.
	m2 := AcquireMemory()
	require.Len(t, m2.data, 0)
	ReleaseMemory(m2)
}

func TestAcquireReleaseScratch(t *testing.T) {
	w := AcquireScratch()
	require.NotNil(t, w)
	require.Len(t, w.readSet, 0)
	require.Len(t, w.writeSet, 0)
	require.Len(t, w.writeBuffer, 0)

	// Use it.
	w.readSet = append(w.readSet, ReadEntry{})
	w.writeSet = append(w.writeSet, WriteEntry{})
	loc := MemoryLocation{Type: LocationBalance}
	w.writeBuffer[loc] = MemoryValue{}

	ReleaseScratch(w)

	// Re-acquire -- should be reset.
	w2 := AcquireScratch()
	require.Len(t, w2.readSet, 0)
	require.Len(t, w2.writeSet, 0)
	require.Len(t, w2.writeBuffer, 0)
	ReleaseScratch(w2)
}

func TestArenaConcurrentAccess(t *testing.T) {
	// Verify pools are safe under concurrent access.
	const workers = 16
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				s := AcquireStack()
				m := AcquireMemory()
				w := AcquireScratch()

				// Light use.
				s.data[0] = common.Hash{}
				s.top = 1
				m.data = append(m.data, byte(j))
				w.readSet = append(w.readSet, ReadEntry{})

				ReleaseScratch(w)
				ReleaseMemory(m)
				ReleaseStack(s)
			}
		}()
	}

	wg.Wait()
}

func TestPinUnpinWorker(t *testing.T) {
	// Just verify it doesn't panic.
	PinWorker()
	UnpinWorker()
}

func BenchmarkAcquireReleaseScratch(b *testing.B) {
	for i := 0; i < b.N; i++ {
		w := AcquireScratch()
		w.readSet = append(w.readSet, ReadEntry{})
		w.writeSet = append(w.writeSet, WriteEntry{})
		ReleaseScratch(w)
	}
}

func BenchmarkAcquireReleaseStack(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s := AcquireStack()
		s.top = 10
		ReleaseStack(s)
	}
}
