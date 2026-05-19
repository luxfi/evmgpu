// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"hash"
	"runtime"
	"sync"

	"github.com/luxfi/geth/common"
	"golang.org/x/crypto/sha3"
)

// keccakState wraps hash.Hash with Read for extracting the digest without
// copying internal state. sha3.NewLegacyKeccak256() satisfies this.
type keccakState interface {
	hash.Hash
	Read([]byte) (int, error)
}

// Hasher computes Keccak-256 hashes for trie node batches.
// The CPU implementation is the only one in Go; GPU keccak lives in
// luxcpp (Metal/CUDA kernels) and is reached through the luxgpu cgo bridge,
// not through a Go-native hasher.
type Hasher interface {
	// BatchHash computes Keccak-256 hashes for a batch of inputs.
	// Returns one hash per input, in the same order.
	BatchHash(inputs [][]byte) []common.Hash

	// Backend returns the name of the compute backend ("Metal", "CUDA", "CPU").
	Backend() string
}

// cpuHasher computes Keccak-256 in parallel across CPU cores.
type cpuHasher struct{}

var keccakPool = sync.Pool{
	New: func() any {
		return sha3.NewLegacyKeccak256()
	},
}

// NewCPUHasher returns a Hasher that uses parallel CPU goroutines.
func NewCPUHasher() Hasher {
	return &cpuHasher{}
}

// BatchHash computes Keccak-256 for each input using parallel goroutines.
// Splits the work across GOMAXPROCS workers for large batches.
func (c *cpuHasher) BatchHash(inputs [][]byte) []common.Hash {
	n := len(inputs)
	if n == 0 {
		return nil
	}

	results := make([]common.Hash, n)

	// For small batches, hash sequentially to avoid goroutine overhead.
	if n <= 16 {
		for i, inp := range inputs {
			d := keccakPool.Get().(keccakState)
			d.Reset()
			d.Write(inp)
			d.Read(results[i][:])
			keccakPool.Put(d)
		}
		return results
	}

	// Parallel hashing across CPU cores.
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > n {
		numWorkers = n
	}
	chunkSize := (n + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			wg.Done()
			continue
		}

		go func(start, end int) {
			defer wg.Done()
			d := keccakPool.Get().(keccakState)
			defer keccakPool.Put(d)

			for i := start; i < end; i++ {
				d.Reset()
				d.Write(inputs[i])
				d.Read(results[i][:])
			}
		}(start, end)
	}

	wg.Wait()
	return results
}

// Backend returns "CPU".
func (c *cpuHasher) Backend() string {
	return "CPU"
}
