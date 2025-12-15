// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu

package parallel

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/luxfi/geth/common"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/sha3"
)

// referenceKeccak256 computes a single Keccak-256 hash using the standard library.
// This is the ground truth for correctness tests.
func referenceKeccak256(data []byte) common.Hash {
	d := sha3.NewLegacyKeccak256()
	d.Write(data)
	var h common.Hash
	// Sum appends the hash to nil, giving us the 32-byte digest.
	copy(h[:], d.Sum(nil))
	return h
}

func TestGPUHasher_BatchHash_Empty(t *testing.T) {
	h := NewGPUHasher()
	result := h.BatchHash(nil)
	require.Nil(t, result)

	result = h.BatchHash([][]byte{})
	require.Nil(t, result)
}

func TestGPUHasher_BatchHash_Single(t *testing.T) {
	h := NewGPUHasher()
	input := []byte("hello world")
	expected := referenceKeccak256(input)

	results := h.BatchHash([][]byte{input})
	require.Len(t, results, 1)
	require.Equal(t, expected, results[0])
}

func TestGPUHasher_BatchHash_MatchesCPU(t *testing.T) {
	h := NewGPUHasher()

	// Generate varied-size inputs to exercise padding paths.
	sizes := []int{0, 1, 31, 32, 33, 64, 100, 136, 200, 500, 1024}
	inputs := make([][]byte, len(sizes))
	expected := make([]common.Hash, len(sizes))

	for i, size := range sizes {
		inputs[i] = make([]byte, size)
		if size > 0 {
			rand.Read(inputs[i])
		}
		expected[i] = referenceKeccak256(inputs[i])
	}

	results := h.BatchHash(inputs)
	require.Len(t, results, len(inputs))

	for i := range inputs {
		require.Equal(t, expected[i], results[i],
			"mismatch at index %d (input size %d)", i, sizes[i])
	}
}

func TestGPUHasher_BatchHash_LargeBatch(t *testing.T) {
	h := NewGPUHasher()

	// Test with a batch large enough to exercise the parallel CPU path.
	const n = 1000
	inputs := make([][]byte, n)
	expected := make([]common.Hash, n)

	for i := 0; i < n; i++ {
		inputs[i] = make([]byte, 32+i%100)
		rand.Read(inputs[i])
		expected[i] = referenceKeccak256(inputs[i])
	}

	results := h.BatchHash(inputs)
	require.Len(t, results, n)

	for i := 0; i < n; i++ {
		require.Equal(t, expected[i], results[i], "mismatch at index %d", i)
	}
}

func TestGPUHasher_Backend(t *testing.T) {
	h := NewGPUHasher()
	backend := h.Backend()
	// Must be one of the known backends.
	require.Contains(t, []string{"Auto", "CPU", "Metal", "CUDA", "WebGPU"}, backend)
	t.Logf("GPUHasher backend: %s, available: %v", backend, h.Available())
}

func TestCPUHasher_BatchHash_MatchesReference(t *testing.T) {
	c := &cpuHasher{}

	const n = 200
	inputs := make([][]byte, n)
	expected := make([]common.Hash, n)

	for i := 0; i < n; i++ {
		inputs[i] = make([]byte, 32)
		rand.Read(inputs[i])
		expected[i] = referenceKeccak256(inputs[i])
	}

	results := c.BatchHash(inputs)
	require.Len(t, results, n)

	for i := 0; i < n; i++ {
		require.Equal(t, expected[i], results[i], "mismatch at index %d", i)
	}
}

func TestCPUHasher_SmallBatch_Sequential(t *testing.T) {
	// Batch of 16 or fewer should use the sequential path.
	c := &cpuHasher{}

	inputs := make([][]byte, 16)
	expected := make([]common.Hash, 16)

	for i := range inputs {
		inputs[i] = []byte(fmt.Sprintf("input-%d", i))
		expected[i] = referenceKeccak256(inputs[i])
	}

	results := c.BatchHash(inputs)
	require.Len(t, results, 16)

	for i := range results {
		require.Equal(t, expected[i], results[i])
	}
}

// Benchmarks: measure CPU batch hashing throughput at various batch sizes.
// When GPU kernel dispatch is integrated, add BenchmarkGPUHasher_* variants.

func BenchmarkCPUHasher_32B(b *testing.B) {
	benchmarkBatchHash(b, 32, 1000)
}

func BenchmarkCPUHasher_100B(b *testing.B) {
	benchmarkBatchHash(b, 100, 1000)
}

func BenchmarkCPUHasher_1KB(b *testing.B) {
	benchmarkBatchHash(b, 1024, 1000)
}

func BenchmarkGPUHasher_32B_x100(b *testing.B) {
	benchmarkGPUBatchHash(b, 32, 100)
}

func BenchmarkGPUHasher_32B_x1000(b *testing.B) {
	benchmarkGPUBatchHash(b, 32, 1000)
}

func BenchmarkGPUHasher_32B_x10000(b *testing.B) {
	benchmarkGPUBatchHash(b, 32, 10000)
}

func benchmarkBatchHash(b *testing.B, inputSize, batchSize int) {
	c := &cpuHasher{}

	inputs := make([][]byte, batchSize)
	for i := range inputs {
		inputs[i] = make([]byte, inputSize)
		rand.Read(inputs[i])
	}

	b.SetBytes(int64(inputSize * batchSize))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		c.BatchHash(inputs)
	}
}

func benchmarkGPUBatchHash(b *testing.B, inputSize, batchSize int) {
	h := NewGPUHasher()

	inputs := make([][]byte, batchSize)
	for i := range inputs {
		inputs[i] = make([]byte, inputSize)
		rand.Read(inputs[i])
	}

	b.SetBytes(int64(inputSize * batchSize))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		h.BatchHash(inputs)
	}
}
