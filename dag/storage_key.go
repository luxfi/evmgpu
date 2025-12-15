// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

// StorageKeySet tracks EVM storage slots as a bitmap for fast intersection.
//
// Layout: fixed 4096-bit (512-byte) bitmap. Each storage key hashes to a bit
// position via the lower 12 bits of its first 2 bytes. This is a Bloom-style
// approximation: false positives cause conservative conflict detection (safe),
// false negatives are impossible because every key sets its bit.
//
// For typical EVM transactions touching 1-20 storage slots, the false positive
// rate at 4096 bits is <1% per slot, which is acceptable for speculative
// execution where false conflicts only cost a re-execution.

import (
	"math/bits"

	"github.com/luxfi/geth/common"
)

const (
	bitmapWords = 64  // 64 uint64 = 512 bytes = 4096 bits
	bitmapBits  = bitmapWords * 64
)

// StorageKeySet is a fixed-size bitmap representing a set of EVM storage keys.
type StorageKeySet struct {
	bits [bitmapWords]uint64
	n    int // number of keys inserted (for stats)
}

// Add inserts a storage key into the set.
func (s *StorageKeySet) Add(key common.Hash) {
	bit := keyToBit(key)
	word := bit / 64
	mask := uint64(1) << (bit % 64)
	if s.bits[word]&mask == 0 {
		s.n++
	}
	s.bits[word] |= mask
}

// Contains checks if a storage key might be in the set.
func (s *StorageKeySet) Contains(key common.Hash) bool {
	bit := keyToBit(key)
	word := bit / 64
	mask := uint64(1) << (bit % 64)
	return s.bits[word]&mask != 0
}

// Len returns the number of keys inserted.
func (s *StorageKeySet) Len() int { return s.n }

// IsEmpty returns true if no keys have been inserted.
func (s *StorageKeySet) IsEmpty() bool { return s.n == 0 }

// Intersects returns true if two sets share any bits.
// This is the CPU path; the GPU path uses conflicts_gpu.go.
func (s *StorageKeySet) Intersects(other *StorageKeySet) bool {
	for i := 0; i < bitmapWords; i++ {
		if s.bits[i]&other.bits[i] != 0 {
			return true
		}
	}
	return false
}

// IntersectionPopcount returns the number of shared bits.
func (s *StorageKeySet) IntersectionPopcount(other *StorageKeySet) int {
	count := 0
	for i := 0; i < bitmapWords; i++ {
		count += bits.OnesCount64(s.bits[i] & other.bits[i])
	}
	return count
}

// Union merges another set into this one.
func (s *StorageKeySet) Union(other *StorageKeySet) {
	for i := 0; i < bitmapWords; i++ {
		s.bits[i] |= other.bits[i]
	}
	s.n = 0
	for i := 0; i < bitmapWords; i++ {
		s.n += bits.OnesCount64(s.bits[i])
	}
}

// Words returns the raw bitmap for GPU kernel consumption.
func (s *StorageKeySet) Words() *[bitmapWords]uint64 {
	return &s.bits
}

// keyToBit hashes a common.Hash down to a bit index in [0, bitmapBits).
// Uses FNV-1a over the full 32-byte hash; the key is already a keccak digest
// so uniform distribution is inherent and we just need to fold to 12 bits.
// Earlier implementations XOR'd two 16-bit windows, which collapsed to 0 for
// repeated-byte keys (e.g. 0x1111…1111 vs 0x2222…2222) — a false positive
// that wedged disjoint slots into the same bit.
func keyToBit(key common.Hash) uint {
	const (
		fnvOffset = uint64(14695981039346656037)
		fnvPrime  = uint64(1099511628211)
	)
	h := fnvOffset
	for i := 0; i < len(key); i++ {
		h ^= uint64(key[i])
		h *= fnvPrime
	}
	return uint(h % bitmapBits)
}
