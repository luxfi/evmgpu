// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo || nogpu

package dag

// conflictsImpl is the CPU fallback for bitmap intersection conflict detection.
// Uses native Go popcount (math/bits) which compiles to POPCNT on amd64 and
// CNT on arm64 when available.
func conflictsImpl(aWrite, aRead, bWrite, bRead *StorageKeySet) bool {
	if aWrite == nil || aRead == nil || bWrite == nil || bRead == nil {
		return false
	}

	// WAW: a writes and b writes to same slot
	if aWrite.Intersects(bWrite) {
		return true
	}
	// RAW: a writes, b reads same slot
	if aWrite.Intersects(bRead) {
		return true
	}
	// WAR: a reads, b writes same slot
	if aRead.Intersects(bWrite) {
		return true
	}
	return false
}

// BatchConflicts returns the pairwise conflict adjacency matrix for a slice
// of vertices. CPU path uses native Go popcount; GPU build tag provides a
// parallel implementation that dispatches to Metal/CUDA when available.
func BatchConflicts(vertices []*EVMVertex) [][]bool {
	n := len(vertices)
	result := make([][]bool, n)
	for i := range result {
		result[i] = make([]bool, n)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if Conflicts(vertices[i], vertices[j]) {
				result[i][j] = true
				result[j][i] = true
			}
		}
	}
	return result
}
