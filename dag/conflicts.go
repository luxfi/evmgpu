// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

// Conflicts checks whether two EVM vertices have overlapping storage access
// that would create a data hazard if executed concurrently.
//
// Conflict exists when:
//   - a.WriteSet intersects b.ReadSet  (write-read / RAW hazard)
//   - a.ReadSet intersects b.WriteSet  (read-write / WAR hazard)
//   - a.WriteSet intersects b.WriteSet (write-write / WAW hazard)
//
// Implementation lives in conflicts_cpu.go (Go popcount); GPU bitmap
// intersection lives in luxcpp kernels and is reached through the luxgpu
// cgo bridge from a higher-level dispatcher, not from Go.
func Conflicts(a, b *EVMVertex) bool {
	if a == nil || b == nil {
		return false
	}
	return conflictsImpl(a.writeSet, a.readSet, b.writeSet, b.readSet)
}

// ConflictsSets checks conflicts using raw storage key sets.
// Useful when vertices have not been constructed yet (e.g., during builder
// speculative grouping).
func ConflictsSets(aWrite, aRead, bWrite, bRead *StorageKeySet) bool {
	return conflictsImpl(aWrite, aRead, bWrite, bRead)
}
