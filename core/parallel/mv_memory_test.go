// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"testing"

	"github.com/luxfi/geth/common"
)

// TestMvMemoryFullKeyPreventsHashCollision verifies that MvMemory uses the
// full MemoryLocation as key, not just LocationHash. Two locations that
// produce the same LocationHash must be stored in separate version chains.
// This is the fix for Red Team F02 — LocationHash collision vulnerability.
func TestMvMemoryFullKeyPreventsHashCollision(t *testing.T) {
	// Construct two MemoryLocations that have the same LocationHash.
	// Hash() only uses the last 8 bytes of the address XOR'd with slot,
	// so we craft two (address, slot) pairs that collide.
	//
	// Location A: address with last 8 bytes = 0x0102030405060708
	//             slot with last 8 bytes    = 0x0000000000000000
	// Location B: address with last 8 bytes = 0x0000000000000000
	//             slot with last 8 bytes    = 0x0102030405060708
	//
	// Hash(A) = 0x0102030405060708 ^ type_bits ^ 0x0000000000000000
	// Hash(B) = 0x0000000000000000 ^ type_bits ^ 0x0102030405060708
	// These differ because of XOR ordering — so use a simpler approach:
	// same slot bytes, different address prefix (first 12 bytes differ,
	// last 8 bytes identical). Hash() ignores the first 12 bytes.

	var addrA, addrB common.Address
	// Last 8 bytes identical
	for i := 12; i < 20; i++ {
		addrA[i] = byte(i)
		addrB[i] = byte(i)
	}
	// First 12 bytes differ
	addrA[0] = 0xAA
	addrB[0] = 0xBB

	locA := MemoryLocation{Address: addrA, Type: LocationStorage, Slot: common.Hash{}}
	locB := MemoryLocation{Address: addrB, Type: LocationStorage, Slot: common.Hash{}}

	// Verify they produce the same LocationHash
	if locA.Hash() != locB.Hash() {
		t.Fatalf("test setup error: locA.Hash()=%x != locB.Hash()=%x", locA.Hash(), locB.Hash())
	}

	// Verify they are NOT equal as MemoryLocations
	if locA == locB {
		t.Fatal("test setup error: locA == locB, but they should differ")
	}

	// Create MvMemory and write different values at each location
	mvMem := NewMvMemory(4)

	valA := MemoryValue{Type: ValueAbsolute, Storage: common.HexToHash("0xAAAA")}
	valB := MemoryValue{Type: ValueAbsolute, Storage: common.HexToHash("0xBBBB")}

	mvMem.Write(locA, 0, 0, valA)
	mvMem.Write(locB, 1, 0, valB)

	// Read locA from tx 1 — should find valA (written by tx 0)
	entryA, foundA := mvMem.Read(locA, 1)
	if !foundA {
		t.Fatal("expected to find locA in MvMemory")
	}
	if entryA.Value.Storage != valA.Storage {
		t.Fatalf("locA read wrong value: got %x, want %x", entryA.Value.Storage, valA.Storage)
	}

	// Read locB from tx 2 — should find valB (written by tx 1)
	entryB, foundB := mvMem.Read(locB, 2)
	if !foundB {
		t.Fatal("expected to find locB in MvMemory")
	}
	if entryB.Value.Storage != valB.Storage {
		t.Fatalf("locB read wrong value: got %x, want %x", entryB.Value.Storage, valB.Storage)
	}

	// Read locB from tx 1 — should NOT find anything (tx 1 wrote it, tx 0 did not)
	_, foundB1 := mvMem.Read(locB, 1)
	if foundB1 {
		t.Fatal("locB should not be visible to tx 1 (only tx 1 wrote to locB)")
	}
}

// TestMvMemoryDistinctChains verifies that two locations with different hashes
// still get distinct chains (basic sanity check).
func TestMvMemoryDistinctChains(t *testing.T) {
	var addr common.Address
	addr[19] = 0x01

	locBalance := MemoryLocation{Address: addr, Type: LocationBalance}
	locStorage := MemoryLocation{Address: addr, Type: LocationStorage, Slot: common.HexToHash("0x01")}

	mvMem := NewMvMemory(4)

	valBal := MemoryValue{Type: ValueAbsolute, Balance: common.HexToHash("0x1000")}
	valSto := MemoryValue{Type: ValueAbsolute, Storage: common.HexToHash("0x2000")}

	mvMem.Write(locBalance, 0, 0, valBal)
	mvMem.Write(locStorage, 0, 0, valSto)

	entryBal, found := mvMem.Read(locBalance, 1)
	if !found {
		t.Fatal("expected to find balance location")
	}
	if entryBal.Value.Balance != valBal.Balance {
		t.Fatalf("balance wrong: got %x, want %x", entryBal.Value.Balance, valBal.Balance)
	}

	entrySto, found := mvMem.Read(locStorage, 1)
	if !found {
		t.Fatal("expected to find storage location")
	}
	if entrySto.Value.Storage != valSto.Storage {
		t.Fatalf("storage wrong: got %x, want %x", entrySto.Value.Storage, valSto.Storage)
	}
}

// TestMvMemoryValidateReadSetWithFullKey verifies that ValidateReadSet uses
// the full MemoryLocation for re-validation, not just the hash.
func TestMvMemoryValidateReadSetWithFullKey(t *testing.T) {
	var addrA, addrB common.Address
	// Same last 8 bytes → same LocationHash
	for i := 12; i < 20; i++ {
		addrA[i] = 0x42
		addrB[i] = 0x42
	}
	addrA[0] = 0x11
	addrB[0] = 0x22

	locA := MemoryLocation{Address: addrA, Type: LocationStorage, Slot: common.Hash{}}
	locB := MemoryLocation{Address: addrB, Type: LocationStorage, Slot: common.Hash{}}

	if locA.Hash() != locB.Hash() {
		t.Fatalf("test setup: hashes should match")
	}

	mvMem := NewMvMemory(4)

	// Tx 0 writes to locA
	valA := MemoryValue{Type: ValueAbsolute, Storage: common.HexToHash("0xAA")}
	mvMem.Write(locA, 0, 0, valA)

	// Tx 1 reads locA from MvMemory and locB from storage
	mvMem.RecordReadSet(1, []ReadEntry{
		{
			Location: locA,
			Origin: ReadOrigin{
				FromMvMemory: true,
				Version:      TxVersion{TxIdx: 0, Incarnation: 0},
			},
		},
		{
			Location: locB,
			Origin:   ReadOrigin{FromMvMemory: false},
		},
	})

	// Validate tx 1 — should pass (locA still written by tx 0, locB still not in MvMemory)
	if !mvMem.ValidateReadSet(1) {
		t.Fatal("expected read set to be valid")
	}

	// Now tx 0 writes to locB as well
	valB := MemoryValue{Type: ValueAbsolute, Storage: common.HexToHash("0xBB")}
	mvMem.Write(locB, 0, 0, valB)

	// Validate tx 1 again — should now FAIL because locB appeared in MvMemory
	if mvMem.ValidateReadSet(1) {
		t.Fatal("expected read set to be INVALID after locB was written")
	}
}
