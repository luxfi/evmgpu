// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

import (
	"context"
	"math/big"
	"testing"

	"github.com/luxfi/consensus/core/choices"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
)

func testTx(nonce uint64, to common.Address) *types.Transaction {
	return types.NewTransaction(nonce, to, big.NewInt(1000), 21000, big.NewInt(1e9), nil)
}

func testContractTx(nonce uint64, data []byte) *types.Transaction {
	return types.NewContractCreation(nonce, big.NewInt(0), 100000, big.NewInt(1e9), data)
}

// --- StorageKeySet tests ---

func TestStorageKeySetAddContains(t *testing.T) {
	s := &StorageKeySet{}
	key := common.HexToHash("0xdeadbeef")
	if s.Contains(key) {
		t.Fatal("empty set should not contain key")
	}
	s.Add(key)
	if !s.Contains(key) {
		t.Fatal("set should contain added key")
	}
	if s.Len() != 1 {
		t.Fatalf("expected len 1, got %d", s.Len())
	}
}

func TestStorageKeySetIntersects(t *testing.T) {
	a := &StorageKeySet{}
	b := &StorageKeySet{}

	key := common.HexToHash("0x1234")
	a.Add(key)
	b.Add(key)

	if !a.Intersects(b) {
		t.Fatal("sets with same key should intersect")
	}
}

func TestStorageKeySetDisjoint(t *testing.T) {
	a := &StorageKeySet{}
	b := &StorageKeySet{}

	a.Add(common.HexToHash("0x1111111111111111111111111111111111111111111111111111111111111111"))
	b.Add(common.HexToHash("0x2222222222222222222222222222222222222222222222222222222222222222"))

	if a.Intersects(b) {
		// This could be a Bloom false positive; verify with popcount.
		if a.IntersectionPopcount(b) > 0 {
			t.Fatal("disjoint sets with different hash positions should not intersect")
		}
	}
}

func TestStorageKeySetUnion(t *testing.T) {
	a := &StorageKeySet{}
	b := &StorageKeySet{}
	k1 := common.HexToHash("0xaaaa")
	k2 := common.HexToHash("0xbbbb")
	a.Add(k1)
	b.Add(k2)
	a.Union(b)
	if !a.Contains(k1) || !a.Contains(k2) {
		t.Fatal("union should contain keys from both sets")
	}
}

// --- EVMVertex tests ---

func TestNewEVMVertex(t *testing.T) {
	txs := []*types.Transaction{
		testTx(0, common.HexToAddress("0xaaaa")),
		testTx(1, common.HexToAddress("0xbbbb")),
	}
	rs := &StorageKeySet{}
	ws := &StorageKeySet{}
	rs.Add(common.HexToHash("0x01"))
	ws.Add(common.HexToHash("0x02"))

	v := NewEVMVertex(42, 1, []ids.ID{ids.Empty}, txs, rs, ws)

	if v.ID() == ids.Empty {
		t.Fatal("vertex ID should not be empty")
	}
	if v.Height() != 42 {
		t.Fatalf("expected height 42, got %d", v.Height())
	}
	if v.Epoch() != 1 {
		t.Fatalf("expected epoch 1, got %d", v.Epoch())
	}
	if len(v.Txs()) != 2 {
		t.Fatalf("expected 2 tx IDs, got %d", len(v.Txs()))
	}
	if len(v.Transactions()) != 2 {
		t.Fatalf("expected 2 transactions, got %d", len(v.Transactions()))
	}
	if v.Status() != choices.Processing {
		t.Fatalf("expected Processing status, got %s", v.Status())
	}
	if len(v.Bytes()) == 0 {
		t.Fatal("vertex bytes should not be empty")
	}
}

func TestEVMVertexAcceptReject(t *testing.T) {
	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	v := NewEVMVertex(1, 0, nil, txs, &StorageKeySet{}, &StorageKeySet{})

	if err := v.Accept(context.Background()); err != nil {
		t.Fatalf("Accept failed: %v", err)
	}
	if v.Status() != choices.Accepted {
		t.Fatal("expected Accepted status")
	}
	if err := v.Reject(context.Background()); err == nil {
		t.Fatal("Reject after Accept should fail")
	}

	// Test reject path.
	v2 := NewEVMVertex(2, 0, nil, txs, &StorageKeySet{}, &StorageKeySet{})
	if err := v2.Reject(context.Background()); err != nil {
		t.Fatalf("Reject failed: %v", err)
	}
	if v2.Status() != choices.Rejected {
		t.Fatal("expected Rejected status")
	}
	if err := v2.Accept(context.Background()); err == nil {
		t.Fatal("Accept after Reject should fail")
	}
}

func TestEVMVertexVerify(t *testing.T) {
	// Valid vertex.
	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	v := NewEVMVertex(1, 0, nil, txs, &StorageKeySet{}, &StorageKeySet{})
	if err := v.Verify(context.Background()); err != nil {
		t.Fatalf("valid vertex should verify: %v", err)
	}

	// Empty txs.
	v2 := NewEVMVertex(1, 0, nil, nil, &StorageKeySet{}, &StorageKeySet{})
	if err := v2.Verify(context.Background()); err == nil {
		t.Fatal("vertex with no txs should fail verification")
	}
}

func TestEVMVertexDeterministicID(t *testing.T) {
	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	parents := []ids.ID{{0x01}}

	v1 := NewEVMVertex(10, 1, parents, txs, &StorageKeySet{}, &StorageKeySet{})
	v2 := NewEVMVertex(10, 1, parents, txs, &StorageKeySet{}, &StorageKeySet{})

	if v1.ID() != v2.ID() {
		t.Fatalf("same inputs should produce same vertex ID: %s vs %s", v1.ID(), v2.ID())
	}
}

// --- Conflict detection tests ---

func TestConflictsDisjointRWSets(t *testing.T) {
	// Two vertices touching completely different addresses.
	txsA := []*types.Transaction{testTx(0, common.HexToAddress("0x1111111111111111111111111111111111111111"))}
	txsB := []*types.Transaction{testTx(1, common.HexToAddress("0x2222222222222222222222222222222222222222"))}

	rsA := &StorageKeySet{}
	wsA := &StorageKeySet{}
	rsA.Add(common.BytesToHash(common.HexToAddress("0x1111111111111111111111111111111111111111").Bytes()))
	wsA.Add(common.BytesToHash(common.HexToAddress("0x1111111111111111111111111111111111111111").Bytes()))

	rsB := &StorageKeySet{}
	wsB := &StorageKeySet{}
	rsB.Add(common.BytesToHash(common.HexToAddress("0x2222222222222222222222222222222222222222").Bytes()))
	wsB.Add(common.BytesToHash(common.HexToAddress("0x2222222222222222222222222222222222222222").Bytes()))

	vA := NewEVMVertex(1, 0, nil, txsA, rsA, wsA)
	vB := NewEVMVertex(2, 0, nil, txsB, rsB, wsB)

	if Conflicts(vA, vB) {
		// Due to Bloom hashing, there's a small chance of false positive.
		// Verify it's actually a false positive by checking raw bitmap intersection.
		if wsA.IntersectionPopcount(wsB) == 0 &&
			wsA.IntersectionPopcount(rsB) == 0 &&
			rsA.IntersectionPopcount(wsB) == 0 {
			t.Fatal("disjoint addresses should not conflict")
		}
		t.Log("Bloom false positive detected (expected at ~1% rate per slot)")
	}
}

func TestConflictsWriteReadOverlap(t *testing.T) {
	// Vertex A writes to slot X, vertex B reads slot X.
	sharedKey := common.HexToHash("0xdeadbeef00000000000000000000000000000000000000000000000000000001")

	txsA := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	txsB := []*types.Transaction{testTx(1, common.HexToAddress("0xbbbb"))}

	rsA := &StorageKeySet{}
	wsA := &StorageKeySet{}
	wsA.Add(sharedKey)

	rsB := &StorageKeySet{}
	wsB := &StorageKeySet{}
	rsB.Add(sharedKey)

	vA := NewEVMVertex(1, 0, nil, txsA, rsA, wsA)
	vB := NewEVMVertex(2, 0, nil, txsB, rsB, wsB)

	if !Conflicts(vA, vB) {
		t.Fatal("write-read overlap must produce a conflict")
	}
}

func TestConflictsWriteWriteOverlap(t *testing.T) {
	// Both vertices write to the same slot.
	sharedKey := common.HexToHash("0xcafebabe00000000000000000000000000000000000000000000000000000002")

	txsA := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	txsB := []*types.Transaction{testTx(1, common.HexToAddress("0xbbbb"))}

	rsA := &StorageKeySet{}
	wsA := &StorageKeySet{}
	wsA.Add(sharedKey)

	rsB := &StorageKeySet{}
	wsB := &StorageKeySet{}
	wsB.Add(sharedKey)

	vA := NewEVMVertex(1, 0, nil, txsA, rsA, wsA)
	vB := NewEVMVertex(2, 0, nil, txsB, rsB, wsB)

	if !Conflicts(vA, vB) {
		t.Fatal("write-write overlap must produce a conflict")
	}
}

func TestConflictsNilVertices(t *testing.T) {
	txs := []*types.Transaction{testTx(0, common.HexToAddress("0xaaaa"))}
	v := NewEVMVertex(1, 0, nil, txs, &StorageKeySet{}, &StorageKeySet{})

	if Conflicts(nil, v) {
		t.Fatal("nil vertex should not conflict")
	}
	if Conflicts(v, nil) {
		t.Fatal("nil vertex should not conflict")
	}
	if Conflicts(nil, nil) {
		t.Fatal("two nil vertices should not conflict")
	}
}

func TestConflictsSetsDirectly(t *testing.T) {
	key := common.HexToHash("0x9999")
	aWrite := &StorageKeySet{}
	aRead := &StorageKeySet{}
	bWrite := &StorageKeySet{}
	bRead := &StorageKeySet{}

	aWrite.Add(key)
	bRead.Add(key)

	if !ConflictsSets(aWrite, aRead, bWrite, bRead) {
		t.Fatal("ConflictsSets should detect write-read overlap")
	}
}

// --- Builder tests ---

func TestBuilderBuildVertex(t *testing.T) {
	frontier := []ids.ID{{0x01}, {0x02}}
	b := NewBuilder(BuilderConfig{
		Workers:         2,
		MaxTxsPerVertex: 100,
		FrontierFn:      func() []ids.ID { return frontier },
		HeightFn:        func() uint64 { return 42 },
		EpochFn:         func() uint32 { return 1 },
	})

	txs := []*types.Transaction{
		testTx(0, common.HexToAddress("0xaaaa")),
		testTx(1, common.HexToAddress("0xbbbb")),
		testTx(2, common.HexToAddress("0xcccc")),
	}

	v := b.BuildVertex(txs)
	if v == nil {
		t.Fatal("BuildVertex returned nil")
	}
	if len(v.Transactions()) != 3 {
		t.Fatalf("expected 3 txs, got %d", len(v.Transactions()))
	}
	if v.Height() != 42 {
		t.Fatalf("expected height 42, got %d", v.Height())
	}
	if len(v.Parents()) != 2 {
		t.Fatalf("expected 2 parents, got %d", len(v.Parents()))
	}
	if v.ReadSet() == nil || v.WriteSet() == nil {
		t.Fatal("r/w sets should not be nil")
	}
}

func TestBuilderEmptyTxs(t *testing.T) {
	b := NewBuilder(BuilderConfig{})
	if v := b.BuildVertex(nil); v != nil {
		t.Fatal("nil txs should return nil vertex")
	}
	if v := b.BuildVertex([]*types.Transaction{}); v != nil {
		t.Fatal("empty txs should return nil vertex")
	}
}

func TestBuilderMaxTxsCap(t *testing.T) {
	b := NewBuilder(BuilderConfig{MaxTxsPerVertex: 2})

	txs := []*types.Transaction{
		testTx(0, common.HexToAddress("0xaaaa")),
		testTx(1, common.HexToAddress("0xbbbb")),
		testTx(2, common.HexToAddress("0xcccc")),
	}

	v := b.BuildVertex(txs)
	if v == nil {
		t.Fatal("BuildVertex returned nil")
	}
	if len(v.Transactions()) != 2 {
		t.Fatalf("expected 2 txs (capped), got %d", len(v.Transactions()))
	}
}

// --- ExtractRWSet tests ---

func TestExtractRWSetTransfer(t *testing.T) {
	to := common.HexToAddress("0xbbbb")
	tx := testTx(0, to)
	rw := extractRWSet(tx)

	addrHash := common.BytesToHash(to.Bytes())
	if !rw.readSet.Contains(addrHash) {
		t.Fatal("read set should contain recipient")
	}
	if !rw.writeSet.Contains(addrHash) {
		t.Fatal("write set should contain recipient")
	}
}

func TestExtractRWSetContractCreation(t *testing.T) {
	tx := testContractTx(0, []byte{0x60, 0x00})
	rw := extractRWSet(tx)

	if !rw.writeSet.Contains(tx.Hash()) {
		t.Fatal("contract creation write set should contain tx hash")
	}
}
