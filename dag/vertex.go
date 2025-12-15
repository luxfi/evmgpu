// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dag

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/luxfi/consensus/core/choices"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
)

// EVMVertex is a DAG vertex that carries EVM transactions along with their
// speculative read/write sets. It implements the consensus vertex.Vertex
// interface so the nebula DAG engine can order and finalize it.
//
// The read/write sets are computed during BuildVertex via Block-STM speculative
// execution. Conflict detection between vertices uses bitmap intersection on
// these sets (see conflicts.go).
type EVMVertex struct {
	id       ids.ID
	bytes    []byte
	height   uint64
	epoch    uint32
	parents  []ids.ID
	txIDs    []ids.ID
	txs      []*types.Transaction
	readSet  *StorageKeySet
	writeSet *StorageKeySet
	status   choices.Status
}

// NewEVMVertex creates a vertex from transactions and their speculative r/w sets.
// The vertex ID is derived deterministically from parent IDs, tx hashes, and height.
func NewEVMVertex(
	height uint64,
	epoch uint32,
	parents []ids.ID,
	txs []*types.Transaction,
	readSet *StorageKeySet,
	writeSet *StorageKeySet,
) *EVMVertex {
	txIDs := make([]ids.ID, len(txs))
	for i, tx := range txs {
		h := tx.Hash()
		copy(txIDs[i][:], h[:])
	}

	v := &EVMVertex{
		height:   height,
		epoch:    epoch,
		parents:  parents,
		txIDs:    txIDs,
		txs:      txs,
		readSet:  readSet,
		writeSet: writeSet,
		status:   choices.Processing,
	}

	v.id = v.computeID()
	v.bytes = v.marshal()
	return v
}

// computeID produces a deterministic vertex ID from its contents.
func (v *EVMVertex) computeID() ids.ID {
	h := sha256.New()
	buf := make([]byte, 12)
	binary.BigEndian.PutUint64(buf[:8], v.height)
	binary.BigEndian.PutUint32(buf[8:], v.epoch)
	h.Write(buf)
	for _, p := range v.parents {
		h.Write(p[:])
	}
	for _, tx := range v.txs {
		txh := tx.Hash()
		h.Write(txh[:])
	}
	var id ids.ID
	copy(id[:], h.Sum(nil))
	return id
}

// marshal serializes the vertex for wire format.
// Format: [8 height][4 epoch][1 parentCount][32*N parentIDs][txRLP...]
func (v *EVMVertex) marshal() []byte {
	// Estimate size: header + parents + tx hashes
	size := 8 + 4 + 1 + len(v.parents)*32 + len(v.txIDs)*32
	buf := make([]byte, 0, size)

	b8 := make([]byte, 8)
	binary.BigEndian.PutUint64(b8, v.height)
	buf = append(buf, b8...)

	b4 := make([]byte, 4)
	binary.BigEndian.PutUint32(b4, v.epoch)
	buf = append(buf, b4...)

	buf = append(buf, byte(len(v.parents)))
	for _, p := range v.parents {
		buf = append(buf, p[:]...)
	}

	buf = append(buf, byte(len(v.txIDs)))
	for _, txID := range v.txIDs {
		buf = append(buf, txID[:]...)
	}

	return buf
}

// --- vertex.Vertex interface ---

func (v *EVMVertex) ID() ids.ID          { return v.id }
func (v *EVMVertex) Bytes() []byte        { return v.bytes }
func (v *EVMVertex) Height() uint64       { return v.height }
func (v *EVMVertex) Epoch() uint32        { return v.epoch }
func (v *EVMVertex) Parents() []ids.ID    { return v.parents }
func (v *EVMVertex) Txs() []ids.ID        { return v.txIDs }
func (v *EVMVertex) Status() choices.Status { return v.status }

func (v *EVMVertex) Accept(_ context.Context) error {
	if v.status == choices.Rejected {
		return fmt.Errorf("vertex %s already rejected", v.id)
	}
	v.status = choices.Accepted
	return nil
}

func (v *EVMVertex) Reject(_ context.Context) error {
	if v.status == choices.Accepted {
		return fmt.Errorf("vertex %s already accepted", v.id)
	}
	v.status = choices.Rejected
	return nil
}

func (v *EVMVertex) Verify(_ context.Context) error {
	if v.id == ids.Empty {
		return fmt.Errorf("vertex has empty ID")
	}
	if len(v.txs) == 0 {
		return fmt.Errorf("vertex %s has no transactions", v.id)
	}
	if v.readSet == nil || v.writeSet == nil {
		return fmt.Errorf("vertex %s missing r/w sets", v.id)
	}
	return nil
}

// --- EVM-specific accessors ---

// Transactions returns the EVM transactions in this vertex.
func (v *EVMVertex) Transactions() []*types.Transaction { return v.txs }

// ReadSet returns the speculative read set bitmap.
func (v *EVMVertex) ReadSet() *StorageKeySet { return v.readSet }

// WriteSet returns the speculative write set bitmap.
func (v *EVMVertex) WriteSet() *StorageKeySet { return v.writeSet }
