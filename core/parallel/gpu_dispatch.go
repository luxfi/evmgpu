// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"encoding/binary"
	"errors"
	"math/big"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	ethparams "github.com/luxfi/geth/params"
)

// ErrGPUDeclined is what a GPUDispatcher returns for a batch the GPU EVM does
// not run: one go_bridge.h's gpu_execute_block answered ok=0 for, and one that
// never reached it because its 64-bit wire cannot carry it. The engine then
// runs the block on its sequential path; a declined batch has no result.
var ErrGPUDeclined = errors.New("parallel: the GPU EVM declined the batch; the block runs on the sequential path")

// gpuTx is one transaction as go_bridge.h's CGpuTx carries it. Only plain
// value transfers reach the GPU (IsGPUEligible), so there is no calldata and
// no code, and every one has a recipient.
type gpuTx struct {
	From, To [20]byte
	GasLimit uint64
	Value    uint64
	Nonce    uint64
	GasPrice uint64
}

// gpuAccount is one row of the state before the block, as CGpuStateAccount
// carries it. Balance is little-endian limbs (Balance[0] = low 64 bits). Every
// row is an account without code (an account with code declines the batch),
// so no row has code bytes.
type gpuAccount struct {
	Address     [20]byte
	Nonce       uint64
	Balance     [4]uint64
	CodeHash    [32]byte
	StorageRoot [32]byte
}

// gpuBlockContext is the part of CBlockContext a batch of plain transfers
// depends on; the rest crosses as zero.
type gpuBlockContext struct {
	Timestamp  uint64
	Number     uint64
	GasLimit   uint64
	ChainID    uint64
	BaseFee    uint64
	Coinbase   [20]byte
	Prevrandao [32]byte
}

// gpuBatch is everything one gpu_execute_block call takes.
type gpuBatch struct {
	txs      []gpuTx
	ctx      gpuBlockContext
	accounts []gpuAccount
}

// GPUEVMDispatcher dispatches GPU-eligible transactions to the C++ Metal/CUDA
// EVM through go_bridge.h's gpu_execute_block. Construct it with
// NewGPUEVMDispatcher (gpu build tag).
type GPUEVMDispatcher struct {
	backend uint8
	// execute is the one call into gpu_execute_block. It is a field so that
	// what reaches the GPU can be shown without one.
	execute func(backend uint8, batch *gpuBatch) ([]GPUEVMResult, error)
}

// Available returns true if a GPU backend was detected and the dispatcher can
// call it: one without a device call (a library of another ABI, or the zero
// value) is not.
func (d *GPUEVMDispatcher) Available() bool {
	return d.execute != nil && d.backend >= 2 // Metal=2, CUDA=3
}

// Backend returns the name of the active backend.
func (d *GPUEVMDispatcher) Backend() string {
	return gpuBackendName(d.backend)
}

// ExecuteBlock dispatches a batch of GPU-eligible transactions and returns
// per-transaction gas and outcome, or ErrGPUDeclined.
//
// Nothing the wire cannot carry reaches the GPU: a value, price, base fee or
// chain id wider than 64 bits, a transaction that is not eligible, a state
// the getter cannot answer, or an account with code, whose bytes the state
// getter does not give. Truncating any of them would run a different block.
// A dispatcher with no device call (the zero value) declines everything.
func (d *GPUEVMDispatcher) ExecuteBlock(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs []*types.Transaction,
	senders []common.Address,
	state StateGetter,
) ([]GPUEVMResult, error) {
	if len(txs) == 0 {
		return nil, nil
	}
	batch, ok := shapeGPUBatch(config, header, txs, senders, state)
	if !ok || d.execute == nil {
		return nil, ErrGPUDeclined
	}
	return d.execute(d.backend, batch)
}

// shapeGPUBatch puts txs into gpu_execute_block's wire form, with the block
// context and the state before the block that go_bridge.h asks for: every
// account a transaction touches (sender, recipient, and the coinbase its fee
// goes to), each with its code hash and storage root. It reports false for a
// batch the wire cannot carry.
func shapeGPUBatch(
	config *ethparams.ChainConfig,
	header *types.Header,
	txs []*types.Transaction,
	senders []common.Address,
	state StateGetter,
) (*gpuBatch, bool) {
	if len(senders) != len(txs) || state == nil {
		return nil, false
	}
	chainID, ok := fitsUint64(config.ChainID)
	if !ok {
		return nil, false
	}
	baseFee, ok := fitsUint64(header.BaseFee)
	if !ok {
		return nil, false
	}
	b := &gpuBatch{
		txs: make([]gpuTx, len(txs)),
		ctx: gpuBlockContext{
			Timestamp: header.Time,
			Number:    header.Number.Uint64(),
			GasLimit:  header.GasLimit,
			ChainID:   chainID,
			BaseFee:   baseFee,
			Coinbase:  header.Coinbase,
		},
	}
	// PREVRANDAO (0x44) is the header's difficulty as a 32-byte word:
	// NewEVMBlockContext sets Random to it from Shanghai on, and DIFFICULTY
	// before that answers the same number. A Lux header's MixDigest is zero.
	if header.Difficulty != nil {
		b.ctx.Prevrandao = common.BigToHash(header.Difficulty)
	}

	seen := make(map[common.Address]struct{}, 2*len(txs)+1)
	add := func(addr common.Address) bool {
		if _, dup := seen[addr]; dup {
			return true
		}
		seen[addr] = struct{}{}
		acct, ok := gpuAccountOf(addr, state)
		if ok {
			b.accounts = append(b.accounts, acct)
		}
		return ok
	}
	for i, tx := range txs {
		if !IsGPUEligible(tx) {
			return nil, false
		}
		value, ok := fitsUint64(tx.Value())
		if !ok {
			return nil, false
		}
		// tx.GasPrice() is a dynamic-fee tx's fee cap: what the EVM's buy-gas
		// balance check charges, and what cevm checks against the base fee.
		price, ok := fitsUint64(tx.GasPrice())
		if !ok {
			return nil, false
		}
		to := *tx.To()
		b.txs[i] = gpuTx{
			From:     senders[i],
			To:       to,
			GasLimit: tx.Gas(),
			Value:    value,
			Nonce:    tx.Nonce(),
			GasPrice: price,
		}
		if !add(senders[i]) || !add(to) {
			return nil, false
		}
	}
	if !add(header.Coinbase) {
		return nil, false
	}
	return b, true
}

// gpuAccountOf reads addr's row from the state before the block. It reports
// false when the getter cannot answer, or when the account has code: the
// getter does not give its bytes, and a row without them is not the account.
//
// An account that does not exist has a zero code hash and storage root in the
// state; it has no code and no storage, and its row says so with keccak256 of
// nothing and the empty trie root. Zero would say nothing, which cevm
// declines.
func gpuAccountOf(addr common.Address, state StateGetter) (gpuAccount, bool) {
	codeHash, ok := state(MemoryLocation{Address: addr, Type: LocationCodeHash})
	if !ok {
		return gpuAccount{}, false
	}
	switch codeHash.Storage {
	case common.Hash{}, types.EmptyCodeHash:
	default:
		return gpuAccount{}, false
	}
	root, ok := state(MemoryLocation{Address: addr, Type: LocationStorageRoot})
	if !ok {
		return gpuAccount{}, false
	}
	nonce, ok := state(MemoryLocation{Address: addr, Type: LocationNonce})
	if !ok {
		return gpuAccount{}, false
	}
	balance, ok := state(MemoryLocation{Address: addr, Type: LocationBalance})
	if !ok {
		return gpuAccount{}, false
	}
	acct := gpuAccount{
		Address:     addr,
		Nonce:       nonce.Nonce,
		CodeHash:    types.EmptyCodeHash,
		StorageRoot: types.EmptyRootHash,
	}
	if root.Storage != (common.Hash{}) {
		acct.StorageRoot = root.Storage
	}
	// MemoryValue.Balance is the balance's 32 big-endian bytes; the wire
	// wants little-endian 64-bit limbs.
	for i := 0; i < 4; i++ {
		acct.Balance[i] = binary.BigEndian.Uint64(balance.Balance[24-8*i : 32-8*i])
	}
	return acct, true
}

// fitsUint64 reads v as the 64-bit wire carries it. A nil v is zero; one wider
// than 64 bits does not fit.
func fitsUint64(v *big.Int) (uint64, bool) {
	if v == nil {
		return 0, true
	}
	if !v.IsUint64() {
		return 0, false
	}
	return v.Uint64(), true
}

func gpuBackendName(b uint8) string {
	switch b {
	case 0:
		return "CPU-Sequential"
	case 1:
		return "CPU-Parallel"
	case 2:
		return "Metal"
	case 3:
		return "CUDA"
	default:
		return "Unknown"
	}
}
