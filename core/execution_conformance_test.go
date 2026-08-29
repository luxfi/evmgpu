// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package core

import (
	"crypto/ecdsa"
	"fmt"
	"math/big"
	"testing"

	"github.com/holiman/uint256"
	"github.com/luxfi/constants"
	"github.com/luxfi/crypto"
	"github.com/luxfi/evmgpu/consensus/dummy"
	"github.com/luxfi/evmgpu/params"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/core/vm"
)

// The execution plane has no conformance vector.
//
// The corpus in luxfi/conformance covers the EVM with three vectors -- chain
// id, the genesis block, and the exact header bytes the genesis hash is taken
// over -- plus the pure per-transaction functions: signing hash, wire bytes,
// sender recovery, transaction-list root, intrinsic gas, contract address. Every
// one of those is a pure function of its input, and none of them observes what
// executing a block does to state.
//
// So a block processor that returned a full set of receipts while leaving
// canonical state untouched passed the entire corpus. It shipped in this
// repository, reachable at four transactions per block, and what caught it was
// an unrelated gas-price test failing on a merkle root.
//
// This test closes that. It executes a block of transfers through the real
// insert path and pins what came out: the post-execution state root and the
// receipt root. A backend that does not move state cannot produce these roots,
// whatever receipts it hands back.
//
// The values are what luxfi/evm v1.104.49 produces for the identical scenario.
// They are a cross-implementation agreement, not a local snapshot -- if a change
// here moves them, this EVM has diverged from the Go line and the question is
// which one is right, not which constant to update.
const (
	// State root after one block of transferConformanceTxCount transfers.
	wantExecutedStateRoot = "0xcfcbb36538cdab67b730c404ce8fc329f8864685aebe80e0f9a64538284c7993"
	// Receipt root of that same block.
	wantExecutedReceiptRoot = "0xc45910b88e011b192e2e89e58f3fdac8190eee24626de6d546e1788f7d444d30"

	// The removed parallel dispatch handed any block of four or more
	// transactions to the Block-STM engine, but the engine itself only went
	// parallel once the block had at least GOMAXPROCS transactions and a gas
	// limit of 4M -- below that it returned a fallback error and the dispatch
	// dropped through to the sequential loop. A block of eight therefore
	// executes correctly on a 32-core machine and proves nothing.
	//
	// Sixty-four clears GOMAXPROCS on any machine this is likely to run on, so
	// the block is the shape that actually produced receipts against unmoved
	// state. Verified: this test fails at the commit before the dispatch was
	// removed and passes after.
	transferConformanceTxCount = 64
)

// conformanceKey returns the private key for sender i: the scalar i+1, so the
// account set is fixed and reproducible across implementations without carrying
// a table of literals.
func conformanceKey(i int) string {
	return fmt.Sprintf("%064x", i+1)
}

// transferRecipient receives every transfer, so the recipient balance is a
// single accumulated value rather than eight separate ones.
var transferRecipient = common.HexToAddress("0x00000000000000000000000000000000000000ff")

// buildTransferBlock genesises a chain funding one account per key, then
// produces a single block in which each account sends one transfer to
// transferRecipient. It returns the inserted chain and the senders.
func buildTransferBlock(t *testing.T) (*BlockChain, []common.Address, *types.Block) {
	t.Helper()

	config := params.Copy(params.TestChainConfig)
	signer := types.LatestSigner(config)

	alloc := types.GenesisAlloc{}
	senders := make([]common.Address, 0, transferConformanceTxCount)
	keys := make([]*ecdsa.PrivateKey, 0, transferConformanceTxCount)
	for i := 0; i < transferConformanceTxCount; i++ {
		key, err := crypto.HexToECDSA(conformanceKey(i))
		if err != nil {
			t.Fatalf("key %d: %v", i, err)
		}
		// crypto and geth each carry their own 20-byte Address type.
		cryptoAddr := crypto.PubkeyToAddress(key.PublicKey)
		addr := common.BytesToAddress(cryptoAddr[:])

		alloc[addr] = types.Account{Balance: big.NewInt(1_000_000_000_000_000_000)}
		senders = append(senders, addr)
		keys = append(keys, key)
	}

	gspec := &Genesis{
		Config:   config,
		Alloc:    alloc,
		GasLimit: params.GetExtra(config).FeeConfig.GasLimit.Uint64(),
	}
	engine := dummy.NewETHFaker()

	genDb, blocks, _, err := GenerateChainWithGenesis(gspec, engine, 1, 1, func(i int, b *BlockGen) {
		// Lux requires the blackhole coinbase; fees are burned, not paid out.
		b.SetCoinbase(constants.BlackholeAddr)
		for j := 0; j < transferConformanceTxCount; j++ {
			tx, err := types.SignTx(types.NewTx(&types.LegacyTx{
				Nonce:    0,
				To:       &transferRecipient,
				Value:    big.NewInt(1_000),
				Gas:      21_000,
				GasPrice: b.BaseFee(),
			}), signer, keys[j])
			if err != nil {
				t.Fatalf("sign tx %d: %v", j, err)
			}
			b.AddTx(tx)
		}
	})
	if err != nil {
		t.Fatalf("generate chain: %v", err)
	}

	chain, err := NewBlockChain(genDb, DefaultCacheConfig, gspec, engine, vm.Config{}, common.Hash{}, false)
	if err != nil {
		t.Fatalf("new blockchain: %v", err)
	}
	if _, err := chain.InsertChain(blocks); err != nil {
		t.Fatalf("insert chain: %v", err)
	}
	return chain, senders, blocks[0]
}

// TestBlockExecutionMovesState is the assertion the corpus was missing: after a
// block of transfers is inserted, every sender is poorer and the recipient is
// richer. A processor that finalizes on receipts without committing writes
// fails here and nowhere else.
func TestBlockExecutionMovesState(t *testing.T) {
	chain, senders, block := buildTransferBlock(t)
	defer chain.Stop()

	statedb, err := chain.StateAt(block.Root())
	if err != nil {
		t.Fatalf("state at block root: %v", err)
	}

	funded := uint256.NewInt(1_000_000_000_000_000_000)
	for i, s := range senders {
		if got := statedb.GetBalance(s); got.Cmp(funded) >= 0 {
			t.Errorf("sender %d still holds its full genesis balance (%s); "+
				"the block was accepted without executing it", i, got)
		}
	}

	wantRecipient := uint256.NewInt(uint64(1_000 * transferConformanceTxCount))
	if got := statedb.GetBalance(transferRecipient); got.Cmp(wantRecipient) != 0 {
		t.Errorf("recipient balance %s, want %s", got, wantRecipient)
	}
}

// TestBlockExecutionRootsMatchGoLine pins the post-execution state root and the
// receipt root against luxfi/evm. These are the two values no vector in the
// conformance corpus records, and the only ones that catch an execution backend
// that agrees on every transaction yet disagrees on what the block did.
func TestBlockExecutionRootsMatchGoLine(t *testing.T) {
	chain, _, block := buildTransferBlock(t)
	defer chain.Stop()

	if got := block.Root().Hex(); got != wantExecutedStateRoot {
		t.Errorf("post-execution state root\n got %s\nwant %s", got, wantExecutedStateRoot)
	}
	if got := block.ReceiptHash().Hex(); got != wantExecutedReceiptRoot {
		t.Errorf("receipt root\n got %s\nwant %s", got, wantExecutedReceiptRoot)
	}
}
