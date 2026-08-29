// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"math/big"
	"testing"

	"github.com/holiman/uint256"
	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/state"
	"github.com/luxfi/geth/core/tracing"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/core/vm"
	ethparams "github.com/luxfi/geth/params"
)

// blockGasLimit is above the 4M floor under which ExecuteBlock refuses to go
// parallel, so these tests exercise the Block-STM path rather than the
// sequential fallback.
const blockGasLimit = 15_000_000

// newTestState returns an empty in-memory state database.
func newTestState(t *testing.T) *state.StateDB {
	t.Helper()
	sdb, err := state.New(types.EmptyRootHash, state.NewDatabaseForTesting())
	if err != nil {
		t.Fatalf("new state: %v", err)
	}
	return sdb
}

// fundedTransfers builds n signed value transfers, each from a distinct funded
// sender to a single shared recipient, and credits the senders in sdb. It
// returns the transactions, the senders in transaction order, and the
// recipient.
func fundedTransfers(t *testing.T, sdb *state.StateDB, config *ethparams.ChainConfig, n int) (types.Transactions, []common.Address, common.Address) {
	t.Helper()

	signer := types.LatestSigner(config)
	recipient := common.HexToAddress("0xdeadbeef00000000000000000000000000000001")

	txs := make(types.Transactions, 0, n)
	senders := make([]common.Address, 0, n)
	for i := 0; i < n; i++ {
		key, err := crypto.GenerateKey()
		if err != nil {
			t.Fatalf("generate key: %v", err)
		}
		// crypto and geth each carry their own 20-byte Address type.
		cryptoFrom := crypto.PubkeyToAddress(key.PublicKey)
		from := common.BytesToAddress(cryptoFrom[:])

		// Plenty for value + gas at 1 gwei.
		sdb.SetBalance(from, uint256.NewInt(1_000_000_000_000_000_000), tracing.BalanceChangeUnspecified)

		tx, err := types.SignTx(types.NewTx(&types.LegacyTx{
			Nonce:    0,
			To:       &recipient,
			Value:    big.NewInt(1_000),
			Gas:      ethparams.TxGas,
			GasPrice: big.NewInt(1_000_000_000),
		}), signer, key)
		if err != nil {
			t.Fatalf("sign tx: %v", err)
		}

		txs = append(txs, tx)
		senders = append(senders, from)
	}
	return txs, senders, recipient
}

// blockHeader returns a header whose gas limit clears the parallel threshold.
func blockHeader() *types.Header {
	return &types.Header{
		Number:   big.NewInt(1),
		Time:     1,
		GasLimit: blockGasLimit,
		Coinbase: common.HexToAddress("0xc0ffee0000000000000000000000000000000000"),
		BaseFee:  big.NewInt(1),
	}
}

// productionSeams builds the StateGetter and VMFactory exactly the way a block
// processor would have to: reads come from the canonical state, and every
// transaction is handed its own copy to execute against.
func productionSeams(sdb *state.StateDB, config *ethparams.ChainConfig, header *types.Header) (StateGetter, VMFactory) {
	getter := func(loc MemoryLocation) (MemoryValue, bool) {
		switch loc.Type {
		case LocationBalance:
			var balHash common.Hash
			sdb.GetBalance(loc.Address).WriteToSlice(balHash[:])
			return MemoryValue{Balance: balHash}, true
		case LocationNonce:
			return MemoryValue{Nonce: sdb.GetNonce(loc.Address)}, true
		case LocationStorage:
			return MemoryValue{Storage: sdb.GetState(loc.Address, loc.Slot)}, true
		case LocationCodeHash:
			return MemoryValue{Storage: sdb.GetCodeHash(loc.Address)}, true
		}
		return MemoryValue{}, false
	}

	factory := func(TxIdx) (*vm.EVM, error) {
		blockCtx := vm.BlockContext{
			CanTransfer: func(db vm.StateDB, addr common.Address, amount *uint256.Int) bool {
				return db.GetBalance(addr).Cmp(amount) >= 0
			},
			Transfer: func(db vm.StateDB, sender, recipient common.Address, amount *uint256.Int) {
				db.SubBalance(sender, amount, tracing.BalanceChangeTransfer)
				db.AddBalance(recipient, amount, tracing.BalanceChangeTransfer)
			},
			GetHash:     func(uint64) common.Hash { return common.Hash{} },
			Coinbase:    header.Coinbase,
			GasLimit:    header.GasLimit,
			BlockNumber: new(big.Int).Set(header.Number),
			Time:        header.Time,
			Difficulty:  new(big.Int),
			BaseFee:     new(big.Int).Set(header.BaseFee),
		}
		return vm.NewEVM(blockCtx, sdb.Copy(), config, vm.Config{}), nil
	}

	return getter, factory
}

// TestExecuteBlockDoesNotTouchCanonicalState pins the engine's actual contract.
//
// ExecuteBlock reads canonical state through a StateGetter and executes against
// whatever the VMFactory hands it. It has no write channel back, so it cannot
// and does not mutate the canonical state database. It still returns a full set
// of receipts.
//
// That combination -- receipts for a block whose state never moved -- is what
// made a block processor that trusted these receipts produce the pre-block
// state root for a block full of transfers. The property is asserted here so
// that a future caller has to confront it rather than discover it on a
// validator.
//
// A caller that wants to commit these results has to route the write sets back
// itself. ExecuteBlock does not surface them today, which is why ApplyToState
// still has no caller.
func TestExecuteBlockDoesNotTouchCanonicalState(t *testing.T) {
	const numTxs = 8

	config := ethparams.TestChainConfig
	sdb := newTestState(t)
	header := blockHeader()

	txs, senders, recipient := fundedTransfers(t, sdb, config, numTxs)

	balancesBefore := make([]*uint256.Int, len(senders))
	for i, s := range senders {
		balancesBefore[i] = sdb.GetBalance(s).Clone()
	}
	recipientBefore := sdb.GetBalance(recipient).Clone()

	getter, factory := productionSeams(sdb, config, header)

	// Concurrency below the transaction count keeps ExecuteBlock off its
	// sequential fallback.
	engine := NewEngine(numTxs / 2)
	receipts, err := engine.ExecuteBlock(config, header, txs, getter, factory)
	if err != nil {
		t.Fatalf("ExecuteBlock: %v", err)
	}
	if len(receipts) != numTxs {
		t.Fatalf("got %d receipts, want %d", len(receipts), numTxs)
	}

	for i, s := range senders {
		if got := sdb.GetBalance(s); got.Cmp(balancesBefore[i]) != 0 {
			t.Errorf("sender %d balance moved: before %s, after %s -- "+
				"ExecuteBlock is not supposed to reach canonical state",
				i, balancesBefore[i], got)
		}
	}
	if got := sdb.GetBalance(recipient); got.Cmp(recipientBefore) != 0 {
		t.Errorf("recipient balance moved: before %s, after %s -- "+
			"ExecuteBlock is not supposed to reach canonical state",
			recipientBefore, got)
	}
}

// TestExecuteBlockRefusesSmallBlocks records the two conditions under which
// ExecuteBlock declines to run in parallel: fewer transactions than workers, or
// a gas limit under 4M. In both cases it takes its sequential fallback, which
// does not execute anything -- it returns an error telling the caller to run
// its own sequential loop.
func TestExecuteBlockRefusesSmallBlocks(t *testing.T) {
	config := ethparams.TestChainConfig

	for _, tc := range []struct {
		name     string
		numTxs   int
		gasLimit uint64
	}{
		{name: "fewer txs than workers", numTxs: 2, gasLimit: blockGasLimit},
		{name: "gas limit below 4M", numTxs: 8, gasLimit: 3_999_999},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sdb := newTestState(t)
			header := blockHeader()
			header.GasLimit = tc.gasLimit

			txs, _, _ := fundedTransfers(t, sdb, config, tc.numTxs)
			getter, factory := productionSeams(sdb, config, header)

			engine := NewEngine(4)
			receipts, err := engine.ExecuteBlock(config, header, txs, getter, factory)
			if err == nil {
				t.Fatalf("want the sequential-fallback error, got %d receipts and no error", len(receipts))
			}
			if !engine.stats.FellBack {
				t.Error("engine did not record the fallback")
			}
		})
	}
}

// TestExecuteBlockEmpty covers the empty-block short circuit.
func TestExecuteBlockEmpty(t *testing.T) {
	config := ethparams.TestChainConfig
	sdb := newTestState(t)
	header := blockHeader()
	getter, factory := productionSeams(sdb, config, header)

	receipts, err := NewEngine(4).ExecuteBlock(config, header, nil, getter, factory)
	if err != nil {
		t.Fatalf("ExecuteBlock on an empty block: %v", err)
	}
	if receipts != nil {
		t.Fatalf("want no receipts for an empty block, got %d", len(receipts))
	}
}

// TestApplyToStateCannotRestoreCode records why ApplyToState is not a working
// commit path, so that wiring it up is a deliberate act rather than an
// assumption. It drops LocationCodeHash entries on the stated grounds that
// execution already wrote the code to the base state database -- which is
// untrue when execution ran against copies, as it does under ExecuteBlock.
func TestApplyToStateCannotRestoreCode(t *testing.T) {
	sdb := newTestState(t)
	addr := common.HexToAddress("0xabc0000000000000000000000000000000000001")

	tx := types.NewTx(&types.LegacyTx{Nonce: 0, Gas: ethparams.TxGas, GasPrice: big.NewInt(1)})
	results := []ExecResult{{
		WriteSet: []WriteEntry{{
			Location: MemoryLocation{Address: addr, Type: LocationCodeHash},
			Value:    MemoryValue{Storage: common.HexToHash("0x1234")},
		}},
	}}

	NewEngine(1).ApplyToState(sdb, results, types.Transactions{tx})

	if code := sdb.GetCode(addr); len(code) != 0 {
		t.Fatalf("ApplyToState restored code (%d bytes); update the commit-path "+
			"analysis in state_processor if this behaviour changed", len(code))
	}
}
