// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package parallel

import (
	"errors"
	"math/big"
	"testing"

	"github.com/holiman/uint256"
	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/tracing"
	"github.com/luxfi/geth/core/types"
	ethparams "github.com/luxfi/geth/params"
)

// These run in every build: the GPU call is a field, and here it is a fake
// that records what reached it.

// fakeGPU is a GPUDispatcher whose answer the test chooses.
type fakeGPU struct {
	results []GPUEVMResult
	err     error
	calls   int
}

func (f *fakeGPU) Available() bool { return true }
func (f *fakeGPU) Backend() string { return "fake" }
func (f *fakeGPU) ExecuteBlock(*ethparams.ChainConfig, *types.Header, []*types.Transaction, []common.Address, StateGetter) ([]GPUEVMResult, error) {
	f.calls++
	return f.results, f.err
}

// successes is n results the GPU ran through cleanly.
func successes(n int) []GPUEVMResult {
	out := make([]GPUEVMResult, n)
	for i := range out {
		out[i] = GPUEVMResult{GasUsed: ethparams.TxGas, Success: true}
	}
	return out
}

// A batch the GPU declines has no result: the block runs on the sequential
// path, and what that path returns is what ExecuteBlock returns. Nothing the
// GPU said is counted as executed — not even results that came back with the
// decline, or a set of results that is not one per transaction.
func TestADeclinedBatchRunsTheBlockOnTheSequentialPath(t *testing.T) {
	const n = 8
	for _, tc := range []struct {
		name string
		gpu  *fakeGPU
	}{
		{"ok=0", &fakeGPU{err: ErrGPUDeclined}},
		{"results alongside the decline", &fakeGPU{results: successes(n), err: ErrGPUDeclined}},
		{"a failure that is not a decline", &fakeGPU{err: errors.New("device lost")}},
		{"results for other transactions", &fakeGPU{results: successes(n - 1)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := ethparams.TestChainConfig
			sdb := newTestState(t)
			header := blockHeader()
			txs, _, _ := fundedTransfers(t, sdb, config, n)
			getter, factory := productionSeams(sdb, config, header)

			engine := NewEngine(n / 2)
			engine.gpuEVM = tc.gpu
			receipts, err := engine.ExecuteBlock(config, header, txs, getter, factory)

			wantReceipts, wantErr := NewEngine(n/2).executeSequential(config, header, txs, getter, factory)
			if receipts != nil || wantReceipts != nil {
				t.Fatalf("got %d receipts; the sequential path returns %d", len(receipts), len(wantReceipts))
			}
			if err == nil || err.Error() != wantErr.Error() {
				t.Fatalf("ExecuteBlock = %v, want the sequential path's answer %v", err, wantErr)
			}
			if tc.gpu.calls != 1 {
				t.Errorf("the GPU was called %d times, want 1", tc.gpu.calls)
			}
			st := engine.Stats()
			if !st.FellBack {
				t.Error("the engine did not record the fallback")
			}
			if st.GPUExecuted != 0 {
				t.Errorf("%d txs counted as GPU-executed from a batch with no result", st.GPUExecuted)
			}
			if st.GPUFallback != n {
				t.Errorf("GPUFallback = %d, want %d", st.GPUFallback, n)
			}
		})
	}
}

// A batch the GPU ran is counted, and Block-STM still executes the block: no
// GPU result reaches a receipt.
func TestAnAcceptedBatchIsCountedAndBlockSTMRunsTheBlock(t *testing.T) {
	const n = 8
	config := ethparams.TestChainConfig
	sdb := newTestState(t)
	header := blockHeader()
	txs, _, _ := fundedTransfers(t, sdb, config, n)
	getter, factory := productionSeams(sdb, config, header)

	engine := NewEngine(n / 2)
	engine.gpuEVM = &fakeGPU{results: successes(n)}
	receipts, err := engine.ExecuteBlock(config, header, txs, getter, factory)
	if err != nil {
		t.Fatalf("ExecuteBlock: %v", err)
	}
	if len(receipts) != n {
		t.Fatalf("got %d receipts, want %d", len(receipts), n)
	}
	if st := engine.Stats(); st.FellBack || st.GPUExecuted != n {
		t.Fatalf("FellBack=%v GPUExecuted=%d, want false and %d", st.FellBack, st.GPUExecuted, n)
	}
}

// Nothing the 64-bit wire cannot carry reaches the GPU. A value or price of
// 2^64 is legal; truncated to 0 it is another transaction. A base fee or chain
// id that does not fit, an account with code (whose bytes the state getter
// does not give), and a state the getter cannot answer are the same: the batch
// is declined and the GPU is never called.
func TestWhatTheWireCannotCarryNeverReachesTheGPU(t *testing.T) {
	huge := new(big.Int).Lsh(big.NewInt(1), 64)
	to := common.Address{0x11}
	contract := common.Address{0xCC}
	legacy := func(value, price *big.Int) *types.Transaction {
		return types.NewTx(&types.LegacyTx{To: &to, Value: value, Gas: ethparams.TxGas, GasPrice: price})
	}
	for _, tc := range []struct {
		name   string
		tx     *types.Transaction
		header func(*types.Header)
		config *ethparams.ChainConfig
		state  func(StateGetter) StateGetter
	}{
		{name: "value 2^64", tx: legacy(huge, big.NewInt(1))},
		{name: "gas price 2^64", tx: legacy(big.NewInt(1), huge)},
		{name: "fee cap 2^64", tx: types.NewTx(&types.DynamicFeeTx{To: &to, Value: big.NewInt(1), Gas: ethparams.TxGas, GasFeeCap: huge, GasTipCap: big.NewInt(1)})},
		{name: "base fee 2^64", tx: legacy(big.NewInt(1), big.NewInt(1)), header: func(h *types.Header) { h.BaseFee = new(big.Int).Set(huge) }},
		{name: "chain id 2^64", tx: legacy(big.NewInt(1), big.NewInt(1)), config: &ethparams.ChainConfig{ChainID: new(big.Int).Set(huge)}},
		{name: "calldata", tx: types.NewTx(&types.LegacyTx{To: &to, Gas: 30000, GasPrice: big.NewInt(1), Data: []byte{0x01}})},
		{
			name: "recipient with code",
			tx:   types.NewTx(&types.LegacyTx{To: &contract, Value: big.NewInt(1), Gas: ethparams.TxGas, GasPrice: big.NewInt(1)}),
		},
		{
			name: "a state the getter cannot answer",
			tx:   legacy(big.NewInt(1), big.NewInt(1)),
			state: func(StateGetter) StateGetter {
				return func(MemoryLocation) (MemoryValue, bool) { return MemoryValue{}, false }
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := ethparams.TestChainConfig
			if tc.config != nil {
				config = tc.config
			}
			header := blockHeader()
			if tc.header != nil {
				tc.header(header)
			}
			sdb := newTestState(t)
			sdb.SetCode(contract, []byte{0x00}, tracing.CodeChangeUnspecified)
			getter, _ := productionSeams(sdb, config, header)
			if tc.state != nil {
				getter = tc.state(getter)
			}

			d := &GPUEVMDispatcher{
				backend: 2,
				execute: func(uint8, *gpuBatch) ([]GPUEVMResult, error) {
					t.Fatal("the GPU was called with a batch its wire cannot carry")
					return nil, nil
				},
			}
			results, err := d.ExecuteBlock(config, header, []*types.Transaction{tc.tx}, []common.Address{{0x01}}, getter)
			if !errors.Is(err, ErrGPUDeclined) {
				t.Fatalf("ExecuteBlock = %v, want ErrGPUDeclined", err)
			}
			if results != nil {
				t.Errorf("a declined batch came back with %d results", len(results))
			}
		})
	}
}

// End to end: a block with one transfer of 2^64 goes to the sequential path,
// and the GPU never sees any of it.
func TestABlockWithAValueTooWideRunsOnTheSequentialPath(t *testing.T) {
	const n = 8
	config := ethparams.TestChainConfig
	sdb := newTestState(t)
	header := blockHeader()
	txs, _, _ := fundedTransfers(t, sdb, config, n-1)

	key, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	cryptoFrom := crypto.PubkeyToAddress(key.PublicKey)
	from := common.BytesToAddress(cryptoFrom[:])
	sdb.SetBalance(from, uint256.MustFromDecimal("100000000000000000000000"), tracing.BalanceChangeUnspecified)
	to := common.Address{0x11}
	wide, err := types.SignTx(types.NewTx(&types.LegacyTx{
		To: &to, Value: new(big.Int).Lsh(big.NewInt(1), 64), Gas: ethparams.TxGas, GasPrice: big.NewInt(1_000_000_000),
	}), types.LatestSigner(config), key)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	txs = append(txs, wide)
	getter, factory := productionSeams(sdb, config, header)

	engine := NewEngine(n / 2)
	engine.gpuEVM = &GPUEVMDispatcher{
		backend: 2,
		execute: func(uint8, *gpuBatch) ([]GPUEVMResult, error) {
			t.Fatal("the GPU was called for a block holding a value it cannot carry")
			return nil, nil
		},
	}
	receipts, err := engine.ExecuteBlock(config, header, txs, getter, factory)
	if err == nil || receipts != nil {
		t.Fatalf("ExecuteBlock = (%d receipts, %v), want the sequential path's answer", len(receipts), err)
	}
	if !engine.Stats().FellBack {
		t.Error("the engine did not record the fallback")
	}
}

// The batch is what go_bridge.h asks for: every touched account (senders,
// recipients, the coinbase), once, each with its balance in little-endian
// limbs, its nonce, and its code hash and storage root — the hashes of nothing
// for an account that does not exist, the real root for one with storage.
func TestTheBatchCarriesTheStateTheContractAsks(t *testing.T) {
	config := ethparams.TestChainConfig
	header := blockHeader()
	sdb := newTestState(t)

	from := common.Address{0x01}
	bal := new(uint256.Int).SetBytes32([]byte{
		0x44, 0, 0, 0, 0, 0, 0, 0x43,
		0x34, 0, 0, 0, 0, 0, 0, 0x33,
		0x24, 0, 0, 0, 0, 0, 0, 0x23,
		0x14, 0, 0, 0, 0, 0, 0, 0x13,
	})
	sdb.SetBalance(from, bal, tracing.BalanceChangeUnspecified)
	sdb.SetNonce(from, 3, tracing.NonceChangeUnspecified)
	sdb.SetState(from, common.Hash{0x01}, common.Hash{0x02}) // storage, and no code
	sdb.IntermediateRoot(false)
	root := sdb.GetStorageRoot(from)
	if root == (common.Hash{}) || root == types.EmptyRootHash {
		t.Fatalf("test setup: storage root %x is not a written trie's", root)
	}

	to := common.Address{0x02} // does not exist
	tx := types.NewTx(&types.DynamicFeeTx{
		Nonce: 3, To: &to, Value: big.NewInt(7), Gas: ethparams.TxGas,
		GasFeeCap: big.NewInt(9), GasTipCap: big.NewInt(2),
	})
	getter, _ := productionSeams(sdb, config, header)

	b, ok := shapeGPUBatch(config, header, []*types.Transaction{tx, tx}, []common.Address{from, from}, getter)
	if !ok {
		t.Fatal("shapeGPUBatch declined a batch the wire carries")
	}

	want := gpuTx{From: from, To: to, GasLimit: ethparams.TxGas, Value: 7, Nonce: 3, GasPrice: 9}
	if len(b.txs) != 2 || b.txs[0] != want || b.txs[1] != want {
		t.Fatalf("txs = %+v, want two of %+v (the price is the fee cap)", b.txs, want)
	}
	if b.ctx.GasLimit != header.GasLimit || b.ctx.BaseFee != header.BaseFee.Uint64() ||
		b.ctx.ChainID != config.ChainID.Uint64() || b.ctx.Number != header.Number.Uint64() ||
		b.ctx.Timestamp != header.Time || common.Address(b.ctx.Coinbase) != header.Coinbase {
		t.Errorf("context = %+v does not carry the header", b.ctx)
	}

	if len(b.accounts) != 3 {
		t.Fatalf("%d accounts, want 3: the sender, the recipient and the coinbase, once each", len(b.accounts))
	}
	sender, recipient, coinbase := b.accounts[0], b.accounts[1], b.accounts[2]
	if common.Address(sender.Address) != from || common.Address(recipient.Address) != to ||
		common.Address(coinbase.Address) != header.Coinbase {
		t.Fatalf("accounts are %x %x %x", sender.Address, recipient.Address, coinbase.Address)
	}
	if sender.Balance != [4]uint64{0x1400000000000013, 0x2400000000000023, 0x3400000000000033, 0x4400000000000043} {
		t.Errorf("sender balance limbs = %#x", sender.Balance)
	}
	if sender.Nonce != 3 {
		t.Errorf("sender nonce = %d, want 3", sender.Nonce)
	}
	if common.Hash(sender.StorageRoot) != root {
		t.Errorf("sender storage root = %x, want %x", sender.StorageRoot, root)
	}
	for _, a := range b.accounts {
		if common.Hash(a.CodeHash) != types.EmptyCodeHash {
			t.Errorf("account %x code hash = %x, want keccak256 of nothing", a.Address, a.CodeHash)
		}
	}
	for _, a := range []gpuAccount{recipient, coinbase} {
		if common.Hash(a.StorageRoot) != types.EmptyRootHash {
			t.Errorf("absent account %x storage root = %x, want the empty trie root", a.Address, a.StorageRoot)
		}
	}
}

// PREVRANDAO (0x44) is what the Go EVM answers for the header: its difficulty
// as a 32-byte word, which NewEVMBlockContext makes Random from Shanghai on and
// DIFFICULTY answers before it. The MixDigest, zero on a Lux header, is not it.
func TestTheBatchPrevrandaoIsTheHeadersDifficulty(t *testing.T) {
	config := ethparams.TestChainConfig
	getter, _ := productionSeams(newTestState(t), config, blockHeader())
	to := common.Address{0x11}
	tx := types.NewTx(&types.LegacyTx{To: &to, Value: big.NewInt(1), Gas: ethparams.TxGas, GasPrice: big.NewInt(1)})

	header := blockHeader()
	header.Difficulty = big.NewInt(0x1234)
	header.MixDigest = common.Hash{0xAB, 0xCD}
	b, ok := shapeGPUBatch(config, header, []*types.Transaction{tx}, []common.Address{{0x01}}, getter)
	if !ok {
		t.Fatal("shapeGPUBatch declined a batch the wire carries")
	}
	if want := common.BigToHash(header.Difficulty); common.Hash(b.ctx.Prevrandao) != want {
		t.Errorf("Prevrandao = %x, want the difficulty %x, not the MixDigest %x", b.ctx.Prevrandao, want, header.MixDigest)
	}

	header.Difficulty = nil
	b, ok = shapeGPUBatch(config, header, []*types.Transaction{tx}, []common.Address{{0x01}}, getter)
	if !ok {
		t.Fatal("shapeGPUBatch declined a header without a difficulty")
	}
	if b.ctx.Prevrandao != ([32]byte{}) {
		t.Errorf("a header without a difficulty gave Prevrandao %x, want zero", b.ctx.Prevrandao)
	}
}

// A dispatcher with no device call is not available, and declines rather than
// calling nil.
func TestAZeroDispatcherDeclines(t *testing.T) {
	if (&GPUEVMDispatcher{backend: 2}).Available() {
		t.Error("a dispatcher with no device call says it is available")
	}
	config := ethparams.TestChainConfig
	header := blockHeader()
	getter, _ := productionSeams(newTestState(t), config, header)
	to := common.Address{0x11}
	tx := types.NewTx(&types.LegacyTx{To: &to, Value: big.NewInt(1), Gas: ethparams.TxGas, GasPrice: big.NewInt(1)})
	results, err := (&GPUEVMDispatcher{}).ExecuteBlock(config, header, []*types.Transaction{tx}, []common.Address{{0x01}}, getter)
	if !errors.Is(err, ErrGPUDeclined) || results != nil {
		t.Fatalf("zero dispatcher = (%v, %v), want ErrGPUDeclined", results, err)
	}
}

// A plain transfer is eligible; anything CGpuTx cannot carry is not.
func TestIsGPUEligible(t *testing.T) {
	to := common.Address{0x12}
	one := big.NewInt(1)
	for _, tc := range []struct {
		name string
		tx   *types.Transaction
		want bool
	}{
		{"legacy transfer", types.NewTx(&types.LegacyTx{To: &to, Value: one, Gas: ethparams.TxGas, GasPrice: one}), true},
		{"dynamic-fee transfer", types.NewTx(&types.DynamicFeeTx{To: &to, Value: one, Gas: ethparams.TxGas, GasFeeCap: big.NewInt(2), GasTipCap: one}), true},
		{"access-list type, empty list", types.NewTx(&types.AccessListTx{To: &to, Value: one, Gas: ethparams.TxGas, GasPrice: one}), true},
		{"calldata", types.NewTx(&types.LegacyTx{To: &to, Gas: 30000, GasPrice: one, Data: []byte{0x01}}), false},
		{"creation", types.NewTx(&types.LegacyTx{Gas: 60000, GasPrice: one}), false},
		{
			"an access list",
			types.NewTx(&types.AccessListTx{
				To: &to, Value: one, Gas: 23400, GasPrice: one,
				AccessList: types.AccessList{{Address: common.Address{0x22}}},
			}),
			false,
		},
		{"tip above fee cap", types.NewTx(&types.DynamicFeeTx{To: &to, Value: one, Gas: ethparams.TxGas, GasFeeCap: one, GasTipCap: big.NewInt(2)}), false},
		{
			"blob",
			types.NewTx(&types.BlobTx{
				To: to, Value: uint256.NewInt(1), Gas: ethparams.TxGas,
				GasFeeCap: uint256.NewInt(1), GasTipCap: uint256.NewInt(1), BlobFeeCap: uint256.NewInt(1),
				BlobHashes: []common.Hash{{0x01}},
			}),
			false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsGPUEligible(tc.tx); got != tc.want {
				t.Fatalf("IsGPUEligible = %v, want %v", got, tc.want)
			}
		})
	}
}
