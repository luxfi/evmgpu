// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu && cgo && darwin && arm64

package parallel

import (
	"errors"
	"math/big"
	"testing"

	"github.com/holiman/uint256"
	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/state"
	"github.com/luxfi/geth/core/tracing"
	"github.com/luxfi/geth/core/types"
	ethparams "github.com/luxfi/geth/params"
	"github.com/stretchr/testify/require"
)

// These run against the linked library and hold it to go_bridge.h (ABI 6).

// transferBatch is n signed plain transfers from one funded sender, with the
// state getter and header the dispatcher needs.
func transferBatch(tb testing.TB, n int) (*ethparams.ChainConfig, *types.Header, []*types.Transaction, []common.Address, StateGetter) {
	tb.Helper()
	config := ethparams.TestChainConfig
	header := blockHeader()
	header.GasLimit = max(header.GasLimit, uint64(n)*ethparams.TxGas) // the device refuses limits past the block's
	sdb, err := state.New(types.EmptyRootHash, state.NewDatabaseForTesting())
	require.NoError(tb, err)

	key, err := crypto.GenerateKey()
	require.NoError(tb, err)
	cryptoFrom := crypto.PubkeyToAddress(key.PublicKey)
	sender := common.BytesToAddress(cryptoFrom[:])
	sdb.SetBalance(sender, uint256.MustFromDecimal("1000000000000000000000000"), tracing.BalanceChangeUnspecified)

	signer := types.LatestSigner(config)
	to := common.HexToAddress("0x1234567890abcdef1234567890abcdef12345678")
	txs := make([]*types.Transaction, n)
	senders := make([]common.Address, n)
	for i := range txs {
		tx, err := types.SignTx(types.NewTransaction(uint64(i), to, big.NewInt(1e18), ethparams.TxGas, big.NewInt(1_000_000_000), nil), signer, key)
		require.NoError(tb, err)
		txs[i] = tx
		senders[i] = sender
	}
	getter, _ := productionSeams(sdb, config, header)
	return config, header, txs, senders, getter
}

func TestGPUEVMDispatcher_Init(t *testing.T) {
	d := NewGPUEVMDispatcher()
	require.NotNil(t, d)

	backend := d.Backend()
	require.Contains(t, []string{"CPU-Sequential", "CPU-Parallel", "Metal", "CUDA"}, backend)
	t.Logf("GPU EVM backend: %s, available: %v", backend, d.Available())
}

// Plain transfers from a funded sender run on the device at 21000 gas each,
// or come back declined. Never a result with something else in it.
func TestGPUEVMDispatcher_ExecuteBlock_SimpleTransfers(t *testing.T) {
	d := NewGPUEVMDispatcher()
	if !d.Available() {
		t.Skip("no GPU backend")
	}
	const numTxs = 10
	config, header, txs, senders, getter := transferBatch(t, numTxs)

	results, err := d.ExecuteBlock(config, header, txs, senders, getter)
	if errors.Is(err, ErrGPUDeclined) {
		t.Skipf("the device declined the batch: %v", err)
	}
	require.NoError(t, err)
	require.Len(t, results, numTxs)
	for i, r := range results {
		require.True(t, r.Success, "tx %d should succeed", i)
		require.Equal(t, ethparams.TxGas, r.GasUsed, "tx %d is a plain transfer", i)
	}
}

func TestGPUEVMDispatcher_ExecuteBlock_Empty(t *testing.T) {
	d := NewGPUEVMDispatcher()
	results, err := d.ExecuteBlock(ethparams.TestChainConfig, blockHeader(), nil, nil, nil)
	require.NoError(t, err)
	require.Nil(t, results)
}

// A sender the state does not fund cannot pay for its transfer: the device
// declines, and nothing comes back.
func TestGPUEVMDispatcher_AnUnfundedBatchIsDeclined(t *testing.T) {
	d := NewGPUEVMDispatcher()
	if !d.Available() {
		t.Skip("no GPU backend")
	}
	config, header, txs, _, getter := transferBatch(t, 1)
	results, err := d.ExecuteBlock(config, header, txs, []common.Address{{0xDE, 0xAD}}, getter)
	require.ErrorIs(t, err, ErrGPUDeclined)
	require.Nil(t, results)
}

// Benchmarks: CGo GPU dispatch for simple transfers.

func BenchmarkGPUEVM_100Transfers(b *testing.B) {
	benchmarkGPUEVM(b, 100)
}

func BenchmarkGPUEVM_1000Transfers(b *testing.B) {
	benchmarkGPUEVM(b, 1000)
}

func BenchmarkGPUEVM_10000Transfers(b *testing.B) {
	benchmarkGPUEVM(b, 10000)
}

func benchmarkGPUEVM(b *testing.B, numTxs int) {
	d := NewGPUEVMDispatcher()
	config, header, txs, senders, getter := transferBatch(b, numTxs)
	if _, err := d.ExecuteBlock(config, header, txs, senders, getter); errors.Is(err, ErrGPUDeclined) {
		b.Skipf("the device declines %d transfers at once: %v", numTxs, err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := d.ExecuteBlock(config, header, txs, senders, getter); err != nil {
			b.Fatal(err)
		}
	}
}
