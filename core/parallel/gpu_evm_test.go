// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu

package parallel

import (
	"crypto/ecdsa"
	"math/big"
	"testing"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	ethparams "github.com/luxfi/geth/params"
	"github.com/stretchr/testify/require"
)

// makeTestTransfer creates a signed simple ETH transfer (GPU-eligible).
func makeTestTransfer(t *testing.T, key *ecdsa.PrivateKey, signer types.Signer, nonce uint64, to common.Address, value *big.Int) *types.Transaction {
	t.Helper()
	tx := types.NewTransaction(nonce, to, value, 21000, big.NewInt(1_000_000_000), nil)
	signed, err := types.SignTx(tx, signer, key)
	require.NoError(t, err)
	return signed
}

// makeTestContractCall creates a signed tx with calldata (not GPU-eligible).
func makeTestContractCall(t *testing.T, key *ecdsa.PrivateKey, signer types.Signer, nonce uint64, to common.Address) *types.Transaction {
	t.Helper()
	tx := types.NewTransaction(nonce, to, big.NewInt(0), 100000, big.NewInt(1_000_000_000), []byte{0x01, 0x02, 0x03})
	signed, err := types.SignTx(tx, signer, key)
	require.NoError(t, err)
	return signed
}

func TestIsGPUEligible(t *testing.T) {
	key, err := crypto.GenerateKey()
	require.NoError(t, err)

	signer := types.NewEIP155Signer(big.NewInt(1))
	to := common.HexToAddress("0x1234567890abcdef1234567890abcdef12345678")

	// Simple transfer -- eligible.
	transfer := makeTestTransfer(t, key, signer, 0, to, big.NewInt(1e18))
	require.True(t, IsGPUEligible(transfer))

	// Contract call with data -- not eligible.
	call := makeTestContractCall(t, key, signer, 1, to)
	require.False(t, IsGPUEligible(call))

	// Contract creation (no To) -- not eligible.
	createTx := types.NewContractCreation(2, big.NewInt(0), 100000, big.NewInt(1e9), []byte{0x60, 0x00})
	signedCreate, err := types.SignTx(createTx, signer, key)
	require.NoError(t, err)
	require.False(t, IsGPUEligible(signedCreate))
}

func TestGPUEVMDispatcher_Init(t *testing.T) {
	d := NewGPUEVMDispatcher()
	require.NotNil(t, d)

	backend := d.Backend()
	require.Contains(t, []string{"CPU-Sequential", "CPU-Parallel", "Metal", "CUDA"}, backend)
	t.Logf("GPU EVM backend: %s, available: %v", backend, d.Available())
}

func TestGPUEVMDispatcher_ExecuteBlock_SimpleTransfers(t *testing.T) {
	d := NewGPUEVMDispatcher()
	require.NotNil(t, d)

	key, err := crypto.GenerateKey()
	require.NoError(t, err)

	signer := types.NewEIP155Signer(big.NewInt(1))
	to := common.HexToAddress("0x1234567890abcdef1234567890abcdef12345678")
	sender := crypto.PubkeyToAddress(key.PublicKey)

	// Create 10 simple transfers.
	const numTxs = 10
	txs := make([]*types.Transaction, numTxs)
	senders := make([]common.Address, numTxs)
	for i := 0; i < numTxs; i++ {
		txs[i] = makeTestTransfer(t, key, signer, uint64(i), to, big.NewInt(1e18))
		senders[i] = sender
	}

	results, err := d.ExecuteBlock(signer, txs, senders)
	require.NoError(t, err)
	require.Len(t, results, numTxs)

	for i, r := range results {
		require.True(t, r.Success, "tx %d should succeed", i)
		require.Greater(t, r.GasUsed, uint64(0), "tx %d should use gas", i)
	}
}

func TestGPUEVMDispatcher_ExecuteBlock_Empty(t *testing.T) {
	d := NewGPUEVMDispatcher()
	results, err := d.ExecuteBlock(
		types.NewEIP155Signer(big.NewInt(1)),
		nil,
		nil,
	)
	require.NoError(t, err)
	require.Nil(t, results)
}

func TestGPUEVMDispatcher_MatchesCPUGas(t *testing.T) {
	// Simple transfers always cost exactly 21000 gas.
	// Verify that the GPU kernel returns the same value.
	d := NewGPUEVMDispatcher()

	key, err := crypto.GenerateKey()
	require.NoError(t, err)

	signer := types.NewEIP155Signer(big.NewInt(1))
	to := common.HexToAddress("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
	sender := crypto.PubkeyToAddress(key.PublicKey)

	tx := makeTestTransfer(t, key, signer, 0, to, big.NewInt(1000))
	results, err := d.ExecuteBlock(signer, []*types.Transaction{tx}, []common.Address{sender})
	require.NoError(t, err)
	require.Len(t, results, 1)

	// GPU returns gas_limit for gas-estimation-only mode (no host).
	// The C++ engine returns tx.gas_limit when state==nullptr.
	expectedGas := tx.Gas()
	require.Equal(t, expectedGas, results[0].GasUsed,
		"GPU gas should match expected (gas-estimation mode returns gas_limit)")
}

// Benchmarks: compare Go EVM overhead vs CGo GPU dispatch for simple transfers.

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

	key, err := crypto.GenerateKey()
	if err != nil {
		b.Fatal(err)
	}

	signer := types.NewEIP155Signer(big.NewInt(1))
	to := common.HexToAddress("0x1234567890abcdef1234567890abcdef12345678")
	sender := crypto.PubkeyToAddress(key.PublicKey)

	txs := make([]*types.Transaction, numTxs)
	senders := make([]common.Address, numTxs)
	for i := 0; i < numTxs; i++ {
		raw := types.NewTransaction(uint64(i), to, big.NewInt(1e18), 21000, big.NewInt(1e9), nil)
		txs[i], _ = types.SignTx(raw, signer, key)
		senders[i] = sender
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := d.ExecuteBlock(signer, txs, senders)
		if err != nil {
			b.Fatal(err)
		}
	}

	_ = ethparams.MainnetChainConfig // reference to confirm import resolves
}
