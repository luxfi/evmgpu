// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

package parallel

import (
	"sync"
	"time"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/log"
	"github.com/luxfi/gpu"
)

// SenderRecovery holds a pre-recovered transaction sender address.
type SenderRecovery struct {
	Sender common.Address
	Valid  bool
}

// GPUEcrecoverStats tracks performance metrics for sender recovery.
type GPUEcrecoverStats struct {
	NumTxs     int
	Duration   time.Duration
	Backend    string  // "Metal", "CUDA", "CPU"
	Throughput float64 // signatures/second
}

// SenderCache is a concurrent map of pre-recovered transaction senders.
// Used to populate the EVM's sender cache before block execution.
type SenderCache struct {
	mu      sync.RWMutex
	senders map[common.Hash]common.Address // tx hash → sender address
}

// NewSenderCache creates a new sender cache.
func NewSenderCache(capacity int) *SenderCache {
	return &SenderCache{
		senders: make(map[common.Hash]common.Address, capacity),
	}
}

// Get retrieves a cached sender address for a transaction hash.
func (c *SenderCache) Get(txHash common.Hash) (common.Address, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	addr, ok := c.senders[txHash]
	return addr, ok
}

// Set stores a sender address for a transaction hash.
func (c *SenderCache) Set(txHash common.Hash, sender common.Address) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.senders[txHash] = sender
}

// PreRecoverSenders recovers all transaction sender addresses using GPU
// batch ecrecover. Returns a SenderCache that maps tx hash to sender address.
//
// Call this before block execution. During EVM execution, look up senders
// from the cache instead of computing ecrecover per-transaction.
//
// The GPU batches all signatures into a single kernel dispatch, processing
// them in parallel across GPU shader cores. Falls back to parallel goroutines
// on CPU if no GPU is available.
//
// Performance context:
//
//	CPU sequential: ~334ms for 10k txs (87% of block processing)
//	CPU parallel (10 cores): ~48ms for 10k txs
//	GPU target: <5ms for 10k txs
func PreRecoverSenders(signer types.Signer, txs types.Transactions) (*SenderCache, *GPUEcrecoverStats) {
	n := len(txs)
	if n == 0 {
		return NewSenderCache(0), &GPUEcrecoverStats{}
	}

	ctx := gpu.DefaultContext
	backend := "CPU"
	if ctx != nil {
		backend = ctx.GetBackend().String()
	}

	stats := &GPUEcrecoverStats{
		NumTxs:  n,
		Backend: backend,
	}

	start := time.Now()

	// Build GPU signature batch from transactions.
	gpuSigs := make([]gpu.Signature, n)
	for i, tx := range txs {
		hash := signer.Hash(tx)
		v, r, s := tx.RawSignatureValues()
		if r == nil || s == nil {
			continue
		}

		// Pack r as 32-byte big-endian (zero-padded left)
		rBytes := r.Bytes()
		if len(rBytes) <= 32 {
			copy(gpuSigs[i].R[32-len(rBytes):], rBytes)
		}

		// Pack s as 32-byte big-endian
		sBytes := s.Bytes()
		if len(sBytes) <= 32 {
			copy(gpuSigs[i].S[32-len(sBytes):], sBytes)
		}

		// Normalize v: Ethereum uses 27/28 (legacy) or 0/1 (EIP-2930/1559)
		vVal := v.Uint64()
		switch {
		case vVal == 0 || vVal == 1:
			gpuSigs[i].V = uint8(vVal)
		case vVal == 27 || vVal == 28:
			gpuSigs[i].V = uint8(vVal - 27)
		default:
			// EIP-155: v = chainId*2 + 35 + recId
			gpuSigs[i].V = uint8((vVal - 35) & 1)
		}

		copy(gpuSigs[i].MsgHash[:], hash[:])
	}

	// Dispatch batch ecrecover (GPU or CPU parallel via C library)
	results, err := gpu.BatchEcrecover(gpuSigs)
	if err != nil {
		log.Warn("GPU batch ecrecover failed, using types.Sender fallback",
			"err", err, "txs", n)
		// Fall back to types.Sender which does per-tx ecrecover + caching
		cache := NewSenderCache(n)
		for _, tx := range txs {
			sender, sErr := types.Sender(signer, tx)
			if sErr == nil {
				cache.Set(tx.Hash(), sender)
			}
		}
		stats.Duration = time.Since(start)
		return cache, stats
	}

	// Build sender cache from GPU results
	cache := NewSenderCache(n)
	recovered := 0
	for i, res := range results {
		if res.Valid {
			addr := common.BytesToAddress(res.Address[:])
			cache.Set(txs[i].Hash(), addr)
			recovered++
		}
	}

	stats.Duration = time.Since(start)
	if stats.Duration > 0 {
		stats.Throughput = float64(n) / stats.Duration.Seconds()
	}

	log.Debug("GPU batch ecrecover complete",
		"txs", n,
		"recovered", recovered,
		"backend", stats.Backend,
		"elapsed", stats.Duration,
		"sigs/sec", int(stats.Throughput),
	)

	return cache, stats
}

// WithGPUEcrecover returns an EngineOption that enables GPU-accelerated
// sender recovery before block execution. Sender recovery itself dispatches
// to the luxgpu cgo bridge (which calls the C++ Metal/CUDA kernel); the
// trie-node hasher stays on CPU because there is no Go-native GPU keccak
// entry point.
func WithGPUEcrecover() EngineOption {
	return func(e *Engine) {
		e.UseGPU = true
		// e.hasher left as the default CPU hasher.
	}
}
