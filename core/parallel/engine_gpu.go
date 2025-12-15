// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu

package parallel

// WithGPU enables GPU-accelerated batch hashing for state root computation.
// When the GPU is available, trie node Keccak-256 hashes are dispatched to
// Metal/CUDA. Falls back to parallel CPU hashing if no GPU is detected.
func WithGPU() EngineOption {
	return func(e *Engine) {
		e.UseGPU = true
		e.hasher = NewGPUHasher()
	}
}
