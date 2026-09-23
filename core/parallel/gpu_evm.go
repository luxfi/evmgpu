// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu && cgo && darwin && arm64

package parallel

// Linkage goes through the lux-cevm pkg-config bundle (libevm + libevm-gpu).
// The C header "go_bridge.h" ships under
// $LUXCPP_PREFIX/include/cevm/lib/evm/gpu/, which is in the .pc Cflags.
//
// Build + install once with:
//
//   cmake -S ~/work/luxcpp/cevm -B ~/work/luxcpp/cevm/build \
//         -DCMAKE_INSTALL_PREFIX=$HOME/work/luxcpp/install
//   cmake --build ~/work/luxcpp/cevm/build
//   cmake --install ~/work/luxcpp/cevm/build
//
// Then `export PKG_CONFIG_PATH=$HOME/work/luxcpp/install/lib/pkgconfig`.

/*
#cgo pkg-config: lux-cevm
#cgo darwin LDFLAGS: -framework Metal -framework Foundation -lstdc++

#include <stdlib.h>
#include "go_bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/luxfi/geth/log"
)

// abiVersion is the go_bridge.h ABI executeOnDevice is written to: 7, in which
// ok says whether the result is the block's. A version 6 library answers ok=1
// for results that are not.
const abiVersion uint32 = 7

// The header this builds against names the same ABI, or this file does not
// compile: each line fails when EVM_GPU_ABI_VERSION is smaller (the first) or
// larger (the second) than abiVersion.
var (
	_ [abiVersion - C.EVM_GPU_ABI_VERSION]struct{}
	_ [C.EVM_GPU_ABI_VERSION - abiVersion]struct{}
)

// The C structs executeOnDevice fills and reads, at the sizes go_bridge.h
// (ABI 7) gives them on LP64. A header that adds, drops or widens a field
// changes a size, and then this file does not compile until it has been read
// against the new header: a field it never sets would otherwise cross as
// zero. Each pair fails when the size is larger (the first) or smaller (the
// second).
var (
	_ [unsafe.Sizeof(C.CGpuTx{}) - 120]struct{}
	_ [120 - unsafe.Sizeof(C.CGpuTx{})]struct{}
	_ [unsafe.Sizeof(C.CGpuStateAccount{}) - 136]struct{}
	_ [136 - unsafe.Sizeof(C.CGpuStateAccount{})]struct{}
	_ [unsafe.Sizeof(C.CGpuBlockResult{}) - 88]struct{}
	_ [88 - unsafe.Sizeof(C.CGpuBlockResult{})]struct{}
	_ [unsafe.Sizeof(C.CBlockContext{}) - 392]struct{}
	_ [392 - unsafe.Sizeof(C.CBlockContext{})]struct{}
)

// libraryABI is the ABI the loaded library speaks, read once in init: the one
// it reports (gpu_abi_version), or 0 where its CGpuTx is not this file's
// (gpu_abi_tx_size), as it is not in a version 7 library built before the tx
// carried its fee cap and tip. The library a binary loads at run time need not
// be the one whose header it was built against.
var libraryABI uint32

func init() {
	libraryABI = uint32(C.gpu_abi_version())
	if uint32(C.gpu_abi_tx_size()) != uint32(unsafe.Sizeof(C.CGpuTx{})) {
		libraryABI = 0
	}
}

// NewGPUEVMDispatcher creates a dispatcher that routes eligible transactions
// to the C++ GPU kernel. Auto-detects the best available backend.
//
// A loaded library of another ABI lays the structs out and means ok
// differently, so it is sent nothing: the dispatcher has no device call, is
// not Available, and declines every batch, and every block runs on the CPU.
func NewGPUEVMDispatcher() *GPUEVMDispatcher {
	backend := uint8(C.gpu_auto_detect_backend())
	if libraryABI != abiVersion {
		log.Warn("GPU EVM dispatcher disabled: the loaded library speaks another ABI",
			"library", libraryABI, "reads", abiVersion)
		return &GPUEVMDispatcher{backend: backend}
	}
	log.Info("GPU EVM dispatcher initialized",
		"backend", gpuBackendName(backend),
	)
	return &GPUEVMDispatcher{backend: backend, execute: executeOnDevice}
}

// executeOnDevice runs batch through go_bridge.h's gpu_execute_block, under
// Cancun, the one revision the kernels implement.
//
// ok=0 names no gas or status the caller may use (every status reads
// EVM_GPU_TX_ERROR, and the arrays may be NULL), so it comes back as
// ErrGPUDeclined and nothing in it is read; so does a result of another ABI,
// whose ok means something else. A loaded library of another ABI is sent
// nothing.
func executeOnDevice(backend uint8, b *gpuBatch) ([]GPUEVMResult, error) {
	n := len(b.txs)
	if n == 0 {
		return nil, nil
	}
	if libraryABI != abiVersion {
		return nil, ErrGPUDeclined
	}

	// No Go pointer is stored in any of these: no calldata or code crosses,
	// so there is nothing to pin.
	cTxs := make([]C.CGpuTx, n)
	for i := range b.txs {
		t := &b.txs[i]
		cTxs[i].from = *(*[20]C.uint8_t)(unsafe.Pointer(&t.From[0]))
		cTxs[i].to = *(*[20]C.uint8_t)(unsafe.Pointer(&t.To[0]))
		cTxs[i].has_to = 1
		cTxs[i].gas_limit = C.uint64_t(t.GasLimit)
		cTxs[i].value = C.uint64_t(t.Value)
		cTxs[i].nonce = C.uint64_t(t.Nonce)
		cTxs[i].max_fee_per_gas = C.uint64_t(t.GasFeeCap)
		cTxs[i].max_priority_fee_per_gas = C.uint64_t(t.GasTipCap)
	}

	var cctx C.CBlockContext
	cctx.timestamp = C.uint64_t(b.ctx.Timestamp)
	cctx.number = C.uint64_t(b.ctx.Number)
	cctx.gas_limit = C.uint64_t(b.ctx.GasLimit)
	cctx.chain_id = C.uint64_t(b.ctx.ChainID)
	cctx.base_fee = C.uint64_t(b.ctx.BaseFee)
	cctx.coinbase = *(*[20]C.uint8_t)(unsafe.Pointer(&b.ctx.Coinbase[0]))
	cctx.prevrandao = *(*[32]C.uint8_t)(unsafe.Pointer(&b.ctx.Prevrandao[0]))

	cAccts := make([]C.CGpuStateAccount, len(b.accounts))
	for i := range b.accounts {
		a := &b.accounts[i]
		cAccts[i].address = *(*[20]C.uint8_t)(unsafe.Pointer(&a.Address[0]))
		cAccts[i].nonce = C.uint64_t(a.Nonce)
		for j := 0; j < 4; j++ {
			cAccts[i].balance[j] = C.uint64_t(a.Balance[j])
		}
		cAccts[i].code_hash = *(*[32]C.uint8_t)(unsafe.Pointer(&a.CodeHash[0]))
		cAccts[i].storage_root = *(*[32]C.uint8_t)(unsafe.Pointer(&a.StorageRoot[0]))
	}
	var cAcctsPtr *C.CGpuStateAccount
	if len(cAccts) > 0 {
		cAcctsPtr = &cAccts[0]
	}

	result := C.gpu_execute_block(
		&cTxs[0],
		C.uint32_t(n),
		C.uint8_t(backend),
		0, // num_threads: hardware concurrency
		C.uint8_t(C.EVM_GPU_REV_CANCUN),
		&cctx,
		cAcctsPtr,
		C.uint32_t(len(cAccts)),
		nil, // code_blob: no row has code
		0,
	)
	defer C.gpu_free_result(&result)
	runtime.KeepAlive(cTxs)
	runtime.KeepAlive(cAccts)

	if uint32(result.abi_version) != abiVersion || result.ok == 0 {
		return nil, ErrGPUDeclined
	}
	if int(result.num_txs) != n || result.gas_used == nil || result.status == nil {
		return nil, fmt.Errorf("gpu evm: result for %d txs is not the batch's %d", uint32(result.num_txs), n)
	}
	gas := unsafe.Slice((*C.uint64_t)(unsafe.Pointer(result.gas_used)), n)
	status := unsafe.Slice((*C.uint8_t)(unsafe.Pointer(result.status)), n)
	out := make([]GPUEVMResult, n)
	for i := range out {
		out[i] = GPUEVMResult{
			GasUsed: uint64(gas[i]),
			Success: status[i] == C.EVM_GPU_TX_OK || status[i] == C.EVM_GPU_TX_RETURN,
		}
	}
	return out, nil
}

// WithGPUOpcodes returns an EngineOption that enables GPU EVM opcode dispatch
// for eligible transactions. Requires the gpu build tag and luxcpp/evm.
//
// Trie-node hashing stays on the CPU hasher: GPU keccak dispatch belongs in
// luxcpp (Metal/CUDA kernels) and is not implemented in Go. Once
// luxcpp/gpu exposes a batch keccak entry point, the dispatcher can be
// wired through the existing luxgpu cgo bridge in lux/gpu.
func WithGPUOpcodes() EngineOption {
	return func(e *Engine) {
		e.gpuEVM = NewGPUEVMDispatcher()
		e.UseGPU = true
		// e.hasher left as the default CPU hasher.
	}
}
