// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build gpu && cgo && darwin && arm64

package parallel

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../luxcpp/cevm/include -I${SRCDIR}/../../../../luxcpp/cevm/lib/evm/gpu
#cgo LDFLAGS: -L${SRCDIR}/../../../../luxcpp/cevm/build/lib/evm -levm-gpu -L${SRCDIR}/../../../../luxcpp/cevm/build/lib -levm -lstdc++ -framework Metal -framework Foundation

#include <stdlib.h>
#include "go_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/log"
)

// GPUEVMDispatcher dispatches GPU-eligible transactions to the C++
// Metal/CUDA EVM kernel via CGo. Transactions that contain CALL, CREATE,
// or DELEGATECALL opcodes are not eligible and fall back to Go EVM.
type GPUEVMDispatcher struct {
	backend C.uint8_t
}

// NewGPUEVMDispatcher creates a dispatcher that routes eligible transactions
// to the C++ GPU kernel. Auto-detects the best available backend.
func NewGPUEVMDispatcher() *GPUEVMDispatcher {
	backend := C.gpu_auto_detect_backend()
	log.Info("GPU EVM dispatcher initialized",
		"backend", gpuBackendName(uint8(backend)),
	)
	return &GPUEVMDispatcher{
		backend: backend,
	}
}

// Available returns true if a GPU backend was detected.
func (d *GPUEVMDispatcher) Available() bool {
	return uint8(d.backend) >= 2 // Metal=2, CUDA=3
}

// Backend returns the name of the active backend.
func (d *GPUEVMDispatcher) Backend() string {
	return gpuBackendName(uint8(d.backend))
}

// ExecuteBlock dispatches a batch of GPU-eligible transactions to the C++
// Metal/CUDA kernel. Returns per-transaction gas used.
//
// Caller must ensure all txs passed here are GPU-eligible (IsGPUEligible).
// The signer is needed to recover sender addresses.
func (d *GPUEVMDispatcher) ExecuteBlock(
	signer types.Signer,
	txs []*types.Transaction,
	senders []common.Address,
) ([]GPUEVMResult, error) {
	n := len(txs)
	if n == 0 {
		return nil, nil
	}

	// Pack transactions into C structs.
	cTxs := make([]C.CGpuTx, n)
	for i, tx := range txs {
		// From (sender)
		copy(cTxs[i].from[:], senders[i][:])

		// To
		if tx.To() != nil {
			copy(cTxs[i].to[:], tx.To()[:])
			cTxs[i].has_to = 1
		}

		// Data -- for GPU-eligible txs this is empty, but handle it.
		if len(tx.Data()) > 0 {
			cTxs[i].data = (*C.uint8_t)(C.CBytes(tx.Data()))
			cTxs[i].data_len = C.uint32_t(len(tx.Data()))
		}

		cTxs[i].gas_limit = C.uint64_t(tx.Gas())
		cTxs[i].value = C.uint64_t(tx.Value().Uint64())
		cTxs[i].nonce = C.uint64_t(tx.Nonce())
		if tx.GasPrice() != nil {
			cTxs[i].gas_price = C.uint64_t(tx.GasPrice().Uint64())
		}
	}

	// Call C++ via CGo.
	result := C.gpu_execute_block(
		(*C.CGpuTx)(unsafe.Pointer(&cTxs[0])),
		C.uint32_t(n),
		d.backend,
	)

	// Free any calldata we allocated.
	for i := range cTxs {
		if cTxs[i].data != nil {
			C.free(unsafe.Pointer(cTxs[i].data))
		}
	}

	if result.ok == 0 {
		C.gpu_free_result(&result)
		return nil, fmt.Errorf("gpu evm execute_block failed")
	}

	// Unpack results.
	gasUsedSlice := unsafe.Slice((*C.uint64_t)(result.gas_used), n)
	results := make([]GPUEVMResult, n)
	for i := 0; i < n; i++ {
		results[i] = GPUEVMResult{
			GasUsed: uint64(gasUsedSlice[i]),
			Success: true,
		}
	}

	C.gpu_free_result(&result)
	return results, nil
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

// WithGPUOpcodes returns an EngineOption that enables GPU EVM opcode dispatch
// for eligible transactions. Requires the gpu build tag and luxcpp/evm.
func WithGPUOpcodes() EngineOption {
	return func(e *Engine) {
		e.gpuEVM = NewGPUEVMDispatcher()
		e.UseGPU = true
		e.hasher = NewGPUHasher()
	}
}
