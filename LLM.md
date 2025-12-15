# GPUEVM — GPU-Accelerated Parallel EVM

**Module**: `github.com/luxfi/evmgpu`
**Fork of**: `github.com/luxfi/evm`
**Status**: Active development

## Overview

GPUEVM is a parallel EVM execution engine designed to run entirely in GPU memory. It implements modified Block-STM (optimistic concurrency control) for parallel transaction execution, with all core data structures designed for GPU kernel offload.

The key insight: EVM state (accounts, storage slots) fits in GPU VRAM. When the multi-version memory, scheduler, and eventually the EVM opcode interpreter all run as GPU kernels, we achieve the first true GPUEVM — executing EVM transactions at GPU throughput.

## Architecture

```
Block of N transactions
        |
        v
+-------------------+
| Scheduler         |  <- GPU atomics (execution_idx, validation_idx)
| (collaborative)   |
+--------+----------+
         |
    +---------+---------+--- ... ---+
    |         |         |           |
    v         v         v           v
 Worker0   Worker1   Worker2    WorkerN
    |         |         |           |
    v         v         v           v
+-------------------------------------------+
| MvMemory (Multi-Version Data Structure)   |  <- GPU VRAM (zapdb GPU cache)
| location_hash -> [tx0, tx1, ..., txN]     |
+-------------------------------------------+
         |
         v
+-------------------------------------------+
| ZapDB GPU Cache (Hot State in VRAM)       |  <- Pre-block state
| Robin Hood hash table + blob arena        |
+-------------------------------------------+
         |
         v
+-------------------------------------------+
| ZapDB Disk (Cold State)                   |  <- BadgerDB/LSM-tree
+-------------------------------------------+
```

## Block-STM Algorithm

1. **Initialize**: Create MvMemory (version chains per location), Scheduler (all txs ReadyToExecute), pre-allocate coinbase with ESTIMATE markers
2. **Parallel Execute**: Workers grab execution tasks, run EVM with intercepted state access through MvMemory, record read/write sets
3. **Validate**: Workers validate completed txs by re-checking read origins. If stale -> abort, replace writes with ESTIMATE markers, re-execute
4. **Terminate**: When all txs validated
5. **Post-process**: Evaluate lazy addresses (beneficiary, raw transfers) sequentially

### Key EVM Optimization: Lazy Beneficiary

Every EVM transaction pays gas to `coinbase`. Without lazy evaluation, ALL transactions conflict on coinbase balance -> zero parallelism. Solution:

- Pre-allocate coinbase in MvMemory with ESTIMATE markers for all tx indices
- During execution, record gas payment as `LazyCredit` delta (not absolute value)
- After all txs validate, walk the delta chain sequentially to compute final balance

This single optimization enables 5-22x speedup on real EVM blocks.

## Core Files

| File | Purpose |
|------|---------|
| `core/parallel/types.go` | MemoryLocation, MemoryValue, ReadSet, WriteSet, Task, TxStatus |
| `core/parallel/mv_memory.go` | Multi-version data structure (version chains, validation) |
| `core/parallel/scheduler.go` | Collaborative task scheduler (Block-STM state machine) |
| `core/parallel/engine.go` | Main engine, worker loop, ParallelStateDB |
| `core/state_processor.go` | Block processing (sequential, to be wired to parallel) |

## GPU Kernel Roadmap

### Phase 1: CPU Parallel (Current)
- Go goroutines for workers
- MvMemory in Go maps
- All data in CPU memory
- Speedup: ~2-6x over sequential

### Phase 2: GPU State Backend
- MvMemory backed by zapdb GPU cache
- Hot EVM state (accounts, storage) in GPU VRAM
- CPU reads/writes via unified memory (Apple Silicon zero-copy)
- Speedup: ~5-10x (eliminates disk I/O for hot state)

### Phase 3: GPU Scheduler + Validation
- `execution_idx`, `validation_idx` as GPU atomics
- Validation kernel: parallel scan of read sets against MvMemory
- ESTIMATE conversion kernel: parallel mark of aborted writes
- Eviction scan in GPU kernel
- Speedup: ~10-20x

### Phase 4: GPU EVM Interpreter (GPUEVM)
- EVM opcode interpreter as GPU compute kernel
- Each transaction = one GPU thread group
- Stack, memory, calldata in GPU shared memory
- State reads/writes via MvMemory in GPU global memory
- PRECOMPILE dispatch to CPU for complex crypto (pairing, etc.)
- Speedup: ~50-100x (GPU throughput for compute-heavy contracts)

### Phase 5: Multi-VM GPU Runtime
- **EVM**: Full GPU interpreter (Phase 4)
- **SVM** (Solana): BPF/SBF interpreter on GPU (parallel by design)
- **TON VM**: TVM interpreter on GPU (actor model maps to GPU threads)
- Shared MvMemory across all VMs -> cross-VM atomic composability
- Native DEX venues: `venue_v4.go` (Uniswap V4 style), `venue_zap` (ZAP binary protocol)

## DEX Venue Architecture

### venue_v4 (Uniswap V4 Compatible)
- EVM precompile implementing concentrated liquidity AMM
- Hook system for custom pool logic
- Compatible with Uniswap V4 interfaces for easy migration
- State stored in EVM storage slots (parallel-friendly via MvMemory)

### venue_zap (Native ZAP Binary Protocol)
- Fastest path: native binary protocol, not EVM precompile
- Direct MvMemory access (no EVM overhead)
- GPU kernel for order matching (from luxcpp/dex)
- Sub-microsecond matching latency
- ZAP wire format for minimal serialization

## GPU Data Structure Design

### MvMemory on GPU
```
Slot Table: [num_locations] x [block_size] entries
Each entry: { tx_idx: u32, incarnation: u32, is_estimate: bool, value: MemoryValue }

On unified memory (Metal/Apple Silicon):
  - CPU and GPU share same physical memory
  - buffer_get_host_ptr() returns direct pointer
  - Go code reads MvMemory at GPU memory speed
  - GPU kernels access same memory for batch validation

On discrete GPU (CUDA):
  - GPU kernels read/write directly
  - CPU communicates via staging buffers
  - Batch operations amortize DMA cost
```

### Scheduler on GPU
```
execution_idx:   GPU atomic u32
validation_idx:  GPU atomic u32
num_validated:   GPU atomic u32
tx_states[N]:    GPU atomic u32 (packed: status + incarnation)
dependents[N]:   GPU u32 arrays (fixed-size, max deps per tx)
```

## Integration with Lux Consensus

GPUEVM is the C-Chain upgrade path -- same chain ID (96369), same VM ID (`evm`). The parallel engine and GPU cache are internal optimizations transparent to consensus. Non-GPU validators continue to work identically.

Integration points:

1. **Block building** (`miner/worker.go`): Replace sequential `commitTransactions()` with `Engine.ExecuteBlock()`
2. **Block verification** (`core/state_processor.go`): Replace sequential `Process()` with `Engine.ExecuteBlock()`
3. **State commitment**: After parallel execution, commit final state to trie in original tx order
4. **Consensus engine**: No changes needed -- parallel execution is transparent to consensus
5. **Rollout**: Opt-in via config (`gpuMemoryBudget > 0`), validators upgrade at their own pace

## Performance Targets

| Workload | Sequential | Phase 1 | Phase 2 | Phase 4 |
|----------|-----------|---------|---------|---------|
| Raw transfers (1B gas) | 160ms | 55ms | 30ms | 5ms |
| ERC-20 transfers | 250ms | 60ms | 35ms | 8ms |
| Uniswap swaps | 400ms | 20ms | 12ms | 4ms |
| Mixed mainnet block | 50ms | 25ms | 15ms | 3ms |

## Dependencies

- `github.com/luxfi/geth` -- EVM types, VM, ethdb
- `github.com/luxfi/database` -- ZapDB with GPU cache
- `github.com/luxfi/consensus` -- Lux consensus integration
- `luxcpp/gpu` -- GPU backend (Metal/CUDA/WebGPU)
- `luxcpp/zapdb` -- C GPU hash table cache
- `luxcpp/dex` -- DEX matching engine (for venue integration)

## Building

```bash
cd ~/work/lux/gpuevm
GOWORK=off go build ./...
GOWORK=off go test ./core/parallel/...
```

## risechain/pevm Reference

Studied at `~/work/rise/pevm/`. Key learnings:
- Block-STM with lazy beneficiary is the right algorithm
- DashMap + BTreeMap works but isn't GPU-friendly (pointer chasing)
- Our fixed-size array approach (indexed by TxIdx) is better for GPU
- Their 2-22x speedup on CPU validates the approach
- No existing GPU path -- we are first movers on GPUEVM

## Multi-VM Vision (Phase 5)

```
+------------------------------------------+
| Lux GPUEVM Runtime                       |
|                                          |
| +--------+ +--------+ +--------+        |
| | EVM    | | SVM    | | TVM    |        |
| | (GPU)  | | (GPU)  | | (GPU)  |        |
| +---+----+ +---+----+ +---+----+        |
|     |          |          |              |
|     v          v          v              |
| +--------------------------------------+|
| | Shared MvMemory (GPU VRAM)           ||
| | Cross-VM atomic composability        ||
| +--------------------------------------+|
|     |                                    |
|     v                                    |
| +--------------------------------------+|
| | ZapDB GPU Cache                      ||
| | Hot state for all VMs                ||
| +--------------------------------------+|
|     |                                    |
|     v                                    |
| +--------------------------------------+|
| | Lux Consensus                        ||
| | Block ordering + finality            ||
| +--------------------------------------+|
+------------------------------------------+
```

EVM+SOL+TON executing in the same GPU memory space, with shared state access through MvMemory. Cross-VM calls (e.g., Solana program calling EVM contract) become MvMemory reads -- no bridge, no IBC, just shared memory.
