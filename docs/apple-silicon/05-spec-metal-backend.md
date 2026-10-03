# 05 — Spec: Metal Inference Backend

**Release:** `as.2`. **Depends on:** 03, plus the 04 CPU reference.
**This is the headline feature.**

## 1. Goals

- A new `ForwardPipe` implementation, `MetalScheduler<T>` (T = `float` or
  `half`), that runs the whole residual tower and both heads on the M4 GPU.
- Zero host↔device copies per evaluation, using unified memory (requirement N3).
- ≥ 2.5× the throughput of the OpenCL-on-macOS baseline for large nets (N1).
- It becomes the default backend on macOS.

## 2. Why not keep OpenCL

OpenCL on macOS has been deprecated since 10.14. On Apple Silicon it is a
translation layer over Metal, capped at OpenCL 1.2. It has no fp16 storage
extensions and no access to `simdgroup_matrix` (the GPU's matrix hardware), and
it adds an extra buffer copy per batch (`enqueueWriteBuffer` / `enqueueReadBuffer`).
See ADR-001.

## 3. Design

### 3.1 Components

```
src/BatchQueue.h            template<class Entry> class BatchQueue
                              – extracted from OpenCLScheduler::batch_worker
                              – adaptive wait (m_waittime), single-eval fallback,
                                drain()/resume(), batch statistics
src/metal/MetalContext.mm   device, command queue, library (runtime-compiled MSL),
                            capability probe (family, simdgroup_matrix, max threads)
src/metal/MetalNetwork.mm   builds the inference graph for fixed batch sizes
                            (Strategy A: MPSGraph; Strategy B: custom kernels)
src/metal/MetalScheduler.mm ForwardPipe impl: owns BatchQueue, slot ring, workers
src/metal/kernels/*.metal   MSL sources embedded as raw string literals
```

`OpenCLScheduler` is refactored to use `BatchQueue`, with identical behavior and
covered by existing tests. This refactor lands first, as its own PR.

### 3.2 Compute strategy

**Strategy A (primary): MPSGraph**
- Build one `MPSGraph` per supported batch size {1, 2, 4, 8, 16, 32, 64}, or one
  graph with a symbolic batch dimension if the measured cost is equal. Compile
  it to an `MPSGraphExecutable` at `push_weights` time.
- Ops: `convolution2D` (NCHW, 3×3, pad 1), with BN folded into the conv weights
  and bias at load time (`w' = w·γ/σ`, `b' = (b-μ)·γ/σ + β`; LZ format has γ=1),
  then `addition` (residual), `reLU`, the 1×1 convs for the heads,
  `matrixMultiplication` (FC), `tanh`.
- Weights become `MPSGraphTensorData` constants backed by private `MTLBuffer`s.
  Graph constants let MPSGraph pre-transform them, including Winograd if it
  chooses.
- fp16: build the graph with `MPSDataTypeFloat16`. MPSGraph accumulates in fp32
  internally for conv.

**Strategy B (fallback / tuning): custom MSL kernels**
- Port `convolve3.opencl` (Winograd in/out transforms) to MSL, and replace the
  CLBlast sgemm with a `simdgroup_matrix<half,8,8>` batched GEMM.
- Port `convolve1.opencl` for the heads.
- Use B only if A misses N1 by more than 15% on the 40b×256 net, or if MPSGraph
  shows latency issues at batch 1. The decision is recorded in BENCHMARKS.md.

### 3.3 Unified-memory I/O (zero-copy)

```
            Slot ring (per worker): K = 3 in-flight batches
 ┌────────────────────── MTLBuffer (StorageModeShared) ───────────────────────┐
 │ batch b0: [B × 18 × 361] T in │ [B × 362] f32 pol │ [B] f32 val │ status   │
 │ batch b1: ...                                                           │
 │ batch b2: ...                                                           │
 └─────────────────────────────────────────────────────────────────────────┘
```

1. `forward(input, out_pol, out_val)` reserves the next index in the current
   open batch (atomic), converts and writes the input planes straight into
   `slot.in + idx*18*361`, and parks on a futex/condition variable.
   - Optimization (phase 2b): add an overload, `forward_into(SlotWriter&)`,
     that `Network::get_output_internal` calls so it gathers planes directly
     into the slot. This removes the `std::vector<float>` intermediate as well.
2. When the batch is full, or the wait expires, the worker encodes a command
   buffer that reads `slot.in` and writes `slot.pol` / `slot.val`, then commits
   it. It doesn't block: the next slot is opened immediately (triple buffering).
3. `addCompletedHandler` (or `MTLSharedEvent` with a listener) wakes the waiting
   search threads. Each thread copies its 362+1 floats out into its own
   `std::vector`. That copy goes to the caller's vector (a cache-resident
   ~1.5 KB copy), not a device transfer.
4. Weights use `StorageModePrivate`, uploaded once per `push_weights` with a
   blit from a temporary shared buffer.

Rules:
- No `didModifyRange` / `synchronizeResource` (those are for discrete GPUs only).
- Allocate the slot ring once at `initialize()`, sized for `max_batch × K`.
- Use `MTLResourceHazardTrackingModeUntracked` on slots. Ordering is enforced by
  the slot state machine (`FREE → FILLING → SUBMITTED → DONE → FREE`).

### 3.4 Concurrency

- Workers per device: 2, which is enough to keep one batch encoding while one
  executes. This is configurable by the existing `--batchsize` and
  `--gpu-threads` mapping.
- Use the existing `BatchQueue` heuristics. Default `cfg_batch_size` on Metal
  is chosen by startup autotune (see 3.6) and is typically 16–32 on M4.

### 3.5 Precision and self-check

- `--precision auto|single|half`. Auto runs 3 batches of each precision and
  picks half if it passes the tolerance and is faster.
- Add a `USE_METAL_SELFCHECK` compile option, mirroring `USE_OPENCL_SELFCHECK`.
  It compares 1 in N evaluations against `CPUPipe` and aborts on mismatch
  beyond the N6 tolerances.

### 3.6 Autotune (lightweight)

On first run per (net shape, device), time batch sizes {8, 16, 32, 64} ×
{fp16, fp32}. Persist the result to
`~/Library/Application Support/leela-zero/metal_tuning` (same idea as
`leelaz_opencl_tuning`). `--tune-only` works for Metal too.

### 3.7 CLI and GTP surface

| Flag | Behavior |
|------|----------|
| `--backend metal` | Force Metal |
| `--gpu N` | Ignored on Metal (single GPU), with a warning |
| `--precision` | As in 3.5 |
| `--batchsize` | Overrides autotune |
| GTP `version` / startup log | `Metal: Apple M4 (10-core GPU), MPSGraph, fp16, batch 32` |

## 4. Out of scope for `as.2`

- Multiple GPUs (none exist on a Mac mini).
- The Core ML / ANE backend (ADR-004, `as.4` experimental).
- Changing NN input encoding or symmetry handling.

## 5. Acceptance criteria

- [ ] Gate G2 (spec 07): Metal fp32 vs CPU reference ≤ 1e-4, and fp16 within
      the N6 tolerance on 3 nets × 100 positions × 8 symmetries.
- [ ] Gate G3: 1,000-move self-play with `USE_METAL_SELFCHECK` shows 0 mismatches.
- [ ] Throughput: ≥ 2.5× the OpenCL baseline on 40b×256, and ≥ 2× on 15b×192.
      Latency at batch 1 is ≤ the OpenCL batch-1 latency.
- [ ] Instruments "Metal System Trace" shows no `blit` encoders per inference
      batch (N3).
- [ ] A 2-hour soak at `-t 16` shows no leaks (Instruments "Leaks") and flat RSS.
- [ ] Linux/Windows builds unaffected (`USE_METAL` defaults OFF off-Apple).

## 6. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MPSGraph graph compile time per batch size is slow at startup | Med | Compile lazily per batch size on first use. Cache executables (`MPSGraphExecutable serialize` on macOS 14+) |
| Completion-handler wake latency hurts batch-1 play | Med | Use `MTLSharedEvent` + spin-then-wait for batch ≤ 2 |
| fp16 overflow in value head on some nets | Low | Run FC layers in fp32 (mixed graph) |
| Objective-C++ in a C++14 codebase | Low | Keep ObjC confined to `src/metal/*.mm`, behind a pure C++ header (pimpl) |
