# 12 — Build Plan: Step 2.11, `MetalWinograd`

Written by Opus for a Sonnet implementer (model plan: "S plans → M builds").
Do the sub-steps in order. Each ends with its gate green and a commit, before
the next starts. Context: ADR-009 in [09-decisions.md](09-decisions.md), and
the step 2.9 numbers in [BENCHMARKS.md](BENCHMARKS.md).

**Goal:** a second Metal network implementation that uses Winograd
F(4×4, 3×3) like the OpenCL backend, behind the existing `MetalScheduler`,
slots and autotune.

**Targets** (random nets, same protocol as step 2.9):
- Must: ≥ 1.0× OpenCL on 15b×192 and 40b×256 (OpenCL: 455 and 107 n/s on
  2026-10-04).
- Spec: ≥ 2× on 15b×192 and ≥ 2.5× on 40b×256.

**Out of scope:** Neural Engine (`--ane` stays on MPSGraph), new CLI beyond
the one flag below, and training.

## 0. Facts to respect

Verified against the code on 2026-10-04 (`d71a763`).

| # | Fact | Where |
|---|------|-------|
| F1 | One tower layer is `y = max(0, scale[k] * (conv(x)[k] - mean[k]) [+ skip])`. Means already include the conv bias; "stddevs" are already the scale `1/sqrt(var+eps)`. Same as steps 2.3/2.5. | `Network.cpp` bias folding, `process_bn_var` |
| F2 | The pipe receives **raw** OIHW 3×3 weights (since 2.3a). `Network::winograd_transform_f(w, outputs, inputs)` (public static) turns them into U, laid out `U[e][c][k]` (e = 0..35 Winograd element, C×K per element). | `Network.h`, `CPUPipe::push_weights` |
| F3 | OpenCL template layouts, with B the batch, P = 25 tiles per board, `N = B·P` columns and padded sizes `Cpad, Kpad, Npad`: **V** is `[36][Cpad][Npad]`, `V[e][c][n]` (`in_transform`: `offset = ch*Ppad + block`). **M** is `[36][Npad][Kpad]`, `M[e][n][k]` (`out_transform_fused_bn`: `offset = block*Kpad + k`, element stride `Kpad*Ppad`). So per element e: `M_e (N×K) = V_eᵀ (N×C) · U_e (C×K)`. | `src/kernels/convolve3.opencl`, `OpenCL.cpp::convolve3` |
| F4 | Tile n = `batch*P + block_y*WTILES + block_x`; the input tile starts at `(4·block_y − 1, 4·block_x − 1)` with zero padding outside the board; output tile rows/cols past 19 are dropped. | `in_transform`, `out_transform_fused_bn` |
| F5 | BN is applied in the **output transform** (not folded into U): `r = (Aᵀ·m·A − mean) * scale`, then `+ residual`, then ReLU. The transform constants (`Bt`, `At`, `SQ2`) are in `convolve3.opencl` and match `CPUPipe.cpp`'s `multiply_bt` / `multiply_at`. | `convolve3.opencl`, `CPUPipe.cpp` |
| F6 | Heads: raw 1×1 convolutions only, 2 and 1 outputs, **no** BN/ReLU (head BN/FC/softmax run on the CPU). Outputs fp32. | spec 11 F7, `MetalNetwork.mm` |
| F7 | `MetalScheduler` workers each own their slots (ADR-007). `MetalNetwork::run(slot)` must be thread-safe across *different* slots. Compute pipelines (`MTLComputePipelineState`) are immutable and thread-safe, so the Winograd path needs **no encode mutex**: each `run()` builds its own command buffer. | ADR-007, `MetalScheduler.cpp` |
| F8 | `BatchQueue::pickup()` returns 1 or `max_batch` entries; only those two batch sizes need buffers. | ADR-007 |
| F9 | Do not build on the exFAT copy (`/Volumes/RED/...`): its `._*.cpp` sidecars get globbed into the build. Build and benchmark in an internal-disk checkout. | 2026-10-04 session |

## 1. Shape of the change

Keep the public API (`MetalSlot`, `MetalNetwork::run/benchmark/make_slot/forward`)
and add an engine choice:

```cpp
// MetalNetwork.h
enum class MetalEngine { Graph, Winograd };   // Graph = MPSGraph (today)
MetalNetwork(const MetalContext&, int channels, int residual_blocks,
             const ForwardPipe::ForwardPipeWeights&,
             const std::vector<int>& batch_sizes,
             MetalPrecision precision = MetalPrecision::Single,
             bool ane = false,
             MetalEngine engine = MetalEngine::Graph);
```

- `MetalNetwork::Impl` dispatches to `GraphImpl` (the current code, moved, not
  rewritten) or `WinogradImpl` (new, in `src/metal/MetalWinograd.mm`). `ane`
  with `Winograd` is an error.
- `MetalSlot::Impl` gains per-slot scratch for Winograd: V, M, and two
  activation buffers of `B × C × 361` (ping-pong) plus a residual buffer. A
  slot is owned by one worker, so this scratch needs no locking (F7).
- MSL source lives in `src/metal/kernels/winograd.metal`, embedded as a raw
  string (like `convolve3.opencl`'s `R"(...)"` trick) and compiled at runtime
  through `MetalContext` (this already works, as the 2.2 self-test proves).
  Use `#define` constants for `C, K, B` per pipeline only if it measurably
  helps; start with runtime arguments.

## 2. Sub-steps

### 2.11a — Transforms in MSL, GEMM via `MPSMatrixMultiplication`, fp32

The lowest-risk route to a working Winograd network: Apple's tuned GEMM,
custom code only for the transforms.

1. **Weights:** in the constructor, for each tower layer compute
   `U = Network::winograd_transform_f(...)` (F2) and upload it to one
   `MTLBuffer` per layer (shared storage is fine to start; try private plus a
   blit later and measure). Means and scales go up as per-layer float
   buffers. Head weights stay as in the graph path.
2. **Kernel `in_transform`** (port of OpenCL `in_transform` + `__in_transform_eq`):
   grid `(N, C)`, writes `V[e][c][n]` with the `Bt` transform (F4/F5).
   Zero-pad rows and columns up to Cpad and Npad only if the GEMM needs it (MPS
   does not).
3. **GEMM:** one `MPSMatrixMultiplication` per layer, batched over the 36
   elements (`matrixDescriptorWithRows:columns:matrices:36:rowBytes:matrixBytes:dataType:`,
   the current non-deprecated variant, with `batchStart = 0, batchSize = 36`
   on the kernel): left = V
   (`transposeLeft = YES`, C×N per element), right = U (C×K), result = M
   (N×K per element). Encode it into the same command buffer.
4. **Kernel `out_transform_bn`** (port of `out_transform_fused_bn`): grid
   `(K, N)`, reads `M[e][n][k]`, applies `At`, then BN, then optional
   residual, then ReLU (F1/F5), and writes `Y[b][k][361]` with the edge
   clipping (F4). Start without the threadgroup-memory staging the OpenCL
   kernel uses; add it in 2.11d if the profile asks.
5. **Heads:** kernel `conv1x1` that computes `out[b][o][p] = Σ_c w[o][c]·x[b][c][p]`
   for o ∈ {2 policy, 1 value}, writing the slot's fp32 pol/val buffers
   directly.
6. **Run:** one command buffer per `run()`: for each layer, in_transform →
   GEMM → out_transform (the residual is the block input buffer), then the
   heads. Commit, then wait on the completion handler as the graph path does.

**Gate 2.11a:**
- New unit test `MetalWinogradTest.MatchesCpu`: C=32, 3 blocks, non-trivial
  BN (`make_test_weights`), fp32, batch 1 and 4. Max abs diff vs `CPUPipe` ≤
  1e-4 relative to the output scale; batch 4 = batch 1 (≤ 1e-6).
- Mutation check: drop the BN mean subtraction in `out_transform_bn` → the
  test must fail.
- `scripts/parity/full_gate.sh` with Winograd forced (see 2.11c for the flag;
  until then, a temporary env var): fp32 G2 ≤ 1e-4 on all three shapes.
- Benchmark vs OpenCL and MPSGraph (2.9 protocol), recorded even if below
  target.

### 2.11b — fp16 storage, fp32 math

1. U, V and M in `half`; transforms compute in `float` registers (load half,
   compute float, store half), like OpenCL's `net_t`/`real` split.
   Activations between layers in half; heads read half and write fp32.
2. GEMM: `MPSMatrixMultiplication` with `MPSDataTypeFloat16` inputs. Check
   whether MPS accumulates in fp32 (compare against an fp32 reference on a
   C=256 layer). If it accumulates in fp16 and G2 fails, this is where 2.11c's
   custom GEMM becomes mandatory.

**Gate 2.11b:** unit test fp16 vs CPU ≤ 2% of the output scale (same as the
graph fp16 test); `full_gate.sh` fp16 at N6 (policy 1e-2, value 5e-3) on all
three shapes; concurrency stress test (`run_concurrency_test`) with Winograd
in both precisions; benchmark.

### 2.11c — Integration: autotune, flag, scheduler

1. `MetalScheduler` takes `MetalEngine` and passes it through. Its describe()
   string names the engine ("MPSGraph" or "Winograd").
2. CLI: `--metal-kernels auto|mpsgraph|winograd` (default auto) in the shared
   GPU option block (`Leela.cpp`). `--ane` implies `mpsgraph`.
3. Autotune (`MetalTuning`): add `engine` to `Measurement`, measure both
   engines × both precisions × candidate batches, and choose the fastest
   engine/precision pair (keeping the existing 5%/smallest-batch rules within
   an engine). Bump the cache schema to v2 (`SCHEMA` string). Old v1 rows are
   simply not loaded, which costs one re-measurement. The fp16 accuracy check
   in `Network::init_metal` must run on the chosen engine.
4. `full_gate.sh`: a `--test-flags` option, so nightly can run G2 for both
   engines; add the Winograd rows to `nightly.yml`.

**Gate 2.11c:** all unit tests (no regressions in the graph path); TSan on the
Metal tests with Winograd (expect 0: no shared mutable state, F7); ASan/UBSan;
CPU, OpenCL, Metal and Metal+OpenCL builds; autotune picks Winograd on 15b×192
if it is faster; CI and nightly green.

### 2.11d — Custom `simdgroup_matrix` GEMM (only if needed)

Do this if, after 2.11c, Winograd is below the **spec** target, or 2.11b found
fp16 accumulation. Otherwise record "not needed" with the numbers.

1. Kernel `batched_gemm`: per threadgroup, a 32×32 (start) output tile of
   `M_e`, built from 8×8 `simdgroup_float8x8` / `simdgroup_half8x8`
   fragments; accumulate in `simdgroup_float8x8`; stage V and U tiles through
   threadgroup memory; K loop in steps of 8 (or 16). Pad C, K and N to
   multiples of 32 in the buffers (zero-filled) so the kernel needs no bounds
   checks; the transforms already write the padded layout.
2. Tune tile sizes (32×32 vs 64×32, threadgroup 128 vs 256) with a tiny
   timing harness on the 192- and 256-channel shapes. Record the table.
3. Keep it behind the same engine (Winograd) and switch the GEMM by a
   constructor parameter only during development. Ship the faster one.

**Gate 2.11d:** the same as 2.11b, plus a direct GEMM unit test against a
CPU `cblas_sgemm` on odd sizes (C=18, N=25, K=192) to catch padding and edge
bugs.

### 2.11e — Optional fusions (only with a profile)

- Fuse out_transform of layer i with in_transform of layer i+1 (OpenCL's
  `out_transform_fused_bn_in`), saving one activation write and read per
  layer.
- Private-storage weights with a blit, if the profile shows weight bandwidth.

## 2b. Implementation notes (2.11a)

- V and M are stored `[element][tile][channel]` (N×C and N×K row-major per
  element), not OpenCL's `[element][channel][tile]`: the GEMM then needs no
  transposes, and channel-fastest threads give coalesced writes. F3 above
  describes OpenCL's layout, not this one.
- The residual add writes in place: each output pixel is read and written by
  one thread, so the block's output buffer is also its residual input.
- Kernels live in `src/metal/WinogradKernels.h`, the host side in
  `MetalWinograd.{h,mm}`. The engine is chosen by autotune, or with
  `--metal-kernels` (2.11c); the temporary env var is gone.

## 3. Pitfalls

- [ ] U from `winograd_transform_f` is `[e][c][k]` (C×K); OpenCL's
      `zeropad_U` pads it to Cpad×Kpad. Pad the same way if 2.11d needs it.
- [ ] M is `[e][n][k]`, not `[e][k][n]` (the CPU path uses `[e][k][n]`).
      Don't copy CPUPipe's indexing.
- [ ] Input-tile origin is `4·block − 1` (F4). Off by one here passes
      symmetric random tests poorly but fails G2 badly. Test batch > 1 and the
      board edge (tile 24).
- [ ] Residual add before ReLU; the skip is the block input (F1).
- [ ] Heads read the tower output, not a BN'd copy; no head BN (F6).
- [ ] fp16: compute the transforms in float. Winograd F(4×4) amplifies
      rounding (constants up to 5/2 and √2 products).
- [ ] One command buffer per `run()`; no shared mutable state, and no
      encode lock (F7). If you think you need a lock, stop and escalate.
- [ ] `@autoreleasepool` around per-run Objective-C.
- [ ] Build and benchmark on the internal disk (F9).

## 4. Done means

- 2.11a–c gates green (2.11d if triggered), CI and nightly green.
- BENCHMARKS.md: Winograd vs MPSGraph vs OpenCL vs CPU on 15b×192 and
  40b×256, fp32 and fp16, plus the autotune choice.
- Tracking rows (2.11 with sub-steps) in `10-model-plan.md`, ROADMAP, and an
  ADR-009 update with the outcome against the targets.
- Escalate to Opus if: G2 fails twice with a cause you can't explain; fp16
  accumulation in MPS forces 2.11d and the simdgroup GEMM misses OpenCL
  parity; or a lock seems necessary in `run()`.
