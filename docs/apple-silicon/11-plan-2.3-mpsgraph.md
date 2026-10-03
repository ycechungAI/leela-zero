# 11 — Build Plan: Step 2.3, MetalNetwork via MPSGraph

Written by Opus for a Sonnet implementer (model plan: "S plans → M builds").
Follow the sub-steps in order; each ends green before the next starts. Spec
context: [05-spec-metal-backend.md](05-spec-metal-backend.md) §3.2.

**Scope:** fp32 MPSGraph network plus a temporary synchronous pipe, so gate G2
can run. Out of scope: async slot ring, triple buffering, batching across
search threads (step 2.4), fp16 (2.5), `--backend` (2.6), autotune (2.7).

## 0. Facts the implementation must respect

Verified against the code on 2026-10-03.

| # | Fact | Where |
|---|------|-------|
| F1 | Backends currently receive the **3×3 tower weights already Winograd-transformed** (`Network::winograd_transform_f`, 6×6 tiles). MPSGraph needs the raw 3×3 kernels. | `Network.cpp` `initialize()`, the loop before "Move biases" |
| F2 | Raw conv weights are OIHW: `w[((k * C + c) * 3 + ky) * 3 + kx]`, K outputs, C inputs. This is cross-correlation (no kernel flip), the same as MPSGraph's `convolution2D`. | file format; `Im2Col.h` |
| F3 | Conv biases are already folded into the BN means (`means[k] -= bias[k]`; biases set to 0). | `Network.cpp`, "Move biases to batchnorm means" |
| F4 | `m_batchnorm_stddevs` already holds the **scale** `1 / sqrt(var + 1e-5)`, not a std dev. Do not redo `process_bn_var`. | `process_bn_var` |
| F5 | Tower layer: `y = max(0, scale[k] * (conv(x)[k] - mean[k]))`. Second conv of each residual block: `y = max(0, scale[k] * (conv(x)[k] - mean[k]) + skip)`, where `skip` is the block's input. Residual add comes **before** the ReLU. | `CPUPipe::forward`, `transform_out_tiles` |
| F6 | Layer list: `m_conv_weights[0]` = input conv (18 → C); then pairs `[1,2], [3,4], …` = residual blocks (C → C). | `CPUPipe::forward` |
| F7 | The `ForwardPipe` output is the **raw 1×1 head convolutions**, no BN, no ReLU: `output_pol` = 2 × 361 (`m_conv_pol_w`, shape [2, C]), `output_val` = 1 × 361 (`m_conv_val_w`, shape [1, C]). Head BN, FC, softmax and tanh run on the CPU in `Network::get_output_internal`. Head conv biases are zero (folded, like F3). | `Network.cpp` ~line 834; `CPUPipe::push_weights` |
| F8 | Input: one evaluation = 18 planes × 361 floats, plane-major (NCHW with N=1). Symmetries are applied on the CPU before `forward()`. | `Network::gather_features` |
| F9 | `forward()` is called concurrently from every search thread. | `UCTSearch` worker threads |

## 1. Sub-step 2.3a — Pass raw weights; each backend transforms its own

Fixes F1 without changing numbers.

1. `Network::initialize`: delete the two `winograd_transform_f` loops. The
   `ForwardPipeWeights` now carry raw OIHW 3×3 weights. Keep the bias folding
   (F3) exactly as is.
2. `CPUPipe::push_weights`: build a private `std::vector<std::vector<float>>
   m_conv_u` by calling `Network::winograd_transform_f(w, outputs, inputs)` for
   each tower layer: layer 0 is `(channels, Network::INPUT_CHANNELS)`, the rest
   `(channels, channels)`. `forward()` uses `m_conv_u[i]` instead of
   `m_weights->m_conv_weights[i]`.
3. `OpenCLScheduler::push_weights`: same transform, before `push_input_convolution`
   / `push_residual`.
4. Update comments that say "Winograd transformed" on the weights struct.
5. Leave `init_net`'s `push_weights(WINOGRAD_ALPHA, ...)` call and its
   `filter_size` argument alone. OpenCL still means "Winograd tiles" by it,
   and `zeropad_U` (OpenCLScheduler.cpp) must keep receiving transformed
   weights. Metal ignores `filter_size`; the raw kernels are always 3×3.
6. Also note in `Network.h` that the `winograd_transform_in/out/sgemm/convolve3`
   declarations there are dead (never defined), and delete them.

**Gate 2.3a:**
- G1 on `build/` (CPU): max |Δ prior| = **0** against the pre-change binary
  (the same arithmetic in the same order; any non-zero diff is a bug).
- `build-opencl`: OpenCL vs CPU parity unchanged (≈1.8e-7, as in step 2.1).
- ctest + ASan.

Commit on its own: "Pass raw conv weights to backends".

## 2. Sub-step 2.3b — `MetalNetwork` (the MPSGraph)

Files: `src/metal/MetalNetwork.h` (plain C++, pimpl, like `MetalContext.h`)
and `src/metal/MetalNetwork.mm`. Link `-framework MetalPerformanceShadersGraph`
and `-framework MetalPerformanceShaders` in the `USE_METAL` block of
`CMakeLists.txt`.

```cpp
class MetalNetwork {
public:
    // Builds the graph from raw (not Winograd) weights. Throws
    // std::runtime_error with a readable message on failure.
    MetalNetwork(const MetalContext& ctx, int channels, int residual_blocks,
                 const ForwardPipe::ForwardPipeWeights& weights);
    ~MetalNetwork();
    // in: batch × 18 × 361; pol: batch × 2 × 361; val: batch × 1 × 361.
    // Synchronous. Not thread-safe (callers serialize; 2.4 changes this).
    void forward(const float* in, int batch, float* pol, float* val);
};
```

`MetalContext` needs a way to hand its `id<MTLDevice>` and queue to other
`.mm` files: add a header `src/metal/MetalContextImpl.h` (ObjC++, included
only by `.mm` files) that defines `MetalContext::Impl`, and a
`const Impl& impl() const` accessor. Do not put Metal types in
`MetalContext.h`.

### Graph construction (once, in the constructor)

- Input placeholder: shape `[-1, 18, 19, 19]` (batch dimension symbolic),
  `MPSDataTypeFloat32`. If symbolic batch fails to compile or is slower, use one
  graph per batch size, built lazily and cached in a map. Record which you
  chose in the commit message.
- Conv descriptor for every 3×3: `MPSGraphConvolution2DOpDescriptor` with
  strides 1, dilations 1, groups 1, `paddingStyle = MPSGraphPaddingStyleExplicit`,
  padding 1 on all four sides, `dataLayout = NCHW`,
  `weightsLayout = OIHW`.
- **BN folding** (per output channel k, using F3/F4 as they stand):
  - `w'[k][c][ky][kx] = w[k][c][ky][kx] * scale[k]`
  - `b'[k] = -mean[k] * scale[k]`
  - Derivation: `scale*(conv(x) - mean) = conv_{w*scale}(x) - mean*scale`.
- Each tower conv: `t = conv(x, w') + reshape(b', [1, K, 1, 1])` (broadcast add).
  - Input conv and first conv of a block: `relu(t)`.
  - Second conv of a block: `relu(t + block_input)`.
- Heads: 1×1 conv (padding 0, OIHW `[2, C, 1, 1]` and `[1, C, 1, 1]`) of the
  tower output. **No bias, no BN, no ReLU** (F7).
- Weights as `constantWithData:shape:dataType:` (MPSGraph copies the NSData;
  it may pre-transform constants, including into Winograd form).

### Execution

- Compile to `MPSGraphExecutable` once (`compileWithDevice:feeds:targetTensors:
  targetOperations:compilationDescriptor:`), then run with
  `runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:`.
- Wrap **shared `MTLBuffer`s** in `MPSGraphTensorData` for the input and both
  outputs, and pass the outputs in `resultsArray`, so MPSGraph writes straight
  into memory the CPU reads (requirement N3: no copies). For 2.3, allocate these
  buffers per batch size and cache them; `forward()` memcpys `in` into the input
  buffer and the outputs out. (2.4 replaces this with the slot ring.)
- Wrap every per-call path in `@autoreleasepool { }`; long runs leak otherwise.

## 3. Sub-step 2.3c — `MetalPipe` (temporary) and wiring

`src/metal/MetalPipe.h/.mm`, a `ForwardPipe`:

- `initialize(channels)`: create the `MetalContext` (error → throw).
- `push_weights(...)`: store a `shared_ptr` to the weights, build `MetalNetwork`.
  The residual block count is `(m_conv_weights.size() - 1) / 2`.
- `forward(...)`: lock a mutex (F9), call `MetalNetwork::forward` with batch 1.
- This pipe is deliberately simple; step 2.4 replaces it with `MetalScheduler`,
  which batches through `BatchQueue`.

Wiring in `Network::initialize`:
- `#ifdef USE_METAL` and `!cfg_cpu_only` → `MetalPipe`. Log
  `<MetalContext::describe()>, MPSGraph, fp32, batch 1 (synchronous)`.
- Metal takes precedence over OpenCL when both are compiled in. Keep the
  `--cpu-only` path working (it is how G2 gets its reference).
- Restructure the `#ifdef USE_OPENCL` block into an explicit if/else chain so
  every combination of `USE_METAL` × `USE_OPENCL` builds:
  `macos-cpu`, `macos-metal`, `macos-opencl`, and `USE_METAL=ON` + OpenCL.

## 4. Sub-step 2.3d — Tests and gate G2

1. Unit test (`USE_METAL` only) in `src/tests/gtests.cpp`. Build a tiny random
   network in memory (C = 8, 2 blocks, random raw weights, random means, scales
   in [0.5, 2]). Then:
   - Metal batch 1 vs `CPUPipe` on 4 random inputs: max abs diff ≤ 1e-4.
   - Metal batch 4 (the same 4 inputs) vs Metal batch 1: ≤ 1e-6. This checks
     the batch layout.
   - Non-zero means and scales ≠ 1 are required, or a BN-folding bug passes.
2. G2 with `scripts/parity/compare_backends.py`:
   `--ref "build-metal/leelaz --cpu-only" --test "build-metal/leelaz"`
   (check how the script appends its own args first; don't pass `-t`). Run on
   both random nets (6b×64 and 15b×192) with `--tol 1e-4`.
3. ASan build with `USE_METAL=ON` runs the unit tests clean. Metal plus ASan is
   supported; if MPS reports false positives, record them, don't suppress
   blindly.
4. Benchmark (record only, no target): `--benchmark -v 400` with Metal vs CPU.
   Batch-1 synchronous Metal is expected to be **slower** than the CPU; the
   speedup comes from 2.4. Say so in BENCHMARKS.md.

## 5. Pitfalls checklist (read before coding)

- [ ] Weights reaching the pipe are Winograd tiles until 2.3a lands (F1).
- [ ] Means already include the bias; stddevs are already scales (F3, F4).
- [ ] Residual add before ReLU; the skip is the block input (F5).
- [ ] Heads are raw conv outputs only (F7). Applying head BN on the GPU
      double-applies it, and G2 will fail in a confusing way.
- [ ] Output layout `pol[b][plane][361]`: channel-major per evaluation, the same
      as the CPU `convolve<1>` result.
- [ ] `-ffast-math` applies to `.mm` files too (target-wide flags). It is fine
      here, but don't rely on NaN checks.
- [ ] `@autoreleasepool` around every per-call Objective-C path.
- [ ] Only `.mm` files may include Metal/MPSGraph headers.

## 6. Done means

- 2.3a gate (G1 Δ = 0), the unit tests, G2 ≤ 1e-4 on both nets, ASan clean,
  and all four build combinations compile.
- Rows updated in `10-model-plan.md` (2.3: "Opus 5.5 plan → Sonnet 5.5
  build"), `ROADMAP.md`, `BENCHMARKS.md`.
- Escalate to Opus if G2 fails twice with a cause you can't explain.
