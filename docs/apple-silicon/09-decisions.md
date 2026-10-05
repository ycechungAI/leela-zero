# 09 — Architecture Decision Records

## ADR-001: Metal (MPSGraph-first) as the GPU backend, instead of OpenCL

- **Status:** Accepted. The Phase 2 benchmark triggered the custom-MSL
  fallback; see ADR-009.
- **Context:** On Apple Silicon, OpenCL is a deprecated translation layer capped
  at 1.2. It has no fp16 storage extension and no `simdgroup_matrix`, and it
  forces explicit host↔device copies even though memory is unified.
- **Decision:** Add a native Metal backend. Implement it first with MPSGraph,
  which is Apple-tuned, fuses conv/BN/ReLU and makes fp16 trivial. Keep a
  custom-MSL Winograd path as a fallback if MPSGraph misses targets by more than
  15%. OpenCL stays as an opt-in build for comparison and portability.
- **Consequences:** Adds Objective-C++ (confined to `src/metal`). Requires
  macOS 14+. Upstream is unlikely to take the Metal backend, so it remains a
  fork feature, kept isolated so rebases stay cheap.

## ADR-002: MLX for training (alternatives: PyTorch-MPS, TF2 + tensorflow-metal)

- **Status:** Accepted
- **Context:** The TF1 trainer can't run on macOS arm64. A rewrite is required
  whichever framework is chosen.
- **Options:**
  | Option | Unified memory | Maturity on Mac | Notes |
  |--------|----------------|-----------------|-------|
  | **MLX** | Native: arrays are shared by CPU and GPU, no copies | Apple-maintained, fast release cadence | Lazy eval + `mx.compile`. Smaller ecosystem |
  | PyTorch MPS | Partial: still `.to("mps")` semantics, and some ops fall back to CPU | Large ecosystem | Most portable to CUDA |
  | TF2 + tensorflow-metal | No: plugin copies | Plugin updates have lagged | Closest to existing code, but still a rewrite |
- **Decision:** MLX. It best fits "use unified memory for training speed".
  Model and export code stay framework-thin (plain dataclasses for weights), so
  a PyTorch backend could be added later.
- **Consequences:** Training on Mac and on CUDA uses different code
  (`training/tf` vs `training/mlx`). Parity gates T2/T3/T6 guard against drift.

## ADR-003: Reuse `chunkparser.py` unchanged; change only the batch transport

- **Status:** Accepted
- **Decision:** Keep the data-format logic in one place. Only the serialization
  of batches (bytes for `tf.decode_raw`) is replaced with shared-memory numpy
  slots, inside `training/mlx/lz/data.py`.

## ADR-004: Neural Engine (Core ML) is experimental only

- **Status:** Proposed (for `as.4`)
- **Context:** The M4's 16-core ANE has high TOPS/W, but it is reachable only
  through Core ML, which needs fixed shapes, compiles models ahead of time, and
  offers no guarantee an op runs on the ANE. Dispatch latency at small batch is
  uncertain.
- **Decision:** Prototype a `CoreMLPipe` behind `USE_COREML`. The pipeline:
  export via `coremltools` from the MLX model, at fixed batch sizes, in
  `.mlpackage` form, with `MLComputeUnits.cpuAndNeuralEngine`. Ship it only if
  it beats Metal fp16 at the same power, or on n/s per watt for long self-play
  runs.
- **Consequences:** Adds a Python export step and a model cache directory.
  Not a default.
- **Addendum (2026-10-03, step 2.5):** MPSGraph can reach the ANE without
  Core ML. Its default optimization level adds a placement pass that ran the
  fp16 tower on the Neural Engine in our experiments: about 760 n/s against
  about 400 for fp16 on the GPU on a random 15b×192 net (BENCHMARKS.md). Three
  problems keep it off for now, so `MetalNetwork` compiles at level 0
  (GPU only): the first run on a never-seen network compiled for **325 s**
  (cached by the OS afterwards: 2.7 s); MPSGraph prints `error: Incompatible
  element type for ANE` lines to **stdout**, which corrupts GTP; and accuracy
  has not been checked on a real network (the 6b×64 case showed no gain, as it
  is CPU-bound). A follow-up should add an opt-in flag with a startup warm-up
  that explains the wait, stdout protection around the compile, and the G2
  fp16 gate on real networks. This may make `CoreMLPipe` unnecessary.
- **Addendum (2026-10-04): `--ane` opt-in implemented, off by default.** Level 1
  placement is requested only with `--ane` and fp16. `MetalNetwork` prints a
  notice to stderr, redirects fd 1 to /dev/null during compile, and runs every
  graph once in the constructor, so no compile happens during search. The fp16
  accuracy gate always runs for the ANE (a failure falls back to GPU fp16, or
  fp32 with auto). The tuning cache has a separate `ane` key, and ANE autotune
  tries only batches 8 and 16. Measured on a random 15b×192 net: first run
  compiled 330 s (batch 8), 681 s (16), 1132 s (batch 1); ~803 n/s end to end;
  stdout clean. Still to do: G2 on a real network, and the abort at exit.

## ADR-005: Raise the engine to C++17

- **Status:** Accepted (revised in Phase 0; originally "keep C++14")
- **Context:** Current Boost (1.92, Homebrew) Spirit X3, used by
  `Network.cpp` to parse weights, needs C++17. libc++ in C++17 mode also drops
  `std::binary_function`.
- **Decision:** Set `CMAKE_CXX_STANDARD 17` everywhere. Replace the removed or
  deprecated constructs (`std::binary_function`, `std::result_of` →
  `std::invoke_result_t`). GCC 7+, Clang 5+ and MSVC 2017+ all support it.
- **Consequences:** The VS2015 AppVeyor image can no longer build. ObjC++ files
  in `src/metal` use the same standard.

## ADR-006: One build system (CMake + presets) on every platform

- **Status:** Accepted (Phase 0)
- **Context:** The repo carried five overlapping build/CI paths: CMake, a
  hand-written `src/Makefile`, Visual Studio 2015/2017 solutions with NuGet
  packages, AppVeyor and Travis (with Ubuntu 16.04 Dockerfiles). All but CMake
  were stale (C++14, dead services, EOL images), and the vendored
  `cmake/Modules` shadowed CMake's own newer find-modules.
- **Decision:** CMake is the only build system. `CMakePresets.json` holds the
  per-platform configurations, and vcpkg manifest mode supplies Windows
  dependencies. One GitHub Actions workflow covers macOS arm64, Linux and
  Windows. Legacy files are deleted rather than kept "just in case".
- **Consequences:** Windows users need VS 2019+ and vcpkg instead of opening a
  `.sln`. There are no Docker images until someone needs one; a single modern
  Dockerfile can be added and built in CI then. Fewer moving parts means less
  to keep green.

## ADR-007: Metal scheduler with worker-owned slots and two graphs

- **Status:** Accepted (Phase 2, step 2.4). Refines spec 05 §3.3–3.4.
- **Context:** Spec 05 proposed a shared ring of K = 3 slots per worker with
  an explicit `FREE → FILLING → SUBMITTED → DONE` state machine, untracked
  hazards, and one compiled graph per batch size {1, 2, …, 64}. Two facts
  changed the trade-offs: `BatchQueue::pickup()` only ever returns a full batch
  or a single entry, and on the M4 the GPU saturates at 15b×192 with one batch
  in flight (sweep in BENCHMARKS.md).
- **Decision:**
  - Each worker thread owns its slots (shared `MTLBuffer`s) for life. A
    slot's states are that thread's program order, so no slot state is shared
    between threads and there is no lock-free state machine.
  - One set of compiled `MPSGraphExecutable`s is shared by all workers.
    Encoding is serialized by a mutex in `MetalNetwork::run()`; execution
    overlaps (`runAsync` + a completion semaphore per call).
  - Only two graphs: batch 1 and `cfg_batch_size`. The network does not keep
    the raw weights after compiling them.
  - Defaults: 2 workers, batch 8, 16 search threads. A third worker
    (triple buffering) measured no faster; batch 16 gave ~1% more throughput
    for twice the threads.
- **Consequences:** The scheduler is about 150 lines of plain C++ with no
  Objective-C and no shared mutable slot state; TSan reports no races. GPU
  memory holds two copies of the folded weights instead of seven. If
  `BatchQueue` ever returns partial batches, the full-size slot still works
  (unused rows are ignored), at the cost of computing padding. Step 2.7
  (autotune) should revisit the batch size per network.

## ADR-008: Metal autotune caches speed per network shape, checks accuracy per network

- **Status:** Accepted (Phase 2, step 2.7). Refines spec 05 §3.6.
- **Context:** The best batch size and precision depend on the device and the
  network's size, so measuring on every start costs seconds (20 s for
  15b×192), which is unacceptable for a GUI engine and for self-play, where a
  new network generation starts a new process. But whether fp16 is accurate
  depends on the weight *values*, so caching that verdict by shape could
  approve a network where fp16 fails.
- **Decision:** The cache (`~/Library/Application Support/leela-zero/
  metal_tuning`, plain text, tab separated, one row per device × channels ×
  blocks × precision × batch) holds throughput only. The fp16 accuracy check
  (N6 tolerances, six positions through the full head pipeline) runs on every
  start, in about 0.2 s. `--tune-only` forces a re-measurement. Writes go to a
  temporary file and are renamed, and unreadable rows are skipped, so a
  crashed or concurrent process cannot corrupt it. A user-fixed `--batchsize`
  or `-t` skips the table and times fp32/fp16 at that batch size instead.
- **Addendum (step 2.11c):** the key also holds the engine (MPSGraph or
  Winograd), stored as an optional seventh field; rows without it are
  MPSGraph rows, so existing caches keep working and only the Winograd
  measurements are new.
- **Consequences:** Starts are fast after the first, and a new training
  generation costs nothing extra. The thread pool is created after the network
  because autotune can change the thread count. The cache is not invalidated
  by OS or driver updates; `--tune-only` is the remedy, and a stale table only
  costs a slightly suboptimal batch size.

## ADR-009: MPSGraph missed the target; build a Metal Winograd backend

- **Status:** Accepted (Phase 2, step 2.9). Triggers the ADR-001 fallback.
- **Context:** Measured on the M4 (BENCHMARKS.md, step 2.9), the MPSGraph
  backend reaches 0.75× OpenCL on random 15b×192 and 40b×256 nets, against
  targets of 2× and 2.5×. MPSGraph runs direct convolution at ~63% of the
  GPU's fp32 peak; OpenCL wins because its kernels use Winograd F(4×4, 3×3).
  ADR-001 set the trigger at missing the target by more than 15%.
- **Decision:** Add a second Metal network implementation, `MetalWinograd`,
  behind the same `MetalScheduler`, slots and autotune: port the OpenCL input
  and output transforms (with fused BN/ReLU/residual) to MSL, and do the 36
  batched GEMMs with `simdgroup_matrix` (fp16 inputs, fp32 accumulation), then
  fp32. Keep the MPSGraph network as the correctness reference and as an
  autotune candidate; autotune picks whichever is faster per shape. Gates as
  for MPSGraph: G2 at fp32 1e-4 and fp16 N6, the concurrency tests, ASan/TSan.
- **Outcome (steps 2.11a–c, 2026-10-05):** Winograd with Apple's stock batched
  matrix multiply and fp16 storage reaches 1.57× OpenCL on 15b×192 and 1.52×
  on 40b×256 at the autotuned defaults (about 2× the MPSGraph default), and is
  now the autotune choice. The spec's 2× / 2.5× targets are not met yet; the
  custom `simdgroup_matrix` GEMM (2.11d) is the remaining lever and needs a
  profile first. Winograd amplifies fp16 rounding (20×256: 8.0e-3 against the
  1e-2 limit), so the per-start accuracy check stays essential.
- **Outcome (step 2.11d, 2026-10-05): custom GEMM not adopted.** A profile
  (skipping stages) puts the GEMMs at ~65–70% of GPU time and the transforms
  at ~30–35%; MPS's batched GEMM already runs at 2.1–2.7 TFLOP/s (fp16 in,
  fp32 out). The go/no-go `simdgroup_matrix` kernels (direct and
  threadgroup-staged, 8 tilings) reached 0.84–0.94× MPS on the 256-channel
  shapes, so MPS stays. Even a GEMM at ~80% of peak would give at most ~1.3×,
  i.e. ~1.75× OpenCL on 40b×256: the 2.5× target is out of reach on this path.
  Earlier note: since the GEMM result moved to fp32 (`558de70`) the margin is
  1.4–1.5× OpenCL, at a 7% cost for accuracy on real nets.
- **Consequences:** Hand-written kernels to maintain (the OpenCL ones already
  exist as the template). Until it lands, OpenCL is about 1.33× faster than
  the default Metal backend on these nets, while Metal is 1.9× the CPU.
  Neural Engine placement (ADR-004 addendum) is a separate, possibly larger
  win, and does not replace this: it needs fp16, a long first compile and
  stdout protection.

## ADR-010: Lower the Metal throughput target (N1) to the measured 1.4× OpenCL

- **Status:** Accepted (2026-10-05, after step 2.11d). Amends N1 in
  01-requirements.md and the Phase 2 exit in 05-spec-metal-backend.md and
  ROADMAP.md.
- **Context:** N1 asked for ≥ 2.5× the OpenCL baseline on 40b×256 (and ≥ 2× on
  15b×192). Both engines have now been built and measured on the M4: MPSGraph
  reaches 0.75× OpenCL; the Winograd engine 1.4–1.5× (fp32 GEMM result, kept
  for fp16 accuracy on real nets). The profile puts ~65–70% of the GPU time in
  the batched GEMMs, which MPS already runs at 2.1–2.7 TFLOP/s, and custom
  `simdgroup_matrix` kernels did not beat it (BENCHMARKS.md, step 2.11d). A
  perfect GEMM would give ~1.75×, and fusing the transforms (the other ~30%)
  perhaps 1.6–1.8× in total. 2.5× is not reachable on this hardware path.
  OpenCL on macOS is deprecated and stays capped at version 1.2, so the default
  path still has to be Metal.
- **Decision:** N1 becomes: the default Metal backend is **≥ 1.3× the tuned
  OpenCL baseline** on 15b×192 and 40b×256 (measured 1.4–1.5×, leaving room for
  run-to-run noise), and ≥ 1.5× the CPU backend. Phase 2 exits on that, G2/G3/G4
  and a clean soak. Step 2.11e (transform fusions) is optional future work, not
  an exit condition.
- **Consequences:** `as.2` can be tagged without more kernel work. The
  OpenCL backend stays buildable for comparison but is no longer a reason to
  chase throughput. If a future macOS or MPS release changes GEMM speed, the
  autotune picks it up without code changes.
