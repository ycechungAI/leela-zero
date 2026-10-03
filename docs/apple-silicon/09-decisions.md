# 09 — Architecture Decision Records

## ADR-001: Metal (MPSGraph-first) as the GPU backend, instead of OpenCL

- **Status:** Accepted (pending the Phase 2 benchmark confirmation)
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
