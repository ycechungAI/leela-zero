# 03 — Spec: Build System & Toolchain

**Release:** `as.0`. **Depends on:** nothing. **Blocks:** everything else.

## 1. Problem

The upstream build doesn't configure on the M4 (blockers B1–B5 in 00-repo-overview).

## 2. Changes

### 2.1 CMake modernization (`CMakeLists.txt`)

| Change | Detail |
|--------|--------|
| Minimum version | `cmake_minimum_required(VERSION 3.13...3.31)` (done in Phase 0). The range syntax keeps CMake 4.x happy and still runs on Ubuntu 20.04 |
| Options | Add `option(USE_METAL ...)`, `option(USE_ACCELERATE ...)` and `option(USE_COREML ...)`. Change the default of `USE_OPENCL` to OFF on APPLE and ON elsewhere (today OpenCL is implied unless `USE_CPU_ONLY` is set) |
| OpenCL | `find_package(OpenCL)` only if `USE_OPENCL`. Keep `USE_CPU_ONLY` as an alias that sets `USE_OPENCL=OFF USE_METAL=OFF`, for backward compatibility |
| Accelerate | Replace the hard-coded header path with `find_library(ACCELERATE_FRAMEWORK Accelerate)` and `target_link_libraries(... ${ACCELERATE_FRAMEWORK})`, plus `target_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64=0)` |
| Metal | When `USE_METAL`: `enable_language(OBJCXX)`, add `src/metal/*.mm`, link `-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation`, and compile `.mm` with `-fobjc-arc -std=c++17` |
| Arch flags | On APPLE arm64: `-mcpu=apple-m4` when `LZ_TUNE_M4=ON`, else `-mcpu=native` for local builds and `-mcpu=apple-m1` for distributable builds, so M1–M3 users don't crash on M4-only instructions. Use `CMAKE_OSX_ARCHITECTURES=arm64` and `CMAKE_OSX_DEPLOYMENT_TARGET=14.0` |
| `-march=native` | Keep it for non-Apple GCC/Clang. On Apple, replace it with the `-mcpu` rule above |
| LTO | Use `CMAKE_INTERPROCEDURAL_OPTIMIZATION` (portable) instead of the raw `-flto` in two places |
| Target-based | Move the global `include_directories` / `add_definitions` to `target_*` on the `objs` OBJECT library, so `tests` inherits them cleanly |
| Qt | `find_package(Qt6 COMPONENTS Core)` first, falling back to `Qt5`. Spec 08 covers porting autogtp/validation to Qt 6 (mostly `QRegExp` → `QRegularExpression` and `QProcess` API changes) |

### 2.2 Submodules

- `src/Eigen`: the current mirror `eigenteam/eigen-git-mirror` is archived.
  Repoint it to `https://gitlab.com/libeigen/eigen.git` at tag **3.4.0**,
  which has better NEON (aarch64) GEMM kernels and fixes for clang ≥ 15.
  Verify bit-identical CPU results against 3.3 on the 0k test (spec 07, gate G1).
- `gtest`: bump to `v1.14.x`. The old commit uses `std::tr1` paths that warn on
  clang 21.
- Add a `BUILD.md` step: `git submodule update --init --recursive`.

### 2.3 Dependencies (Homebrew, `/opt/homebrew`)

```
brew install cmake boost zlib qt          # qt = Qt 6
# optional legacy path
brew install qt@5                          # only if Qt 6 port slips
```

`find_package(Boost)` must use `CONFIG` mode on Boost ≥ 1.70 with CMake ≥ 3.30
(policy CMP0167). Pin the policy to NEW and use `Boost::program_options` and
`Boost::filesystem` imported targets. Note: `filesystem` usage is limited, and
it can be replaced with `std::filesystem` when the macOS target is ≥ 10.15.

### 2.4 Presets

Add a `CMakePresets.json` with:

| Preset | Options |
|--------|---------|
| `macos-metal` (default on Mac) | `USE_METAL=ON USE_ACCELERATE=ON USE_BLAS=ON` |
| `macos-cpu` | `USE_ACCELERATE=ON USE_BLAS=ON`, no GPU |
| `macos-opencl` | `USE_OPENCL=ON USE_ACCELERATE=ON`. This is the benchmark baseline |
| `macos-debug-asan` | Debug + `-fsanitize=address,undefined` |
| `linux-*`, `windows-*` | Mirror existing behavior |

## 3. Acceptance criteria

- [ ] From a clean clone on the M4, these commands produce `leelaz` and
      `tests`, and `./tests` passes:
  - `git submodule update --init --recursive`
  - `cmake --preset macos-cpu`
  - `cmake --build --preset macos-cpu`
- [ ] `cmake --preset macos-opencl` also builds, and `leelaz --tune-only`
      completes. This gives the OpenCL baseline for N1.
- [ ] A Linux docker build (`Dockerfile.tests`, updated to Ubuntu 22.04) still passes.
- [ ] `otool -L leelaz` shows only system frameworks plus Homebrew boost/zlib.
- [ ] No new compiler warnings at `-Wall -Wextra` on Apple clang 21.

## 4. Risks

| Risk | Mitigation |
|------|------------|
| Eigen 3.4 changes CPU numerics | Gate G1 compares against 3.3 output. Keep the 3.3 SHA available behind `-DLZ_EIGEN_LEGACY=ON` for one release |
| Upstream merges get harder | Keep CMake changes additive. Open the CMake 4 fix as a separate upstream PR, since it helps every platform |
