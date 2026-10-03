# ROADMAP — Release-First Plan for Leela Zero on Apple Silicon (M4)

Each phase ends in a **tagged, usable release**. Steps inside a phase are
ordered. Every step is a PR into `apple-silicon` unless noted otherwise.

---

## Phase 0 — `v0.17.1-as.0` "Builds on M4" (spec 03)

Status key: ✅ done · 🟡 partial · ⏸ deferred · ⬜ not started

1. ✅ Fork hygiene: spec set added, and the `apple-silicon` integration branch
   was created. No `upstream` remote, by owner's choice.
2. ✅ Submodules initialized. Eigen moved to GitLab 3.4.0 (parity with 3.3:
   ≤1e-9, same speed). gtest bumped to v1.15.2.
3. ✅ CMake rewritten target-based (C++17, IPO in Release, `LZ_NATIVE_ARCH`
   with `-mcpu=apple-m1` redistributable builds, `LZ_SANITIZE`, ctest). Legacy
   build systems removed (Makefile, msvc, AppVeyor, Travis, Dockerfiles).
   Boost.Filesystem replaced by `std::filesystem`. See ADR-006.
4. ✅ `CMakePresets.json` (`macos-cpu`, `macos-opencl`, `macos-debug`,
   `macos-asan`, `linux-cpu`).
5. ✅ `BUILD.md` + `scripts/macos/` (build, start, debug, train).
6. 🟡 `lz-nn-eval` GTP hook + `scripts/parity/compare_backends.py` done. G1
   (Accelerate vs Eigen) passes. Still to do: test-net fetch script (needs
   real networks).
7. ⏸ `BENCHMARKS.md` started with provisional random-network numbers. The
   real baseline with public nets is deferred, by owner's choice.
8. ✅ CI: `.github/workflows/apple-silicon.yml` (macOS CPU + G1 parity, ASan,
   OpenCL compile, Linux).

**Exit:** gate G0 green on M4 and CI. Baseline recorded. Tag `as.0`.

## Phase 1 — `v0.17.1-as.1` "Fast CPU" (spec 04)

> Which model does each step, and the tracking log: [10-model-plan.md](10-model-plan.md).

1. `src/Platform.h`: core counts, QoS, and feature detection.
2. Accelerate as the default BLAS, single-threaded per call, with a startup log line.
3. Default `-t` and QoS for P/E cores.
4. NEON-friendly Winograd transforms and a fused BN+ReLU+residual pass.
5. Gate G1 + benchmark.

**Exit:** ≥1.5× Eigen on 15b×192, and G1 passes. Tag `as.1`.

## Phase 2 — `v0.18.0-as.2` "Metal" (spec 05) ★

> Model per step and tracking: [10-model-plan.md](10-model-plan.md).

1. Extract `BatchQueue.h` from `OpenCLScheduler`. No behavior change, and it is
   sent upstream as well.
2. `MetalContext`: device, queue, runtime MSL compile, and the `USE_METAL` CMake option.
3. `MetalNetwork` via MPSGraph, fp32, fixed batch sizes, with BN folded into conv.
4. `MetalScheduler`: shared-buffer slot ring, triple buffering, completion wakeups.
5. Gate G2 (fp32). Then add fp16 + `--precision auto` + `USE_METAL_SELFCHECK`.
6. `--backend` flag, plus Metal as the default on macOS.
7. Autotune of batch size and precision, with a persisted cache.
8. Optional 2b: `forward_into` zero-staging input path.
9. Benchmark. If MPSGraph is under target by more than 15%, start the custom
   MSL Winograd fallback (ADR-001).
10. CI `build-metal`, nightly `parity-full` and `asan`.

**Exit:** ≥2.5× OpenCL on 40b×256, G2/G3/G4 pass, and the soak test is clean.
Tag `as.2`.

## Phase 3 — `v0.18.0-as.3` "Train on Mac" (spec 06)

This phase can start in parallel with Phase 2 after Phase 0.

1. Scaffold `training/mlx` (uv, Python 3.12, pytest).
2. `model.py` with TF parity, plus T1.
3. `import_weights.py` / `export.py`, plus T2 and T3 (uses the Phase 0 eval hook).
4. Shared-memory data pipeline reusing `chunkparser.py`.
5. `train.py`: compiled step, bf16, macrobatch, LR schedule, checkpoints,
   memory guard.
6. SWA with BN recalc.
7. T4 (learning sanity) and T5 (performance and memory).
8. T6 strength match (400 games). It may finish after the tag and be reported
   in `as.4`.

**Exit:** T1–T5 pass. Tag `as.3`.

## Phase 4 — `v0.18.0-as.4` "Polish & Distribution" (spec 08, ADR-004)

1. Port autogtp/validation to Qt 6 (keep a Qt5 fallback).
2. Release workflow: tarball with bundled Qt, static Boost, `@rpath`.
3. Homebrew tap + formula.
4. Codesign and notarize (needs an Apple Developer ID).
5. Experimental `CoreMLPipe` (ANE) prototype and benchmark. Ship only if it wins.
6. Send the small upstream PRs (CMake fix, BatchQueue, Accelerate link).

**Exit:** a clean-machine install in under 10 minutes. Tag `as.4`.

---

## Dependency graph

```
P0 ──► P1
 │
 ├───► P2 ──────────┐
 │                  ├──► P4
 └───► P3 ──────────┘
```

## Open questions for the owner

1. Is an Apple Developer ID available for notarization (Phase 4 step 4)?
2. Is the public LZ server still the target for autogtp, or a private or local
   training loop?
3. Which network size should training target on the 16 GB machine? The
   recommendation is ≤ 20b×256 (see the memory budget in 02-architecture).
