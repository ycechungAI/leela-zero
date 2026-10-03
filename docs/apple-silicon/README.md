# Leela Zero on Apple Silicon (M4) — Spec Set

This folder holds the design specs and release plan for porting Leela Zero to the
Apple M4 Mac mini (arm64 / aarch64). The port has three goals:

1. **Build and run natively** on arm64 macOS with a modern toolchain.
2. **Accelerate inference** with Apple's hardware: Accelerate (SME/AMX) on the
   CPU and Metal on the GPU. The Neural Engine is an experimental option.
3. **Train on-device** with a framework that uses unified memory, replacing the
   TensorFlow 1.x pipeline.

> Status: **planning only**. No code has been written yet. Each spec describes
> what to build and how we will know it is done.

## Fork

| | |
|---|---|
| Upstream | https://github.com/leela-zero/leela-zero (branch `next`, last tag `v0.17`) |
| Fork | https://github.com/ycechungAI/leela-zero (already forked; `origin` points here) |
| Working branch (proposed) | `apple-silicon` off `next` |

## Using it now

**[BUILD.md](BUILD.md)** covers building, running, debugging and the training-data workflow on an M4, with the helper scripts in `scripts/macos/`.
Measured numbers live in **[BENCHMARKS.md](BENCHMARKS.md)**.

## Reading order

| # | Doc | What it covers |
|---|-----|----------------|
| 0 | [00-repo-overview.md](00-repo-overview.md) | How the repo is put together today, and where Apple Silicon breaks it |
| 1 | [01-requirements.md](01-requirements.md) | Goals, non-goals, target hardware, success metrics |
| 2 | [02-architecture.md](02-architecture.md) | Target architecture: backends, unified memory model, training stack |
| 3 | [03-spec-build-system.md](03-spec-build-system.md) | CMake / toolchain / dependency modernization |
| 4 | [04-spec-cpu-accelerate.md](04-spec-cpu-accelerate.md) | CPU path: Accelerate BLAS, NEON, P/E-core threading |
| 5 | [05-spec-metal-backend.md](05-spec-metal-backend.md) | New Metal GPU inference backend using zero-copy unified memory |
| 6 | [06-spec-training-mlx.md](06-spec-training-mlx.md) | New MLX training pipeline, weight import/export |
| 7 | [07-spec-testing-benchmarks.md](07-spec-testing-benchmarks.md) | Correctness gates, benchmark protocol, perf targets |
| 8 | [08-spec-release-ci.md](08-spec-release-ci.md) | CI on arm64 runners, packaging, versioning, signing |
| 9 | [09-decisions.md](09-decisions.md) | Architecture decision records (Metal vs OpenCL, MLX vs PyTorch/TF, …) |
| 10 | [10-model-plan.md](10-model-plan.md) | Which Claude model does each Phase 1–2 step, and a log of what was used |
| R | [ROADMAP.md](ROADMAP.md) | **Phased release plan with ordered steps and exit criteria** |

## Release summary

| Release | Theme | Headline |
|---------|-------|----------|
| `v0.17.1-as.0` | Builds on M4 | CMake 4 / Homebrew / arm64 build of the existing code, with CPU-only and OpenCL options |
| `v0.17.1-as.1` | Fast CPU | Accelerate BLAS (SME), tuned threading, NEON-clean Eigen |
| `v0.18.0-as.2` | Metal | Native Metal backend with fp16 and zero-copy batching. It becomes the default on macOS |
| `v0.18.0-as.3` | Train on Mac | MLX training pipeline, compatible with leelaz weights |
| `v0.18.0-as.4` | Polish | autogtp/validation on Qt 6, signed and notarized bundle, Homebrew tap, optional Core ML/ANE |

See [ROADMAP.md](ROADMAP.md) for the step-by-step breakdown.
