# 01 — Requirements

## 1. Target platform

| Item | Requirement |
|------|-------------|
| Primary hardware | Apple M4 Mac mini, 16 GB unified memory (minimum supported config) |
| Also supported | Any Apple Silicon (M1–M4, Pro/Max). Performance is tuned for M4 |
| OS | macOS 14 Sonoma or later (needed for current MPSGraph and MLX). Primary test target is the latest macOS |
| Architecture | `arm64` only. No Rosetta, no universal binary in the first releases |
| Toolchain | Apple clang (Command Line Tools), CMake ≥ 3.24, Homebrew. Full Xcode is **not** required |
| Python (training) | 3.11 or 3.12 via `uv` or pyenv. The system 3.9 is not used |

## 2. Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | `leelaz` builds natively on arm64 macOS with one documented command sequence |
| F2 | `leelaz` plays GTP games with every public LZ network (v1 and v2 weight formats) and gives the same move choices as the reference CPU backend (within numeric tolerance) |
| F3 | Runtime backend selection: `--backend=metal|cpu|opencl`. On macOS the default is `metal` when it is compiled in |
| F4 | The Metal backend supports fp32 and fp16, chosen automatically or forced with `--precision` |
| F5 | The CPU backend uses Accelerate (BLAS on SME/AMX) by default on macOS |
| F6 | `autogtp` and `validation` build and run on macOS arm64 |
| F7 | Training runs on the Mac GPU: it reads existing `*.gz` training chunks and writes leelaz-format weights that load in `leelaz` |
| F8 | Training can resume from an existing leelaz weights file (fine-tuning public nets) |
| F9 | The existing Linux and Windows builds keep working. Apple-specific code is fenced behind CMake options and `__APPLE__` |

## 3. Non-functional requirements

| ID | Requirement | Target |
|----|-------------|--------|
| N1 | Metal inference throughput, 15b×192 and 40b×256 nets | ≥ 1.2× the tuned OpenCL-on-macOS baseline on the same M4 on real networks, and ≥ 1.5× the CPU backend (was ≥ 2.5×; lowered by ADR-010 and ADR-011 after measuring both Metal engines) |
| N2 | CPU-only throughput, 15b×192 net | ≥ 1.5× the Eigen-only baseline |
| N3 | Host↔device copies per NN eval in the Metal path | **0** buffer copies. Inputs are written in place into shared `MTLBuffer`s and outputs are read in place |
| N4 | Training throughput, 20b×256, batch 256, bf16/fp16 | Baseline is set in Phase 3. Target: GPU utilization ≥ 80% in steady state, with no input-pipeline stalls above 5% |
| N5 | Peak resident memory, training a 20b×256 net on a 16 GB machine | ≤ 11 GB, which leaves room for the OS and one `leelaz` self-play process |
| N6 | Numerical agreement with the CPU fp32 reference | fp32: max abs policy diff ≤ 1e-4 and value diff ≤ 1e-4. fp16: ≤ 1e-2 policy and ≤ 5e-3 value (same gates as the existing OpenCL self-check) |
| N7 | Build from clean checkout | ≤ 3 minutes on M4 (Release, without autogtp) |
| N8 | License | Remains GPLv3. New dependencies must be GPL-compatible: MLX (MIT), Qt (LGPL), Boost (BSL) |

## 4. Non-goals (for now)

- iOS or iPadOS builds.
- Universal (x86_64 + arm64) binaries. These may come after `as.4`.
- Distributed or multi-machine training.
- Rewriting the MCTS or changing playing strength. This port is about
  performance and platform support only.
- Supporting the TF1 trainer on macOS. It stays in-tree for Linux/CUDA users,
  unmodified.
- Neural Engine as a default backend. It is explicitly an experimental stretch
  goal (see ADR-004).

## 5. Success metrics for the full program

1. A new user goes from `git clone` to a running `leelaz --backend=metal` in
   under 10 minutes, following `docs/apple-silicon/BUILD.md` (written in Phase 0).
2. `leelaz` self-play on M4 with the Metal backend is at least 2.5× faster than
   the upstream build running through macOS OpenCL.
3. A 10b×128 net trained from scratch on the Mac with the MLX pipeline beats the
   equivalent TF-trained net in a 400-game `validation` match at 50:50 or better
   (proves training parity).
4. CI is green on `macos-15` arm64 runners for every PR to the fork's `apple-silicon` branch.
