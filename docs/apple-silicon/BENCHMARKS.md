# Benchmarks (Apple M4 Mac mini)

Protocol: [07-spec-testing-benchmarks.md §3](07-spec-testing-benchmarks.md).
Every row records the commit, the network and how it was measured.

## Host

| | |
|---|---|
| Machine | Mac mini, Apple M4 (4P + 6E CPU, 10-core GPU), 16 GB |
| OS / toolchain | macOS 27, Apple clang 21, CMake 4.4.3, Boost 1.92 |

## Provisional: CPU backends with a random network (2026-10-03)

> ⚠️ This is **not** the official baseline. It uses a random-weights network
> from `make_random_net.py`, so the numbers show the relative cost of the
> backends' math but not playing strength. The official baseline (public
> 15b×192 and 40b×256 nets, plus OpenCL) is still Phase 0 step 7.

- Network: random 6 blocks × 64 filters (`make_random_net.py --blocks 6 --filters 64`, seed 1)
- Command: `leelaz -w net --benchmark -v 1600` (default threads = 10), median of runs
- Commit: `as/p0-build` working tree after `9eef2e9`, Release, `-O3 -flto -march=native`

| Backend | n/s (median) | Runs | vs Eigen 3.3 |
|---------|-------------:|-----:|-------------:|
| CPU, Eigen 3.3.7 (`USE_CPU_ONLY`, no BLAS) | 1,512 | 5 | 1.00× |
| CPU, Eigen 3.4.0 (`USE_CPU_ONLY`, no BLAS) | 1,525 | 5 | 1.01× |
| CPU, Accelerate BLAS (`USE_BLAS`, Eigen 3.4) | 2,710 | 3 | **1.79×** |

Observations:
- Accelerate is already ~1.8× faster than Eigen before any Phase 1 tuning. The
  Phase 1 target (≥1.5× on 15b×192) still has to be confirmed on a real network.
- Eigen 3.4 is speed-neutral against 3.3. A first 3-run sample suggested 3.4
  was slower (1,248 vs 1,468), but 5 interleaved runs showed that was noise.
  **Lesson:** always interleave runs and use ≥5 samples.

## Phase 1 step 1.4a: vectorized Winograd transforms (2026-10-03)

Random networks again, so the numbers are relative. Same protocol, but A/B
interleaved: the pre-change binary (`d9c8ded`) and the new one alternate, 5
rounds each. Accelerate BLAS, Release, `-O3 -flto -mcpu=native`.

| Network | Threads | Before n/s (median) | After n/s (median) | Speedup |
|---------|--------:|--------------------:|-------------------:|--------:|
| random 15b×192 | 1 (`-v 300`) | 62 | 78 | **1.26×** |
| random 15b×192 | 10 (default, `-v 1600`) | 184 | 207 | 1.13× |
| random 6b×64 | 10 (default, `-v 1600`) | 2,642 | 3,198 | 1.21× |

Observations:
- Profile (`sample`, 15b×192, `-t 1`): the transforms were 48% of CPU time
  (24% in, 24% out) before and about 40% after, with sgemm most of the rest.
  They are about 1.5× faster per eval, not 4×. They are now limited by stores,
  not math: `transform_in` scatters each 4-channel result into V's
  `[element][channel][tile]` layout one lane at a time.
- Tried and rejected: a `[element][tile][channel]` V layout, where each result
  is one vector store and sgemm reads B transposed. It was 9% slower overall
  (71 vs 78 n/s), because the stores land in a new cache line each time.
  Fixing that properly means an NHWC-blocked restructure. Not worth it before
  Metal (Phase 2) takes over inference.
- Tried and rejected: fixed-trip-count store fast paths. No measurable
  change.
- Multi-thread gains are smaller, because 10 threads share memory bandwidth
  and include the E-cores.

## Phase 1 step 1.4b: fused BN + ReLU + residual (2026-10-03)

A/B interleaved against 1.4a, 5 rounds, single thread, random nets.

| Network | 1.4a n/s (median) | 1.4b n/s (median) | Change |
|---------|------------------:|------------------:|-------:|
| random 15b×192 | 79 | 79 | 0% |
| random 6b×64 | 836 | 844 | +1% (noise) |

Speed-neutral. The separate pass was cheap next to the sgemm and the
transforms. Kept because it removes a full read/write sweep of each layer's
output and deletes the duplicate `batchnorm` template from `CPUPipe.cpp`.
Parity unchanged (G1 vs pre-1.4 build: 3.7e-7 on 15b×192).

## Numerical parity

| Gate | Compared | Network / positions | max abs Δ prior | max abs Δ winrate | Result |
|------|----------|---------------------|----------------:|------------------:|--------|
| G1 | Accelerate vs Eigen 3.3 | random 6×64, 8 positions × 8 symmetries | 5.6e-9 | 1.5e-7 | PASS (tol 1e-5) |
| — | Eigen 3.4 vs Eigen 3.3 | same | 9.3e-10 | 6.0e-8 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized vs pre-change (Accelerate) | random 15b×192, 3 positions × 8 symmetries | 3.7e-7 | 0 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized vs pre-change (Accelerate) | random 6b×64, same | 2.8e-9 | 1.2e-7 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized (Accelerate) vs Eigen 3.4 | random 15b×192, same | 3.4e-7 | 0 | PASS (tol 1e-5) |
| G1 | 1.4a scalar fallback (MSVC path) vs pre-change | random 6b×64, 2 positions × 8 symmetries | 3.0e-9 | 6.0e-8 | PASS (tol 1e-5) |

## Official baseline (to do)

| Backend | 15b×192 n/s | 40b×256 n/s | Notes |
|---------|------------:|------------:|-------|
| OpenCL (upstream path) | — | — | |
| CPU Eigen | — | — | |
| CPU Accelerate | — | — | |
