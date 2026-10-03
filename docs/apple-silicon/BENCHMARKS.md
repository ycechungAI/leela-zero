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

## Numerical parity

| Gate | Compared | Network / positions | max abs Δ prior | max abs Δ winrate | Result |
|------|----------|---------------------|----------------:|------------------:|--------|
| G1 | Accelerate vs Eigen 3.3 | random 6×64, 8 positions × 8 symmetries | 5.6e-9 | 1.5e-7 | PASS (tol 1e-5) |
| — | Eigen 3.4 vs Eigen 3.3 | same | 9.3e-10 | 6.0e-8 | PASS (tol 1e-5) |

## Official baseline (to do)

| Backend | 15b×192 n/s | 40b×256 n/s | Notes |
|---------|------------:|------------:|-------|
| OpenCL (upstream path) | — | — | |
| CPU Eigen | — | — | |
| CPU Accelerate | — | — | |
