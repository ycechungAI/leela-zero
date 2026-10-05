# 07 — Spec: Testing, Correctness Gates & Benchmarks

## 1. Test networks (fixtures)

Download once and cache them. They are **not** committed (large files).

| ID | Net | Size | Why |
|----|-----|------|-----|
| N-S | a 6b×64 bootstrap net | ~1 MB | Fast CI |
| N-M | a 15b×192 public net | ~45 MB | Mid-size, CPU benchmark |
| N-L | a 40b×256 public net (e.g. the final LZ best) | ~90 MB | GPU benchmark, fp16 range stress |
| N-E | an ELF v2 net | ~90 MB | v2 value-head format path |

`scripts/fetch_test_nets.sh` downloads by SHA-256 into `~/.cache/leela-zero/nets`.
CI caches this directory.

## 2. Correctness gates

| Gate | Compares | Tolerance | When |
|------|----------|-----------|------|
| G0 | Existing gtests (`./tests`) | exact | every PR |
| G1 | CPU Accelerate vs CPU Eigen (3.3 SHA) | ≤1e-5 abs | `as.1` + every PR touching `CPUPipe` |
| G2 | Metal fp32 / fp16 vs CPU reference: 100 positions × 8 symmetries × {N-S, N-M, N-L, N-E} | fp32 ≤1e-4. fp16 ≤1e-2 policy prob / ≤5e-3 value | every PR touching `src/metal` |
| G3 | `USE_METAL_SELFCHECK` 1000-move self-play | 0 mismatches | release candidates |
| G4 | GTP regression (`scripts/parity/gtp_regression.py`): fixed-seed `genmove` with `-p 1600 --noponder --randomcnt 0 -s 1 -t 1` on 20 positions | Same move as the CPU, or a move within 1% winrate of the CPU's choice in the CPU's own search, in ≥ 19/20 (ADR-011) | release candidates |
| T1–T6 | Training gates | see spec 06 | `as.3` |

### Implementation notes

- **Implemented (Phase 0):** the unlisted GTP command `lz-nn-eval [symmetry]`
  evaluates the current position on the loaded backend, bypassing the cache.
  It prints the winrate, the pass prior and the 361 priors at `%.9g`.
  `scripts/parity/compare_backends.py` drives two `leelaz` processes, which can
  be different builds, backends or (with `--test-weights`) networks. It loads
  the same SGF positions into both, diffs all 8 symmetries, and exits non-zero
  past the tolerance. Choosing the backend per process replaced the original
  `lz-nn-eval <backend> <symmetry>` design, because one process holds one
  backend.
- The same hook serves T2/T3: the MLX side dumps its outputs for the same
  positions (`lz.tools.dump_eval`).
- Positions come from SGFs under `src/tests/` plus 100 positions sampled from
  the 0k test file already in the repo (`src/tests/0k.txt`).

## 3. Benchmark protocol (`BENCHMARKS.md`)

Every number in release notes must come from this protocol.

1. Mac mini M4, 16 GB, on AC power, no other user apps open, `caffeinate -i`,
   and 2 minutes of cool-down before each run.
2. **NN throughput**: `leelaz -w <net> --benchmark` (upstream feature: fixed
   positions, fixed playouts) reporting n/s. Run 3 times and take the median.
   Matrix: backends {opencl, cpu-eigen, cpu-accelerate, metal-fp32, metal-fp16}
   × nets {N-M, N-L} × batch {1, 8, 16, 32, 64}.
3. **Search throughput**: `genmove` from the empty board with `-v 6400`,
   measuring visits/s at `-t {4, 7, 10, 16}`.
4. **Power/thermals**: `sudo powermetrics --samplers gpu_power,cpu_power -i 1000`
   during (2). Report average watts and n/s per watt.
5. **Training**: steps/s and positions/s at {6b×64, 15b×192, 20b×256} ×
   batch {128, 256, 512} × {bf16, fp32}, plus peak RSS (`/usr/bin/time -l`).
6. Record the exact commit, macOS build, MLX version and net SHA.

**Phase 0 records the baseline** (OpenCL and Eigen on the unmodified upstream
code, built with the minimal CMake fix) before any optimization lands.

## 4. Performance targets (summary)

| Metric | Target | Spec |
|--------|--------|------|
| Metal (autotuned) vs tuned OpenCL, N-L and N-M | ≥ 1.2× n/s (ADR-010, ADR-011) | 05 |
| Metal batch-1 latency vs OpenCL batch-1 | ≤ 1.0× | 05 |
| CPU Accelerate vs Eigen, N-M | ≥ 1.5× n/s | 04 |
| Training GPU utilization | ≥ 80% | 06 |
| Training peak RSS, 20b×256 b256 bf16 | ≤ 11 GB | 06 |

## 5. Tooling

- Instruments: Metal System Trace (copies and encoder gaps), Time Profiler
  (CPU hot spots), Leaks.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` in debug CI runs.
- ASan/UBSan preset (`macos-debug-asan`) runs G0 + G2 on N-S nightly.
