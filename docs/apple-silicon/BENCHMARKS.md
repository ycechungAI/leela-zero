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

## Phase 1 steps 1.1–1.3: threads, QoS, Accelerate threading (2026-10-03)

Random nets, M4 (4P + 6E), `--benchmark`, interleaved runs, n/s.

Thread-count sweep (4 runs each, medians; Accelerate BLAS):

| Network | `-t 4` | `-t 6` | `-t 7` | `-t 8` | `-t 10` (default) |
|---------|-------:|-------:|-------:|-------:|------------------:|
| random 15b×192 | 170 | 192 | 201 | 193 | 201 |
| random 6b×64 | 2,515 | 2,500 | 2,950 | 2,915 | 3,055 |

- The spec's proposed default of `perf + eff/2 = 7` threads gave no gain in
  throughput, so **the default stays at all logical CPUs**. (Spec C6 predicted
  E-cores would hurt the MCTS tail; that is a latency effect this throughput
  benchmark does not measure. Revisit with a fixed-time strength test.)
- `pthread_set_qos_class_self_np(USER_INTERACTIVE)` on search threads
  (5 runs, default threads): 15b×192 median 207 vs 203; 6b×64 3,246 vs 3,181.
  Neutral within noise; kept as a harmless scheduling hint.
- `BLASSetThreading(SINGLE_THREADED)` (macOS 15+): 15b×192 205 vs 190, 6b×64
  3,451 vs 3,299 (medians of 5). Small gain, within noise; kept to match the
  OpenBLAS/MKL behaviour of one BLAS thread per search thread.

## Phase 2 step 2.9: Metal (MPSGraph) vs OpenCL vs CPU (2026-10-04)

Random nets, M4, `--benchmark` (`-v 1600` on 15b×192, `-v 800` on 40b×256),
3 interleaved rounds, medians, n/s. OpenCL tuned first (`--tune-only`; it
reports no fp16 compute, and its autodetect picks half *storage*). Metal's
autotune cache was warm. All backends ran about 14% below the previous day's
numbers (machine state), and interleaving keeps the comparison fair.

| Network | OpenCL default (B5, 10 thr) | OpenCL B16, 32 thr | CPU (Accelerate, 10 thr) | Metal fp32 (B8, 16 thr) | Metal autotuned (fp16, B8, 16 thr) | Metal ÷ best OpenCL | Target |
|---------|---:|---:|---:|---:|---:|---:|---:|
| random 15b×192 | 432 | **455** | 181 | 310 | 339 | **0.75×** | ≥ 2× |
| random 40b×256 | 106 | **107** | 48 | 69 | 80 | **0.75×** | ≥ 2.5× |

**Result: MPSGraph misses the target by far.** Metal is 1.9× the CPU but only
0.75× OpenCL, on both sizes.

Why: one 40b×256 evaluation is ~34.5 GFLOP of direct 3×3 convolution. Metal's
80 n/s is ~2.8 TFLOPS, about 63% of the M4 GPU's ~4.4 TFLOPS fp32 peak, so
MPSGraph runs direct convolution well. OpenCL's 106 n/s would be ~3.7 TFLOPS of
direct-convolution work. It gets there because Leela Zero's OpenCL kernels use
Winograd F(4×4, 3×3), which needs about 2.5–4× fewer multiplications, and
MPSGraph does not use Winograd for these layers.

Decision (ADR-009): start the custom MSL Winograd backend (spec 05 Strategy B).
Porting the OpenCL Winograd kernels to Metal should recover OpenCL's speed
without OpenCL's copies and translation layer. A `simdgroup_matrix` batched
GEMM is the route toward the spec's 2–2.5× target: at MPSGraph's 63% of peak,
Winograd would put 40b×256 near 2× OpenCL. Separately, Neural Engine
placement measured ~760 n/s on 15b×192 (step 2.5), which would already be
1.7× OpenCL, if its startup and stdout problems are solved (follow-up task).

Found while running this: OpenCL builds hung at startup since step 2.7 (the
thread pool was created after the network, but OpenCL's precision autodetect
runs on it). Fixed in `d0ebb96`, and CI now starts the OpenCL binary.

## Phase 2 step 2.8: zero-staging input (measured, not implemented) (2026-10-04)

Question: would letting `Network` gather the input planes straight into the
GPU slot (`forward_into`, spec 05 §3.3) speed anything up? Profile of a
Metal search on the smallest net, random 6b×64 at the autotuned defaults
(fp16, batch 16, 32 search threads), `sample` for 8 s:

| | Samples | Share |
|---|---:|---:|
| All threads | 25,579 | 100% |
| Waiting (condition variables, semaphores, work queues) | 25,196 | 98.5% |
| Busy | 383 | 1.5% |
| `MetalScheduler::worker` input/output copies | 13 | 0.05% |

The search is GPU-bound even on the smallest net: threads spend almost all
their time waiting for evaluations. The copy that `forward_into` would remove
is 0.05% of samples (3% of the little CPU time there is), so it cannot raise
throughput, and on 15b×192 and larger the ratio is smaller still. It would also
need the batching redesigned so a search thread can own a slot row before a
worker picks the batch up (ADR-007 deliberately avoided that shared state).
Decision: not implemented. Revisit only if a profile ever shows the CPU side
as the bottleneck.

## Phase 2 step 2.7: autotune (2026-10-03)

Random nets, M4. Default run (autotuned) against the step 2.4 defaults (fp32,
batch 8, 16 threads) and against batch 8 with `--precision auto`. n/s from
`--benchmark`, 3 runs each, the cache warm.

| Network | Autotune picks | Autotuned default | 2.4 defaults (fp32 B8) | B8 + auto precision |
|---------|----------------|------------------:|-----------------------:|--------------------:|
| random 6b×64 | fp16, batch 16, 32 threads | **7,450** | 6,150 (+21%) | 6,725 (+11%) |
| random 15b×192 | fp16, batch 8, 16 threads | **399** | 357 (+12%) | 392 (+2%) |

Against the CPU (Accelerate, 10 threads: about 3,100 and 200 n/s) that is
2.4× and 2.0×.

Measurements the choice came from (evals/s, GPU only, 2 streams):

| 6b×64 | B=8 | B=16 | B=32 | B=64 |
|-------|----:|-----:|-----:|-----:|
| single | 6,916 | 7,267 | 7,233 | 6,784 |
| half | 7,382 | 7,802 | 7,995 | 7,634 |

| 15b×192 | B=8 | B=16 | B=32 | B=64 |
|---------|----:|-----:|-----:|-----:|
| single | 357 | 349 | 359 | 364 |
| half | 397 | 387 | 386 | 400 |

- One-off cost on a cold cache: 6 s (6b×64), 19 s (15b×192); a cached start
  takes 0.2 s and 0.6 s, including the fp16 accuracy check.
- The 5% rule picks 16 for the small net (8 is 7% below its best) and 8 for
  the big one (flat across batch sizes). Larger batches would add search
  threads for little GPU gain.
- Not measured: a real 40b×256 network. Tuning time grows with the network,
  but each measurement is bounded to about 0.25 s, so the cost is dominated by
  compiling the 8 graphs.

## Phase 2 step 2.5: fp16 and `--precision auto` (2026-10-03)

Random nets, M4. fp16 means the residual tower runs in fp16; the 1×1 heads
and the input/output buffers stay fp32. n/s from `--benchmark`, 3 runs.

| Network | fp32 (n/s) | fp16, GPU only (n/s) | Gain |
|---------|-----------:|---------------------:|-----:|
| random 15b×192 | 358–365 | 402 (all 3 runs) | **+11%** |
| random 6b×64 | ~6,100 | ~6,600 | +8% |

- Gate G2 at the N6 fp16 tolerances (policy 1e-2, value 5e-3), 3 positions × 8
  symmetries: 15b×192 policy 3.4e-4 and value 2.5e-5; 6b×64 policy 3.2e-6 and
  value 1.1e-4. fp32 still passes at 1e-4 (6.4e-7 and 3.5e-9).
- `--precision auto` times both precisions at the full batch size with one
  stream per worker (what the scheduler will see), alternates the
  measurements and takes medians (15b×192: fp32 355, fp16 409 evals/s), and
  takes fp16 only if it is ≥ 5% faster and within the N6 tolerances of fp32
  on six positions through the full head pipeline. Startup cost: about 5 s on
  15b×192, which step 2.7 caches.
- A first version measured one batch at a time and picked fp32 on 15b×192:
  sequential runs hide the CPU-side work that fp16 overlaps. Single-stream
  runs also favored whichever precision went first (GPU warm-up), hence the
  alternation.
- Self-check (`-DUSE_METAL_SELFCHECK=ON`): with the check probability forced to 1,
  every search evaluation was compared with the CPU, 0 mismatches in both
  precisions (throughput drops to ~2,300 n/s while it runs).

### Neural Engine placement (experiment, not enabled)

MPSGraph's default optimization level may run the fp16 tower on the Neural
Engine:

| Network | fp32 GPU | fp16 GPU only | fp16 + ANE placement |
|---------|---------:|--------------:|---------------------:|
| random 15b×192 | 358 | 402 | **~760** (758, 769, 731) |
| random 6b×64 | 6,100 | 6,600 | 6,560 (CPU-bound) |

- First run on a never-seen 15b×192 network: **325 s** (5 n/s), then 2.7 s
  once the OS has cached the compiled graph. Each batch size is a separate
  graph.
- MPSGraph prints `error: Incompatible element type for ANE …` to stdout
  during compilation, which breaks the GTP stream (`compare_backends.py`
  could not parse it).
- Accuracy on a real network is untested, so `MetalNetwork` pins level 0
  (GPU only). See ADR-004 for the follow-up.

## Phase 2 step 2.4: MetalScheduler (batched, asynchronous) (2026-10-03)

Random nets, M4, fp32. Interleaved, 3 runs, medians, n/s.

| Network | CPU (Accelerate, 10 threads) | Metal 2.4 defaults (batch 8, 2 workers, 16 threads) | Metal / CPU | Metal batch 1 (`-t 1`) |
|---------|-----------------------------:|----------------------------------------------------:|------------:|-----------------------:|
| random 15b×192 | 198 | 326 | **1.65×** | 209 |
| random 6b×64 | 3,117 | 5,850 | **1.88×** | 1,377 |

Worker × batch sweep (threads = batch × workers, 2 runs each):

| | B=4 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|
| 15b×192, W=1 | 284–321 | 330–343 | 344–348 | 350–352 |
| 15b×192, W=2 | 339–348 | **349–353** | 341–351 | 340–347 |
| 15b×192, W=3 | 284–342 | 349 | 331–338 | 326 |
| 6b×64, W=1 | 3,351–3,459 | 4,089–4,288 | 4,937–4,951 | 4,600–4,810 |
| 6b×64, W=2 | 5,040–5,328 | **5,750–5,803** | 5,865–6,167 | 6,240–6,485 |
| 6b×64, W=3 | 4,954–5,281 | 5,713–5,890 | 5,299–6,089 | 5,890–6,213 |

- 15b×192 is GPU-bound: every configuration with batch ≥ 8 plateaus near
  350 n/s, even with one worker. That is ~2.6 TFLOPS of conv work (about 60%
  of the GPU's fp32 peak), so fp16 (step 2.5) is the next lever, not deeper
  pipelining.
- The small net is CPU-side bound: a second worker adds ~25%, a third adds
  nothing. Hence 2 workers, not the spec's triple buffering (ADR-007).
- Batch-1 latency (`-t 1`): 209 n/s vs 179 for the 2.3 synchronous pipe, so
  the asynchronous completion path costs nothing. The spec's spin-then-wait
  mitigation is not needed.
- Batching works: a debug run reported 188 full batches and 8 single
  evaluations.
- Peak memory footprint (15b×192, `-v 1600`): Metal 550 MB vs CPU 635 MB.
  Not yet measured on 40b×256.

## Phase 2 step 2.3: Metal (MPSGraph, fp32, batch 1, synchronous) (2026-10-03)

Random nets, M4, `--benchmark`, 3 runs each, n/s. The Metal pipe evaluates one
position at a time behind a mutex; batching and asynchronous submission arrive
in step 2.4.

| Network | Threads | CPU (Accelerate) | Metal | Metal / CPU |
|---------|--------:|-----------------:|------:|------------:|
| random 15b×192 | 1 | 72 | 179 | **2.5×** |
| random 15b×192 | 10 | 228 | 193 | 0.85× |
| random 6b×64 | 1 | 899 | 1,326 | 1.5× |
| random 6b×64 | 10 | 4,433 | 1,433 | 0.32× |

- Even unbatched and single-threaded, the GPU beats one CPU thread, by 2.5× on
  the big net. With 10 search threads the CPU scales out while Metal stays
  flat (the mutex serializes it), so the CPU wins until 2.4 batches
  evaluations across threads.
- Metal's `-t 1` figure is the latency-bound number: one dispatch per
  evaluation. That is the floor 2.4 has to improve on with batching.

## Numerical parity

| Gate | Compared | Network / positions | max abs Δ prior | max abs Δ winrate | Result |
|------|----------|---------------------|----------------:|------------------:|--------|
| G1 | Accelerate vs Eigen 3.3 | random 6×64, 8 positions × 8 symmetries | 5.6e-9 | 1.5e-7 | PASS (tol 1e-5) |
| — | Eigen 3.4 vs Eigen 3.3 | same | 9.3e-10 | 6.0e-8 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized vs pre-change (Accelerate) | random 15b×192, 3 positions × 8 symmetries | 3.7e-7 | 0 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized vs pre-change (Accelerate) | random 6b×64, same | 2.8e-9 | 1.2e-7 | PASS (tol 1e-5) |
| G1 | 1.4a vectorized (Accelerate) vs Eigen 3.4 | random 15b×192, same | 3.4e-7 | 0 | PASS (tol 1e-5) |
| G2 | Metal fp16 vs CPU (`--cpu-only`) | random 15b×192, 3 positions × 8 symmetries | 3.4e-4 | 2.5e-5 | PASS (N6: 1e-2 / 5e-3) |
| G2 | Metal fp16 vs CPU (`--cpu-only`) | random 6b×64, same | 3.2e-6 | 1.1e-4 | PASS (N6: 1e-2 / 5e-3) |
| G2 | Metal fp32 vs CPU (`--cpu-only`) | random 6b×64, 3 positions × 8 symmetries | 3.0e-9 | 1.8e-7 | PASS (tol 1e-4) |
| G2 | Metal fp32 vs CPU (`--cpu-only`) | random 15b×192, same | 3.1e-7 | 0 | PASS (tol 1e-4) |
| — | Metal vs CPU raw head outputs (unit test, C=8, 2 blocks, non-trivial BN) | 4 random inputs | 6.0e-7 | — | PASS (tol 1e-4; batch 4 = batch 1 exactly) |
| G1 | 1.4a scalar fallback (MSVC path) vs pre-change | random 6b×64, 2 positions × 8 symmetries | 3.0e-9 | 6.0e-8 | PASS (tol 1e-5) |

## Official baseline (to do)

| Backend | 15b×192 n/s | 40b×256 n/s | Notes |
|---------|------------:|------------:|-------|
| OpenCL (upstream path) | — | — | |
| CPU Eigen | — | — | |
| CPU Accelerate | — | — | |
