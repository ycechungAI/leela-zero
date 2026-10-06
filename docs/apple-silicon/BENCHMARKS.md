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

## Phase 2 step 2.11c: Winograd by default (autotune picks the engine) (2026-10-05)

Autotune now measures both engines (MPSGraph and Winograd) × both precisions ×
batch 8/16/32/64 per network shape, takes Winograd only if it is at least 5%
faster, and `--metal-kernels auto|mpsgraph|winograd` can force one. Default
run (`--benchmark`, autotuned, cache warm), medians of 3 interleaved rounds,
n/s:

| Network | New default | Autotune's choice | MPSGraph only (the old default) | OpenCL (B16, 32 thr) | CPU (10 thr) | ÷ OpenCL | ÷ old default | ÷ CPU |
|---------|---:|---|---:|---:|---:|---:|---:|---:|
| random 15b×192 | **795** | Winograd fp16, batch 16 | 400 | 508 | 226 | **1.57×** | 1.99× | 3.5× |
| random 40b×256 | **170** | Winograd fp16, batch 32 | 83 | 112 | 51 | **1.52×** | 2.05× | 3.3× |

- Raw GPU throughput per engine (15b×192, evals/s at the best batch size):
  MPSGraph 381 (fp16) vs Winograd 726 (fp16) and 491 (fp32).
- The spec targets are still open: 2× OpenCL on 15b×192 and 2.5× on 40b×256.
  The default is at 1.5× on both. What is left is the GEMM (Apple's stock
  `MPSMatrixMultiplication` at about a third of the GPU's fp16 peak, estimated
  from the 8.6 GFLOP of Winograd multiplies per 40b×256 evaluation) and the
  transforms (memory bound); step 2.11d (custom `simdgroup_matrix` GEMM) or
  2.11e (fusions) would go after them, after a profile.
- G2 on 3 shapes × both engines × both precisions, all pass. fp16 Winograd on
  20×256: 8.0e-3 against the 1e-2 limit (MPSGraph 4.1e-3).
- Sanitizers: ASan/UBSan 36 tests clean; TSan 0 warnings on the 22 Metal tests
  and on real 16-thread searches through the Winograd engine in both
  precisions.

## Phase 2 step 2.11b: Winograd engine, fp16 storage (2026-10-05)

Weights, V, M and activations in fp16; the transforms compute in float
registers, the heads and batch norm stay fp32, and `MPSMatrixMultiplication`
multiplies fp16 matrices. Same protocol as 2.11a, medians of 3, n/s.

| Network | OpenCL (B16, 32 thr) | MPSGraph fp16 (B8) | Winograd fp32 (B16) | Winograd fp16 (B8) | Winograd fp16 (B16) | fp16 B16 ÷ OpenCL | ÷ MPSGraph fp16 |
|---------|---:|---:|---:|---:|---:|---:|---:|
| random 15b×192 | 395 | 319 | 378 | 593 | **591** | **1.50×** | 1.85× |
| random 40b×256 | 99 | 73 | 100 | 150 | **150** | **1.52×** | 2.05× |

- fp16 is worth 1.5× over Winograd fp32 here (the engine is memory-bound,
  not math-bound), and it makes Metal 1.5× OpenCL on both sizes. The spec
  target is 2× / 2.5×; the remaining gap is what the custom GEMM (2.11d) goes
  after.
- Accuracy: the unit test is within 1.3–2.1e-3 of the CPU (0.1% of the
  output scale) for C = 17, 32 and **256**; the error does not grow with C, so
  MPS accumulates in fp32. G2 at the N6 tolerances (policy 1e-2, value 5e-3):
  6×64 1.2e-5, 15×192 1.8e-3, **20×256 8.0e-3** (MPSGraph fp16: 4.5e-3).
  Winograd amplifies fp16 rounding more, so the margin on large nets is thin:
  `--precision auto` still checks fp16 against fp32 on the actual network at
  every start, and takes fp32 if it is outside the tolerance.
- The scheduler stress tests (12 threads, 1,800 evaluations) pass with the
  Winograd engine in both precisions.

## Phase 2 step 2.11a: Winograd engine, fp32 (2026-10-05)

First working Winograd network (MSL transforms + `MPSMatrixMultiplication`,
fp32 storage and math). Random nets, M4, `--benchmark`, 3 interleaved rounds,
medians, n/s. fp32 only; fp16 and the custom GEMM are later sub-steps. OpenCL
tuned (`--tune-only`) and run at batch 16 / 32 threads.

| Network | OpenCL (B16, 32 thr) | MPSGraph fp32 (B8) | Winograd fp32 (B8) | Winograd fp32 (B16) | Winograd B16 ÷ OpenCL | ÷ MPSGraph fp32 |
|---------|---:|---:|---:|---:|---:|---:|
| random 15b×192 | 363 | 248 | 377 | **411** | **1.13×** | 1.66× |
| random 40b×256 | 80 | 55 | 93 | **97** | **1.21×** | 1.76× |

- Already past the plan's minimum (at least OpenCL) with fp32 and Apple's
  stock matrix multiply: no hand-written GEMM, no fusion, one compute encoder
  per kernel. The spec target (2–2.5× OpenCL) needs fp16 (2.11b) and probably
  the custom GEMM (2.11d).
- The day's absolute numbers are lower than on 2026-10-04 for every backend
  (machine state), so compare within the table.
- Correctness: the unit test matches the CPU to 5–7e-7 for C = 8, 17, 32 and
  64 (batch 4 equals batch 1 exactly); dropping the BN mean or the tile
  border makes it fail (diffs of 0.47 and 0.88). Gate G2 fp32 on 6×64, 15×192
  and 20×256: worst prior difference 8.3e-6 (limit 1e-4).
- Layout note: V and M are `[element][tile][channel]` (N×C and N×K per
  element), so the multiply needs no transposes; this differs from OpenCL's
  `[element][channel][tile]` in the plan's F3.

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

### Neural Engine placement (opt-in: `--ane`, 2026-10-04)

With `--ane`: random 15b×192, `--benchmark -v 1600`: **803 n/s** (autotune table,
fp16 ANE: batch 8 771, batch 16 827; fp32: 296 / 287). First-run compile: batch 8
330 s, batch 16 681 s, batch 1 1132 s; cached runs take ~0 s. Stdout stayed clean.

Batch 1 (`-t 1 --batchsize 1`), n/s: 15b×192 ANE 411-430 vs GPU fp16 126-240;
official 40b×256 ANE 159 vs 49. The ANE stays on at batch 1.

Official 40b×256 autotune (n/s): fp16 ANE batch 8 181, batch 16 180; fp16 GPU
batch 8/16/32/64 77/82/71/81; fp32 65-71. `--ane --benchmark -v 1600`: **186 n/s**
(about 2.3x GPU fp16). First-run compile of the batch-8 graph: 728 s.

G2 with `--ane` against `--backend cpu` (`--tol 1e-2 --tol-value 5e-3`), all pass:

| Net, positions | Engine | Prior diff | Winrate diff |
|---|---|---:|---:|
| random 15b×192, random play | ANE | 2.8e-3 | 1.9e-3 |
| official 40b×256, random play | ANE | 1.8e-3 | 4.2e-3 |
| official 40b×256, random play | GPU fp16 | 2.5e-3 | 2.3e-3 |
| official 40b×256, its own games | ANE | 3.3e-3 | 3.9e-3 |
| official 40b×256, its own games | GPU fp16 | 3.3e-3 | 4.8e-3 |

Small nets (32×3, 64×3, 128×3) are not placed on the ANE; 128×6 is.

### Neural Engine placement (experiment, original notes)

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

## Step 2.11d: profile and custom GEMM go/no-go (2026-10-05)

Stage profile (search n/s, Winograd fp16, batch 16, two rounds; a stage is
skipped to see its share):

| net | full | no GEMM | no transforms |
| --- | --- | --- | --- |
| 15b×192 | 627 / 665 | 1739 / 1758 | 1082 / 1106 |
| 40b×256 | 143 / 151 | 462 / 479 | 227 / 257 |

GEMM ≈ 65–70% of GPU time, transforms ≈ 30–35%. Batched GEMM alone (36
elements, half × half → float, GPU timestamps, median of 20):

| N×C×K | MPS | best custom (direct 2×2 simdgroups of 32×32) | best staged (64×64, BK 16/32) |
| --- | --- | --- | --- |
| 400×192×192 | 0.388 ms, 2.74 TFLOP/s | 0.93× | 0.70× |
| 400×256×256 | 0.793 ms, 2.38 TFLOP/s | 0.84× | 0.76× |
| 800×256×256 | 1.801 ms, 2.10 TFLOP/s | 0.94× | 0.88× |
| 1600×256×256 | 3.035 ms, 2.49 TFLOP/s | 0.91× | 0.83× |

No custom kernel beat MPS, so 2.11d stops here (ADR-009).

## Phase 3 gates T2–T5 (2026-10-06)

Networks from zero.sjeng.org, verified by SHA-256 and deleted afterwards:
15b×192 `d351f06e…` (T2/T3) and 6b×128 `b3b00c6d…` (self-play data for T4).

- **T3** (MLX fp32 eval vs `leelaz --backend cpu`, 30 positions from real
  self-play): winrate 8.9e-7, priors 2.2e-6 (tol 1e-4). **Pass.**
- **T2** (import → export → leelaz CPU, 100 positions × 8 symmetries): priors
  3.5e-6, winrate 2.0e-6 against the spec's 1e-6. 623 of 10.4 M values (all
  in the BN "bias" lines, β·√(var+ε)) come back one float32 step off: when
  √(var+ε) > 1 no float32 β reproduces every file bias exactly, so 1e-6 is not
  reachable with a β parameterisation. **Open: needs a tolerance decision.**
- **T5** (lz-train on real chunks; GPU utilisation from `ioreg`, no sudo):

  | config | positions/s | GPU | input stall | peak MLX |
  | --- | ---: | ---: | ---: | ---: |
  | 20b×256, batch 64, bf16 | 33.6 | 99% | — | 3.1 GiB |
  | 20b×256, batch 128, bf16 | 33.7 | 100% | — | 5.3 GiB |
  | 20b×256, batch 256, bf16 (CLI) | 4.4 | 71% | 0.1% | 9.15 GiB |
  | 6b×64, batch 256, bf16 (CLI) | 760 | 98% | 0.1% (10.6% first window) | 1.55 GiB |

  Batch 256 at 20b×256 fits the 11 GB budget (N5) but, with everyday apps
  open on a 16 GB machine, the system swaps (8 GB of swap) and the step is 8×
  slower; batch 128 × macrobatch 2 gives the same effective batch at full
  speed. lz-train now warns when a config needs over half the memory.
  **Pass** (GPU ≥ 80%, stall < 5%, peak ≤ 11 GB), at batch 128 for 20b×256.
- **T4** (learning sanity; ADR-012: no TF reference, thresholds from this
  first run): 6b×64, batch 128, 20k steps, lr 0.05 → 0.005 at 12k, bf16,
  sample 1, on 243 self-play games (42k positions; 6b×128 net, 100–200 visits
  with noise, so the targets are flat: mean entropy 3.9 nats, top move 16%).
  10% of chunks held out.

  | step | test policy loss | test accuracy | test mse/4 |
  | ---: | ---: | ---: | ---: |
  | 2000 | 5.60 | 4.0% | 0.40 |
  | 6000 | 4.99 | 31.3% | 0.43 |
  | 12000 | 4.88 | 37.3% | 0.46 |
  | 16000 | 4.80 | 41.5% | 0.47 |
  | 20000 | **4.80** | **42.2%** | 0.46 |

  Train at 20k: policy 4.76, accuracy 46.7%. 773 positions/s, peak 1.4 GiB.
  The value head memorises the 219 training games (train mse/4 0.005, test
  0.46), so value is not gated at this data size. **Thresholds for T4 on this
  recipe: test policy loss ≤ 4.85 and test accuracy ≥ 40% at 20k steps.**
  Both exported nets (plain and SWA) load in leelaz.

## Step 3.5: MLX training step, speed and memory (2026-10-06)

`lz.train.Trainer` (compiled grad step, bf16 compute, fp32 master weights),
random batches, M4 (16 GB). Peak memory is `mx.get_peak_memory()`.

| net | batch | dtype | step | positions/s | peak |
| --- | ---: | --- | ---: | ---: | ---: |
| 6b×64 | 256 | bf16 | 0.26–0.30 s | 850–980 | 1.6–1.8 GiB |
| 6b×64 | 256 | fp32 | 0.31 s | 830 | 2.8 GiB |
| 10b×128 | 256 | bf16 | 1.15 s | 223 | 3.9 GiB (before the BN change) |
| 20b×256 | 64 | bf16 | 1.7 s | 36–39 | 3.3 GiB |
| 20b×256 | 256 | bf16 | 7.9 s | 33 | **10.55 GiB** (N5 budget 11) |

Compiled vs eager: 0.26 vs 0.33 s/step (6b×64). 20b×256 is compute-bound at
about 2 TFLOP/s effective (17 GFLOP/position forward, ~3× with backward), so
roughly 35 positions/s on this GPU: a 200k-step run at batch 128 would take
about 8 days. The first measurement was 12.9 GiB at batch 256 because batch
norm upcast every activation to fp32 and the backward pass kept those copies;
checkpointing the batch norm body (recompute from the bf16 input) brought it to
10.55 GiB at the same speed. The memory guard's estimate is calibrated on the
two 20b×256 points (3.27 and 10.59 GiB projected).

## Phase 2 soak (2026-10-05)

`scripts/macos/soak.py --minutes 120 -t 16`, random 15b×192 network, Metal
autotuned (Winograd fp16), 800 visits per move: 12 games, 5,114 moves, no crash
or hang. RSS rises with the search tree during a game and drops back to
~120 MB when the next one starts (peak 638 MB in a long game); medians by
quarter 381 / 375 / 124 MB (second, third, last). A 20-minute run on the
review-fixed build ended with `leaks`: 0 leaks for 0 bytes. The Accelerate
per-thread fix from 1.R is speed-neutral: 15b×192 CPU 241→242, 174→173 and
83→83 n/s at -t 10/4/1 (medians of 3, interleaved).

## Winograd fp16 accuracy on a real network (2026-10-05)

The nightly failed G2 for fp16 Winograd on the random 20×256 stand-in (policy
0.0137 vs the 1e-2 limit). Emulating fp16 rounding one stage at a time in fp32
showed no dominant stage (weights, V, M and activations each add 3-4e-3 on the
random net); on the real 40×256 network M mattered most for the value. M (the
GEMM result) is now always float: about 7% slower (15×192 B16: ~730 to ~690
n/s; 40×256: ~171 to ~160), still ~1.4-1.5x OpenCL.

Real 40×256 network (the official best network, downloaded for this check and
deleted), Metal vs CPU, 8 symmetries x 80 positions of random-play games:

| engine / precision | max policy diff | max winrate diff |
| --- | --- | --- |
| MPSGraph fp32 | 2.3e-6 | 3.1e-6 |
| MPSGraph fp16 | 2.7e-3 | 3.0e-3 |
| Winograd fp32 | 1.5e-6 | 3.4e-6 |
| Winograd fp16 before (half M) | 2.3e-3 | **6.0e-3 (over the 5e-3 limit)** |
| Winograd fp16 after (float M) | 2.8e-3 | 2.5e-3 |

Random-weight stand-ins are a worst case whose fp16 error varies about 2x with
the positions played, so `full_gate.sh` gives them a 2e-2 policy limit
(`FP16_POLICY_TOL`); real networks keep N6 (1e-2 / 5e-3). The random 20×256
fp16 Winograd row is 5.6e-3 locally after the change (6.5e-3 before).

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
| G2 | Metal Winograd vs CPU, fp32 and fp16 | random 6×64, 15×192, 20×256, 100 positions × 8 symmetries | fp32 ≤ 6.8e-6; fp16 ≤ 8.0e-3 (20×256) | — | PASS (1e-4 / N6 1e-2) |
| G2 | Metal fp16 vs CPU (`--cpu-only`) | random 15b×192, 3 positions × 8 symmetries | 3.4e-4 | 2.5e-5 | PASS (N6: 1e-2 / 5e-3) |
| G2 | Metal fp16 vs CPU (`--cpu-only`) | random 6b×64, same | 3.2e-6 | 1.1e-4 | PASS (N6: 1e-2 / 5e-3) |
| G2 | Metal fp32 vs CPU (`--cpu-only`) | random 6b×64, 3 positions × 8 symmetries | 3.0e-9 | 1.8e-7 | PASS (tol 1e-4) |
| G2 | Metal fp32 vs CPU (`--cpu-only`) | random 15b×192, same | 3.1e-7 | 0 | PASS (tol 1e-4) |
| — | Metal vs CPU raw head outputs (unit test, C=8, 2 blocks, non-trivial BN) | 4 random inputs | 6.0e-7 | — | PASS (tol 1e-4; batch 4 = batch 1 exactly) |
| G1 | 1.4a scalar fallback (MSVC path) vs pre-change | random 6b×64, 2 positions × 8 symmetries | 3.0e-9 | 6.0e-8 | PASS (tol 1e-5) |

## Official baseline: real networks (step 1.5, 2026-10-05)

M4 Mac mini (4P+6E, 10-core GPU), macOS 27.0.1, commit `c44465b`. Networks
from zero.sjeng.org, verified by SHA-256 and deleted afterwards: 15b×192
`d351f06e…` (the last 15-block net) and 40b×256 `0e9ea880…` (the final best
network). `--benchmark` (n/s), 3 interleaved rounds, medians; OpenCL tuned
first, Metal autotuned first (fresh cache).

| Backend | 15b×192, 1600 v | 40b×256, 800 v | Notes |
|---------|----------------:|---------------:|-------|
| OpenCL (B16, t32) | 558 | 117 | upstream kernels, tuned |
| CPU Eigen (t10) | 99 | 22 | |
| CPU Accelerate (t10) | 225 | 50 | **2.3× Eigen** (Phase 1 exit ≥ 1.5×) |
| Metal (autotuned: Winograd fp16) | 682 | 157 | **1.22× / 1.34× OpenCL**, 3.0× / 3.1× Accelerate |

The first Metal round on 15b×192 (377) was an outlier and is excluded by the
median. On the real 15b×192 net Metal's lead depends on search length: at
1600 visits 1.25× (710 vs 568, a second set of rounds), at 6400 visits 1.38×
(775 vs 560). Fixing the batch and threads by hand (B8/16/32) gives the same
690–720 n/s, so the autotune choice is not the cause. Random nets of the same
shapes gave 1.57× / 1.52× (step 2.11c), so random-net ratios overstate Metal
somewhat on real nets.

### Gates G3 and G4 on the real 15b×192 network (2026-10-05)

- **G3** (`USE_METAL_SELFCHECK`, 1 in 2000 evaluations checked against the
  CPU): 6 self-play games at 200 visits, 1,330 moves (~130 checks), no
  mismatch. Pass.
- **G4** (`scripts/parity/gtp_regression.py`: 20 positions from the G3 games,
  `genmove` at 1600 playouts, `-t 1 -s 1`, no noise, ample time): leelaz's
  search was not reproducible across processes (pool threads seeded their RNG
  from the thread id; the default one-hour time budget cut long searches
  short); both are fixed. Exact same move as the CPU: OpenCL 18/20, Metal fp32
  17/20, Metal fp16 16–18/20, and the CPU against itself with another seed
  16/20 and 17/20, so exact agreement measures search noise on near-ties.
  With the ADR-011 rule (same move, or within 1% winrate in the CPU's
  search): Metal fp16 19/20 (18 same), **pass**; the reseeded CPU 19/20 and
  20/20.
