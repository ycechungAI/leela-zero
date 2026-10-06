# 10 — Model Plan & Tracking (Phases 1–3)

Which Claude model should do each step of Phases 1 and 2. Every step is
logged in the tracking tables below, so you can see what was done by which
model and how it was verified.

## 1. Why lesser models are safe here

Every step has to pass the same automatic gates, whichever model wrote it:

| Gate | Catches |
|------|---------|
| ctest (13 unit tests) | Engine logic regressions |
| `scripts/parity/compare_backends.py` (G1 CPU, G2 Metal) | Any numeric drift in NN output, beyond 1e-5 (CPU) or the fp16 tolerance |
| ASan + UBSan build | Memory errors, undefined behavior |
| CI on macOS / Linux / Windows | Breaking other platforms |
| Benchmark (BENCHMARKS.md protocol) | "Optimizations" that aren't faster |

A model can't land wrong numbers or memory bugs without a red gate. The
limit is what the gates **can't** see: races that rarely fire, design
mistakes, and code that passes but is slower or harder to maintain than it
should be. Those steps get the strongest model.

## 2. Model tiers

| Tier | Model | Use for |
|------|-------|---------|
| **S** (strongest) | Opus 5.5 (`claude-opus-5-5`) | Concurrency, lock-free/GPU sync, numerics-critical kernels, design calls, phase-end review |
| **M** (default) | Sonnet 5.5 (`claude-sonnet-5-5`) | Well-specified implementation against a spec + gates |
| **L** (light) | Haiku 4.5 (`claude-haiku-4-5-20251001`) | Mechanical edits: CI YAML, flags, docs, log lines, benchmark runs |

**Escalation rule:** if a step's gates fail twice on M or L, hand the step to
the next tier up, with the failure output. Note the escalation in the log.

**Review rule:** at the end of each phase, run `/code-review high` on the
phase diff with **S**, before tagging the release.

## 3. How to switch models

- **Desktop app:** model picker in the session header.
- **CLI:** `claude --model sonnet` (or `haiku`, `opus`), or `/model` inside a session.
- **All three in one flow:** the `/claude-code-toolkit:build <task>` skill
  plans with Opus, implements with Sonnet and reviews with Opus. It fits the
  steps marked "S plans → M builds" below.

When handing a step to a lesser model, start the session with this:

> Implement Phase N step K from docs/apple-silicon/ROADMAP.md, following
> spec 04/05. Done = ctest + G1/G2 parity + ASan pass, benchmark recorded.
> Update the tracking row in docs/apple-silicon/10-model-plan.md.

## 4. Phase 1 — "Fast CPU" (spec 04)

| Step | Work | Model | Why |
|------|------|:-----:|-----|
| 1.1 | `src/Platform.h`: core counts, QoS, feature detection | **M** | Small, well-specified `sysctl` and pthread calls, with portable fallbacks |
| 1.2 | Accelerate single-threaded per call + startup log line | **L** | A guarded API call and a printf. G1 catches any mistake |
| 1.3 | Default `-t` and QoS for P/E cores | **M** | Touches search thread startup. Needs care, but the scope is narrow |
| 1.4a | NEON-friendly Winograd transforms | **S** | Hot-loop numerics. Rearranging float ops changes rounding; needs judgment on vectorization vs exactness |
| 1.4b | Fused BN + ReLU + residual pass | **M** | Mechanical loop fusion, fully covered by G1 |
| 1.5 | Gate G1 + benchmark + BENCHMARKS.md | **L** | Run the scripts and record the numbers |
| 1.R | Phase review before tag `as.1` | **S** | `/code-review high` on the phase diff |

Expected split: about 70% of Phase 1 on M/L.

## 5. Phase 2 — "Metal" (spec 05)

| Step | Work | Model | Why |
|------|------|:-----:|-----|
| 2.1 | Extract `BatchQueue.h` from `OpenCLScheduler` | **S** | Moves mutex/condition-variable batching logic shared by two backends. Races here are rare and invisible to tests |
| 2.2 | `MetalContext` + `USE_METAL` CMake option | **M** | Boilerplate Objective-C++ (device, queue, runtime shader compile) |
| 2.3 | `MetalNetwork` via MPSGraph (fp32, BN folding) | **S plans → M builds** | The graph code is straightforward with a plan. Getting BN folding and tensor layouts right is the risk, and G2 catches it |
| 2.4 | `MetalScheduler`: zero-copy slot ring, triple buffering, completion wakeups | **S** | The hardest step: GPU/CPU synchronization over unified memory, a slot state machine, and wakeup latency |
| 2.5 | fp16 + `--precision auto` + `USE_METAL_SELFCHECK` | **M** | Follows the existing OpenCL patterns. G2 tolerances catch errors |
| 2.6 | `--backend` flag, Metal default on macOS | **L** | CLI option plumbing |
| 2.7 | Autotune batch size/precision + cache file | **M** | Timing loop plus a small file format |
| 2.8 | (optional) `forward_into` zero-staging input | **S** | Changes the `ForwardPipe` interface across backends |
| 2.9 | Benchmark; decide on the MSL Winograd fallback | **L** runs, **S** decides | Numbers are mechanical. The fallback decision and any custom kernels are S work |
| 2.11 | `MetalWinograd`: MSL Winograd transforms + `simdgroup_matrix` batched GEMM | **S plans → M builds** | Kernel port from the OpenCL template is mechanical with a plan; the GEMM tiling and fp16 accumulation choices are the risk, and G2 catches errors |
| 2.10 | CI `build-metal`, nightly parity, ASan | **L** | YAML |
| 2.R | Phase review + 2-hour soak before tag `as.2` | **S** | Review, and read the soak/Instruments results |

Expected split: about half the steps on M/L. The critical path (2.1, 2.4)
stays on S.

## 5b. Phase 3 — "Train on Mac" (spec 06, plan 13)

| Step | Work | Model | Why |
|------|------|:-----:|-----|
| 3.1 | Scaffold `training/mlx` (uv, pytest, CI job) | **L** | Boilerplate |
| 3.2 | `model.py` + T1 | **S** | Layer-by-layer parity with tfprocess (BN, flatten order, init, loss) decides everything downstream |
| 3.3 | weights I/O + convert, T2/T3 | **M** (S reviews the gate) | Mechanical given plan 13 §0; the gates catch layout bugs |
| 3.4 | Shared-memory data pipeline | **S** | Multiprocess shared memory, backpressure and cleanup on macOS `spawn` |
| 3.5 | `train.py` (compile, dtypes, macrobatch, checkpoints, guard) | **M** (S reviews compile + guard) | Mostly CLI and plumbing; stateful `mx.compile` and memory need a careful eye |
| 3.6 | SWA + BN refinement | **M** | Small, fully specified (F9) |
| 3.7 | T4, T5 | **S** | Judging learning curves and profiles |
| 3.8 | T6 match | **L** run, **S** judge | Long unattended run |
| 3.R | Phase review before `as.3` | **S** | `/code-review high` |

## 6. Tracking log

Update a row when a step lands. Status: ⬜ todo · 🔄 in progress · ✅ done · ⏫ escalated.

### Phase 1

| Step | Planned | Used | Status | Gates passed | Commit | Notes |
|------|:-------:|:----:|:------:|--------------|--------|-------|
| 1.1 | M | Sonnet 5.5 | ✅ | ctest (new `PlatformTest`), ASan+UBSan clean | `7fafb4b` | `src/Platform.h`; shipped together with 1.2 and 1.3 |
| 1.2 | L | Sonnet 5.5 | ✅ | G1 3.7e-7 vs pre-1.4 build, ASan clean | `7fafb4b` | Done by Sonnet, not Haiku, because it was bundled with 1.1/1.3. Startup log `BLAS Core: Apple Accelerate (SME)` |
| 1.3 | M | Sonnet 5.5 | ✅ | ctest, ASan clean | `7fafb4b` | QoS on search threads + P/E core log. The spec's `-t 7` default was measured and **not adopted** (no throughput gain) |
| 1.4a | S | Opus 5.5 | ✅ | ctest, G1 vs pre-change 3.7e-7 and vs Eigen 3.4e-7, scalar (MSVC) fallback G1 3.0e-9, ASan+UBSan clean, x86-64 SSE compile | `004f7d6` | 1.26× single-thread, 1.13× at 10 threads (random 15b×192). Now store-bound; a `[tile][channel]` V layout was tried and was 9% slower (see BENCHMARKS.md) |
| 1.4b | M | Sonnet 5.5 | ✅ | ctest, G1 vs pre-1.4 build 3.7e-7 and vs Eigen 3.4e-7, ASan+UBSan clean | `aec4bd0` | Speed-neutral (±1%); kept for less memory traffic and one fewer duplicate function |
| 1.5 | L | Opus 5.5 | ✅ | G1 on real 15b×192 and 40b×256 (winrate 4.2e-7 / 3.3e-6, tol 1e-5); real-network baseline in BENCHMARKS.md: Accelerate 2.3× Eigen (exit ≥ 1.5×) | `c44465b` (measured), this commit | Done by Opus with step 2.R/2.11d. Metal on real nets: 1.22× OpenCL on 15b×192 at 1600 visits (1.38× at 6400), 1.34× on 40b×256 |
| 1.R | S | Opus 5.5 | ✅ | `/code-review high` on 7fafb4b/004f7d6/aec4bd0: 8 findings, 7 fixed; CPU output bit-identical to before (6b×64); ctest; CPU speed unchanged (15b×192: 242/173/83 n/s at -t 10/4/1, medians of 3) | `9f0ecca` | Real bugs: Accelerate single-threading was set on the main thread only (per-thread setting), so pool threads ran multi-threaded sgemm (no measurable speed effect); the vectorized input transform formed a pointer past the end of the 18-plane input. Skipped: hoisting the residual branch (cleanup) |

### Phase 2

| Step | Planned | Used | Status | Gates passed | Commit | Notes |
|------|:-------:|:----:|:------:|--------------|--------|-------|
| 2.1 | S | Opus 5.5 | ✅ | ctest, OpenCL↔CPU parity 1.8e-7, TSan (batching + drain): 0 races | `5a4182f` | Also fixed a spurious-wakeup race in forward() |
| 2.2 | M | Sonnet 5.5 | ✅ | ctest (new `MetalContextTest` compiles and runs an MSL kernel on the M4 GPU via shared buffers), default build still green | `bce8066` | Pure-C++ header, ObjC++ only in `src/metal/*.mm` (ARC). `USE_METAL` defaults OFF and errors off-Apple |
| 2.3 | S→M | Opus 5.5 plan → Sonnet 5.5 build | ✅ | 2.3a: G1 diff 0 (exact), OpenCL old/new diff 0, ASan. 2.3b–d: G2 vs CPU 3.1e-7 (15b×192) and 1.8e-7 (6b×64), unit test (6e-7; mutation check confirmed it catches a BN-fold bug), ASan+UBSan with Metal, 4 build combinations | `4732b73` (2.3a), `6778cca` (2.3b–d) | Plan: [11-plan-2.3-mpsgraph.md](11-plan-2.3-mpsgraph.md). Followed as written, no escalation. Per-batch-size graphs (not symbolic batch), built lazily |
| 2.4 | S | Opus 5.5 (extra-high) | ✅ | ctest 18/18; scheduler stress test (12 threads, 1,800 distinct evals, 0 wrong; a row-offset mutation makes it fail); drain/resume test; TSan 0 races (unit tests, 16-thread search, GTP session); ASan+UBSan; G2 unchanged | `5b20383` | Worker-owned slots, shared executables, 2 graphs, defaults W=2 / B=8 from a sweep (ADR-007). 1.65× CPU on 15b×192, 1.88× on 6b×64 |
| 2.5 | M | Sonnet 5.5 | ✅ | ctest 20/20; G2 fp32 ≤ 1e-4 and fp16 at N6 (1e-2/5e-3) on both nets; fp16 concurrency stress test; ASan+UBSan; TSan 0 races; self-check run with probability 1 (0 mismatches); CPU, OpenCL, OpenCL+HALF and Metal+OpenCL builds | `0fe6189` | Found by measurement: the first auto benchmark picked the wrong precision (sequential, ordering bias), fixed with per-worker streams + alternating medians. Also found MPSGraph can use the ANE (~2× on 15b×192) but with a 325 s first compile and stdout pollution: pinned to GPU-only, follow-up in ADR-004 |
| 2.6 | L | Sonnet 5.5 | ✅ | Backend selection checked in 4 builds (CPU-only, OpenCL-only, Metal-only, Metal+OpenCL): default, explicit, conflict, unknown and not-built-in cases; ctest 20/20 (Metal) and 14/14 (CPU); scripts exercised end to end (`build.sh`, `start.sh`, `dist` build) | `52d9641` | Done by Sonnet, not Haiku, in the same session as 2.5/2.7. Replaced `cfg_cpu_only` with `cfg_backend`; `--gpu` is ignored with a warning when Metal is chosen in a Metal+OpenCL build |
| 2.7 | M | Sonnet 5.5 | ✅ | ctest 26/26 (chooser, cache round trip, key isolation and garbage tolerance, GPU measurement; mutation checks on the 5% threshold and the cache replace logic fail as they should); ASan+UBSan clean; TSan 0 warnings (unit tests and an autotuned search); modes checked by hand (cold/cached, `--tune-only`, `--batchsize`, `-t`, `--precision single/half`, `--backend cpu`); CPU, OpenCL and Metal+OpenCL builds | `ffd306e` | End to end +21% (6b×64) and +12% (15b×192) over the 2.4 defaults. Design in ADR-008: the cache holds speed only; the fp16 accuracy gate runs every start |
| 2.5b | M | Sonnet 5.5 | 🟡 | ctest passes without ANE; ANE unit tests pass with LZ_TEST_ANE=1 (small net, probably not placed on the ANE); 15b×192 ANE run 803 n/s, stdout clean | `169924b`, `f5af9dc` | `--ane` opt-in, see ADR-004 addendum. Not done: G2 on ANE, TSan/ASan, abort at exit of the ANE benchmark (`mutex lock failed`; not seen without ANE) |
| 2.8 | S | Opus 5.5 | ✅ | Profile (`sample`, 6b×64, autotuned defaults): 98.5% of thread samples waiting on the GPU, copies 0.05% | `bcc973c` | Decided not to implement: no measurable gain possible, and it would reintroduce the shared slot state ADR-007 avoided |
| 2.9 | L / S | Opus 5.5 | ✅ | 3 interleaved rounds, OpenCL tuned, on random 15b×192 and 40b×256: Metal 0.75× OpenCL on both (target 2–2.5×) | `991b3a5` | Decision: start the MSL Winograd backend (ADR-009). Found and fixed an OpenCL startup hang from step 2.7 (`d0ebb96`) |
| 2.11 | S plans → M builds | Opus 5.5 (plan) → Sonnet 5.5 (build) | ✅ | 2.11a ✅: unit test vs CPU (5–7e-7, 4 shapes, batch = single), 2 mutation checks, G2 fp32 on 3 shapes (≤ 8.3e-6), 1.13× / 1.21× OpenCL. 2.11b ✅: fp16 unit test (≤ 2.1e-3 incl. C=256), G2 fp16 at N6 on 3 shapes (≤ 8.0e-3 on 20×256), 1.50× / 1.52× OpenCL. 2.11c ✅: engine in the tuning key + choice, `--metal-kernels`, G2 on 3 shapes × 2 engines × 2 precisions in `full_gate.sh` (so nightly), scheduler stress tests per engine, ASan/UBSan 36 tests, TSan 0 warnings (unit tests and real searches), CPU/OpenCL/Metal+OpenCL builds; default is now 1.57× / 1.52× OpenCL and 2× the old default | `393df9b` (2.11a), `f09d5de` (2.11b), `29ef320` (2.11c) | 2.11d (Opus 5.5): profile (GEMM ~65–70% of GPU time) and a custom `simdgroup_matrix` GEMM go/no-go: 0.84–0.94× MPS, not adopted; the GEMM result is fp32 since `558de70` (real-net accuracy). Spec target (2× / 2.5× OpenCL) not reachable on this path; 2.11e (fusions, ~30% of GPU time) is the only lever left |
| 2.10 | L | Opus 5.5 | ✅ | CI job `macOS arm64 / Metal`: build + ctest + G2 when the runner has a GPU; nightly `parity-full` (3 shapes × 2 engines × 2 precisions) and Metal ASan + UBSan + API validation green on GitHub (run 37257334371) | `7631ddc`, `ba53da7` | The CI parity step compares fp32 only (`50903c8`); fp16 is gated in the nightly |
| 2.R | S | Opus 5.5 | ✅ | `/code-review high` on the Phase 2 source diff: 10 findings, 9 fixed; ctest; G2 on 6×64 and 15×192 (both engines and precisions); soak `scripts/macos/soak.py` at -t 16 on 15b×192: 120 min, 12 games, no crash, RSS returns to ~120 MB after every game (peak 638 MB), 0 leaks (`leaks`, 20-min run) | `141f385` | Real bugs: NaN passed the fp16 check and the parity script; a fixed batch size took its precision from other batch sizes; an fp32 fallback kept the fp16 engine/batch; the CI probe matched the CPU-fallback error; the MSL hardcoded 19×19; unit tests wrote the user's tuning cache. Phase exit met after ADR-010/011: 1.22× / 1.34× OpenCL on real nets, G3 (1,330 self-check moves) and G4 (19/20, near-tie rule) pass on the real 15b×192 net |

### Phase 3

| Step | Planned | Used | Status | Gates passed | Commit | Notes |
|------|:-------:|:----:|:------:|--------------|--------|-------|
| plan | S | Opus 5.5 | ✅ | — | this commit | [13-plan-phase3-mlx.md](13-plan-phase3-mlx.md); decisions D1–D3 resolved in ADR-012 |
| 3.1 | L | Opus 5.5 | ✅ | `uv run pytest` (MLX 0.32.3 on the M4 GPU); CI job `macOS arm64 / MLX trainer` | this commit | Done by Opus in the planning session (small) |
| 3.2 | S | Opus 5.5 | ✅ | T1: 10 pytest tests — loss = NumPy reference (NCHW, written from tfprocess) ≤ 1e-5 in train and eval mode, parameter count = leelaz file size (1×8, 6×64), NCHW head flatten, TF batch-norm running stats (unbiased, 0.99), Xavier truncated init, bf16 forward; mutation checks (NHWC flatten, biased running variance) fail as they should | this commit | Custom Conv/Linear/BatchNorm layers cast fp32 master weights to the compute dtype per call (nn.Conv2d would promote bf16 back to fp32) |
| 3.3 | M | | ⬜ | | | |
| 3.4 | S | | ⬜ | | | |
| 3.5 | M | | ⬜ | | | |
| 3.6 | M | | ⬜ | | | |
| 3.7 | S | | ⬜ | | | |
| 3.8 | L/S | | ⬜ | | | |
| 3.R | S | | ⬜ | | | |
