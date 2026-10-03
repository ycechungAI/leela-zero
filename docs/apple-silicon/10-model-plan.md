# 10 — Model Plan & Tracking (Phases 1–2)

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
| 2.10 | CI `build-metal`, nightly parity, ASan | **L** | YAML |
| 2.R | Phase review + 2-hour soak before tag `as.2` | **S** | Review, and read the soak/Instruments results |

Expected split: about half the steps on M/L. The critical path (2.1, 2.4)
stays on S.

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
| 1.5 | L | | ⬜ | | | |
| 1.R | S | | ⬜ | | | |

### Phase 2

| Step | Planned | Used | Status | Gates passed | Commit | Notes |
|------|:-------:|:----:|:------:|--------------|--------|-------|
| 2.1 | S | Opus 5.5 | ✅ | ctest, OpenCL↔CPU parity 1.8e-7, TSan (batching + drain): 0 races | `5a4182f` | Also fixed a spurious-wakeup race in forward() |
| 2.2 | M | Sonnet 5.5 | ✅ | ctest (new `MetalContextTest` compiles and runs an MSL kernel on the M4 GPU via shared buffers), default build still green | `bce8066` | Pure-C++ header, ObjC++ only in `src/metal/*.mm` (ARC). `USE_METAL` defaults OFF and errors off-Apple |
| 2.3 | S→M | Opus 5.5 plan → Sonnet 5.5 build | ✅ | 2.3a: G1 diff 0 (exact), OpenCL old/new diff 0, ASan. 2.3b–d: G2 vs CPU 3.1e-7 (15b×192) and 1.8e-7 (6b×64), unit test (6e-7; mutation check confirmed it catches a BN-fold bug), ASan+UBSan with Metal, 4 build combinations | `4732b73` (2.3a), `6778cca` (2.3b–d) | Plan: [11-plan-2.3-mpsgraph.md](11-plan-2.3-mpsgraph.md). Followed as written, no escalation. Per-batch-size graphs (not symbolic batch), built lazily |
| 2.4 | S | | ⬜ | | | |
| 2.5 | M | | ⬜ | | | |
| 2.6 | L | | ⬜ | | | |
| 2.7 | M | | ⬜ | | | |
| 2.8 | S | | ⬜ | | | |
| 2.9 | L / S | | ⬜ | | | |
| 2.10 | L | | ⬜ | | | |
| 2.R | S | | ⬜ | | | |
