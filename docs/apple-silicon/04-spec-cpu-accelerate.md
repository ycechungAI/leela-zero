# 04 — Spec: CPU Path (Accelerate, NEON, P/E Cores)

**Release:** `as.1`. **Depends on:** 03. **Blocks:** nothing. The CPU path is
also the numeric reference for 05 and 06.

## 1. Why the CPU path still matters

- It is the **reference implementation** for all correctness gates (spec 07).
- It is the fallback when Metal is unavailable (CI VMs without a GPU, headless
  sandboxes).
- On M4 the CPU has SME (Scalable Matrix Extension) units. Accelerate's
  `cblas_sgemm` uses them, and they give substantial fp32 GEMM throughput.
  Hand-writing SME is a non-goal; we get it through Accelerate.

## 2. Current state

`CPUPipe.cpp` runs Winograd F(4×4, 3×3) transforms in scalar C++, then one
`sgemm` per Winograd tile position (36 batched GEMMs) via either
`cblas_sgemm` (`USE_BLAS`) or Eigen. The heads use `cblas_sgemv` or Eigen. On
Apple, `USE_BLAS` already pulls in `<Accelerate/Accelerate.h>`, but the build
does not link it correctly (B3).

## 3. Changes

| # | Change | Notes |
|---|--------|-------|
| C1 | Make Accelerate the default BLAS on Apple (`USE_ACCELERATE` implies `USE_BLAS`) | Define `ACCELERATE_NEW_LAPACK` before the include. Calls stay `cblas_sgemm`. No signature changes are needed for LP64 |
| C2 | Single-threaded BLAS per call | Use the `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` API (macOS 15+), guarded by `__builtin_available`. This mirrors `openblas_set_num_threads(1)`: parallelism comes from search threads, and nested threading oversubscribes the CPU |
| C3 | Startup log line | `BLAS Core: Apple Accelerate (SME)` or `(AMX)`, detected with `sysctlbyname("hw.optional.arm.FEAT_SME")` |
| C4 | Winograd transforms vectorized | Rewrite `winograd_transform_in/out` inner loops so clang auto-vectorizes them to NEON. Check with `-Rpass=loop-vectorize`. Fallback: explicit `<arm_neon.h>` behind `__ARM_NEON`. Expected gain is 10–20% on small nets, where transforms are a large share of the time |
| C5 | BN+ReLU+residual fused pass | Today these run as separate passes over memory. One fused loop cuts memory traffic. The change is mechanical and backend-independent |
| C6 | Default thread count (**measured, not adopted**: `-t 7` was no faster than `-t 10` in nodes/s, see BENCHMARKS.md; the default stays at all logical CPUs and the startup log reports the P/E split) | `cfg_num_threads` default on Apple = `hw.perflevel0.physicalcpu + hw.perflevel1.physicalcpu / 2` (=7 on the base M4) instead of `hardware_concurrency()` (=10), because E-cores slow the MCTS tail. The value stays tunable via `-t` |
| C7 | QoS | Search threads call `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` so the scheduler favors P-cores |
| C8 | Optional fp16 CPU path | **Deferred.** The M4 CPU supports FP16 arithmetic, but the Accelerate sgemm path is fp32. Revisit only if the CPU becomes a release target for strength play |

## 4. Interfaces

No changes to `ForwardPipe` or `CPUPipe`'s public API. Helpers go into a new
internal header `src/Platform.h`: `num_perf_cores()`, `num_eff_cores()`,
`set_thread_qos_interactive()` and `cpu_feature_string()`. Each one has
portable fallbacks.

## 5. Acceptance criteria

- [ ] Gate G1 (spec 07): CPU Accelerate output equals CPU Eigen output within
      1e-5 for 100 random positions on 3 networks.
- [ ] `leelaz --benchmark` (spec 07 protocol) on a 15b×192 net: ≥ 1.5× the
      Eigen-only nodes/s.
- [ ] `-t` default reported in the startup log, with core counts.
- [ ] No regression in the Linux OpenBLAS CI job.
