# 02 — Target Architecture

## 1. Big picture

```
                         ┌───────────────────────── leelaz (C++17 / ObjC++) ─────────────────┐
 GTP / autogtp  ───────► │ UCTSearch threads ──► Network ──► NNCache                          │
                         │                           │                                         │
                         │                           ▼                                         │
                         │                ForwardPipe (interface, unchanged API)               │
                         │       ┌───────────────┬────────────────────┬──────────────────┐     │
                         │       ▼               ▼                    ▼                  ▼     │
                         │   CPUPipe        MetalScheduler      OpenCLScheduler    (CoreMLPipe)│
                         │  Accelerate      ★ NEW, default      legacy, opt-in      experimental│
                         │  sgemm on SME    on macOS                                          │
                         │       │               │                    │                         │
                         │       │      BatchQueue<Entry> ◄──── extracted from OpenCLScheduler  │
                         └───────┼───────────────┼─────────────────────────────────────────────┘
                                 ▼               ▼
                    ┌──────────────────── Unified memory (LPDDR5X, 16 GB) ───────────────────┐
                    │  CPU P/E cores    │   GPU (10 cores)   │   Neural Engine (16 cores)       │
                    │  SME / NEON       │   Metal / MPSGraph │   Core ML only                   │
                    └──────────────────────────────────────────────────────────────────────────┘

 Training (Python 3.12)
   chunks *.gz ──► chunkparser (multiprocess, numpy) ──► shared-memory ring ──► MLX model on GPU
                                                                         └──► export leelaz weights
```

## 2. Key design principles

1. **Keep `ForwardPipe` as the seam.** All acceleration goes behind the existing
   interface (`initialize`, `push_weights`, `forward`, `drain`, `resume`).
   `Network`, `UCTSearch` and `GTP` change only to add backend selection.
2. **Extract, don't duplicate, the batching logic.** `OpenCLScheduler::batch_worker`
   holds an adaptive batching algorithm that doesn't depend on any device. It
   moves into `BatchQueue` (header-only template), which both OpenCL and Metal
   then use. This is the only refactor of shared code.
3. **Unified memory, used deliberately:**
   - *Inference*: each batch slot is a region of a `MTLStorageModeShared`
     buffer. Search threads write their 18×361 input planes **directly** into
     the slot (no `std::vector` staging), and the GPU reads the same pages.
     Policy and value outputs are read in place after a completion handler or
     `MTLSharedEvent` fires. Weights are uploaded once into `Private` storage,
     where the GPU can use its optimal layout and compression.
   - *Training*: MLX arrays live in unified memory, so there is no
     `.to(device)` step. The data loader produces numpy batches in a
     `multiprocessing.shared_memory` ring, which MLX wraps without an extra
     host→device transfer. Optimizer state, weights and activations share one
     pool, so batch size is limited by total RAM rather than by VRAM.
4. **Precision follows the hardware.** The M4 GPU runs fp16 at about twice the
   fp32 rate. Default: fp16 storage and math with fp32 accumulation for
   inference, and bf16 mixed precision for training. Both stay within the N6
   tolerances.
5. **Apple frameworks over hand-written kernels where they are competitive.**
   Use MPSGraph for conv/BN/ReLU fusion and Accelerate for CPU GEMM, which picks
   SME on M4. Hand-written MSL Winograd kernels are a fallback, used only if
   MPSGraph can't hit N1 (see ADR-001).
6. **Fence everything.** CMake options `USE_METAL`, `USE_ACCELERATE`, `USE_OPENCL`
   and `USE_COREML`, with `__APPLE__` guards in sources. Linux and Windows
   builds are byte-for-byte unaffected when the options are off.

## 3. Backend selection logic

```
--backend given? ── yes ──► use it (error if not compiled in)
        │ no
        ▼
USE_METAL && MTLCreateSystemDefaultDevice() != nil ──► metal
        │ else
        ▼
USE_OPENCL && a device is found ──► opencl
        │ else
        ▼
cpu
```

`--precision auto` on Metal benchmarks fp16 against fp32 at startup, like the
existing OpenCL `needs_autodetect()` path, and keeps fp16 if it passes the
self-check tolerance.

## 4. Threading model on M4 (4P + 6E)

| Work | Placement |
|------|-----------|
| MCTS search threads | Default `-t` = P-cores + E-cores/2. Threads use QoS `USER_INTERACTIVE` so they land on P-cores first |
| Metal batch workers | 2 per device (one encodes while one is in flight) for double buffering |
| CPUPipe GEMM | Accelerate is single-threaded per call (`BLAS_THREADING` off), because the parallelism comes from search threads, the same way `openblas_set_num_threads(1)` is used today |
| Training data loader | Processes on E-cores (`taskpolicy -b` or `QOS_CLASS_UTILITY`), so P-cores stay free to feed the GPU |

## 5. Memory budget (16 GB machine)

| Consumer | Inference | Training (20b×256) |
|----------|-----------|--------------------|
| Network weights | 40b×256 fp16 ≈ 47 MB | fp32 master + bf16 copy + momentum ≈ 280 MB |
| Activations | batch 32 ≈ 30 MB | batch 256 with autograd ≈ 4–6 GB |
| NNCache | `--cache-size` (existing heuristic, capped at 10% RAM) | — |
| Shuffle buffer | — | Configurable. Default 250k positions ≈ 1.6 GB (uint8-packed planes) |
| Search tree | Grows with playouts. Typical < 2 GB | — |
| Headroom for OS | ≥ 4 GB | ≥ 4 GB |

## 6. Repository layout after the port

```
src/
  BatchQueue.h              (new, extracted)
  metal/MetalScheduler.{h,mm}   (new, Objective-C++)
  metal/MetalNetwork.{h,mm}     (new, MPSGraph builder)
  metal/kernels/*.metal         (new, embedded as strings; fallback kernels)
  coreml/CoreMLPipe.{h,mm}      (new, experimental, behind USE_COREML)
training/
  tf/                       (unchanged)
  mlx/                      (new: model.py, data.py, train.py, export.py, import.py, tests/)
docs/apple-silicon/         (this spec set + BUILD.md + BENCHMARKS.md)
.github/workflows/macos-arm64.yml (new)
cmake/Modules/FindMetal.cmake     (new)
```
