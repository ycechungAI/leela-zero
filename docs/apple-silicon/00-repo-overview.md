# 00 — Repository Overview (as of `next` @ 3ee6d20, `v0.17-31`)

Leela Zero is a Go engine that reimplements AlphaGo Zero. It is a C++14 engine
(`leelaz`), plus Qt tools for distributed self-play (`autogtp`) and match
validation (`validation`), plus a Python TensorFlow 1.x training pipeline.

## 1. Components

```
leela-zero/
├── src/                 C++ engine (≈11k LOC, excluding Eigen)
│   ├── Leela.cpp        main(), CLI parsing (boost::program_options)
│   ├── GTP.cpp          GTP protocol, config globals (cfg_*)
│   ├── UCTSearch.cpp    MCTS (PUCT), multi-threaded via ThreadPool
│   ├── UCTNode*.cpp     tree nodes, lock-free virtual loss
│   ├── Network.cpp      weight loading, NN eval dispatch, symmetries, NNCache
│   ├── ForwardPipe.h    ★ backend interface (initialize / forward / push_weights)
│   ├── CPUPipe.cpp      CPU backend: Winograd F(4x4,3x3) + BLAS/Eigen sgemm
│   ├── OpenCL*.cpp      GPU backend: OpenCL kernels + CLBlast-derived sgemm
│   ├── OpenCLScheduler  batching scheduler: queue → batch_worker threads
│   ├── Tuner.cpp        OpenCL sgemm auto-tuner (writes leelaz_opencl_tuning)
│   ├── kernels/*.opencl OpenCL C sources (stringified into the binary)
│   ├── Training.cpp     dumps self-play training chunks (gzip text)
│   ├── Eigen/           git submodule (eigen-git-mirror, 3.3.x)
│   └── tests/           gtest unit tests
├── gtest/               git submodule (googletest)
├── autogtp/             Qt5 client: fetches nets/jobs from server, runs self-play
├── validation/          Qt5 SPRT match runner
├── training/tf/         TF 1.x trainer (parse.py, tfprocess.py, chunkparser.py …)
├── training/{elf,minigo,caffe}/  weight converters and legacy prototxt
├── Dockerfiles/         Ubuntu 16.04 CPU/GPU images (Travis)
├── msvc/, appveyor.yml  Windows build
└── .travis.yml          Linux CI (docker)
```

## 2. Neural network

- **Input**: 18 planes × 19×19 (8 history steps × own and opponent stones, plus
  2 side-to-move planes). `INPUT_CHANNELS = 2*INPUT_MOVES + 2`.
- **Tower**: 3×3 conv input block, then *N* residual blocks of *F* filters
  (conv-BN-ReLU-conv-BN-add-ReLU). Public networks range from 6b×64f to 40b×256f.
- **Policy head**: 1×1 conv (2 filters) → BN → ReLU → FC → 362 logits (361 points + pass).
- **Value head**: 1×1 conv (1 filter) → BN → ReLU → FC 256 → ReLU → FC 1 → tanh.
- **Weights file**: text, version `1` (or `2` for ELF-derived nets with a
  different value-head encoding), optionally gzipped. BN `beta` is stored as a
  pre-BN conv bias (`beta * sqrt(var + 1e-5)`). Convs are stored as
  `[out, in, kh, kw]` and FC layers as `[out, in]`.
- **Inference precision**: fp32, or fp16 storage/compute on OpenCL (`USE_HALF`).
  Precision is auto-detected unless forced.

## 3. Inference data flow

```
UCTSearch worker threads (cfg_num_threads)
   │   Network::get_output()  → NNCache lookup (hit → return)
   ▼
Network::get_output_internal() → gather 18 planes (+ random symmetry)
   ▼
ForwardPipe::forward(input, out_pol, out_val)    ← blocking call per position
   ├── CPUPipe          : synchronous, single position, Winograd + sgemm
   └── OpenCLScheduler  : enqueue entry; batch_worker forms a batch of
                          up to cfg_batch_size (waits ≤ m_waittime ms),
                          copies host→device, runs kernels, copies back,
                          signals each entry's condition_variable
   ▼
policy softmax (temperature) + value → UCTNode expansion
```

The batching scheduler in `OpenCLScheduler` (queue, adaptive wait time,
single-eval fallback, drain/resume) does not depend on OpenCL. It is the
natural piece to reuse for a Metal backend (see spec 05).

## 4. Training data flow

```
leelaz self-play (autogtp) → Training.cpp → *.gz chunks (v1 text: 16 hex plane lines,
                                               side-to-move, 362 probs, winner)
   ▼
training/tf/chunkparser.py  (multiprocessing, 8 symmetries, shuffle buffer)
   ▼  serialized bytes via tf.placeholder(tf.string) + decode_raw
training/tf/tfprocess.py    TF1 graph: NCHW conv, fused BN, MomentumOptimizer
                            (lr 0.05, nesterov, momentum 0.9), L2 1e-4,
                            loss = policy CE + value MSE + reg, SWA, fp16 loss scaling
   ▼
save_leelaz_weights()  → leelaz-format text weights
```

## 5. Where it breaks on an M4 Mac mini (verified on this machine)

| # | Problem | Evidence | Fixed in |
|---|---------|----------|---------|
| B1 | `cmake_minimum_required(VERSION 3.1)` is rejected by CMake 4.x | `Compatibility with CMake < 3.5 has been removed` (cmake 4.4.3) | Spec 03 |
| B2 | Submodules `src/Eigen` and `gtest` are not initialized, and `eigen-git-mirror` is archived | `git submodule status` shows `-` prefix | Spec 03 |
| B3 | Hard-coded include path `/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers` no longer exists, and Accelerate is never linked explicitly | `ls` shows no such directory | Spec 03/04 |
| B4 | `find_package(OpenCL REQUIRED)` is unconditional, even for `USE_CPU_ONLY`. OpenCL is deprecated on macOS and runs as a translation layer over Metal on Apple Silicon | CMakeLists.txt L28 | Spec 03 |
| B5 | Boost and Qt are not installed. The tools require **Qt5** (`Qt5::Core`), which is legacy on Homebrew | `brew list` empty | Spec 03/08 |
| B6 | TF 1.x training needs TF ≥1.12 graph mode, `tf.contrib`-era APIs and NCHW conv (GPU-only in TF). There is no TF1 wheel for macOS arm64, and the pinned `numpy==1.13.3` / `protobuf==3.4.0` don't build on Python ≥3.9 | requirements.txt | Spec 06 |
| B7 | Old Accelerate CBLAS API is used. The new `ACCELERATE_NEW_LAPACK` interface (macOS 13.3+) is preferred and is the path that gets SME-tuned kernels on M4 | CPUPipe.cpp, Network.cpp | Spec 04 |
| B8 | CI covers only Linux x86 (Travis, now defunct for OSS) and Windows (AppVeyor). There is no macOS or arm64 coverage | `.travis.yml`, `appveyor.yml` | Spec 08 |
| B9 | System Python is 3.9.6 and no ML frameworks are installed | `python3 --version` | Spec 06 |

Works as-is: Apple clang 21 accepts `-march=native` and `-mcpu=apple-m4`. Eigen
3.3 has NEON paths. The engine has no x86 intrinsics (`immintrin`/`_mm_*`), so
there is no SSE code to port.

## 6. Host facts (this machine)

- Apple M4 Mac mini, `arm64`, macOS Darwin 27, **16 GB unified memory**
- 10-core CPU (4 performance + 6 efficiency), 10-core GPU, 16-core Neural Engine
- About 120 GB/s memory bandwidth, shared by CPU, GPU and ANE
- Toolchain: Apple clang 21, CMake 4.4.3, Homebrew at `/opt/homebrew`
- The Xcode `metal` CLI compiler is **not** installed (Command Line Tools only).
  Spec 05 therefore compiles Metal shaders from source at runtime, so full Xcode
  is not a build requirement.
