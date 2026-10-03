# Leela Zero on Apple Silicon: Build, Run, Debug, Train

How to build and use Leela Zero on an arm64 Mac (M1–M4) with this fork.
Tested on a Mac mini M4 (16 GB) running macOS 27, Apple clang 21, CMake 4.4
and Boost 1.92.

> **Current state (Phase 0):** CPU (Accelerate) and OpenCL backends build and
> pass all tests. The Metal backend (Phase 2) and the MLX trainer (Phase 3) are
> not written yet. See [ROADMAP.md](ROADMAP.md).

---

## 1. Quick start

```bash
# one-time setup
xcode-select --install          # Command Line Tools (full Xcode not needed)
brew install cmake boost
git clone https://github.com/ycechungAI/leela-zero
cd leela-zero
git checkout as/p0-build

# build + run unit tests
scripts/macos/build.sh

# check the engine end to end, without downloading a network
scripts/macos/debug.sh smoke
```

`build.sh` initializes the submodules (Eigen, googletest) if needed, builds
`build/leelaz` and `build/tests`, and runs the tests. A good run ends with
`[  PASSED  ] 13 tests.`

## 2. Scripts

Everything lives in `scripts/macos/`. Pass `-h` to any script for its usage.

| Script | Purpose |
|--------|---------|
| `build.sh [cpu\|opencl\|debug\|asan\|dist] [--clean] [--no-test]` | Configure, build and test one configuration |
| `start.sh [-w net] [-b cpu\|opencl] [-t threads] [-v visits] [-- leelaz args]` | Run `leelaz` in GTP mode, for GUIs or by hand |
| `debug.sh lldb\|asan\|tests\|smoke` | Debugger, sanitizers, unit tests under lldb, quick smoke test |
| `train.sh selfplay\|sgf\|split\|fit` | Training-data workflow (section 6) |
| `make_random_net.py out.txt` | Writes a random-weights network for tests (it plays nonsense) |
| `selfplay.py` | Local self-play driver used by `train.sh selfplay` |

Each configuration gets its own build directory, so they don't clobber each other:

| Config | Directory | Notes |
|--------|-----------|-------|
| `cpu` (default) | `build/` | Release, LTO, `-mcpu=native`, Accelerate BLAS. **Use this one** |
| `opencl` | `build-opencl/` | Release, OpenCL GPU backend. Deprecated by Apple; kept as the speed baseline |
| `debug` | `build-debug/` | `-Og -g`, no LTO, for lldb |
| `asan` | `build-asan/` | Debug + AddressSanitizer + UBSan |
| `dist` | `build-dist/` | Release with `-mcpu=apple-m1` instead of `native`, so the binary runs on any Apple Silicon Mac. Use it for binaries you hand to others |

Generated data goes in `data/` and logs go in `logs/`. Git ignores both.

### CMake presets (without the scripts)

The same configurations are available as CMake presets, which IDEs such as
CLion and VS Code pick up automatically:

```bash
cmake --list-presets
cmake --preset macos-cpu            # or macos-opencl, macos-metal (Metal backend), macos-debug, macos-asan, macos-dist
cmake --build --preset macos-cpu
ctest --preset macos-cpu            # unit tests; works from any directory
```

## 3. Get a network

`leelaz` needs a weights file. It looks for one in this order:

1. `-w <file>` on the command line
2. `$LZ_WEIGHTS` (environment variable, read by the scripts)
3. `~/.local/share/leela-zero/best-network` (the leelaz default)

Weights are text files, optionally gzipped (`.gz`); both work as-is. Sources:

- The historical Leela Zero networks were published at `https://zero.sjeng.org/`.
  The final 40-block network is the strongest. If that site is unavailable, use
  any mirror you trust.
- Leela Zero also loads converted ELF and Minigo networks (see `training/elf`
  and `training/minigo`).

To install a network as the default:

```bash
mkdir -p ~/.local/share/leela-zero
cp ~/Downloads/<network>.gz ~/.local/share/leela-zero/best-network
```

On a 16 GB M4 with the CPU backend, 15b×192 or smaller networks give the best
balance of strength and speed. 40b×256 works, but each move is slow until the
Metal backend lands.

## 4. Run

### By hand (GTP)

```bash
scripts/macos/start.sh -w ~/nets/best.gz -v 800
```

Type GTP commands such as `genmove b`, `play w Q16`, `showboard`,
`lz-analyze 50` and `quit`. Status lines go to stderr, so stdout carries pure
GTP.

One-shot example:

```bash
printf 'genmove b\nquit\n' | scripts/macos/start.sh -w ~/nets/best.gz -v 400
```

### With a GUI (Sabaki, Lizzie, GoGui, KaTrain)

Point the GUI's engine command at the script or at the binary:

```
/path/to/leela-zero/scripts/macos/start.sh -w /path/to/net.gz
```

or

```
/path/to/leela-zero/build/leelaz --gtp -w /path/to/net.gz --noponder
```

Lizzie needs `-g` (`--gtp`) and works best with pondering left on.

### Useful leelaz options

| Option | Meaning |
|--------|---------|
| `-t N` | Search threads. Default is all cores (10 on M4). Fewer can be faster with tiny nets |
| `-v N` / `-p N` | Visits / playouts per move (strength vs time) |
| `--noponder` | Don't think on the opponent's time |
| `--timemanage off` | Use the full visit budget every move |
| `--benchmark` | Fixed-workload speed test, then exit (`start.sh -- --benchmark`) |
| `-l file` | Log file. `start.sh` always writes `logs/leelaz-<time>.log` |
| `--cpu-only` | In an OpenCL build, skip the GPU |
| `--tune-only` | OpenCL: run the kernel tuner and exit |

## 5. Debug

### Smoke test (start here when something is off)

```bash
scripts/macos/debug.sh smoke              # random net
scripts/macos/debug.sh smoke -w net.gz    # your net
```

This plays two moves, prints the board, and writes `logs/smoke.log`. The log
should contain a line like `Detecting residual layers...v1...16 channels...2 blocks`.

### lldb

```bash
scripts/macos/debug.sh lldb                     # random net, 1 thread, no pondering
scripts/macos/debug.sh lldb -w net.gz -- -v 50  # extra leelaz args after --
```

Inside lldb:

```
(lldb) breakpoint set -n UCTSearch::think
(lldb) run
genmove b                  # typed into the program; the breakpoint hits
(lldb) bt
(lldb) frame variable
(lldb) continue
```

Good breakpoints:
- `Network::get_output`: NN evaluation entry
- `CPUPipe::forward`: CPU inference
- `UCTSearch::play_simulation`: one MCTS playout
- `GTP::execute`: every GTP command

Non-interactive (scripted) session:

```bash
printf 'genmove b\nquit\n' > /tmp/gtp.txt
lldb --batch -o "breakpoint set -n UCTSearch::think" -o "process launch -i /tmp/gtp.txt" \
     -o "bt" -o "kill" -- build-debug/leelaz --gtp -w logs/random-net.txt -t 1 -v 5 --noponder
```

### Unit tests under lldb

```bash
scripts/macos/debug.sh tests                    # all tests
scripts/macos/debug.sh tests 'LeelaTest.*'      # gtest filter
```

### AddressSanitizer / UBSan

```bash
scripts/macos/build.sh asan                     # build + run tests under ASan
printf 'genmove b\ngenmove w\nquit\n' | scripts/macos/debug.sh asan -- -v 20
```

ASan reports appear on stderr and abort the run (`abort_on_error=1`). Leak
detection is off by default; enable it with
`ASAN_OPTIONS=detect_leaks=1 scripts/macos/debug.sh asan`.

### Compare backends numerically

The unlisted GTP command `lz-nn-eval [symmetry]` prints the raw network output
for the current position: winrate, pass prior, then 361 priors. It runs
uncached, at full precision. `scripts/parity/compare_backends.py` runs two
engines side by side and diffs them over SGF positions and all 8 symmetries:

```bash
# Accelerate build vs an Eigen-only build (gate G1), tolerance 1e-5
python3 scripts/parity/compare_backends.py \
    --ref "build-eigen/leelaz" --test "build/leelaz" \
    -w net.gz --sgf 'data/selfplay/*.sgf' --moves 0,10,60,200 --tol 1e-5

# The same build with two networks (e.g. a re-exported net); should match exactly
python3 scripts/parity/compare_backends.py --ref build/leelaz --test build/leelaz \
    -w original.gz --test-weights reexported.txt
```

It exits 1 when a difference exceeds the tolerance, so it can be used in CI
(see `.github/workflows/apple-silicon.yml`). By hand:
`printf 'play b Q16\nlz-nn-eval 0\nquit\n' | build/leelaz --gtp -q -w net.gz`.

### Profiling

- `sample <pid> 5` gives a quick 5-second stack sample of a running leelaz. Works with just the Command Line Tools.
- **Instruments → Time Profiler** (needs full Xcode): `xcrun xctrace record --template 'Time Profiler' --launch -- build/leelaz -w net.gz --benchmark`
- For flame-graph style output, profile the `cpu` (Release) build. It keeps `-g`.

### Common problems

| Symptom | Cause / fix |
|---------|-------------|
| `Compatibility with CMake < 3.5 has been removed` | You're on upstream `next`. Use this fork's branch, which fixes it |
| `no template named 'optional' in namespace 'std'` (boost/spirit/x3) | C++14 build against new Boost. This branch builds with C++17 |
| `Could not open weights file: ../src/tests/0k.txt` | Run the tests with `ctest --preset <name>` (it sets the working directory), or run `tests` from `src/` |
| `A network weights file is required` | See section 3, or use `make_random_net.py` for testing |
| Tests or the first OpenCL run take minutes | The OpenCL tuner is running. The result is cached in `~/.local/share/leela-zero/leelaz_opencl_tuning`. Delete that file to re-tune |
| `unknown warning option '-Wno-maybe-uninitialized'` | Old CMakeLists. Fixed on this branch (GCC-only flag now) |
| lldb shows assembly, not source | You're debugging the Release (`-flto`) build. Use `build.sh debug` / `debug.sh lldb` |
| `submodule ... not initialized` errors | `git submodule update --init --recursive` |
| Eigen submodule fetch fails in an older clone | Eigen moved to GitLab: `git submodule sync && git submodule update --init --recursive` |

## 6. Training workflow

The training scripts already produce real training data. Only the final
network-fitting step is waiting on Phase 3.

```bash
# 1. Generate data by playing games locally (no server needed)
scripts/macos/train.sh selfplay -w ~/nets/best.gz -n 50 -v 800

# 1b. ...and/or convert existing SGF games (pro games, your own games)
scripts/macos/train.sh sgf ~/games/collection.sgf

# 2. Split into train/test (symlinks, 10% test by default)
scripts/macos/train.sh split

# 3. Train (needs the Phase 3 MLX trainer; prints an explanation until then)
scripts/macos/train.sh fit --blocks 6 --filters 64
```

What you get:

| Path | Contents |
|------|----------|
| `data/selfplay/game-<time>-NNNN.0.gz` | Training chunk, one per game: 19 lines per position (16 history planes, side to move, 362 move probabilities, winner). This is the same format autogtp produced |
| `data/selfplay/game-<time>-NNNN.sgf` | The game record |
| `data/supervised/<name>.N.gz` | Chunks converted from SGF |
| `data/train/`, `data/test/` | Symlinks used by the trainer |

Notes:
- Self-play speed depends on network size and visits. With a random 2-block
  net at 10 visits, one game takes about 10 s on M4. With real nets at 800
  visits, expect minutes per game on the CPU backend.
- Set `LZ_TRAIN_DATA=/some/dir` to keep data outside the repo.
- The old TF1 trainer in `training/tf` doesn't run on macOS arm64. The data
  format is shared, so chunks made here also work with it on Linux/CUDA.

## 7. autogtp and validation (Qt 6)

These are optional Qt tools: `validation` runs engine-vs-engine matches with
a statistical stopping rule (SPRT), and `autogtp` contributes self-play to a
training server.

```bash
brew install qtbase                      # Qt 6 core module only (no GUI)
cmake --preset macos-cpu                 # re-run so CMake finds Qt 6
cmake --build build --target autogtp validation
```

Local match between two networks (no server involved). Give `-o` once per
network, or the second engine falls back to the 3200-visit default:

```bash
O="-g -v 400 --noponder -t 1 -q -d -r 0 -w"
build/validation/validation -n netA.gz -o "$O" -n netB.gz -o "$O" -g 2 -k sgf-out \
    -- build/leelaz -- build/leelaz
```

It prints a running `W wins, L losses` tally and stops when the test
decides (default hypothesis 0 vs 35 Elo).

## 8. File locations

| What | Where |
|------|-------|
| Default network | `~/.local/share/leela-zero/best-network` |
| OpenCL tuning cache | `~/.local/share/leela-zero/leelaz_opencl_tuning` |
| Script logs | `logs/` (repo root) |
| Training data | `data/` (repo root), or `$LZ_TRAIN_DATA` |
| Builds | `build/`, `build-opencl/`, `build-debug/`, `build-asan/`, `build-dist/` |
