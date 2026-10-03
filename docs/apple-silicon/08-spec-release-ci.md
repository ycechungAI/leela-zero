# 08 — Spec: CI, Packaging & Release Process

## 1. Branching (in fork `ycechungAI/leela-zero`)

| Branch | Purpose |
|--------|---------|
| `next` | Mirrors upstream `leela-zero/next`. Never commit directly. Sync with `git fetch upstream && git merge --ff-only upstream/next` |
| `apple-silicon` | Integration branch for this program. Release tags are cut from here |
| `as/<phase>-<topic>` | Feature branches (e.g. `as/p2-metal-mpsgraph`). PRs target `apple-silicon` |

Add the `upstream` remote: `git remote add upstream https://github.com/leela-zero/leela-zero.git`.

Upstream-friendly changes go back to upstream as separate small PRs, so the
fork's diff stays small:
- the CMake 4 minimum-version fix
- the `BatchQueue` extraction
- the Accelerate link fix

## 2. CI (GitHub Actions)

`.github/workflows/macos-arm64.yml` runs on `macos-15` (Apple Silicon runners,
which have a GPU with Metal available):

| Job | Steps | Trigger |
|-----|-------|---------|
| `build-cpu` | brew deps (cached) → submodules → `cmake --preset macos-cpu` → build → G0 | push/PR |
| `build-metal` | `cmake --preset macos-metal` → build → G0 → G2 on N-S | push/PR |
| `parity-full` | G2 on N-M/N-L/N-E, G4 | nightly + `release/*` tags |
| `asan` | `macos-debug-asan` → G0 + G2(N-S) with `MTL_DEBUG_LAYER=1` | nightly |
| `training` | `uv sync` → `pytest training/mlx/tests` (T1, T2, T3 on N-S) | PRs touching `training/mlx` |
| `linux` | Ubuntu 22.04 CPU + OpenBLAS build + G0 (replaces dead Travis) | push/PR |

Note: GitHub-hosted macOS runners are virtualized. Metal works there, but
performance numbers are meaningless, so benchmarks run only on the physical M4
(section 5).

## 3. Versioning

`v<upstream-base>-as.<n>`, for example `v0.17.1-as.0`. It moves to the
`v0.18.0-as.*` base once the backend default changes, signaling a behavior
change. `git describe` in CMake already derives version numbers from tags. The
CMake regex in `CMakeLists.txt` must accept the `-as.N` suffix.

## 4. Artifacts per release

| Artifact | Contents |
|----------|----------|
| `leela-zero-<ver>-macos-arm64.tar.gz` | `leelaz` (Metal+Accelerate), `autogtp`, `validation`, `LICENSE`, `BUILD.md`, `README-macOS.md` |
| `leela-zero-<ver>-macos-arm64.dmg` (as.4) | Same, signed + notarized |
| Homebrew tap `ycechungAI/homebrew-leela-zero` (as.4) | `brew install ycechungAI/leela-zero/leela-zero`. Formula builds from source with `macos-metal` and bottles for arm64 |
| `training/mlx` | Released as a Python package from the repo (`uv pip install ./training/mlx`). No PyPI upload initially |
| `BENCHMARKS.md` | Updated table for the release |

Packaging details:
- Bundle Qt (for autogtp/validation) with `macdeployqt`, or link statically.
- Use `install_name_tool` / `@rpath` so the tarball runs without Homebrew
  except where documented. Boost is linked statically to avoid that dependency.
- Codesign + notarize (as.4): needs an Apple Developer ID. Until then, document
  `xattr -d com.apple.quarantine`.

## 5. Release checklist (every `as.N`)

1. All CI jobs green on the `apple-silicon` HEAD.
2. Run the full benchmark protocol (spec 07 §3) on the physical M4 and update
   `BENCHMARKS.md`.
3. G3 + G4 pass.
4. Update `CHANGELOG-apple-silicon.md`.
5. Tag `v…-as.N`, then push the tag; a CI workflow builds the artifacts and
   drafts a GitHub Release.
6. Smoke test the downloaded artifact on a clean user account: run `leelaz` in
   GTP with the N-S net, play 10 moves, and check that `autogtp` connects to
   the server in dry-run mode (if the public server is still up; otherwise use
   a local job).
7. Publish the release notes with the benchmark table and known issues.
