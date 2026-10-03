#!/usr/bin/env bash
# Build leelaz + unit tests on Apple Silicon.
#
#   scripts/macos/build.sh [cpu|opencl|debug|asan|dist] [--clean] [--no-test]
#
#   cpu     Release, CPU only, Accelerate BLAS (default)
#   opencl  Release, OpenCL GPU backend (deprecated on macOS; perf baseline)
#   debug   Debug (-Og -g), CPU only, for lldb
#   asan    Debug + AddressSanitizer/UBSan, CPU only
#   dist    Release, CPU only, -mcpu=apple-m1 (runs on any Apple Silicon Mac)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_macos_arm64

CFG="cpu"; CLEAN=0; RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        cpu|opencl|debug|asan|dist) CFG="$arg" ;;
        --clean)   CLEAN=1 ;;
        --no-test) RUN_TESTS=0 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) die "unknown argument: $arg" ;;
    esac
done

command -v cmake >/dev/null || die "cmake not found: brew install cmake"
brew list --versions boost >/dev/null 2>&1 || die "Boost not found: brew install boost"

cd "$REPO_ROOT"
if [[ ! -f src/Eigen/Eigen/Core || ! -f gtest/CMakeLists.txt ]]; then
    info "initializing submodules"
    git submodule update --init --recursive
fi

BUILD_DIR="$(build_dir_for "$CFG")"
[[ $CLEAN -eq 1 ]] && { info "removing $BUILD_DIR"; rm -rf "$BUILD_DIR"; }

# Each config maps to a preset in CMakePresets.json.
PRESET="macos-$CFG"

info "configuring ($PRESET) in ${BUILD_DIR#$REPO_ROOT/}"
cmake --preset "$PRESET" >/dev/null

info "building leelaz and tests"
cmake --build --preset "$PRESET" -j"$(sysctl -n hw.ncpu)"

if [[ $RUN_TESTS -eq 1 ]]; then
    info "running unit tests"
    ctest --preset "$PRESET"
fi

info "done: ${BUILD_DIR#$REPO_ROOT/}/leelaz"
