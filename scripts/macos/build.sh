#!/usr/bin/env bash
# Build leelaz + unit tests on Apple Silicon.
#
#   scripts/macos/build.sh [cpu|opencl|debug|asan] [--clean] [--no-test]
#
#   cpu     Release, CPU only, Accelerate BLAS (default)
#   opencl  Release, OpenCL GPU backend (deprecated on macOS; perf baseline)
#   debug   Debug (-Og -g), CPU only, for lldb
#   asan    Debug + AddressSanitizer/UBSan, CPU only

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_macos_arm64

CFG="cpu"; CLEAN=0; RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        cpu|opencl|debug|asan) CFG="$arg" ;;
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

CMAKE_ARGS=(-DUSE_BLAS=1)
case "$CFG" in
    cpu)    CMAKE_ARGS+=(-DUSE_CPU_ONLY=1 -DCMAKE_BUILD_TYPE=Release) ;;
    opencl) CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Release) ;;
    debug)  CMAKE_ARGS+=(-DUSE_CPU_ONLY=1 -DCMAKE_BUILD_TYPE=Debug) ;;
    asan)   CMAKE_ARGS+=(-DUSE_CPU_ONLY=1 -DCMAKE_BUILD_TYPE=Debug
                         "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"
                         "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined") ;;
esac

info "configuring ($CFG) in ${BUILD_DIR#$REPO_ROOT/}"
cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" -Wno-dev >/dev/null

info "building leelaz and tests"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)" --target leelaz tests

if [[ $RUN_TESTS -eq 1 ]]; then
    info "running unit tests"
    # The tests load ../src/tests/0k.txt, so run them from the build dir.
    (cd "$BUILD_DIR" && ./tests --gtest_brief=1)
fi

info "done: ${BUILD_DIR#$REPO_ROOT/}/leelaz"
