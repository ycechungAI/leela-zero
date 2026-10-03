#!/usr/bin/env bash
# Shared helpers for the macOS (Apple Silicon) scripts. Source, don't run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts/macos"
LZ_DATA_DIR="${LZ_DATA_DIR:-$HOME/.local/share/leela-zero}"   # where leelaz looks by default

# Build directories, one per configuration (see build.sh).
build_dir_for() {
    case "$1" in
        metal)  echo "$REPO_ROOT/build-metal" ;;
        cpu)    echo "$REPO_ROOT/build" ;;
        opencl) echo "$REPO_ROOT/build-opencl" ;;
        debug)  echo "$REPO_ROOT/build-debug" ;;
        asan)   echo "$REPO_ROOT/build-asan" ;;
        dist)   echo "$REPO_ROOT/build-dist" ;;
        *)      die "unknown build config '$1' (expected metal|cpu|opencl|debug|asan|dist)" ;;
    esac
}

if [[ -t 1 ]]; then
    _c_info=$'\033[1;34m'; _c_warn=$'\033[1;33m'; _c_err=$'\033[1;31m'; _c_off=$'\033[0m'
else
    _c_info=""; _c_warn=""; _c_err=""; _c_off=""
fi
info() { echo "${_c_info}==>${_c_off} $*"; }
warn() { echo "${_c_warn}warning:${_c_off} $*" >&2; }
die()  { echo "${_c_err}error:${_c_off} $*" >&2; exit 1; }

require_macos_arm64() {
    [[ "$(uname -s)" == "Darwin" ]] || die "these scripts are for macOS"
    [[ "$(uname -m)" == "arm64" ]] || warn "not running on arm64 ($(uname -m)); are you under Rosetta?"
}

# Resolve the weights file: $1, then $LZ_WEIGHTS, then the leelaz default location.
resolve_weights() {
    local w="${1:-${LZ_WEIGHTS:-$LZ_DATA_DIR/best-network}}"
    [[ -f "$w" ]] || die "weights file not found: $w
  Get a network (see docs/apple-silicon/BUILD.md, 'Get a network'), or make a
  random one for smoke tests:  python3 scripts/macos/make_random_net.py /tmp/rand.txt"
    echo "$w"
}

# Path to a built leelaz for the given config, building it if missing.
leelaz_for() {
    local cfg="$1" dir
    dir="$(build_dir_for "$cfg")"
    if [[ ! -x "$dir/leelaz" ]]; then
        info "leelaz ($cfg) not built yet, building it"
        "$SCRIPTS_DIR/build.sh" "$cfg" >&2
    fi
    echo "$dir/leelaz"
}
