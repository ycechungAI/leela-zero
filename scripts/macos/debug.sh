#!/usr/bin/env bash
# Debug helpers.
#
#   scripts/macos/debug.sh lldb  [-w weights] [-- extra leelaz args]   leelaz (Debug build) under lldb
#   scripts/macos/debug.sh asan  [-w weights] [-- extra leelaz args]   leelaz with AddressSanitizer/UBSan
#   scripts/macos/debug.sh tests [gtest filter]                       unit tests under lldb (Debug build)
#   scripts/macos/debug.sh smoke [-w weights]                         short GTP session, prints the log
#
# Without -w, the lldb/asan/smoke modes create a small random network so you
# can debug the engine with no download.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_macos_arm64

MODE="${1:-}"; shift || true
WEIGHTS=""; EXTRA=(); FILTER="*"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w) WEIGHTS="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *)  FILTER="$1"; shift ;;
    esac
done

weights_or_random() {
    if [[ -n "$WEIGHTS" ]]; then
        resolve_weights "$WEIGHTS"
    else
        local rand="$REPO_ROOT/logs/random-net.txt"
        mkdir -p "$REPO_ROOT/logs"
        [[ -f "$rand" ]] || python3 "$SCRIPTS_DIR/make_random_net.py" "$rand" >&2
        echo "$rand"
    fi
}

case "$MODE" in
    lldb)
        W="$(weights_or_random)"
        LEELAZ="$(leelaz_for debug)"
        info "lldb: 'run' to start, type GTP commands, Ctrl-C to break, 'bt' for a backtrace"
        exec lldb -- "$LEELAZ" --gtp -w "$W" -t 1 --noponder ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
    asan)
        W="$(weights_or_random)"
        LEELAZ="$(leelaz_for asan)"
        export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1}"
        export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1}"
        info "AddressSanitizer build; reports go to stderr"
        exec "$LEELAZ" --gtp -w "$W" --noponder ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
    tests)
        DIR="$(build_dir_for debug)"
        [[ -x "$DIR/tests" ]] || "$SCRIPTS_DIR/build.sh" debug --no-test
        cd "$REPO_ROOT/src"   # tests read ../src/tests/0k.txt (same as ctest)
        exec lldb -- "$DIR/tests" --gtest_filter="$FILTER"
        ;;
    smoke)
        W="$(weights_or_random)"
        LEELAZ="$(leelaz_for cpu)"
        mkdir -p "$REPO_ROOT/logs"
        LOG="$REPO_ROOT/logs/smoke.log"
        info "GTP smoke test with $(basename "$W")"
        printf 'name\nversion\nboardsize 19\nclear_board\ngenmove b\ngenmove w\nshowboard\nquit\n' \
            | "$LEELAZ" --gtp -w "$W" -v 20 --noponder --logfile "$LOG"
        info "log: ${LOG#$REPO_ROOT/}"
        ;;
    *)
        sed -n '2,11p' "$0"; exit 1 ;;
esac
