#!/usr/bin/env bash
# Start leelaz in GTP mode (for Sabaki, Lizzie, GoGui, or typing GTP by hand).
#
#   scripts/macos/start.sh [-w weights] [-b metal|cpu|opencl] [-t threads] [-v visits] [-- extra leelaz args]
#
# Defaults: backend metal, weights $LZ_WEIGHTS or ~/.local/share/leela-zero/best-network,
# leelaz picks the thread count, log written to logs/leelaz-<time>.log.
#
# Examples:
#   scripts/macos/start.sh -w ~/nets/40b.gz
#   scripts/macos/start.sh -b opencl -v 1600 -- --noponder
#   echo -e "genmove b\nquit" | scripts/macos/start.sh -w /tmp/rand.txt -v 50

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_macos_arm64

BACKEND="metal"; WEIGHTS=""; THREADS=""; VISITS=""; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w) WEIGHTS="$2"; shift 2 ;;
        -b) BACKEND="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -v) VISITS="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) die "unknown argument: $1 (put leelaz options after --)" ;;
    esac
done

[[ "$BACKEND" == "metal" || "$BACKEND" == "cpu" || "$BACKEND" == "opencl" ]] || die "backend must be metal, cpu or opencl"
WEIGHTS="$(resolve_weights "$WEIGHTS")"
LEELAZ="$(leelaz_for "$BACKEND")"

mkdir -p "$REPO_ROOT/logs"
LOG="$REPO_ROOT/logs/leelaz-$(date +%Y%m%d-%H%M%S).log"

ARGS=(--gtp -w "$WEIGHTS" --logfile "$LOG" --backend "$BACKEND")
[[ -n "$THREADS" ]] && ARGS+=(-t "$THREADS")
[[ -n "$VISITS" ]] && ARGS+=(-v "$VISITS")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

info "leelaz ($BACKEND) weights=$(basename "$WEIGHTS") log=${LOG#$REPO_ROOT/}" >&2
exec "$LEELAZ" "${ARGS[@]}"
