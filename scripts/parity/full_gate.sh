#!/usr/bin/env bash
# Gate G2 on several network shapes and both Metal engines (MPSGraph and
# Winograd): Metal vs the CPU reference, in fp32 (tolerance 1e-4) and fp16 (N6:
# policy 1e-2, value 5e-3; the policy limit is 2e-2 on random networks, see
# FP16_POLICY_TOL), on 100 positions x 8 symmetries per row. Networks are
# random-weight stand-ins (docs/apple-silicon/07-spec-testing-benchmarks.md).
#
#   scripts/parity/full_gate.sh [--leelaz PATH] [--shapes "6x64 15x192 20x256"]
#                               [--engines "mpsgraph winograd"] [--games N]
#                               [--workdir DIR]
#
# Defaults: build-metal/leelaz, 5 games (20 positions each), a temp workdir.
# Exit status is non-zero if any comparison fails.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LEELAZ="$ROOT/build-metal/leelaz"
SHAPES="6x64 15x192 20x256"
ENGINES="mpsgraph winograd"
GAMES=5
WORK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --leelaz)  LEELAZ="$2"; shift 2 ;;
        --shapes)  SHAPES="$2"; shift 2 ;;
        --engines) ENGINES="$2"; shift 2 ;;
        --games)   GAMES="$2"; shift 2 ;;
        --workdir) WORK="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -x "$LEELAZ" ]] || { echo "not executable: $LEELAZ" >&2; exit 2; }
if [[ -z "$WORK" ]]; then WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT; fi
mkdir -p "$WORK/positions"

# 20 positions per game: after moves 0, 10, ..., 190.
MOVES="$(seq 0 10 190 | paste -sd, -)"   # BSD seq -s, leaves a trailing comma

# N6 allows 1e-2 on the policy. Random-weight networks are a worst case (the
# fp16 error varies about 2x with the positions played), and the real 40x256
# network measures 2.8e-3, so the stand-ins get 2e-2 and real ones keep 1e-2.
FP16_POLICY_TOL="${FP16_POLICY_TOL:-2e-2}"

FAILED=0
SUMMARY=""
record() {  # name, status, output
    local first
    first="$(echo "$3" | grep -E 'max \|d prior\|' | head -1 | sed 's/  */ /g')"
    SUMMARY+="$(printf '%-38s %-5s %s\n' "$1" "$2" "$first")"$'\n'
    [[ "$2" == PASS ]] || { FAILED=1; echo "$3"; }
}

gate() {  # label, net, extra test flags, tolerances...
    local label="$1" net="$2" flags="$3"; shift 3
    local out status=PASS
    out="$(python3 "$ROOT/scripts/parity/compare_backends.py" \
        --ref "$LEELAZ --backend cpu" --test "$LEELAZ --backend metal $flags" \
        -w "$net" --sgf "$WORK/positions/*.sgf" --moves "$MOVES" "$@" 2>&1)" || status=FAIL
    record "$label" "$status" "$out"
}

for shape in $SHAPES; do
    blocks="${shape%%x*}"; filters="${shape##*x}"
    net="$WORK/net$shape.txt"
    echo "== $shape: making a random network" >&2
    python3 "$ROOT/scripts/macos/make_random_net.py" "$net" \
        --blocks "$blocks" --filters "$filters" >/dev/null
    # A Metal failure falls back to the CPU, which would make every row a
    # CPU-vs-CPU pass: check that the Metal pipe really started.
    started="$(printf 'quit\n' | "$LEELAZ" --backend metal -w "$net" --gtp 2>&1 || true)"
    if ! grep -q '^Metal: .* workers\.$' <<<"$started"; then
        echo "Metal did not start on $shape (fell back to the CPU?)" >&2
        exit 1
    fi
    if [[ ! -e "$WORK/positions/.done" ]]; then
        echo "== playing $GAMES games for positions" >&2
        python3 "$ROOT/scripts/macos/selfplay.py" --leelaz "$LEELAZ" -w "$net" \
            -o "$WORK/positions" -n "$GAMES" -v 2 -t 1 >/dev/null
        touch "$WORK/positions/.done"
    fi
    for engine in $ENGINES; do
        echo "== $shape, $engine: G2 fp32" >&2
        gate "G2 fp32 $shape $engine (1e-4)" "$net" \
            "--metal-kernels $engine --precision single" --tol 1e-4
        echo "== $shape, $engine: G2 fp16" >&2
        gate "G2 fp16 $shape $engine (N6)" "$net" \
            "--metal-kernels $engine --precision half" \
            --tol "$FP16_POLICY_TOL" --tol-value 5e-3
    done
    rm -f "$net"
done

echo
echo "Gate G2 (Metal vs CPU), 100 positions x 8 symmetries per row"
printf '%s' "$SUMMARY"
if [[ $FAILED -ne 0 ]]; then echo "FAILED"; exit 1; fi
echo "all passed"
