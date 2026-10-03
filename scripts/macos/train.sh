#!/usr/bin/env bash
# Training workflow on Apple Silicon.
#
#   scripts/macos/train.sh selfplay [-w weights] [-n games] [-v visits] [-o dir]
#       Play games locally and write training chunks (*.gz) + SGFs.
#   scripts/macos/train.sh sgf <games.sgf> [-o dir]
#       Convert existing SGF games into training chunks (supervised data).
#   scripts/macos/train.sh split [-i dir] [--test-pct 10]
#       Split chunks into <dir>/train and <dir>/test for the trainer.
#   scripts/macos/train.sh fit [trainer args...]
#       Train a network on the chunks. Needs the MLX trainer (Phase 3, training/mlx).
#
# Data defaults to $REPO_ROOT/data (ignored by git).
#
# Example end-to-end run:
#   scripts/macos/train.sh selfplay -n 20 -v 400
#   scripts/macos/train.sh split
#   scripts/macos/train.sh fit --blocks 6 --filters 64

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_macos_arm64

DATA_DIR="${LZ_TRAIN_DATA:-$REPO_ROOT/data}"
CMD="${1:-}"; shift || true

case "$CMD" in
    selfplay)
        WEIGHTS=""; GAMES=1; VISITS=800; OUT="$DATA_DIR/selfplay"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -w) WEIGHTS="$2"; shift 2 ;;
                -n) GAMES="$2"; shift 2 ;;
                -v) VISITS="$2"; shift 2 ;;
                -o) OUT="$2"; shift 2 ;;
                *) die "unknown argument: $1" ;;
            esac
        done
        WEIGHTS="$(resolve_weights "$WEIGHTS")"
        LEELAZ="$(leelaz_for cpu)"
        info "self-play: $GAMES game(s), $VISITS visits/move, net $(basename "$WEIGHTS")"
        python3 "$SCRIPTS_DIR/selfplay.py" --leelaz "$LEELAZ" -w "$WEIGHTS" \
            -o "$OUT" -n "$GAMES" -v "$VISITS"
        ;;

    sgf)
        SGF="${1:-}"; [[ -n "$SGF" ]] || die "usage: train.sh sgf <games.sgf> [-o dir]"; shift
        OUT="$DATA_DIR/supervised"
        [[ "${1:-}" == "-o" ]] && OUT="$2"
        [[ -f "$SGF" ]] || die "no such file: $SGF"
        mkdir -p "$OUT"
        # dump_supervised needs a network loaded to start, but never evaluates it.
        RAND="$REPO_ROOT/logs/random-net.txt"
        mkdir -p "$REPO_ROOT/logs"
        [[ -f "$RAND" ]] || python3 "$SCRIPTS_DIR/make_random_net.py" "$RAND" >/dev/null
        LEELAZ="$(leelaz_for cpu)"
        PREFIX="$OUT/$(basename "${SGF%.*}")"
        info "converting $SGF -> $PREFIX.*.gz"
        printf 'dump_supervised %s %s\nquit\n' "$SGF" "$PREFIX" \
            | "$LEELAZ" --gtp -q -w "$RAND" >/dev/null
        ls -1 "$PREFIX".*.gz 2>/dev/null | sed 's/^/  /' || die "no chunks written (bad SGF?)"
        ;;

    split)
        IN="$DATA_DIR"; PCT=10
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -i) IN="$2"; shift 2 ;;
                --test-pct) PCT="$2"; shift 2 ;;
                *) die "unknown argument: $1" ;;
            esac
        done
        mkdir -p "$IN/train" "$IN/test"
        CHUNKS=()
        while IFS= read -r f; do CHUNKS+=("$f"); done < <(
            find "$IN" -name '*.gz' -not -path "$IN/train/*" -not -path "$IN/test/*" | sort)
        [[ ${#CHUNKS[@]} -gt 0 ]] || die "no *.gz chunks under $IN (run selfplay or sgf first)"
        i=0
        for f in "${CHUNKS[@]}"; do
            # Every ~(100/PCT)-th chunk goes to test, spread evenly at any count.
            if (( (i + 1) * PCT / 100 > i * PCT / 100 )); then dest="$IN/test"; else dest="$IN/train"; fi
            # Prefix with the source folder so selfplay/ and supervised/ names can't collide.
            ln -sf "$f" "$dest/$(basename "$(dirname "$f")")-$(basename "$f")"
            i=$((i + 1))
        done
        info "linked $(ls "$IN/train" | wc -l | tr -d ' ') train / $(ls "$IN/test" | wc -l | tr -d ' ') test chunks under $IN"
        ;;

    fit)
        if [[ -f "$REPO_ROOT/training/mlx/pyproject.toml" ]]; then
            command -v uv >/dev/null || die "uv not found: brew install uv"
            cd "$REPO_ROOT/training/mlx"
            exec uv run python -m lz.train --train "$DATA_DIR/train/*" --test "$DATA_DIR/test/*" "$@"
        fi
        die "the MLX trainer (training/mlx) is not implemented yet. It is Phase 3 of
  docs/apple-silicon/ROADMAP.md. The TF1 trainer in training/tf does not run on
  macOS arm64. Training data from 'selfplay' / 'sgf' is already valid input for
  both trainers, so you can keep generating it now."
        ;;

    *)
        sed -n '2,18p' "$0"; exit 1 ;;
esac
