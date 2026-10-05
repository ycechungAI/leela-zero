#!/usr/bin/env python3
"""Gate G4: fixed-seed genmove regression between two leelaz commands.

Loads the same positions in both engines and compares the move each one picks
with `genmove` under a deterministic search (one thread, fixed seed, no noise,
no random opening moves, ample time per move). Passes when at least --min-agree of the positions
give the same move (fp16 may flip near-ties).

    scripts/parity/gtp_regression.py --ref "build-metal/leelaz --backend cpu" \
        --test build-metal/leelaz -w net.txt --sgf 'positions/*.sgf' \
        --moves 30,90,150,210 [-p 1600] [--min-agree 19]
"""
import argparse
import glob
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from gtp import GTPEngine  # noqa: E402


def engine(cmd, weights, playouts):
    return GTPEngine(shlex.split(cmd) + [
        "--gtp", "-q", "-w", weights, "-p", str(playouts), "-v", str(playouts),
        "--noponder", "--randomcnt", "0", "-s", "1", "-t", "1",
        "--timemanage", "off", "--resignpct", "0"], cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="reference leelaz command")
    ap.add_argument("--test", required=True, help="leelaz command under test")
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--sgf", required=True, action="append")
    ap.add_argument("--moves", default="30,90,150,210",
                    help="positions after these move numbers of each game")
    ap.add_argument("-p", "--playouts", type=int, default=1600)
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--min-agree", type=int, default=19)
    args = ap.parse_args()

    sgfs = sorted({f for pat in args.sgf for f in glob.glob(pat)})
    moves = [int(m) for m in args.moves.split(",")]
    positions = [(f, m) for f in sgfs for m in moves][:args.positions]
    if len(positions) < args.positions:
        sys.exit("only %d positions available" % len(positions))

    ref = engine(args.ref, args.weights, args.playouts)
    test = engine(args.test, args.weights, args.playouts)
    agree = 0
    try:
        for sgf, move in positions:
            picks = []
            for e in (ref, test):
                e.send("clear_board")
                e.send("loadsgf %s %d" % (sgf, move + 1))
                # The default is one hour of sudden death, spent across the
                # positions: give every move ample time so playouts alone
                # end the search.
                e.send("time_settings 0 1000000 1")
                picks.append(e.send("genmove %s" % ("b" if move % 2 == 0 else "w")).lower())
            same = picks[0] == picks[1]
            agree += same
            print("%s move %3d: %-6s %-6s %s" % (os.path.basename(sgf), move,
                                                  picks[0], picks[1], "" if same else "DIFFERENT"),
                  flush=True)
    finally:
        ref.close()
        test.close()
    ok = agree >= args.min_agree
    print("same move in %d/%d positions (need %d): %s"
          % (agree, len(positions), args.min_agree, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
