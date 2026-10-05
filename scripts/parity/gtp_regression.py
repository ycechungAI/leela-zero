#!/usr/bin/env python3
"""Gate G4: fixed-seed genmove regression between two leelaz commands.

Loads the same positions in both engines and compares the move each one picks
under a deterministic search (one thread, fixed seed, no noise, no random
opening moves, ample time per move). The reference searches with
lz-genmove_analyze, so its root statistics are known. A position passes when
the test engine plays the same move, or a move the reference searched whose
winrate is within --winrate-tol of the reference's choice: a near-tie, which
small evaluation differences legitimately flip (ADR-011). Passes when at least
--min-agree positions pass.

    scripts/parity/gtp_regression.py --ref "build-metal/leelaz --backend cpu" \
        --test build-metal/leelaz -w net.txt --sgf 'positions/*.sgf' \
        --moves 30,90,150,210 [-p 1600] [--min-agree 19]
"""
import argparse
import glob
import re
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from gtp import GTPEngine  # noqa: E402


def engine(cmd, weights, playouts, seed):
    return GTPEngine(shlex.split(cmd) + [
        "--gtp", "-q", "-w", weights, "-p", str(playouts), "-v", str(playouts),
        "--noponder", "--randomcnt", "0", "-s", str(seed), "-t", "1",
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
    ap.add_argument("--test-seed", type=int, default=1,
                    help="seed of the test engine; a different seed with the "
                         "same backend measures the search's own sensitivity")
    ap.add_argument("--winrate-tol", type=float, default=0.01,
                    help="near-tie margin, as a winrate fraction")
    args = ap.parse_args()

    sgfs = sorted({f for pat in args.sgf for f in glob.glob(pat)})
    moves = [int(m) for m in args.moves.split(",")]
    positions = [(f, m) for f in sgfs for m in moves][:args.positions]
    if len(positions) < args.positions:
        sys.exit("only %d positions available" % len(positions))

    ref = engine(args.ref, args.weights, args.playouts, 1)
    test = engine(args.test, args.weights, args.playouts, args.test_seed)
    agree = same_total = 0
    try:
        for sgf, move in positions:
            color = "b" if move % 2 == 0 else "w"
            for e in (ref, test):
                e.send("clear_board")
                e.send("loadsgf %s %d" % (sgf, move + 1))
                # The default is one hour of sudden death, spent across the
                # positions: give every move ample time so playouts alone
                # end the search.
                e.send("time_settings 0 1000000 1")
            # The reference reports every root move's visits and winrate (in
            # 1/10000), then the move it plays.
            reply = ref.send("lz-genmove_analyze %s 1000000" % color)
            ref_move = re.findall(r"^play (\S+)", reply, re.M)[-1].lower()
            info = reply[:reply.rfind("play ")].split("info move ")
            winrates = {}
            for item in info[1:]:
                f = item.split()
                winrates[f[0].lower()] = int(f[f.index("winrate") + 1]) / 10000.0
            test_move = test.send("genmove %s" % color).lower()
            same = test_move == ref_move
            loss = winrates.get(ref_move, 0.0) - winrates.get(test_move, -1.0)
            close = same or loss <= args.winrate_tol
            same_total += same
            agree += close
            print("%s move %3d: %-6s %-6s %s" % (
                os.path.basename(sgf), move, ref_move, test_move,
                "" if same else ("near-tie (%.1f%% winrate)" % (100 * loss)
                                 if close else "DIFFERENT (%s)" % (
                                     "%.1f%% winrate" % (100 * loss)
                                     if test_move in winrates else "not searched"))),
                  flush=True)
    finally:
        ref.close()
        test.close()
    ok = agree >= args.min_agree
    print("same move in %d/%d positions; same or near-tie in %d/%d (need %d): %s"
          % (same_total, len(positions), agree, len(positions), args.min_agree,
             "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
