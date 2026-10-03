#!/usr/bin/env python3
"""Compare raw network outputs of two leelaz builds/backends (spec 07, gate G1/G2).

Both engines load the same weights, then for each position (SGF file + move
number) and each symmetry the script runs the unlisted GTP command
`lz-nn-eval <symmetry>` and diffs winrate and the 362 move priors.

    python3 scripts/parity/compare_backends.py \
        --ref  "build/leelaz" \
        --test "build-opencl/leelaz --precision single" \
        -w net.gz --sgf data/selfplay/*.sgf --moves 0,30,120 --tol 1e-4

Exit status: 0 if every max abs diff is within --tol (or --tol-value for the
winrate), 1 otherwise.
"""
import argparse
import glob
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from gtp import GTPEngine  # noqa: E402


class Engine(GTPEngine):
    def __init__(self, cmdline, weights):
        cmd = shlex.split(cmdline) + ["--gtp", "-q", "-w", weights, "--noponder",
                                      "-t", "1"]
        super().__init__(cmd, name=cmdline)

    def eval(self, symmetry):
        vals = [float(x) for x in self.send("lz-nn-eval %d" % symmetry).split()]
        if len(vals) != 2 + 361:
            raise RuntimeError("unexpected lz-nn-eval output length %d" % len(vals))
        return vals[0], vals[1:]   # winrate, [pass] + 361 priors



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="reference engine command line")
    ap.add_argument("--test", required=True, help="engine under test command line")
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--test-weights", default=None,
                    help="weights for the test engine (default: same as -w), e.g. a re-exported net")
    ap.add_argument("--sgf", nargs="*", default=[], help="SGF files (globs ok)")
    ap.add_argument("--moves", default="0,10,40,100",
                    help="comma-separated move counts: compare the position after N moves (0 = empty board)")
    ap.add_argument("--symmetries", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--tol", type=float, default=1e-4, help="max abs diff, priors")
    ap.add_argument("--tol-value", type=float, default=None,
                    help="max abs diff, winrate (default: --tol)")
    args = ap.parse_args()
    tol_v = args.tol if args.tol_value is None else args.tol_value

    sgfs = sorted({f for pat in args.sgf for f in glob.glob(pat)})
    moves = [int(m) for m in args.moves.split(",")]
    syms = [int(s) for s in args.symmetries.split(",")]
    positions = [(f, m) for f in sgfs for m in moves] or [(None, 0)]

    ref = Engine(args.ref, args.weights)
    test = Engine(args.test, args.test_weights or args.weights)
    worst_p = worst_v = worst_rel = 0.0
    worst_at = None
    n = 0
    try:
        for sgf, move in positions:
            for e in (ref, test):
                # "loadsgf f N" stops before move N, so N = moves + 1 gives the
                # position after `move` moves (N = 0 would load the whole game).
                e.send("clear_board" if sgf is None else "loadsgf %s %d" % (sgf, move + 1))
            for s in syms:
                v_r, p_r = ref.eval(s)
                v_t, p_t = test.eval(s)
                dp = max(abs(a - b) for a, b in zip(p_r, p_t))
                dv = abs(v_r - v_t)
                # Relative diff on priors that matter (>1e-4), for near-uniform nets.
                worst_rel = max([worst_rel] + [abs(a - b) / max(a, b) for a, b in zip(p_r, p_t)
                                               if max(a, b) > 1e-4])
                n += 1
                if dp > worst_p or dv > worst_v:
                    worst_at = (sgf or "empty board", move, s)
                worst_p, worst_v = max(worst_p, dp), max(worst_v, dv)
    finally:
        ref.close()
        test.close()

    ok = worst_p <= args.tol and worst_v <= tol_v
    print("evaluations: %d (%d positions x %d symmetries)" % (n, len(positions), len(syms)))
    print("max |d prior|   = %.3g (tol %.3g)" % (worst_p, args.tol))
    print("max |d winrate| = %.3g (tol %.3g)" % (worst_v, tol_v))
    print("max rel d prior = %.3g (info)" % worst_rel)
    if worst_at:
        print("worst case: %s move %d symmetry %d" % worst_at)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
