#!/usr/bin/env python3
"""Soak test: one long-lived leelaz plays self-play games over GTP while its
resident memory is sampled, then `leaks` checks the process.

Pass criteria (spec 05, Phase 2 exit): no leaks reported, and the median RSS
of the last quarter at most --max-growth MB above that of the second quarter.
Exit status 0 on pass.

    scripts/macos/soak.py --leelaz build-metal/leelaz -w net.txt \
        --minutes 120 -t 16 [-v 800] [--log soak.csv]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from gtp import GTPEngine  # noqa: E402


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024.0 if out else float("nan")


def leaks(pid):
    """Return (leak count, summary line) from macOS `leaks`."""
    res = subprocess.run(["leaks", str(pid)], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "leaks for" in line:
            return int(line.split(":")[1].split()[0]), line.strip()
    return -1, (res.stdout + res.stderr).strip().splitlines()[-1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leelaz", required=True)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("-t", "--threads", type=int, default=16)
    ap.add_argument("-v", "--visits", type=int, default=800)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--max-growth", type=float, default=50.0,
                    help="allowed median RSS growth (MB), last vs second quarter")
    ap.add_argument("--log", help="CSV of elapsed seconds, RSS MB, moves")
    ap.add_argument("extra", nargs="*", help="extra leelaz arguments after --")
    args = ap.parse_args()

    cmd = [args.leelaz, "-w", args.weights, "--gtp", "--noponder", "-q",
           "-t", str(args.threads), "-v", str(args.visits), "--randomcnt", "30",
           ] + args.extra
    engine = GTPEngine(cmd, "leelaz")
    pid = engine.p.pid
    engine.send("name")  # wait for the network (and any autotune) to load
    start = time.time()
    end = start + args.minutes * 60
    samples = []  # (elapsed, rss, moves)
    moves = games = 0
    last_sample = 0.0
    log = open(args.log, "w") if args.log else None
    if log:
        log.write("seconds,rss_mb,moves\n")

    while time.time() < end:
        engine.send("clear_board")
        color = "b"
        for _ in range(args.max_moves):
            move = engine.send("genmove " + color).lower()
            moves += 1
            color = "w" if color == "b" else "b"
            now = time.time()
            if now - last_sample >= 60:
                last_sample = now
                sample = (now - start, rss_mb(pid), moves)
                samples.append(sample)
                if log:
                    log.write("%.0f,%.1f,%d\n" % sample)
                    log.flush()
                print("%5.0f min  rss %7.1f MB  moves %d  games %d"
                      % (sample[0] / 60, sample[1], moves, games), flush=True)
            if move == "resign" or now >= end:
                break
        games += 1

    count, summary = leaks(pid)
    engine.close()

    # RSS swings by ~100 MB with the size of the search tree, so compare the
    # median of the last quarter with the median of the second quarter.
    def median_rss(lo, hi):
        rss = sorted(s[1] for s in samples
                     if lo * (end - start) <= s[0] < hi * (end - start))
        return rss[len(rss) // 2] if rss else float("nan")
    growth = median_rss(0.75, 1.01) - median_rss(0.25, 0.5)
    peak = max(s[1] for s in samples) if samples else float("nan")
    print("games %d, moves %d (%.1f moves/min)"
          % (games, moves, moves / max(1e-9, (time.time() - start) / 60)))
    print("RSS: first %.1f MB, peak %.1f MB, median growth %.1f MB "
          "(limit %.0f)" % (samples[0][1] if samples else float("nan"), peak,
                            growth, args.max_growth))
    print("leaks: %s" % summary)
    ok = count == 0 and growth == growth and growth < args.max_growth
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
