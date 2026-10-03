#!/usr/bin/env python3
"""Generate self-play training data locally by driving leelaz over GTP.

Each game is written as <outdir>/<prefix>-NNNN.0.gz (training chunk, the same
format autogtp uploads) plus <prefix>-NNNN.sgf. This is a small local
substitute for autogtp; it does not talk to any server.

    python3 scripts/macos/selfplay.py --leelaz build/leelaz -w net.gz -o data/selfplay -n 10
"""
import argparse
import os
import subprocess
import sys
import time


class GTP:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, command):
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("leelaz exited while running: " + command)
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line.strip())
        reply = "\n".join(lines)
        if reply.startswith("?"):
            raise RuntimeError("GTP error for '%s': %s" % (command, reply))
        return reply[1:].strip()

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.p.wait(timeout=10)


def play_game(gtp, max_moves):
    """Returns 'b' or 'w' (winner) and the number of moves played."""
    gtp.send("clear_board")
    colors = ["b", "w"]
    passes = 0
    for move_no in range(max_moves):
        color = colors[move_no % 2]
        move = gtp.send("genmove " + color).lower()
        if move == "resign":
            return ("w" if color == "b" else "b"), move_no
        passes = passes + 1 if move == "pass" else 0
        if passes >= 2:
            break
    score = gtp.send("final_score")
    if score.startswith("B+"):
        return "b", move_no + 1
    if score.startswith("W+"):
        return "w", move_no + 1
    return None, move_no + 1   # jigo: leelaz training data needs a winner, skip


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leelaz", required=True)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("-o", "--outdir", default="data/selfplay")
    ap.add_argument("-n", "--games", type=int, default=1)
    ap.add_argument("-v", "--visits", type=int, default=800,
                    help="visits per move (autogtp used 1600)")
    ap.add_argument("-t", "--threads", type=int, default=0)
    ap.add_argument("--randomcnt", type=int, default=30,
                    help="pick moves proportionally for the first N moves")
    ap.add_argument("--max-moves", type=int, default=19 * 19 * 2)
    ap.add_argument("--prefix", default=time.strftime("game-%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cmd = [args.leelaz, "--gtp", "-q", "-w", args.weights, "-v", str(args.visits),
           "--noponder", "--randomcnt", str(args.randomcnt), "--noise",
           "--resignpct", "5"]
    if args.threads:
        cmd += ["-t", str(args.threads)]

    gtp = GTP(cmd)
    written = 0
    try:
        for i in range(args.games):
            t0 = time.time()
            winner, moves = play_game(gtp, args.max_moves)
            base = os.path.join(args.outdir, "%s-%04d" % (args.prefix, i))
            gtp.send("printsgf %s.sgf" % base)
            if winner is None:
                print("game %d: jigo after %d moves, skipped" % (i + 1, moves))
                continue
            gtp.send("dump_training %s %s" % (winner, base))
            written += 1
            print("game %d/%d: %s wins, %d moves, %.1fs -> %s.0.gz"
                  % (i + 1, args.games, winner.upper(), moves, time.time() - t0, base))
            sys.stdout.flush()
    finally:
        gtp.close()
    print("%d training games written to %s" % (written, args.outdir))


if __name__ == "__main__":
    main()
