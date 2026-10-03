#!/usr/bin/env python3
"""Write a random-weight Leela Zero network (v1 text format) for smoke tests.

It plays nonsense, but it exercises the full load / search / GTP path with no
download. Uses only the standard library.

    python3 scripts/macos/make_random_net.py out.txt [--blocks 2] [--filters 16]
"""
import argparse
import random

BOARD = 19 * 19
INPUT_PLANES = 18


def line(n, fan_in=1, const=None):
    if const is not None:
        return " ".join([const] * n)
    # Scale by fan-in so activations stay in range and outputs don't saturate.
    scale = (1.0 / fan_in) ** 0.5
    return " ".join("%.6g" % random.gauss(0.0, scale) for _ in range(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    f = args.filters

    def conv_bn(inputs, outputs, ksize):
        # conv weights [out, in, k, k], conv bias, BN means, BN variances
        return [line(outputs * inputs * ksize * ksize, inputs * ksize * ksize),
                line(outputs, const="0"),
                line(outputs, const="0"), line(outputs, const="1")]

    rows = ["1"]
    rows += conv_bn(INPUT_PLANES, f, 3)
    for _ in range(args.blocks):
        rows += conv_bn(f, f, 3) + conv_bn(f, f, 3)
    rows += conv_bn(f, 2, 1)                                   # policy conv
    rows += [line((BOARD + 1) * 2 * BOARD, 2 * BOARD), line(BOARD + 1, const="0")]  # policy FC
    rows += conv_bn(f, 1, 1)                                   # value conv
    rows += [line(256 * BOARD, BOARD), line(256, const="0"),    # value FC1
             line(256, 256), line(1, const="0")]                 # value FC2

    with open(args.out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("wrote %s (%d blocks x %d filters)" % (args.out, args.blocks, f))


if __name__ == "__main__":
    main()
