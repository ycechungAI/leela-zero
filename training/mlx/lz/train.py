"""Train a Leela Zero network with MLX (plan 13, step 3.5).

    lz-train --blocks 20 --filters 256 --train 'data/train_*' --test 'data/test_*' \\
             --batch 128 --macrobatch 4 --steps 200000 \\
             --lr-schedule '0:0.05,100000:0.005' --dtype bf16 \\
             --restore ckpt.safetensors --import-weights best.txt.gz --export-dir nets/

A "step" is one micro-batch, as in tfprocess (so schedules carry over). With
--macrobatch k the gradients of k micro-batches are summed and applied once,
like tfprocess's gsum; the effective batch is batch * k.
"""

import argparse
import glob
import os
import random
import sys
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from . import data
from .convert import model_to_tensors, shape_from_tensors, tensors_to_model
from .model import LeelaZeroNet, loss_fn
from .swa import SWA
from .weights import read_weights, write_weights

DTYPES = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
FP16_LOSS_SCALE = 128.0            # tfprocess's static scale for fp16
N5_BYTES = 11 * (1 << 30)          # spec N5: peak memory budget on 16 GB


def parse_schedule(text):
    """'0:0.05,100000:0.005' -> sorted [(step, lr)]; must start at step 0."""
    pairs = sorted((int(a), float(b)) for a, b in
                   (item.split(":") for item in text.split(",")))
    if not pairs or pairs[0][0] != 0:
        raise ValueError("learning rate schedule must start at step 0")
    return pairs


def lr_at(schedule, step):
    lr = schedule[0][1]
    for start, value in schedule:
        if step >= start:
            lr = value
    return lr


def projected_peak_bytes(blocks, filters, batch, dtype):
    """Peak memory estimate, calibrated on the M4 (BENCHMARKS.md, step 3.5):
    20x256 bf16 measured 3.26 GiB at batch 64 and 10.55 GiB at batch 256. About
    5.4 activation tensors per convolution are kept for backward, and the
    fp32 weights exist about six times over (weights, gradients, momentum,
    accumulated gradients, working copies)."""
    weights = (9 * 18 * filters + 2 * blocks * 9 * filters * filters
               + 722 * 362 + 361 * 256 + 3 * filters * 4 * (1 + 2 * blocks))
    act_bytes = {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]
    activations = int(batch * (1 + 2 * blocks) * filters * 361 * act_bytes * 5.4)
    return int(6 * weights * 4 + activations + (300 << 20))


def memory_guard(blocks, filters, batch, dtype, force):
    info = mx.device_info()
    total = int(info["memory_size"])
    budget = min(N5_BYTES, int(0.7 * total)) if total <= (16 << 30) else int(0.7 * total)
    projected = projected_peak_bytes(blocks, filters, batch, dtype)
    print("Projected peak memory %.1f GB (budget %.1f GB of %.0f GB)"
          % (projected / 2**30, budget / 2**30, total / 2**30))
    if projected > budget and not force:
        raise SystemExit("Refusing to start: the projected peak exceeds the "
                         "budget. Lower --batch, use --dtype bf16, or pass --force.")
    mx.set_memory_limit(int(0.9 * total))


class Trainer:
    def __init__(self, model, dtype, schedule, macrobatch=1, swa_c=1,
                 swa_max_n=16):
        self.model = model
        self.dtype = dtype
        self.macrobatch = macrobatch
        self.schedule = schedule
        self.step = 0
        self.swa = SWA(swa_c, swa_max_n)
        self.model.compute_dtype = DTYPES[dtype]
        self.loss_scale = FP16_LOSS_SCALE if dtype == "fp16" else 1.0
        self.optimizer = optim.SGD(learning_rate=lr_at(schedule, 0),
                                   momentum=0.9, nesterov=True)
        self.optimizer.init(model.trainable_parameters())
        scale = self.loss_scale

        def scaled(model, planes, probs, winner):
            total, aux = loss_fn(model, planes, probs, winner)
            return total * scale, aux

        value_and_grad = nn.value_and_grad(model, scaled)
        state = [model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def grad_step(planes, probs, winner):
            (_, aux), grads = value_and_grad(model, planes, probs, winner)
            if scale != 1.0:
                grads = tree_map(lambda g: g / scale, grads)
            return aux, grads

        @partial(mx.compile, inputs=state, outputs=state)
        def apply(grads):
            self.optimizer.update(model, grads)

        self._grad_step = grad_step
        self._apply = apply
        self._pending = None
        self._count = 0

    def train_batch(self, planes, probs, winner):
        """One micro-batch; applies the update every `macrobatch` batches.
        Returns (policy, mse, reg, accuracy) as lazy mx scalars."""
        self.model.train()
        aux, grads = self._grad_step(planes, probs, winner)
        if self._pending is None:
            self._pending = grads
        else:
            self._pending = tree_map(lambda a, b: a + b, self._pending, grads)
        self._count += 1
        self.step += 1
        if self._count == self.macrobatch:
            self.optimizer.learning_rate = mx.array(
                lr_at(self.schedule, self.step - 1), dtype=mx.float32)
            self._apply(self._pending)
            self._pending, self._count = None, 0
        return aux

    def evaluate(self, planes, probs, winner):
        self.model.eval()
        _, aux = loss_fn(self.model, planes, probs, winner)
        return aux

    # -- checkpoints ---------------------------------------------------------

    def save_checkpoint(self, path):
        arrays = {"model/" + k: v for k, v in tree_flatten(self.model.parameters())}
        arrays.update({"opt/" + k: v for k, v in
                       tree_flatten(self.optimizer.state)})
        arrays.update(self.swa.state_arrays())
        mx.save_safetensors(path, arrays, {
            "step": str(self.step), "swa_n": str(self.swa.n),
            "swa_skip": str(self.swa.skip)})

    def load_checkpoint(self, path):
        arrays, meta = mx.load(path, return_metadata=True)
        self.model.update(tree_unflatten(
            [(k[6:], v) for k, v in arrays.items() if k.startswith("model/")]))
        opt = [(k[4:], v) for k, v in arrays.items() if k.startswith("opt/")]
        self.optimizer.state = tree_unflatten(opt)
        self.step = int(meta["step"])
        self.swa.load_state(arrays, meta)
        mx.eval(self.model.parameters(), self.optimizer.state)

    def export(self, path):
        write_weights(path, model_to_tensors(self.model))


class Stats:
    def __init__(self):
        self.clear()

    def clear(self):
        self.sums, self.n = [0.0] * 4, 0

    def add(self, aux):
        for i, v in enumerate(aux):
            self.sums[i] += float(v)
        self.n += 1

    def mean(self):
        policy, mse, reg, acc = (s / max(1, self.n) for s in self.sums)
        # tfprocess reports mse / 4; 'total' uses the unscaled mse.
        return {"policy": policy, "mse": mse / 4, "reg": reg, "accuracy": acc,
                "total": policy + mse + reg}


def get_chunks(prefix):
    return glob.glob(prefix + "*.gz")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train a Leela Zero net with MLX.")
    p.add_argument("blockspref", nargs="?", type=int, help="legacy: blocks")
    p.add_argument("filterspref", nargs="?", type=int, help="legacy: filters")
    p.add_argument("trainpref", nargs="?", help="legacy: training file prefix")
    p.add_argument("restorepref", nargs="?", help="legacy: snapshot to restore")
    p.add_argument("--blocks", "-b", type=int)
    p.add_argument("--filters", "-f", type=int)
    p.add_argument("--train", "-t", help="training chunk prefix")
    p.add_argument("--test", help="test chunk prefix (default: 10%% of --train)")
    p.add_argument("--batch", type=int, default=128, help="micro-batch size")
    p.add_argument("--macrobatch", type=int, default=1,
                   help="micro-batches summed per update")
    p.add_argument("--steps", type=int, default=0,
                   help="stop after this many micro-batches (0 = forever)")
    p.add_argument("--lr-schedule", default="0:0.05")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    p.add_argument("--restore", help="checkpoint (.safetensors) to resume")
    p.add_argument("--import-weights", help="leelaz weights to start from")
    p.add_argument("--export-dir", default="nets")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--export-every", type=int, default=8000)
    p.add_argument("--info-steps", type=int, default=1000)
    p.add_argument("--test-every", type=int, default=8000)
    p.add_argument("--test-batches", type=int, default=800)
    p.add_argument("--sample", type=int, default=16,
                   help="use 1 in N records of each chunk")
    p.add_argument("--shuffle-gb", type=float, default=None)
    p.add_argument("--no-swa", action="store_true",
                   help="do not write stochastic-weight-averaged nets")
    p.add_argument("--swa-c", type=int, default=1,
                   help="sample every c-th checkpoint")
    p.add_argument("--swa-max-n", type=int, default=16)
    p.add_argument("--swa-batches", type=int, default=200,
                   help="batches used to refine batch norm in an SWA net")
    p.add_argument("--force", action="store_true",
                   help="ignore the memory guard")
    return p.parse_args(argv)


def build_model(args):
    blocks = args.blocks or args.blockspref
    filters = args.filters or args.filterspref
    if args.import_weights:
        _, tensors = read_weights(args.import_weights)
        w_blocks, w_filters = shape_from_tensors(tensors)
        if (blocks, filters) not in ((None, None), (w_blocks, w_filters)):
            raise SystemExit("--import-weights is %dx%d, not %dx%d"
                             % (w_blocks, w_filters, blocks, filters))
        return tensors_to_model(tensors)
    if not blocks or not filters:
        raise SystemExit("Must supply the number of blocks and filters")
    return LeelaZeroNet(blocks, filters)


def main(argv=None):
    args = parse_args(argv)
    train_prefix = args.train or args.trainpref
    restore = args.restore or args.restorepref
    model = build_model(args)
    memory_guard(model.blocks, model.filters, args.batch, args.dtype, args.force)
    trainer = Trainer(model, args.dtype, parse_schedule(args.lr_schedule),
                      args.macrobatch, args.swa_c, args.swa_max_n)
    if restore:
        trainer.load_checkpoint(restore)
        print("Restored step", trainer.step)

    training = get_chunks(train_prefix or "")
    if not training:
        raise SystemExit("No data to train on!")
    if args.test:
        test = get_chunks(args.test)
    else:
        random.shuffle(training)
        split = 1 + int(len(training) * 0.9)
        training, test = training[:split], training[split:]
    print("Training with %d chunks, validating on %d chunks"
          % (len(training), len(test)))
    shuffle = None if args.shuffle_gb is None else int(args.shuffle_gb * 2**30)
    os.makedirs(args.export_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    stats, timer = Stats(), time.time()
    with data.Batches(training, args.batch, shuffle_bytes=shuffle,
                      sample=args.sample) as train_data:
        test_data = (data.Batches(test, args.batch, sample=args.sample,
                                  shuffle_bytes=(shuffle or 2**28) // 4)
                     if test else None)
        try:
            while not args.steps or trainer.step < args.steps:
                stats.add(trainer.train_batch(*train_data.next()))
                step = trainer.step
                if step % args.info_steps == 0:
                    m = stats.mean()
                    pos_s = args.info_steps * args.batch / (time.time() - timer)
                    print("step %d, policy=%g mse=%g reg=%g total=%g "
                          "acc=%.2f%% (%g pos/s)" % (
                              step, m["policy"], m["mse"], m["reg"], m["total"],
                              100 * m["accuracy"], pos_s), flush=True)
                    stats.clear()
                    timer = time.time()
                if test_data and step % args.test_every == 0:
                    t = Stats()
                    for _ in range(args.test_batches):
                        t.add(trainer.evaluate(*test_data.next()))
                    m = t.mean()
                    print("step %d, test policy=%g accuracy=%.2f%% mse=%g" % (
                        step, m["policy"], 100 * m["accuracy"], m["mse"]),
                        flush=True)
                if step % args.export_every == 0:
                    trainer.save_checkpoint(os.path.join(
                        args.checkpoint_dir, "step-%d.safetensors" % step))
                    trainer.export(os.path.join(
                        args.export_dir, "lz-%d.txt.gz" % step))
                    if not args.no_swa:
                        n = trainer.swa.update(trainer.model)
                        if n:
                            trainer.swa.export(
                                trainer.model, train_data, os.path.join(
                                    args.export_dir,
                                    "lz-swa-%d-%d.txt.gz" % (n, step)),
                                args.swa_batches)
        finally:
            if test_data:
                test_data.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
