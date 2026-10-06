"""Step 3.5: the training step, macrobatch, schedule, checkpoints, memory guard."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from lz import train
from lz.model import LeelaZeroNet, loss_fn
from test_model import random_batch

PLANE = 361


def batch(seed, n=8):
    planes, probs, winner = random_batch(np.random.default_rng(seed), n)
    return mx.array(planes), mx.array(probs), mx.array(winner)


def make_trainer(dtype="fp32", macrobatch=1, schedule="0:0.05", seed=0):
    mx.random.seed(seed)
    return train.Trainer(LeelaZeroNet(1, 8), dtype,
                         train.parse_schedule(schedule), macrobatch)


def params_flat(trainer):
    return {k: np.asarray(v) for k, v in tree_flatten(trainer.model.parameters())}


def test_schedule_parsing_and_lookup():
    s = train.parse_schedule("100000:0.005,0:0.05")
    assert s == [(0, 0.05), (100000, 0.005)]
    assert train.lr_at(s, 0) == 0.05 and train.lr_at(s, 99999) == 0.05
    assert train.lr_at(s, 100000) == 0.005 and train.lr_at(s, 10**9) == 0.005
    with pytest.raises(ValueError):
        train.parse_schedule("10:0.1")


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_compiled_steps_lower_the_loss(dtype):
    t = make_trainer(dtype, schedule="0:0.02")
    b = batch(1)
    first = float(t.evaluate(*b)[0]) + float(t.evaluate(*b)[1])
    for _ in range(40):
        t.train_batch(*b)
    mx.eval(t.model.parameters())
    last = float(t.evaluate(*b)[0]) + float(t.evaluate(*b)[1])
    assert last < first - 0.1, (first, last)


def test_macrobatch_sums_micro_batch_gradients():
    """k micro-batches with one update = summing their gradients by hand and
    applying one SGD step (tfprocess's gsum)."""
    a, b = batch(2), batch(3)
    t = make_trainer(macrobatch=2)
    before = params_flat(t)
    t.train_batch(*a)
    assert all(np.array_equal(before[k], v) for k, v in params_flat(t).items()
               if "running" not in k), "no update before k batches"
    t.train_batch(*b)
    mx.eval(t.model.parameters())
    after = params_flat(t)

    # By hand, from the same initial weights.
    mx.random.seed(0)
    model = LeelaZeroNet(1, 8)             # the same initial weights
    grad = nn.value_and_grad(model, loss_fn)
    model.train()
    (_, _), ga = grad(model, *a)
    (_, _), gb = grad(model, *b)
    total = tree_map(lambda x, y: x + y, ga, gb)
    expected = {k: v.copy() for k, v in before.items()}
    for k, g in tree_flatten(total):
        # SGD from zero momentum with Nesterov: update = g + 0.9 * g = 1.9 g.
        expected[k] = before[k] - 0.05 * 1.9 * np.asarray(g)
    for k in expected:
        if "running" in k:
            continue
        np.testing.assert_allclose(after[k], expected[k], rtol=2e-4, atol=2e-6,
                                   err_msg=k)


def test_checkpoint_restore_gives_the_same_next_step(tmp_path):
    t = make_trainer(schedule="0:0.02")
    for i in range(3):
        t.train_batch(*batch(10 + i))
    path = str(tmp_path / "ck.safetensors")
    t.save_checkpoint(path)
    nxt = batch(99)
    cont = float(t.train_batch(*nxt)[0])

    r = make_trainer(schedule="0:0.02", seed=5)       # different init
    r.load_checkpoint(path)
    assert r.step == 3
    resumed = float(r.train_batch(*nxt)[0])
    assert resumed == pytest.approx(cont, rel=1e-6)
    mx.eval(t.model.parameters(), r.model.parameters())
    for k, v in params_flat(t).items():
        np.testing.assert_allclose(params_flat(r)[k], v, rtol=1e-5, atol=1e-7)


def test_learning_rate_follows_the_schedule():
    t = make_trainer(schedule="0:0.05,2:0.005")
    for i in range(4):
        t.train_batch(*batch(i))
        mx.eval(t.optimizer.learning_rate)
        want = 0.05 if t.step - 1 < 2 else 0.005
        assert float(t.optimizer.learning_rate) == pytest.approx(want)


def test_fp16_loss_scaling_leaves_gradients_unscaled():
    a = batch(7)
    t32, t16 = make_trainer("fp32"), make_trainer("fp16")
    t32.train_batch(*a)
    t16.train_batch(*a)
    mx.eval(t32.model.parameters(), t16.model.parameters())
    p32, p16 = params_flat(t32), params_flat(t16)
    key = "input.conv.weight"
    d32 = p32[key] - np.asarray(make_trainer("fp32").model.input.conv.weight)
    d16 = p16[key] - np.asarray(make_trainer("fp16").model.input.conv.weight)
    assert np.linalg.norm(d16 - d32) < 0.05 * np.linalg.norm(d32)


def test_memory_guard_refuses_big_configs_and_allows_the_target():
    with pytest.raises(SystemExit):
        train.memory_guard(40, 256, 1024, "fp32", force=False)
    train.memory_guard(20, 256, 256, "bf16", force=False)     # N5 target config
    train.memory_guard(40, 256, 1024, "fp32", force=True)     # --force wins


def test_cli_trains_checkpoints_and_exports_a_net_leelaz_can_load(tmp_path, capsys):
    import os
    import subprocess
    from pathlib import Path

    from lz.convert import tensors_to_model
    from lz.weights import read_weights
    from test_data import write_chunks

    rng = np.random.default_rng(0)
    write_chunks(tmp_path, rng, files=12, per_file=64)
    out = tmp_path / "nets"
    rc = train.main([
        "--blocks", "1", "--filters", "8", "--train", str(tmp_path / "chunk"),
        "--batch", "16", "--steps", "30", "--info-steps", "10",
        "--test-every", "30", "--test-batches", "2", "--export-every", "30",
        "--sample", "1", "--shuffle-gb", "0.01", "--dtype", "bf16",
        "--export-dir", str(out), "--checkpoint-dir", str(tmp_path / "ck")])
    log = capsys.readouterr().out
    assert rc == 0 and "step 30, policy=" in log and "test policy=" in log
    exported = out / "lz-30.txt.gz"
    ckpt = tmp_path / "ck" / "step-30.safetensors"
    assert exported.exists() and ckpt.exists()
    version, tensors = read_weights(exported)
    assert version == 1
    model = tensors_to_model(tensors)
    logits, value = model(mx.zeros((1, 18 * PLANE), dtype=mx.uint8))
    assert np.isfinite(np.asarray(logits)).all()

    leelaz = Path(os.environ.get(
        "LEELAZ", Path(__file__).resolve().parents[3] / "build-metal" / "leelaz"))
    if leelaz.exists():
        res = subprocess.run([str(leelaz), "--backend", "cpu", "--gtp", "-q",
                              "-w", str(exported)], input="name\nquit\n",
                             capture_output=True, text=True, timeout=120)
        assert "= leelaz" in res.stdout.lower() or "leela" in res.stdout.lower(), res

    # And resume from the checkpoint with the exported weights' shape.
    rc = train.main([
        "--train", str(tmp_path / "chunk"), "--batch", "16", "--steps", "40",
        "--info-steps", "10", "--test-every", "1000", "--export-every", "1000",
        "--sample", "1", "--shuffle-gb", "0.01", "--dtype", "bf16",
        "--import-weights", str(exported), "--restore", str(ckpt),
        "--export-dir", str(out), "--checkpoint-dir", str(tmp_path / "ck")])
    assert rc == 0 and "Restored step 30" in capsys.readouterr().out


def test_memory_projection_matches_the_measured_peaks():
    gb = lambda *a: train.projected_peak_bytes(*a) / 2**30
    assert gb(20, 256, 64, "bf16") == pytest.approx(3.26, abs=0.3)
    assert gb(20, 256, 256, "bf16") == pytest.approx(10.55, abs=0.4)
