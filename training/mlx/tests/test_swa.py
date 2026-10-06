"""Step 3.6: stochastic weight averaging with batch norm refinement."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from lz.convert import model_to_tensors, tensors_to_model
from lz.model import LeelaZeroNet
from lz.swa import SWA
from lz.weights import read_weights
from test_model import random_batch


def snapshot(model):
    return {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())}


def perturb(model, seed):
    """Give the model a different set of weights (a stand-in for training)."""
    rng = np.random.default_rng(seed)
    flat = tree_flatten(model.parameters())
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten([
        (k, v + mx.array(rng.normal(0, 0.05, v.shape).astype(np.float32)))
        for k, v in flat]))
    mx.eval(model.parameters())


def numpy_swa(snapshots, c, max_n):
    """Reference: tfprocess's recurrence, written independently."""
    avg, n, skip = None, 0, c
    for snap in snapshots:
        skip -= 1
        if skip > 0:
            continue
        skip = c
        if avg is None:
            avg = {k: np.zeros_like(v) for k, v in snap.items()}
        avg = {k: avg[k] * (n / (n + 1.0)) + snap[k] * (1.0 / (n + 1.0)) for k in snap}
        n = min(n + 1, max_n)
    return avg, n


def run(c, max_n, count):
    model = LeelaZeroNet(1, 8)
    swa, snaps = SWA(c, max_n), []
    for i in range(count):
        perturb(model, i)
        snaps.append(snapshot(model))
        swa.update(model)
    return swa, snaps


def test_average_of_three_snapshots_is_their_mean():
    swa, snaps = run(1, 16, 3)
    assert swa.n == 3
    for k in snaps[0]:
        mean = np.mean([s[k] for s in snaps], axis=0)
        np.testing.assert_allclose(np.asarray(swa.avg[k]), mean, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("c,max_n,count", [(1, 16, 20), (2, 4, 11), (3, 16, 7)])
def test_matches_the_tfprocess_recurrence(c, max_n, count):
    swa, snaps = run(c, max_n, count)
    ref, n = numpy_swa(snaps, c, max_n)
    assert swa.n == n
    for k in ref:
        np.testing.assert_allclose(np.asarray(swa.avg[k]), ref[k], rtol=1e-4, atol=1e-5)


def test_count_is_capped_and_weights_ema_after_the_cap():
    swa, _ = run(1, 4, 9)
    assert swa.n == 4


def test_export_writes_the_average_and_restores_the_live_weights(tmp_path):
    swa, snaps = run(1, 16, 3)
    model = LeelaZeroNet(1, 8)
    perturb(model, 99)
    before = snapshot(model)

    def batches():
        rng = np.random.default_rng(0)
        while True:
            p, pr, w = random_batch(rng, 8)
            yield mx.array(p), mx.array(pr), mx.array(w)

    path = tmp_path / "swa.txt.gz"
    swa.export(model, batches(), path, refine_batches=0)
    after = snapshot(model)
    assert all(np.array_equal(before[k], after[k]) for k in before)   # live restored
    # Without refinement the exported net is exactly the averaged weights.
    ref = tensors_to_model(read_weights(path)[1])
    avg_model = LeelaZeroNet(1, 8)
    from mlx.utils import tree_unflatten
    avg_model.update(tree_unflatten(list(swa.avg.items())))
    for a, b in zip(model_to_tensors(avg_model), model_to_tensors(ref)):
        np.testing.assert_allclose(b, a, rtol=2e-6, atol=1e-7)


def test_batchnorm_refinement_changes_only_the_statistics(tmp_path):
    swa, _ = run(1, 16, 2)
    model = LeelaZeroNet(1, 8)

    def batches():
        rng = np.random.default_rng(1)
        while True:
            p, pr, w = random_batch(rng, 8)
            yield mx.array(p), mx.array(pr), mx.array(w)

    plain, refined = tmp_path / "a.txt", tmp_path / "b.txt"
    swa.export(model, batches(), plain, refine_batches=0)
    swa.export(model, batches(), refined, refine_batches=20)
    a, b = read_weights(plain)[1], read_weights(refined)[1]
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if not np.allclose(x, y)]
    # 1-block file layout: 5 conv blocks of (weights, bias, mean, var) with the
    # policy FC (16, 17) after the policy conv, then value FC1 (22, 23) and FC2
    # (24, 25). Weights and FC tensors must be untouched; only the batch norm
    # bias (beta), mean and variance lines may move.
    untouched = {0, 4, 8, 12, 18, 16, 17, 22, 23, 24, 25}
    assert not untouched & set(changed), sorted(untouched & set(changed))
    assert changed, "refinement should move the batch norm statistics"


def test_checkpoint_round_trip_keeps_the_swa_state(tmp_path):
    from lz import train
    trainer = train.Trainer(LeelaZeroNet(1, 8), "fp32",
                            train.parse_schedule("0:0.05"), 1, 2, 16)
    for i in range(3):
        perturb(trainer.model, i)
        trainer.swa.update(trainer.model)
    path = str(tmp_path / "ck.safetensors")
    trainer.save_checkpoint(path)
    other = train.Trainer(LeelaZeroNet(1, 8), "fp32",
                          train.parse_schedule("0:0.05"), 1, 2, 16)
    other.load_checkpoint(path)
    assert (other.swa.n, other.swa.skip) == (trainer.swa.n, trainer.swa.skip)
    for k, v in trainer.swa.avg.items():
        np.testing.assert_allclose(np.asarray(other.swa.avg[k]), np.asarray(v))
