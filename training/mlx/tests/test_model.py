"""Gate T1: the MLX network and loss against an independent NumPy reference
written from tfprocess.py (NCHW, like TF), plus shapes and batch norm."""

import mlx.core as mx
import numpy as np
import pytest

from lz.model import (BN_EPSILON, BN_MOMENTUM, LeelaZeroNet, PLANE, loss_fn)


def leelaz_tensor_count(blocks, filters):
    """Numbers in a leelaz v1 weights file (fact F1)."""
    def conv(cin, cout, k):
        return cin * cout * k * k + 3 * cout   # W, bias(beta), mean, var
    return (conv(18, filters, 3) + 2 * blocks * conv(filters, filters, 3)
            + conv(filters, 2, 1) + 2 * PLANE * (PLANE + 1) + (PLANE + 1)
            + conv(filters, 1, 1) + PLANE * 256 + 256 + 256 + 1)


def random_batch(rng, n):
    planes = (rng.random((n, 18 * PLANE)) < 0.3).astype(np.uint8)
    probs = rng.random((n, PLANE + 1)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    winner = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return planes, probs, winner


def randomize_bn(model, rng):
    """Non-trivial beta / running stats so eval-mode batch norm is tested."""
    for block in model.conv_blocks():
        c = block.bn.beta.shape[0]
        block.bn.beta = mx.array(rng.normal(0, 0.1, c).astype(np.float32))
        block.bn.running_mean = mx.array(rng.normal(0, 0.1, c).astype(np.float32))
        block.bn.running_var = mx.array(rng.uniform(0.5, 1.5, c).astype(np.float32))


# --- NumPy reference (NCHW, as tfprocess) -----------------------------------

def np_conv(x, w_mlx):
    """x [B,C,19,19], MLX weight [out,kh,kw,in] -> [B,out,19,19], 'same'."""
    w = np.asarray(w_mlx).transpose(0, 3, 1, 2)          # [out, in, kh, kw]
    k = w.shape[2]
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    out = np.zeros((x.shape[0], w.shape[0], 19, 19), np.float64)
    for dy in range(k):
        for dx in range(k):
            patch = xp[:, :, dy:dy + 19, dx:dx + 19]
            out += np.einsum("bchw,oc->bohw", patch, w[:, :, dy, dx])
    return out


def np_block(x, block, training, relu=True):
    y = np_conv(x, block.conv.weight)
    if training:
        mean = y.mean(axis=(0, 2, 3))
        var = y.var(axis=(0, 2, 3))
    else:
        mean = np.asarray(block.bn.running_mean)
        var = np.asarray(block.bn.running_var)
    beta = np.asarray(block.bn.beta)
    y = ((y - mean[None, :, None, None]) / np.sqrt(var + BN_EPSILON)[None, :, None, None]
         + beta[None, :, None, None])
    return np.maximum(y, 0) if relu else y


def np_forward(model, planes, training):
    x = planes.reshape(-1, 18, 19, 19).astype(np.float64)
    x = np_block(x, model.input, training)
    for res in model.tower:
        y = np_block(np_block(x, res.conv1, training), res.conv2, training, relu=False)
        x = np.maximum(y + x, 0)
    pol = np_block(x, model.policy_conv, training).reshape(len(x), -1)  # NCHW flatten
    logits = pol @ np.asarray(model.policy_fc.weight).T + np.asarray(model.policy_fc.bias)
    val = np_block(x, model.value_conv, training).reshape(len(x), -1)
    h = np.maximum(val @ np.asarray(model.value_fc1.weight).T
                   + np.asarray(model.value_fc1.bias), 0)
    v = np.tanh(h @ np.asarray(model.value_fc2.weight).T
                + np.asarray(model.value_fc2.bias)).reshape(-1)
    return logits, v


def np_loss(model, planes, probs, winner, training):
    logits, v = np_forward(model, planes, training)
    z = logits - logits.max(axis=1, keepdims=True)
    log_softmax = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    policy = np.mean(-(probs * log_softmax).sum(axis=1))
    mse = np.mean((winner - v) ** 2)
    reg = 1e-4 * sum(0.5 * np.sum(np.asarray(w, np.float64) ** 2)
                     for w in model.regularized_weights())
    return policy, mse, reg


# --- Tests ------------------------------------------------------------------

@pytest.mark.parametrize("blocks,filters", [(1, 8), (6, 64)])
def test_shapes_and_parameter_count(blocks, filters):
    model = LeelaZeroNet(blocks, filters)
    logits, value = model(mx.zeros((4, 18 * PLANE), dtype=mx.uint8))
    assert logits.shape == (4, PLANE + 1) and logits.dtype == mx.float32
    assert value.shape == (4,)
    leaves = [v for _, v in __import__("mlx.utils", fromlist=["tree_flatten"])
              .tree_flatten(model.parameters())]
    assert sum(v.size for v in leaves) == leelaz_tensor_count(blocks, filters)


@pytest.mark.parametrize("training", [False, True])
def test_loss_matches_numpy(training):
    rng = np.random.default_rng(7)
    mx.random.seed(7)
    model = LeelaZeroNet(1, 8)
    randomize_bn(model, rng)
    model.train(training)
    planes, probs, winner = random_batch(rng, 4)
    ref_policy, ref_mse, ref_reg = np_loss(model, planes, probs, winner, training)
    total, (policy, mse, reg, _) = loss_fn(
        model, mx.array(planes), mx.array(probs), mx.array(winner))
    assert abs(policy.item() - ref_policy) <= 1e-5
    assert abs(mse.item() - ref_mse) <= 1e-5
    assert abs(reg.item() - ref_reg) <= 1e-5 * max(1.0, ref_reg)
    assert abs(total.item() - (ref_policy + ref_mse + ref_reg)) <= 3e-5


def test_head_flatten_is_nchw():
    """A policy-FC weight reading only plane 1 must see plane 1 (fact F4)."""
    rng = np.random.default_rng(1)
    model = LeelaZeroNet(1, 8)
    randomize_bn(model, rng)
    model.eval()
    w = np.zeros((PLANE + 1, 2 * PLANE), np.float32)
    w[0, PLANE:] = 1.0            # output 0 = sum of policy-conv channel 1
    model.policy_fc.weight = mx.array(w)
    planes = (rng.random((2, 18 * PLANE)) < 0.3).astype(np.uint8)
    logits, _ = model(mx.array(planes))
    ref, _ = np_forward(model, planes, training=False)
    assert np.allclose(np.asarray(logits)[:, 0], ref[:, 0], atol=1e-4)


def test_batchnorm_running_stats_follow_tf():
    """TF: moving = 0.99 * moving + 0.01 * batch, variance unbiased."""
    rng = np.random.default_rng(3)
    model = LeelaZeroNet(1, 8)
    model.train()
    planes, _, _ = random_batch(rng, 4)
    x = planes.reshape(-1, 18, 19, 19).astype(np.float64)
    y = np_conv(x, model.input.conv.weight)
    model(mx.array(planes))
    bn = model.input.bn
    mean = y.mean(axis=(0, 2, 3))
    var = y.var(axis=(0, 2, 3), ddof=1)
    assert np.allclose(np.asarray(bn.running_mean), BN_MOMENTUM * mean, atol=1e-5)
    # Recover the batch variance that went into the running average: biased
    # and unbiased differ by only 1/(n-1) = 0.07% here, so compare tightly.
    used = (np.asarray(bn.running_var, np.float64) - (1 - BN_MOMENTUM)) / BN_MOMENTUM
    assert np.max(np.abs(used / var - 1)) < 1e-4


def test_running_stats_are_not_trainable():
    model = LeelaZeroNet(1, 8)
    names = [k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"])
             .tree_flatten(model.trainable_parameters())]
    assert not any("running" in n for n in names)
    assert any(n.endswith("bn.beta") for n in names)


def test_bf16_forward_is_close_to_fp32():
    rng = np.random.default_rng(5)
    model = LeelaZeroNet(2, 16)
    randomize_bn(model, rng)
    model.eval()
    planes = mx.array((rng.random((4, 18 * PLANE)) < 0.3).astype(np.uint8))
    ref_logits, ref_value = model(planes)
    model.compute_dtype = mx.bfloat16
    logits, value = model(planes)
    assert logits.dtype == mx.float32
    # bf16 keeps ~3 significant digits: compare relative to the logits' size.
    diff = np.asarray(logits) - np.asarray(ref_logits)
    assert np.linalg.norm(diff) / np.linalg.norm(np.asarray(ref_logits)) < 0.05
    assert np.max(np.abs(np.asarray(value) - np.asarray(ref_value))) < 0.05


def test_initialisation_follows_tfprocess():
    mx.random.seed(11)
    model = LeelaZeroNet(1, 64)
    w = np.asarray(model.tower[0].conv1.conv.weight)
    stddev = np.sqrt(2.0 / (3 + 3 + 64 + 64))
    assert np.abs(w).max() <= 2 * stddev + 1e-7          # truncated at 2 sigma
    # A normal truncated at +-2 sigma has std 0.880 sigma.
    assert abs(w.std() / stddev - 0.880) < 0.03
    assert np.all(np.asarray(model.policy_fc.bias) == 0)
