"""MLX parameters <-> leelaz weight tensors (plan 13, facts F1-F3).

Per convolution block the file holds: weights [out, in, kh, kw], a "bias"
equal to beta * sqrt(var + eps), the batch norm mean, and its variance. Then
the fully connected layers as [out, in] matrices (which is MLX's layout too).
"""

import mlx.core as mx
import numpy as np

from .model import BN_EPSILON, INPUT_PLANES, LeelaZeroNet


def bias_from_beta(beta, var):
    """File "bias" = beta * sqrt(var + eps) (tfprocess's back-compat trick)."""
    return (beta.astype(np.float64)
            * np.sqrt(var.astype(np.float64) + BN_EPSILON)).astype(np.float32)


def beta_from_bias(bias, var):
    """The inverse. Not always exact: when sqrt(var + eps) > 1, neighbouring
    float32 betas map to biases more than one float32 step apart, so an import
    followed by an export can move a bias by one step (0.006% of the values
    of a 15b x 192 net)."""
    scale = np.sqrt(var.astype(np.float64) + BN_EPSILON)
    return (bias.astype(np.float64) / scale).astype(np.float32)


def shape_from_tensors(tensors):
    """(blocks, filters) from a tensor list: 18 + 8 * blocks lines, and the
    first convolution has filters * 18 * 9 weights."""
    n = len(tensors)
    if n < 18 or (n - 18) % 8:
        raise ValueError("%d weight lines do not form a Leela Zero network" % n)
    blocks = (n - 18) // 8
    size = tensors[0].size
    if size % (INPUT_PLANES * 9):
        raise ValueError("first convolution has %d weights" % size)
    return blocks, size // (INPUT_PLANES * 9)


def model_to_tensors(model):
    """Flat float32 arrays in leelaz file order."""
    out = []
    for block in model.conv_blocks():
        w = np.asarray(block.conv.weight, np.float32)
        out.append(w.transpose(0, 3, 1, 2).ravel())            # [out,in,kh,kw]
        var = np.asarray(block.bn.running_var, np.float32)
        beta = np.asarray(block.bn.beta, np.float32)
        out.append(bias_from_beta(beta, var))
        out.append(np.asarray(block.bn.running_mean, np.float32))
        out.append(var)
    # Policy conv is conv_blocks()[-2] and value conv [-1], but the file puts
    # the policy FC between them: reorder to F1.
    n_conv = len(model.conv_blocks())
    conv_part = out[:4 * (n_conv - 2)]
    policy_conv = out[4 * (n_conv - 2):4 * (n_conv - 1)]
    value_conv = out[4 * (n_conv - 1):]
    fc = lambda layer: [np.asarray(layer.weight, np.float32).ravel(),
                        np.asarray(layer.bias, np.float32)]
    return (conv_part + policy_conv + fc(model.policy_fc) + value_conv
            + fc(model.value_fc1) + fc(model.value_fc2))


def tensors_to_model(tensors, model=None):
    """Build (or fill) a LeelaZeroNet from leelaz tensors."""
    blocks, filters = shape_from_tensors(tensors)
    if model is None:
        model = LeelaZeroNet(blocks, filters)
    elif (model.blocks, model.filters) != (blocks, filters):
        raise ValueError("network is %dx%d, weights are %dx%d"
                         % (model.blocks, model.filters, blocks, filters))
    n_conv = len(model.conv_blocks())
    it = iter(tensors)

    def load_conv(block):
        w, bias, mean, var = next(it), next(it), next(it), next(it)
        out_ch, _, _, in_ch = block.conv.weight.shape
        expect = out_ch * in_ch * block.conv.weight.shape[1] ** 2
        if w.size != expect or not (bias.size == mean.size == var.size == out_ch):
            raise ValueError("convolution tensor has the wrong size")
        k = block.conv.weight.shape[1]
        block.conv.weight = mx.array(
            w.reshape(out_ch, in_ch, k, k).transpose(0, 2, 3, 1))
        block.bn.running_mean = mx.array(mean)
        block.bn.running_var = mx.array(var)
        block.bn.beta = mx.array(beta_from_bias(bias, var))

    def load_fc(layer):
        w, b = next(it), next(it)
        out_f, in_f = layer.weight.shape
        if w.size != out_f * in_f or b.size != out_f:
            raise ValueError("fully connected tensor has the wrong size")
        layer.weight = mx.array(w.reshape(out_f, in_f))
        layer.bias = mx.array(b)

    convs = model.conv_blocks()
    for block in convs[:n_conv - 2]:
        load_conv(block)
    load_conv(model.policy_conv)
    load_fc(model.policy_fc)
    load_conv(model.value_conv)
    load_fc(model.value_fc1)
    load_fc(model.value_fc2)
    return model
