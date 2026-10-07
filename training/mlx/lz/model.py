"""The Leela Zero network in MLX, matching training/tf/tfprocess.py.

Facts F1-F6 of docs/apple-silicon/13-plan-phase3-mlx.md: NHWC inside, input
planes NCHW like tfprocess; batch norm without scale and with a separate beta
added after normalisation; head convolutions flattened in NCHW order; Xavier
truncated-normal initialisation; policy cross entropy + value MSE + 1e-4 L2 on
convolution and fully connected weights only.

Weights stay fp32 (the master copy the optimizer updates). The forward pass
runs in `compute_dtype` (fp32, bf16 or fp16) by casting the weights per call;
batch norm statistics, the heads' outputs and the loss are fp32.
"""

import math

import mlx.core as mx
import mlx.nn as nn

BOARD = 19
PLANE = BOARD * BOARD           # 361
INPUT_PLANES = 18
POLICY_OUTPUTS = PLANE + 1      # 362, the last one is pass
VALUE_HIDDEN = 256
BN_EPSILON = 1e-5
BN_MOMENTUM = 0.01              # TF's 0.99, as MLX's (1 - m) convention
L2_SCALE = 1e-4


def xavier(shape, tf_shape):
    """tfprocess.weight_variable: truncated normal, stddev sqrt(2 / sum)."""
    stddev = math.sqrt(2.0 / sum(tf_shape))
    # tf.truncated_normal re-draws values beyond two standard deviations.
    return stddev * mx.random.truncated_normal(-2.0, 2.0, shape)


@mx.checkpoint
def _bn_train(x, beta):
    """Batch statistics in fp32, output in x's dtype. Checkpointed: the
    backward pass recomputes the fp32 intermediates from x instead of keeping
    an fp32 copy of every activation (about half the training memory)."""
    xf = x.astype(mx.float32)
    axes = tuple(range(x.ndim - 1))
    mean = mx.mean(xf, axis=axes)
    var = mx.var(xf, axis=axes)
    unbiased = mx.var(xf, axis=axes, ddof=1)
    y = (xf - mean) * mx.rsqrt(var + BN_EPSILON) + beta
    return y.astype(x.dtype), mean, unbiased


def _bn_eval(x, beta, mean, var):
    xf = x.astype(mx.float32)
    return ((xf - mean) * mx.rsqrt(var + BN_EPSILON) + beta).astype(x.dtype)


class BatchNorm(nn.Module):
    """tf.layers.batch_normalization(center=True, scale=False, eps=1e-5).

    Statistics are computed in fp32 whatever the input dtype. The running
    variance uses the unbiased batch variance, like TF's fused batch norm.
    """

    def __init__(self, channels):
        super().__init__()
        self.beta = mx.zeros((channels,))
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))
        self.freeze(keys=["running_mean", "running_var"], recurse=False)
        # Fine-tuning on little data: normalise with the stored statistics and
        # leave them alone, even in training mode.
        self.use_running_stats = False

    def __call__(self, x):
        if self.training and not self.use_running_stats:
            y, mean, unbiased = _bn_train(x, self.beta)
            self.running_mean = ((1 - BN_MOMENTUM) * self.running_mean
                                 + BN_MOMENTUM * mean)
            self.running_var = ((1 - BN_MOMENTUM) * self.running_var
                                + BN_MOMENTUM * unbiased)
            return y
        return _bn_eval(x, self.beta, self.running_mean, self.running_var)


class Conv(nn.Module):
    """Bias-free 'same' convolution; weight is [out, kh, kw, in] (MLX)."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.weight = xavier((cout, k, k, cin), (k, k, cin, cout))
        self.padding = k // 2

    def __call__(self, x):
        return mx.conv2d(x, self.weight.astype(x.dtype), padding=self.padding)


class Linear(nn.Module):
    """Fully connected layer; weight is [out, in], like leelaz's files."""

    def __init__(self, cin, cout):
        super().__init__()
        self.weight = xavier((cout, cin), (cin, cout))
        self.bias = mx.zeros((cout,))

    def __call__(self, x):
        return (x @ self.weight.astype(x.dtype).T
                + self.bias.astype(x.dtype))


class ConvBlock(nn.Module):
    """Convolution, batch norm, then ReLU unless relu=False."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = Conv(cin, cout, k)
        self.bn = BatchNorm(cout)

    def __call__(self, x, relu=True):
        y = self.bn(self.conv(x))
        return nn.relu(y) if relu else y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3)
        self.conv2 = ConvBlock(channels, channels, 3)

    def __call__(self, x):
        y = self.conv2(self.conv1(x), relu=False)
        return nn.relu(y + x)


def nchw_flatten(x):
    """[B, H, W, C] -> [B, C*H*W] in TF's NCHW order (fact F4)."""
    return x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)


class LeelaZeroNet(nn.Module):
    def __init__(self, blocks, filters):
        super().__init__()
        self.blocks = blocks
        self.filters = filters
        self.compute_dtype = mx.float32
        self.input = ConvBlock(INPUT_PLANES, filters, 3)
        self.tower = [ResidualBlock(filters) for _ in range(blocks)]
        self.policy_conv = ConvBlock(filters, 2, 1)
        self.policy_fc = Linear(2 * PLANE, POLICY_OUTPUTS)
        self.value_conv = ConvBlock(filters, 1, 1)
        self.value_fc1 = Linear(PLANE, VALUE_HIDDEN)
        self.value_fc2 = Linear(VALUE_HIDDEN, 1)

    def __call__(self, planes):
        """planes: [B, 18*361] or [B, 18, 361], 0/1 values (uint8 is fine).

        Returns fp32 (policy logits [B, 362], value [B] in [-1, 1]).
        """
        x = planes.reshape(-1, INPUT_PLANES, BOARD, BOARD)
        x = x.transpose(0, 2, 3, 1).astype(self.compute_dtype)
        x = self.input(x)
        for block in self.tower:
            x = block(x)
        policy = self.policy_fc(nchw_flatten(self.policy_conv(x)))
        hidden = nn.relu(self.value_fc1(nchw_flatten(self.value_conv(x))))
        value = mx.tanh(self.value_fc2(hidden).astype(mx.float32))
        return policy.astype(mx.float32), value.reshape(-1)

    def conv_blocks(self):
        """Every ConvBlock in leelaz file order (fact F1)."""
        blocks = [self.input]
        for block in self.tower:
            blocks += [block.conv1, block.conv2]
        return blocks + [self.policy_conv, self.value_conv]

    def freeze_batchnorm_statistics(self, frozen=True):
        """Train with the stored batch norm statistics (they are not updated):
        the usual way to fine-tune a strong net on a small dataset, whose
        statistics are not representative (plan 13, T6)."""
        for block in self.conv_blocks():
            block.bn.use_running_stats = frozen

    def regularized_weights(self):
        """The tensors tfprocess puts in tf.GraphKeys.WEIGHTS."""
        return ([b.conv.weight for b in self.conv_blocks()]
                + [self.policy_fc.weight, self.value_fc1.weight,
                   self.value_fc2.weight])


def loss_fn(model, planes, probs, winner, value_weight=1.0):
    """tfprocess.tower_loss. Returns (total, (policy, mse, reg, accuracy)).

    `mse` is the raw mean squared error; tfprocess reports mse / 4.
    `value_weight` scales the value loss in the total; tfprocess advises
    lowering it when training on a small dataset (the value head memorises
    game results).
    """
    logits, value = model(planes)
    policy = mx.mean(-mx.sum(probs * nn.log_softmax(logits, axis=-1), axis=-1))
    mse = mx.mean(mx.square(winner.reshape(-1) - value))
    reg = L2_SCALE * sum(0.5 * mx.sum(mx.square(w))
                         for w in model.regularized_weights())
    accuracy = mx.mean((mx.argmax(logits, axis=-1)
                        == mx.argmax(probs, axis=-1)).astype(mx.float32))
    return policy + value_weight * mse + reg, (policy, mse, reg, accuracy)
