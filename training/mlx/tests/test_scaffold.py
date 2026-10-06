import mlx.core as mx


def test_mlx_runs_on_the_gpu():
    x = mx.arange(4, dtype=mx.float32)
    assert (x * 2).sum().item() == 12.0
    assert mx.default_device() == mx.gpu
