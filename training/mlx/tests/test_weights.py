"""Step 3.3: leelaz weight files <-> MLX, gates T2 (round trip through leelaz)
and T3 (MLX forward vs leelaz CPU). T2/T3 need a leelaz build and are skipped
without one (CI has none); run them by hand, see BENCHMARKS.md."""

import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from lz import data
from lz.convert import model_to_tensors, shape_from_tensors, tensors_to_model
from lz.weights import read_weights, write_weights

REPO = Path(__file__).resolve().parents[3]
LEELAZ = Path(os.environ.get("LEELAZ", REPO / "build-metal" / "leelaz"))
PLANE = 361


def random_tensors(blocks, filters, seed):
    """A leelaz net with non-trivial biases, means and variances (F1 order)."""
    rng = np.random.default_rng(seed)

    def conv(cin, cout, k):
        scale = (1.0 / (cin * k * k)) ** 0.5
        return [rng.normal(0, scale, cout * cin * k * k),
                rng.normal(0, 0.1, cout),            # "bias"
                rng.normal(0, 0.1, cout),            # mean
                rng.uniform(0.5, 1.5, cout)]         # variance

    def fc(cin, cout):
        return [rng.normal(0, (1.0 / cin) ** 0.5, cout * cin),
                rng.normal(0, 0.05, cout)]

    t = conv(18, filters, 3)
    for _ in range(blocks):
        t += conv(filters, filters, 3) + conv(filters, filters, 3)
    t += conv(filters, 2, 1) + fc(2 * PLANE, PLANE + 1)
    t += conv(filters, 1, 1) + fc(PLANE, 256) + fc(256, 1)
    return [np.asarray(x, np.float32) for x in t]


def test_shape_inference():
    t = random_tensors(3, 24, 0)
    assert len(t) == 18 + 8 * 3
    assert shape_from_tensors(t) == (3, 24)
    with pytest.raises(ValueError):
        shape_from_tensors(t[:-1])


@pytest.mark.parametrize("blocks,filters", [(1, 8), (3, 16)])
def test_tensors_round_trip_through_the_model(blocks, filters):
    original = random_tensors(blocks, filters, 1)
    exported = model_to_tensors(tensors_to_model(original))
    assert len(exported) == len(original)
    for a, b in zip(original, exported):
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=2e-6, atol=1e-7)


def test_text_and_gzip_round_trip_is_lossless(tmp_path):
    t = random_tensors(1, 8, 2)
    for name in ("net.txt", "net.txt.gz"):
        write_weights(tmp_path / name, t)
        version, back = read_weights(tmp_path / name)
        assert version == 1
        assert all(np.array_equal(a, b) for a, b in zip(t, back))
    assert gzip.open(tmp_path / "net.txt.gz", "rt").readline() == "1\n"


def test_version_two_is_read_only(tmp_path):
    p = tmp_path / "elf.txt"
    write_weights(p, random_tensors(1, 8, 3))
    p.write_text(p.read_text().replace("1\n", "2\n", 1))
    assert read_weights(p)[0] == 2
    p.write_text(p.read_text().replace("2\n", "3\n", 1))
    with pytest.raises(ValueError):
        read_weights(p)


def test_wrong_sized_weights_are_rejected():
    t = random_tensors(1, 8, 4)
    t[0] = t[0][:-1]
    with pytest.raises(ValueError):
        tensors_to_model(t)


# --- T2 and T3: need leelaz --------------------------------------------------

needs_leelaz = pytest.mark.skipif(not LEELAZ.exists(),
                                  reason="no leelaz build (set LEELAZ)")


@pytest.fixture(scope="module")
def net_and_games(tmp_path_factory):
    work = tmp_path_factory.mktemp("t23")
    net = work / "net.txt"
    write_weights(net, random_tensors(2, 16, 5))
    subprocess.run(
        [sys.executable, str(REPO / "scripts/macos/selfplay.py"),
         "--leelaz", "%s --backend cpu" % LEELAZ, "-w", str(net),
         "-o", str(work / "games"), "-n", "2", "-v", "2", "--max-moves", "120"],
        check=True, capture_output=True)
    return net, work / "games"


@needs_leelaz
def test_t2_export_is_the_same_net_for_leelaz(net_and_games, tmp_path):
    net, games = net_and_games
    reexported = tmp_path / "re.txt.gz"
    write_weights(reexported,
                  model_to_tensors(tensors_to_model(read_weights(net)[1])))
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts/parity/compare_backends.py"),
         "--ref", "%s --backend cpu" % LEELAZ, "--test", "%s --backend cpu" % LEELAZ,
         "-w", str(net), "--test-weights", str(reexported),
         "--sgf", str(games / "*.sgf"), "--moves", "0,10,30,60", "--tol", "1e-6"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def leelaz_eval(net, sgf, moves):
    sys.path.insert(0, str(REPO / "scripts" / "common"))
    from gtp import GTPEngine
    eng = GTPEngine([str(LEELAZ), "--backend", "cpu", "--gtp", "-q", "-w", str(net),
                     "--noponder", "-t", "1"])
    out = []
    try:
        for k in moves:
            eng.send("clear_board")
            eng.send("loadsgf %s %d" % (sgf, k + 1))
            vals = [float(x) for x in eng.send("lz-nn-eval 0").split()]
            out.append((vals[0], np.array(vals[1:], np.float32)))   # winrate, [pass]+361
    finally:
        eng.close()
    return out


@needs_leelaz
def test_t3_mlx_forward_matches_leelaz(net_and_games):
    """Chunk record k holds the input planes leelaz used after k moves."""
    net, games = net_and_games
    model = tensors_to_model(read_weights(net)[1])
    model.eval()
    chunk = sorted(games.glob("*.gz"))[0]
    sgf = chunk.with_name(chunk.name.split(".")[0] + ".sgf")
    parser = data.chunkparser.ChunkParser.__new__(data.chunkparser.ChunkParser)
    parser.flat_planes = [b"\1" * 361 + b"\0" * 361, b"\0" * 361 + b"\1" * 361]
    parser.sample = 1
    parser.init_structs()
    records = list(parser.convert_chunkdata_to_v2(gzip.open(chunk, "rb").read()))
    moves = [0, 1, 7, 20, 50]
    moves = [k for k in moves if k < len(records)]
    planes = np.zeros((len(moves), data.PLANES_BYTES), np.uint8)
    probs = np.zeros((len(moves), data.POLICY), np.float32)
    winner = np.zeros(len(moves), np.float32)
    data.decode_v2(b"".join(records[k] for k in moves), planes, probs, winner)
    logits, value = model(mx.array(planes))
    prior = np.asarray(mx.softmax(logits, axis=-1))
    for row, (win, ref) in enumerate(leelaz_eval(net, sgf, moves)):
        mlx_prior = np.concatenate([[prior[row, 361]], prior[row, :361]])
        assert abs((float(value[row]) + 1) / 2 - win) <= 1e-4, "winrate"
        assert np.max(np.abs(mlx_prior - ref)) <= 1e-4, "policy"
