"""Step 3.4: the shared-memory data pipeline."""

import gzip
import struct
import time
from collections import Counter
from multiprocessing import shared_memory

import numpy as np
import pytest

from lz import data


def make_records(rng, n, first_id=0):
    """n v2 records; the pass probability (never moved by a symmetry) is a
    unique id, so records can be recognised after augmentation."""
    recs = np.zeros(n, dtype=data.V2)
    recs["version"] = 2
    probs = rng.random((n, data.POLICY)).astype(np.float32)
    probs[:, :data.PLANE] /= probs[:, :data.PLANE].sum(axis=1, keepdims=True)
    probs[:, data.PLANE] = np.arange(first_id, first_id + n, dtype=np.float32)
    recs["probs"] = probs
    recs["planes"] = rng.integers(0, 256, (n, 722), dtype=np.uint8)
    recs["stm"] = rng.integers(0, 2, n)
    recs["winner"] = rng.integers(0, 2, n)
    return recs


def write_chunks(tmp_path, rng, files, per_file):
    paths, next_id = [], 0
    for i in range(files):
        recs = make_records(rng, per_file, next_id)
        next_id += per_file
        # Chunk files start with the v2 version bytes (convert_chunkdata_to_v2).
        recs["version"] = 1
        path = tmp_path / "chunk{}.gz".format(i)
        with gzip.open(path, "wb") as f:
            f.write(recs.tobytes())
        paths.append(str(path))
    return paths, next_id


def test_decode_matches_chunkparser():
    rng = np.random.default_rng(0)
    recs = make_records(rng, 16)
    parser = data.chunkparser.ChunkParser.__new__(data.chunkparser.ChunkParser)
    parser.flat_planes = [b"\1" * 361 + b"\0" * 361, b"\0" * 361 + b"\1" * 361]
    parser.init_structs()
    planes = np.zeros((16, data.PLANES_BYTES), np.uint8)
    probs = np.zeros((16, data.POLICY), np.float32)
    winner = np.zeros(16, np.float32)
    data.decode_v2(recs.tobytes(), planes, probs, winner)
    for i in range(16):
        ref_planes, ref_probs, ref_winner = parser.convert_v2_to_tuple(
            recs[i:i + 1].tobytes())
        assert planes[i].tobytes() == ref_planes
        assert probs[i].tobytes() == ref_probs
        assert struct.unpack("f", ref_winner)[0] == winner[i]


def test_every_record_arrives(tmp_path):
    rng = np.random.default_rng(1)
    chunks, total = write_chunks(tmp_path, rng, files=4, per_file=64)
    seen = Counter()
    with data.Batches(chunks, batch=32, shuffle_bytes=64 * data.V2.itemsize,
                      workers=2) as batches:
        for _ in range(40):                      # 1280 records = 5 epochs
            planes, probs, winner = batches.next()
            assert planes.shape == (32, data.PLANES_BYTES)
            assert set(np.unique(np.asarray(winner))) <= {-1.0, 1.0}
            seen.update(int(x) for x in np.asarray(probs)[:, data.PLANE])
    assert set(seen) == set(range(total))
    # Shuffling and worker interleaving make counts uneven, but no record
    # should be missing for long or repeated far too often.
    assert min(seen.values()) >= 1 and max(seen.values()) <= 15


def test_slow_consumer_and_clean_shutdown(tmp_path):
    rng = np.random.default_rng(2)
    chunks, _ = write_chunks(tmp_path, rng, files=2, per_file=32)
    batches = data.Batches(chunks, batch=16, slots=2,
                           shuffle_bytes=16 * data.V2.itemsize, workers=1)
    name = batches._shm.name
    first = None
    for i in range(6):
        planes, _, _ = batches.next()
        if i == 0:
            first = np.asarray(planes).copy()   # must not change under us
        time.sleep(0.2)                           # feeder blocks on free slots
    assert first is not None
    batches.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)    # unlinked, not leaked


def test_default_shuffle_size_is_bounded():
    assert 0 < data.default_shuffle_bytes() <= int(1.5 * (1 << 30))


def test_dead_feeder_raises_instead_of_hanging(tmp_path):
    rng = np.random.default_rng(3)
    chunks, _ = write_chunks(tmp_path, rng, files=1, per_file=16)
    batches = data.Batches(chunks, batch=8, shuffle_bytes=8 * data.V2.itemsize,
                           workers=1)
    batches.next()
    batches._proc.kill()
    batches._proc.join()
    with pytest.raises(RuntimeError, match="feeder"):
        for _ in range(10):          # slots already filled may still drain
            batches.next()
    batches.close()


def test_feeder_exits_when_the_trainer_is_killed(tmp_path):
    """No orphaned feeder or workers after the trainer dies (review 3.R)."""
    import os
    import signal
    import subprocess
    import sys
    rng = np.random.default_rng(4)
    chunks, _ = write_chunks(tmp_path, rng, files=2, per_file=32)
    script = tmp_path / "trainer.py"
    script.write_text(
        "import sys, time\nfrom lz import data\n"
        "if __name__ == '__main__':\n"
        "    b = data.Batches(sys.argv[1:], 8, shuffle_bytes=8 * data.V2.itemsize, workers=1)\n"
        "    b.next()\n"
        "    print(b._proc.pid, flush=True)\n"
        "    time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(script)] + chunks,
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    while not line.strip().isdigit():         # ChunkParser prints a banner
        line = proc.stdout.readline()
    feeder = int(line)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    for _ in range(100):
        try:
            os.kill(feeder, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(feeder, signal.SIGKILL)
        pytest.fail("feeder still running 10 s after the trainer died")
