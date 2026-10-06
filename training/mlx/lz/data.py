"""Training data: chunk files -> shuffled, augmented batches in shared memory.

    ChunkParser workers (training/tf/chunkparser.py, unchanged): read chunks,
        apply a random symmetry, send v2 records over pipes
    feeder process: shuffle buffer (training/tf/shufflebuffer.py), decode a
        whole batch with NumPy, write it into a free slot of the ring
    trainer: Batches.next() -> mx arrays copied out of the slot, slot freed

The ring has R slots in one multiprocessing.shared_memory block; slot indices
travel through two queues (free, ready), so a slow trainer blocks the feeder
and never the other way round. Plan 13, step 3.4.
"""

import gzip
import multiprocessing as mp
import os
import queue
import random
import sys
from multiprocessing import shared_memory

import numpy as np

_TF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tf")
if _TF_DIR not in sys.path:
    sys.path.insert(0, _TF_DIR)
import chunkparser                        # noqa: E402  (TF-free)
import shufflebuffer                      # noqa: E402  (TF-free)

PLANE = 361
PLANES_BYTES = 18 * PLANE                 # 6498 uint8 per position
POLICY = PLANE + 1

# chunkparser's v2 record: int32 version, 362 float32 probabilities, 16x361
# packed bit planes, side to move, winner (1 = side to move won).
V2 = np.dtype([("version", "<i4"), ("probs", "<f4", POLICY),
               ("planes", "u1", 16 * PLANE // 8), ("stm", "u1"),
               ("winner", "u1")])
assert V2.itemsize == 2176


def decode_v2(records, planes, probs, winner):
    """Decode packed v2 records into the given arrays, like
    ChunkParser.convert_v2_to_tuple but for a whole batch at once."""
    r = np.frombuffer(records, dtype=V2)
    planes[:, :16 * PLANE] = np.unpackbits(r["planes"], axis=1)
    # Planes 16 and 17: all ones on the side to move (black = 0 -> plane 16).
    planes[:, 16 * PLANE:17 * PLANE] = (r["stm"] == 0)[:, None]
    planes[:, 17 * PLANE:] = (r["stm"] == 1)[:, None]
    probs[:] = r["probs"]
    winner[:] = r["winner"].astype(np.float32) * 2 - 1


class FileDataSrc:
    """Cycles over gzip chunk files in random order, forever (parse.py's
    FileDataSrc without its TensorFlow import)."""

    def __init__(self, chunks):
        self.chunks = []
        self.done = list(chunks)

    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        while self.chunks:
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, "rb") as f:
                    self.done.append(filename)
                    return f.read()
            except OSError:
                print("failed to parse {}".format(filename), file=sys.stderr)
        return None


def slot_layout(batch):
    """Byte offsets of (planes, probs, winner) in a slot, and its size."""
    planes = batch * PLANES_BYTES
    probs = batch * POLICY * 4
    planes_pad = (planes + 63) // 64 * 64           # keep floats aligned
    return planes_pad, planes_pad + probs, planes_pad + probs + batch * 4


def slot_views(buf, slot, batch):
    probs_at, winner_at, size = slot_layout(batch)
    base = slot * ((size + 63) // 64 * 64)
    planes = np.ndarray((batch, PLANES_BYTES), np.uint8, buf, base)
    probs = np.ndarray((batch, POLICY), np.float32, buf, base + probs_at)
    winner = np.ndarray((batch,), np.float32, buf, base + winner_at)
    return planes, probs, winner


def ring_bytes(batch, slots):
    size = slot_layout(batch)[2]
    return slots * ((size + 63) // 64 * 64)


def _feeder(chunks, shm_name, batch, slots, shuffle_records, sample, workers,
            free, ready, stop):
    """Feeder process: ChunkParser workers -> shuffle buffer -> ring."""
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        parser = chunkparser.ChunkParser(
            FileDataSrc(chunks), shuffle_size=shuffle_records, sample=sample,
            batch_size=batch, workers=workers)
        records = parser.v2_gen()
        record_size = V2.itemsize
        staging = bytearray(batch * record_size)
        while not stop.is_set():
            for i in range(batch):
                staging[i * record_size:(i + 1) * record_size] = next(records)
            while True:
                try:
                    slot = free.get(timeout=0.2)
                    break
                except queue.Empty:
                    if stop.is_set():
                        return
            decode_v2(staging, *slot_views(shm.buf, slot, batch))
            ready.put(slot)
    except StopIteration:
        ready.put(None)                    # out of data (finite sources)
    finally:
        # Stop the ChunkParser workers before their pipes close, so they do
        # not die on a broken pipe mid-write.
        for child in mp.active_children():
            child.terminate()
            child.join()
        shm.close()


class Batches:
    """Shuffled, augmented training batches from gzip chunk files.

    `next()` returns (planes uint8 [B, 6498], probs f32 [B, 362],
    winner f32 [B]) as mx arrays. Use as a context manager, or call close().
    """

    def __init__(self, chunks, batch, slots=4, shuffle_bytes=None, sample=1,
                 workers=None):
        import mlx.core as mx                       # only the trainer needs it
        self._mx = mx
        self.batch = batch
        self.slots = slots
        if shuffle_bytes is None:
            shuffle_bytes = default_shuffle_bytes()
        shuffle_records = max(1, shuffle_bytes // V2.itemsize)
        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 2)
        ctx = mp.get_context("spawn")
        self._shm = shared_memory.SharedMemory(
            create=True, size=ring_bytes(batch, slots))
        self._free = ctx.Queue()
        self._ready = ctx.Queue()
        self._stop = ctx.Event()
        for slot in range(slots):
            self._free.put(slot)
        self._proc = ctx.Process(
            target=_feeder,
            args=(list(chunks), self._shm.name, batch, slots, shuffle_records,
                  sample, workers, self._free, self._ready, self._stop),
            daemon=False)                 # it has children (ChunkParser workers)
        self._proc.start()

    def next(self):
        while True:
            try:
                slot = self._ready.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        "data feeder process exited (code {})".format(
                            self._proc.exitcode)) from None
        if slot is None:
            raise StopIteration
        planes, probs, winner = slot_views(self._shm.buf, slot, self.batch)
        mx = self._mx
        out = (mx.array(planes), mx.array(probs), mx.array(winner))
        mx.eval(out)                      # copy out before the slot is reused
        self._free.put(slot)
        return out

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def close(self):
        if self._proc is None:
            return
        self._stop.set()
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._proc = None
        for q in (self._free, self._ready):
            q.close()
            q.join_thread()
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def default_shuffle_bytes():
    """min(1.5 GB, 10% of physical memory), spec 06 section 6."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = 16 << 30
    return min(int(1.5 * (1 << 30)), total // 10)
