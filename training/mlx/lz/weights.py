"""Leela Zero v1 text weight files: line 1 is the version, then one line of
space-separated floats per tensor (plan 13, fact F1). Plain text or gzip."""

import gzip

import numpy as np

SUPPORTED_READ = (1, 2)      # 2 = ELF: value head not from side to move


def _open(path, mode):
    if str(path).endswith(".gz"):
        # Level 1: about 15x faster than the default 9 for a few % more bytes;
        # export runs on the training thread.
        return gzip.open(path, mode + "t", compresslevel=1)
    return open(path, mode)


def read_weights(path):
    """Return (version, [flat float32 array per tensor line])."""
    with _open(path, "r") as f:
        version = int(f.readline().split()[0])
        if version not in SUPPORTED_READ:
            raise ValueError("unsupported weights version %d" % version)
        tensors = [np.array(line.split(), dtype=np.float32)
                   for line in f if line.strip()]
    return version, tensors


def write_weights(path, tensors):
    """Write version-1 weights; floats as %.9g, so fp32 round-trips exactly."""
    with _open(path, "w") as f:
        f.write("1\n")
        for t in tensors:
            values = np.asarray(t, np.float32).ravel().tolist()
            f.write(" ".join(map("{:.9g}".format, values)))
            f.write("\n")
