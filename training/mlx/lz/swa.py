"""Stochastic weight averaging, as tfprocess.save_swa_network (plan 13, F9).

Every c-th checkpoint is folded into a running mean with weight 1/(n+1), where
n counts the networks so far but is capped at max_n, so after max_n networks it
behaves like an exponential moving average. Batch norm statistics are then
refined by forward passes in training mode on the averaged weights, the net is
exported, and the live weights are put back.
"""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .convert import model_to_tensors
from .weights import write_weights


class SWA:
    def __init__(self, c=1, max_n=16):
        self.c = c
        self.max_n = max_n
        self.skip = c          # networks still to skip before the next sample
        self.n = 0             # networks averaged so far (capped at max_n)
        self.avg = None        # {parameter name: array}

    def update(self, model):
        """Fold the model's weights in if this checkpoint is sampled. Returns
        the capped count n after the update, or 0 if the checkpoint was skipped."""
        self.skip -= 1
        if self.skip > 0:
            return 0
        self.skip = self.c
        weights = dict(tree_flatten(model.parameters()))
        if self.avg is None:
            self.avg = {k: mx.zeros_like(v) for k, v in weights.items()}
        n = self.n
        self.avg = {k: self.avg[k] * (n / (n + 1.0)) + weights[k] * (1.0 / (n + 1.0))
                    for k in weights}
        mx.eval(self.avg)
        self.n = min(n + 1, self.max_n)
        return self.n

    def export(self, model, batches, path, refine_batches=200):
        """Write the averaged net to `path` (.txt or .txt.gz); the model's own
        weights are restored afterwards. `batches` yields (planes, probs,
        winner) like the trainer's data; they are used only to refine batch
        norm statistics."""
        live = dict(tree_flatten(model.parameters()))
        model.update(tree_unflatten(list(self.avg.items())))
        if refine_batches:
            model.train()
            for _ in range(refine_batches):
                planes, _, _ = next(batches)
                model(planes)
                mx.eval(model.state)
        write_weights(path, model_to_tensors(model))
        model.update(tree_unflatten(list(live.items())))
        mx.eval(model.parameters())

    def state_arrays(self):
        return {} if self.avg is None else {"swa/" + k: v for k, v in self.avg.items()}

    def load_state(self, arrays, meta):
        avg = {k[4:]: v for k, v in arrays.items() if k.startswith("swa/")}
        self.avg = avg or None
        self.n = int(meta.get("swa_n", 0))
        self.skip = int(meta.get("swa_skip", self.c))
