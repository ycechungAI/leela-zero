# 13 — Build Plan: Phase 3, MLX trainer (`training/mlx`)

Spec: [06-spec-training-mlx.md](06-spec-training-mlx.md). Gates T1–T6 are
defined there (§9). Model per step: [10-model-plan.md](10-model-plan.md) §5b.
This plan is written so that a Sonnet session can build each step from it.

## 0. Facts to respect (read from `training/tf`, not guessed)

- **F1 Weight order** (`tfprocess.weights`, also the leelaz v1 file order, one
  tensor per line after the `1` version line): input conv (W, β-as-bias, mean,
  var), then per residual block two such quadruples, then policy conv
  quadruple, `w_fc_1` [722→362], `b_fc_1`, value conv quadruple, `w_fc_2`
  [361→256], `b_fc_2`, `w_fc_3` [256→1], `b_fc_3`. Line count =
  4 + 8·blocks + 4 + 2 + 4 + 4.
- **F2 BN**: `center=True, scale=False`, ε = 1e-5, TF momentum 0.99 (MLX
  `momentum=0.01`: MLX's running = (1−m)·running + m·batch). The file stores
  β as a "conv bias" `β·sqrt(var+ε)` (`replace_weights` divides it back).
- **F3 Conv weights**: TF `[kh, kw, in, out]` ↔ leelaz `[out, in, kh, kw]` ↔
  MLX `nn.Conv2d` `[out, kh, kw, in]`. FC: leelaz `[out, in]` = MLX
  `nn.Linear.weight` (no transpose); TF stored `[in, out]`.
- **F4 Flatten order**: TF flattens head convs in NCHW order (c, h, w). In MLX
  (NHWC) transpose the head conv output to NCHW before flattening.
- **F5 Init**: `truncated_normal(stddev = sqrt(2 / sum(shape)))` for conv and
  FC weights (shape as TF writes it, so the sum is the same in any layout);
  biases zero; BN β zero, mean 0, var 1.
- **F6 Loss**: `softmax_cross_entropy(probs, policy_logits)` mean +
  `mean((winner − tanh_value)²)` + `1e-4 · Σ ½‖w‖²` over conv and FC weights
  only (not biases, not BN). Reported `mse` is `mse/4`; `total` is
  policy + mse(unscaled) + reg. Accuracy = argmax(policy) == argmax(probs).
- **F7 Optimizer**: SGD momentum 0.9, Nesterov, lr 0.05 (comment in
  tfprocess: 0.005 when training from self-play); `macrobatch` sums
  gradients over k batches and applies once (sum, not mean, as in
  `gsum`).
- **F8 Data**: `ChunkParser.parse()` yields batches of packed bytes: planes
  uint8 `[B, 18·361]` (16 history planes + 2 side-to-move planes, values
  0/1), probs float32 `[B, 362]`, winner float32 `[B]` (±1). Random
  symmetry is applied in the worker (`v2_apply_symmetry`). `chunkparser.py`
  and `shufflebuffer.py` import no TF and are reused unchanged.
- **F9 SWA**: `swa_c = 1`, `swa_max_n = 16`, running mean
  `swa = swa·n/(n+1) + w/(n+1)` with n capped at 16, BN refined by 200
  training-mode forward passes on the SWA weights, then exported; the live
  weights are restored afterwards.
- **F10 leelaz output**: `lz-nn-eval <sym>` prints winrate, pass prior, then
  361 priors in board order (softmax at temperature 1). This is the T2/T3
  comparison hook (`scripts/parity/compare_backends.py` already parses it).
- **F11 T3 inputs**: self-play chunks (`scripts/macos/selfplay.py`) record,
  for move k of a game, the 18 input planes leelaz used for the position after
  k moves. So T3 evaluates chunk record k in MLX and `loadsgf game.sgf k+1`
  + `lz-nn-eval 0` in leelaz, with no plane-building code in Python.

## 1. Layout (as spec §3, Python 3.12 via uv)

```
training/mlx/
  pyproject.toml        mlx, numpy>=2, pytest; [project.scripts] lz-train
  lz/__init__.py
  lz/model.py           LeelaZeroNet, init, loss, accuracy
  lz/weights.py         read/write leelaz v1 text (+ .gz); ordered tensor list
  lz/convert.py         MLX params <-> weights.py tensor list (F1–F3)
  lz/data.py            shared-memory ring over chunkparser workers
  lz/train.py           CLI, compiled step, schedule, checkpoints, guard
  lz/swa.py             accumulator + BN refinement
  tests/                pytest (T1, T2, T3 helpers; data; swa)
```

`import_weights.py` / `export.py` from the spec become `lz/weights.py` +
`lz/convert.py` (format I/O separate from layout conversion, each testable
alone). `lz.train --import-weights` / `--export-dir` keep the spec's CLI.

## 2. Steps

### 3.1 — Scaffold (L)
`pyproject.toml`, empty package, one passing test, `uv run pytest` in CI on
`macos-15` (new job `macOS arm64 / MLX trainer`, runs only when
`training/mlx/**` changes plus nightly). Gate: CI job green.

### 3.2 — `model.py` + T1 (S)
- `LeelaZeroNet(blocks, filters)`; NHWC inside; input `[B, 18, 361]` uint8 →
  reshape `[B,18,19,19]` → transpose to NHWC → cast to compute dtype.
- `ConvBlock` = `nn.Conv2d(bias=False)` + `nn.BatchNorm(affine=False)` +
  separate `beta` parameter (added after BN) + ReLU (F2).
- Heads per F4. Value: `tanh`.
- `loss_fn(model, planes, probs, winner)` → (total, (policy, mse, reg,
  accuracy)) in fp32 (F6); reg only over `.weight` of conv and linear
  layers — name the regularized set explicitly, do not filter by ndim.
- T1: shapes for 1×8 and 6×64; parameter count = sum of F1 tensor sizes;
  loss on a fixed seeded batch equals a NumPy reference (written in the test
  from the same weights) ≤ 1e-5; reg term equals the NumPy sum; BN train vs
  eval behaviour; momentum update of running stats matches F2.

### 3.3 — Weights I/O, convert, T2, T3 (M build, S gate review)
- `weights.py`: parse/write v1 text and `.gz`; reject v2 for export, accept v2
  (ELF) read-only on import; format floats `%.9g`.
- `convert.py`: params → list (β → `β·sqrt(var+ε)`, conv transpose
  `(0,3,1,2)`), list → params (inverse); infer blocks/filters from line count
  and first conv size.
- T2 (round trip): import a public net → export → `compare_backends.py
  --test-weights` with `leelaz --backend cpu` on both, 100 positions × 8
  symmetries, tol 1e-6. Numeric equality is the gate, not byte equality:
  upstream files use other float formatting.
- T3 (cross-engine): `tests/t3_cross_engine.py` (not in the default pytest
  run: needs leelaz and a net) per F11, MLX fp32 eval vs `leelaz --backend
  cpu`, priors and winrate ≤ 1e-4.
- Unit tests here use a random 2×16 net written by
  `scripts/macos/make_random_net.py`; T2/T3 with real nets run by hand and
  are recorded in BENCHMARKS.md (nets are downloaded, used and deleted, as in
  step 1.5).

### 3.4 — Data pipeline (S)
- Workers: `ChunkParser.task` (unchanged) produces v2 records over pipes, as
  today; a feeder process (or the parent thread) does shuffle + conversion
  and writes **batches** straight into a `multiprocessing.shared_memory`
  ring of R = 4 slots (`planes u8[B,6498]`, `probs f32[B,362]`,
  `winner f32[B]`), with per-slot ready/free semaphores. Backpressure blocks
  the feeder, never the trainer.
- Trainer side: `np.ndarray` views on the slot → `mx.array(view)` (one copy
  into MLX memory), release the slot after the copy.
- Shuffle buffer sized in bytes: default min(1.5 GB, 10% of RAM).
- Tests: a synthetic chunk set (write v2 records with known content), check
  every record arrives exactly once per epoch modulo symmetry; slot reuse
  under a slow consumer; clean shutdown (no leaked `shared_memory` segments:
  check `/dev/shm`-equivalent via `resource_tracker` warnings = none).
- Measure: positions/s of the pipeline alone vs the train step's consumption
  (T5 input stall < 5%). If `mx.array(view)` copy > 3% of step time, note it;
  MLX-owned buffers are a later optimisation, not this step.

### 3.5 — `train.py` (M, with S review of compile + memory guard)
- CLI per spec §8, including the legacy positional form of `parse.py`.
- Step: `nn.value_and_grad` → accumulate over `--macrobatch` (sum, F7) →
  `optim.SGD(lr, momentum=0.9, nesterov=True)` update. `mx.compile` the
  per-batch grad step with `inputs=[model.state, optimizer.state]`,
  `outputs=` the same (MLX's documented pattern for stateful compile).
- dtype: `--dtype bf16|fp16|fp32`: weights fp32, forward in compute dtype,
  loss in fp32; fp16 uses static loss scale 128 (parity with tfprocess).
- LR schedule `'0:0.05,100000:0.005'`; checkpoints with
  `mx.save_safetensors` (params, optimizer state, step, SWA state);
  `--export-every` writes leelaz `.txt.gz`; logs as tfprocess (every 1000
  steps; test eval every 8000 steps over 800 batches).
- Memory guard: projected peak from params + activations
  (B × layers × C × 361 × bytes × ~3 for backward) — print it; refuse when
  above 11 GB on a 16 GB machine unless `--force`; `mx.set_memory_limit`.
- Tests: one compiled step lowers loss on a tiny fixed batch; macrobatch k
  equals one batch of k·B (sum semantics) within fp32 tolerance; checkpoint
  save → restore → identical next-step loss; schedule boundaries.

### 3.6 — SWA (M)
F9 exactly, in `swa.py`; exported as `<prefix>-swa-<n>-<step>.txt.gz`. Test:
three snapshots → average equals the NumPy mean; n capped at 16; live weights
unchanged after export.

### 3.7 — T4, T5 (S)
- T5 (needs nothing external): 20b×256, batch 256, bf16, synthetic data in
  the real pipeline: GPU utilisation (`powermetrics`, needs sudo → the user
  runs it, or Instruments), input stall, peak RSS ≤ 11 GB.
- T4: see decision D2.

### 3.8 — T6 (L to run, S to judge)
See decision D2; may finish after the `as.3` tag (spec).

### 3.R — Phase review (S)
`/code-review high` on `training/mlx`, plus the CI job.

## 3. Decisions (resolved, ADR-012)

- **D1 Tooling:** Homebrew `uv` + system Python 3.12; packages from PyPI into
  `training/mlx/.venv`.
- **D2 No TF reference:** T1/T2/T3 carry equivalence; T4 checks learning
  against thresholds from the first MLX run; T6 plays a public LZ net of the
  same size.
- **D3 Data:** local self-play from `selfplay.py` with a public net.

## 4. Pitfalls

- Flatten order (F4) and FC orientation (F3) are the classic export bugs; T2
  and T3 catch them, unit tests alone do not.
- `mx.compile` with captured state silently freezes values if the state is not
  declared in `inputs/outputs`.
- BN running variance: TF uses the unbiased batch variance for the moving
  average; check MLX's choice and match it (it shows up only in T3 after
  training, not in T2).
- Shared memory segments outlive crashed processes on macOS; unlink in a
  `finally` and in a signal handler.
- `spawn` start method on macOS: worker entry points must be importable at
  module level (no lambdas).
- Do not import `training/tf/tfprocess.py` (it imports TF); only
  `chunkparser.py` and `shufflebuffer.py`, via a path insert.

## 5. Done means

T1–T5 pass (T4 per D2), the CI job is green, BENCHMARKS.md has the T3/T5
numbers, the model-plan rows are filled, and 3.R is done. Tag `as.3` on
request.
