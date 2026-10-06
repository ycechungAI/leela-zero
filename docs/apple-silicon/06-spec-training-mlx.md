# 06 — Spec: MLX Training Pipeline (Unified-Memory Training)

**Release:** `as.3`. **Depends on:** 03, 04 (CPU reference for export
validation), and 07 (gates). It can be developed **in parallel with 05**.

## 1. Problem

`training/tf/` is TensorFlow 1.x graph-mode code: `tf.Session`,
`tf.placeholder`, `tf.layers`, NCHW conv and a custom loss-scaling optimizer.
There is no TF1 wheel for macOS arm64. The `tensorflow-metal` plugin targets TF2
and still copies between host and device. Porting TF1 to TF2 is a rewrite
either way (B6). See ADR-002 for the framework choice.

## 2. Choice: Apple MLX

MLX arrays live in unified memory and every device can use them, so there is no
`.to("mps")` and no staging buffer. Operations are lazy and fused, and
`mx.compile` builds fused GPU kernels for the train step. MLX supports bf16/fp16
natively, uses Metal under the hood, and is MIT-licensed. A PyTorch-MPS port is
kept as an alternative design in ADR-002, not implemented.

## 3. Layout

```
training/mlx/
  pyproject.toml          mlx>=0.2x, numpy>=2, (optional) tensorboardX
  lz/model.py             LeelaZeroNet(blocks, filters): matches tfprocess.construct_net
  lz/data.py              wraps ../tf/chunkparser.py (TF-free, reused as-is) → numpy batches
  lz/train.py             CLI, matching parse.py flags where meaningful
  lz/export.py            MLX params → leelaz v1 text weights (.txt/.gz)
  lz/import_weights.py    leelaz v1 weights → MLX params (resume/fine-tune)
  lz/swa.py               stochastic weight averaging (+ BN recalc)
  tests/                  pytest: shapes, round-trip, parity vs leelaz CPU
```

`chunkparser.py` and `shufflebuffer.py` have no TF imports, so they are reused
without modification. Shared code doesn't fork.

## 4. Model parity requirements (must match `tfprocess.py` exactly)

| Aspect | tfprocess (TF1) | MLX implementation |
|--------|-----------------|--------------------|
| Layout | NCHW | NHWC (the native MLX conv layout). Transpose at the input boundary only |
| Input | 18×19×19 uint8 planes | Same, cast to compute dtype on GPU |
| Conv init | Xavier-ish `truncated_normal(stddev=sqrt(2/(sum(shape))))` | Same formula |
| BN | `center=True, scale=False`, ε=1e-5, momentum default 0.99 | `nn.BatchNorm(affine=False)` + separate β, ε=1e-5, momentum 0.01 (MLX convention = 1-0.99) |
| Policy head | conv1×1(2) → BN → ReLU → FC(722→362) | Same |
| Value head | conv1×1(1) → BN → ReLU → FC(361→256) → ReLU → FC(256→1) → tanh | Same |
| Loss | softmax-CE(policy) + MSE(value) + 1e-4·Σ l2_loss(weights) (l2_loss = ½‖w‖²), biases/BN excluded | Same, computed in fp32 |
| Optimizer | SGD momentum 0.9, Nesterov, lr 0.05 (schedule via script) | `optim.SGD(momentum=0.9, nesterov=True)`, same schedule hooks |
| SWA | c=1, max_n=16, BN recalc | Same semantics |
| Reported MSE | `mse/4` | Same, so logs compare directly |

**Flatten-order gotcha:** TF flattens the head conv output in NCHW order
(c, h, w) before the FC. MLX/NHWC flattens in (h, w, c). Either transpose to
NCHW before flatten (simplest, and recommended), or permute FC input rows at
export. The round-trip test (gate T2) catches mistakes here.

## 5. Export / import (`export.py`, `import_weights.py`)

Write the version line `1`, then one line per tensor in the same order as
`tfprocess.weights`:

| Tensor | MLX shape | leelaz shape | Transform |
|--------|-----------|--------------|-----------|
| Conv weight | `[out, kh, kw, in]` | `[out, in, kh, kw]` | `transpose(0,3,1,2)` |
| BN β → "conv bias" | `[C]` | `[C]` | `β · sqrt(var + 1e-5)` (same back-compat trick as tfprocess) |
| BN mean, BN var | `[C]` | `[C]` | as-is |
| FC weight | `[out, in]` (`nn.Linear`) | `[out, in]` | as-is (note: TF needed a transpose, MLX does not) |
| FC bias | `[out]` | `[out]` | as-is |

Format floats with `repr`-level precision (`np.format_float_positional`, or
`%.9g`) so the text round-trip is lossless in fp32. Also support writing `.gz`.

Import is the inverse. It must accept every public LZ net. v2 (ELF) nets can be
imported read-only for fine-tuning experiments.

## 6. Unified-memory data pipeline

```
 E-cores: N worker processes (chunkparser.task)  ──►  mp.shared_memory ring (R slots)
                                                     slot = {planes u8 [B,18,361],
                                                             probs f32 [B,362],
                                                             winner f32 [B]}
 P-core: training loop  ── np.ndarray view on slot ──► mx.array(view) ──► GPU step
```

- Replace the current `bytes` serialization path (built for `tf.decode_raw`)
  with direct writes into preallocated shared-memory numpy arrays. This removes
  per-batch pickling.
- Ship planes as **uint8** (6.5 KB/position) and cast to bf16 on GPU. This is 4×
  less data than fp32 planes.
- Measure whether `mx.array(np_view)` copies (expected: one memcpy into an MLX
  allocation, which is cheap at 120 GB/s). If profiling shows it above 3% of the
  step time, move to MLX-owned buffers exposed to numpy via the buffer protocol,
  so loaders fill MLX memory directly.
- `R = 4` slots by default (double buffering plus slack). Backpressure blocks
  workers, not the trainer.
- Shuffle buffer sized in **bytes**, not positions. The default is 1.5 GB on
  16 GB machines, auto-scaled to 10% of physical RAM.

## 7. Performance features

| Feature | Detail |
|---------|--------|
| `mx.compile` | Compile the full forward+backward+update step. Shapes are static per run |
| Mixed precision | Weights fp32 master, compute bf16 (no loss scaling needed, unlike the fp16 path in tfprocess). An `--dtype fp16` option keeps static loss scaling 128 for parity tests |
| Gradient accumulation | `--macrobatch k`. Same semantics as `TFProcess.init(macrobatch=)`, which allows an effective batch of 2048 within 16 GB |
| Memory guard | Query `mx.metal.device_info()['memory_size']` and `mx.metal.set_memory_limit()`. Refuse configs that would exceed N5 (11 GB) on a 16 GB machine. Print the projected peak at startup |
| Checkpoints | `mx.save_safetensors`: params + optimizer state + step + SWA accumulators. leelaz weights are exported every `--export-every` steps |
| Logging | Same stats as tfprocess (policy, mse/4, reg, accuracy, steps/s, positions/s). Optional TensorBoard |

## 8. CLI (`python -m lz.train`)

```
lz-train --blocks 20 --filters 256 --train 'data/train_*' --test 'data/test_*' \
         --batch 256 --macrobatch 2 --steps 200000 --lr-schedule '0:0.05,100000:0.005' \
         --dtype bf16 --restore ckpt/ --import-weights best.txt.gz --export-dir nets/
```

The positional args of `parse.py` (`blockspref`, `filterspref`, …) are kept as
an accepted legacy form, so existing scripts keep working.

## 9. Acceptance criteria

- [ ] T1 – unit: `pytest training/mlx/tests` passes (shapes, loss values on a
      fixed batch match a NumPy reference ≤1e-5).
- [ ] T2 – **round-trip parity**: import a public net → export → `leelaz`
      CPU backend output on 100 positions equals the original net ≤1e-5 (ADR-013; was 1e-6).
- [ ] T3 – **cross-engine parity**: MLX forward (fp32, eval mode) vs `leelaz`
      CPU forward on the same exported net ≤1e-4 (policy logits and value).
- [ ] T4 – learning sanity: training a 6b×64 net on local self-play data
      for 20k steps reaches the policy accuracy and losses fixed from the
      first MLX run in BENCHMARKS.md (no TF reference: ADR-012).
- [ ] T5 – performance: GPU utilization ≥ 80% (Instruments / `powermetrics`),
      input stall < 5%, and peak RSS ≤ 11 GB for 20b×256 batch 256 bf16.
- [ ] T6 – strength (program success metric 3): a public LZ 10b×128 net,
      fine-tuned with lz-train on its own self-play at a low learning rate,
      scores ≥ 45% in a 400-game `validation` match against the original
      (ADR-012, ADR-013).
