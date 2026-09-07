# Linear attention with an optional sink

This implements the September 6 experiment in `LOG.md`. Model weights stay
fixed. Every attention layer uses its pretrained queries, keys, values, and
output projection. Layer norms, residual connections, MLPs, embeddings, and the
language-model head stay intact. The [exact-token-tape proposal](tape-proposal.md)
is a separate architecture and is not part of this initial experiment.

## Linear state and normalization

Let `a` be the pretrained logit scale (`1` for GPT-Neo, `1/sqrt(d)` for
GPT-NeoX). Define `f(x) = phi(sqrt(a) x)`, with `phi(x) = ELU(x)+1` by default
or `ReLU(x)` as an ablation. For the first `s` observed tokens, keep exact
key/value pairs. Accumulate only subsequent tokens, including the current one:

```text
S_t = sum_{s <= i <= t} f(k_i) v_i^T
z_t = sum_{s <= i <= t} f(k_i)
n_t = f(q_t)^T S_t
m_t = f(q_t)^T z_t
```

Without sinks, output `n_t / max(m_t, eps)`. With `--sink-mode softmax`, output

```text
               n_t + sum_{i < min(s,t+1)} exp(a q_t^T k_i) v_i
o_t = ----------------------------------------------------------------
               m_t + sum_{i < min(s,t+1)} exp(a q_t^T k_i)
```

The implementation appends `log(m_t)` to the sink logits and applies one stable
softmax, treating the linear average as an extra value. Sink tokens never enter `S`
or `z`; there is no double counting. Accumulators are FP32 even for half-precision
model weights. ReLU can have zero mass, in which case the no-sink output is zero.

The softmax is over a fixed number of explicit sinks plus an aggregate mass.
It preserves constant state size and constant per-token attention cost in
sequence length: `O(d² + sd)` per head. It is still a nonlinear map of a query,
as ordinary normalized linear attention is. The kernel and exponential masses
have different scales; this hybrid is a concrete experimental interpretation of
the notes, not a proven approximation to softmax or a fitted conversion.

`--sink-mode kernel` assigns the same feature-map kernel to the explicit sinks.
This is algebraically equivalent to no-sink linear attention up to rounding,
and is a useful control for simply separating the initial state.

## Run the ablations

```bash
# Identical cached token stream for all six variants. No training.
python -m rttt compare --max-tokens 512 --cache-size 32 --sink-tokens 1 \
  --output results/sink-1.json
python -m rttt compare --max-tokens 512 --cache-size 32 --sink-tokens 4 \
  --output results/sink-4.json
python -m rttt compare --methods linear,linear-sink --sink-mode kernel \
  --max-tokens 512 --output results/kernel-control.json
python -m rttt compare --methods linear,linear-sink --feature-map relu \
  --max-tokens 512 --output results/relu.json
```

In `compare`, `linear` always has zero sinks and `linear-sink` uses
`--sink-tokens`. In individual commands, `--method linear --sink-tokens 0`
turns sinks off. A sink is the first actual input token by default. Pass
`--prepend-bos` in corpus experiments to make the first token BOS; this changes
the evaluated input. No new sink token is trained or inserted implicitly.

## KV baselines and positions

All bounded softmax methods append the current token, attend, and then evict.
`--cache-size B` means **B persistent pairs per head**; the computation can
temporarily contain B+1 pairs. Prefill uses the same sequential policy as decode.

- `full`: all previous tokens, respecting a checkpoint's original local masks.
- `window`: retain the latest B tokens.
- `streaming`: retain `s` initial tokens and B-s recent tokens.
- `h2o`: keep the highest accumulated normalized-attention scores among older
  tokens, plus recent tokens. Selection is independent per head and layer;
  evicted tensors and scores are physically dropped. Default split is
  `floor(B/2)` heavy hitters and the remainder recent. `--heavy-hitter-size`
  changes it. H2O does not also pin the initial tokens.

GPT-Neo uses learned absolute positions and raises at its position-table limit
(2048 for the default model). It does not silently wrap or clamp positions.
`--reset-interval` is an explicit segmented-language-modeling protocol. A bounded
cache alone cannot extend a learned position table.

For vanilla GPT-NeoX (for example `EleutherAI/pythia-70m`), StreamingLLM stores
unrotated keys and applies RoPE again at compact cache positions at each step,
including positioning the query at the current compact index. This follows the
[upstream positional shift implementation](https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/pos_shift/modify_gpt_neox.py).
The other variants use global positions. Linear state therefore aggregates keys
rotated at their original positions. RoPE permits computation beyond training
length, but that alone does not guarantee retained quality. Scaled RoPE and
architectures other than GPT-Neo/GPT-NeoX are rejected.

## Interpreting measurements

`state_bytes` sums actual persistent attention tensors across layers, including
FP32 accumulators and H2O scores. It excludes weights, original model buffers,
and temporary tensors; the `speed` command's `cuda_peak_allocated_bytes`
separately includes CUDA allocations. Position IDs are not stored in the cache.
Equal cache token budgets do not imply equal bytes versus a linear matrix state.
For the default head dimension 64, a linear matrix state is roughly comparable
to 32 FP32 KV tokens per head. Each exact sink adds another full-width KV pair.

The runner is inference-only and uses a Python token loop. It makes correctness,
causality, and physical eviction easy to inspect; its timing measures this
implementation and is not a claim about fused production kernels.
