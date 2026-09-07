# RTTT attention experiments

Small, inference-only experiments that replace attention in pretrained GPT-Neo
and GPT-NeoX models. Compare full softmax, a sliding window, StreamingLLM, H2O,
and normalized linear attention with an optional exact prefix sink.

Start with the [native sink diagnostic](docs/attention-sinks.md). Our TinyStories
checkpoint shows no strong initial-token sink in these tests; Pythia-70M shows
early punctuation sinks and, with BOS prepended, a consistent initial-token sink.

## Run

```bash
# Python 3.10+
python -m pip install -e '.[test]'
python -m rttt compare --device cpu --corpus tinystories \
  --max-tokens 512 --cache-size 32 --sink-tokens 1 \
  --output results/tinystories.json
python -m rttt generate --method streaming --cache-size 32
```

The default is `roneneldan/TinyStories-1Layer-21M`, a GPT-Neo checkpoint with
one global attention layer. Downloads go in `.cache/huggingface`.
For streams beyond its 2048 learned positions, use a GPT-NeoX checkpoint such
as `EleutherAI/pythia-70m`; see the [benchmark commands](docs/benchmarks.md).

In `compare`, `linear` means zero sinks and `linear-sink` uses
`--sink-tokens`. For individual commands, use `--method linear --sink-tokens 0`
to disable sinks. `--sink-mode kernel` provides the normalization control.

## Read the code

- [attention.py](rttt/attention.py): the attention math and cache eviction.
- [sinks.py](rttt/sinks.py): detect sinks in the original pretrained attention.
- [model.py](rttt/model.py): pretrained transformer layers around those policies.
- [benchmarks.py](rttt/benchmarks.py): corpus loading, perplexity, generation, and HELM replay.
- [lm_eval_adapter.py](rttt/lm_eval_adapter.py) and [performance.py](rttt/performance.py):
  downstream tasks and timing. [cli.py](rttt/cli.py) connects the commands.

Each layer owns an `Attention` instance with `step(q, k, v)`, where tensors have
shape `[batch, heads, dim]`. The model runner exposes `step([batch])` and
`prefill([batch, sequence])`; both return the final `[batch, vocab]` logits.
Call `reset()` between independent sequences. To add an attention experiment,
edit `Attention._linear` for kernel experiments or `Attention._evict` for KV policies.

```bash
python -m pytest -q
# Also check against cached pretrained weights:
RTTT_PRETRAINED_TESTS=1 python -m pytest -q
```

The [experiment notes](docs/experiment.md) explain joint sink normalization and
cache budgets. The [initial measurements](docs/initial-run.md) show that this
zero-shot linear conversion substantially worsens perplexity; the sink did not
improve it. The separate [token-tape proposal](docs/tape-proposal.md) remains
design notes.
