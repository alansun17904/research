# Initial validation — September 7, 2026

This is the original run record. Its KV byte counts include position IDs that
have since been removed from the cache; current runs use slightly less memory.

Pretrained checkpoint: `roneneldan/TinyStories-1Layer-21M`, commit
`0d983335d1447a805aa0f4c0c440e257a0a87f44`. It contains 66,155,520 parameters
including embeddings. CPU, FP32, one thread, seed 42; PyTorch 2.10.0,
Transformers 4.44.2, datasets 3.0.0.

The first 513 token IDs from TinyStories validation were used to score 512
next-token predictions. Documents are tokenized with a trailing double newline,
without inserted BOS. All methods saw the same token buffer. There was no
warmup exclusion, sequence reset, training, or hyperparameter search.

```bash
python -m rttt compare --device cpu --corpus tinystories \
  --max-tokens 512 --cache-size 32 --sink-tokens 1 \
  --output results/tinystories-512.json
```

| Method | Perplexity | Persistent attention bytes |
| --- | ---: | ---: |
| Full softmax | 3.7997 | 4,259,840 |
| Recent window, 32 KV pairs | 5.0169 | 266,240 |
| Streaming prefix + recent, 1 + 31 | 5.1917 | 266,240 |
| H2O, 16 heavy + 16 recent | 8.6400 | 268,288 |
| ELU linear attention, no sink | 4,184.4957 | 266,240 |
| ELU linear attention, one softmax sink | 4,232.5114 | 274,432 |

The exponential sink did not improve this small sample. Replacing a pretrained
softmax kernel with ELU features severely degraded likelihood. These results
test one concrete hybrid formula on one short prefix; they do not establish
whether a different kernel, calibration, trained adaptation, or model would
benefit from a sink. Memory values exclude model weights and temporary tensors.
The linear and 32-token KV baselines happen to have similar state bytes because
this model's head dimension is 64.

Full JSON measurements are saved locally at `results/tinystories-512.json`.
Results and model downloads are git-ignored; this small report is tracked as
part of the experiment implementation.

A separate PG19 integration run used `EleutherAI/pythia-70m`, commit
`a39f36b100fe8a5377810d56c3f4789b9c53ac42`, for 2,304 predictions from the first
test book. StreamingLLM with four sinks and 28 recent tokens ran beyond the
2,048-token training context without resets, retained 798,720 attention bytes,
and produced perplexity 38.0332. This validates long-stream execution and the
official PG19 loader; it is not a paper-scale evaluation. The output is
`results/pythia-pg19-smoke.json`.

The real LM Evaluation Harness 0.4.5 also completed a two-example PIQA run
with the linear sink variant (`results/piqa-smoke.json`), and a one-example
smoke run of each of the six H2O tasks using five-shot prompts and a 20% total
cache ratio (`results/h2o-suite-smoke.json`). All 17 candidate likelihood requests
completed without prompt truncation. These sample counts validate integration,
not task accuracy. The test suite
contains mathematical attention references, state/eviction invariants, model
parity, token-scoring and reset checks, HELM replay, and a real harness test
with an offline fixture. All 72 checks passed with the cached pretrained test
enabled. CPU execution was validated; CUDA and MPS were not exercised.
