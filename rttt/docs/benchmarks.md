# Benchmarks and protocols

The same attention runner powers corpus likelihood, harness tasks, generation,
and request replay. Model and tokenizer revisions, seeds, attention settings,
dtype, device, versions, token counts, state sizes, and timings are saved as JSON.
Use explicit model and dataset commit revisions for archival experiments.

## Paper coverage

| Source | Evaluation | Entry point |
| --- | --- | --- |
| H2O | COPA, MathQA, OpenBookQA, PIQA, RTE, WinoGrande; five-shot | `lm-eval --tasks h2o` |
| H2O | XSum, CNN/Daily Mail; HELM zero-shot, 1000 examples | `helm` request replay and original HELM scorer |
| H2O | WikiText-103 attention analysis | `ppl --corpus wikitext103` for likelihood; attention-sparsity plots are not reproduced |
| StreamingLLM | Concatenated PG19 test books, including 400K/4M-token experiments | `ppl --corpus pg19` |
| StreamingLLM repository | WikiText-2 raw language modeling | `ppl --corpus wikitext2` |
| StreamingLLM pretraining evaluation | ARC Easy/Challenge, HellaSwag, LAMBADA, OpenBookQA, PIQA, WinoGrande; zero-shot | `lm-eval --tasks streamingllm` |
| Both papers | State memory, latency, throughput | `speed` |

H2O's task list, shot counts, sample counts, and total cache fractions
(4%, 10%, 20%, 60% of prompt length) are specified in
[Appendix A](https://arxiv.org/html/2306.14048v3#A1).
StreamingLLM's long-stream and downstream protocols are described in its
[paper](https://arxiv.org/abs/2309.17453) and
[evaluation script](https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py).

These are benchmark integrations, not reproductions of the papers' reported
scores. The model size, tokenizer, prefill policy, implementation and hardware
differ. The StreamingLLM task preset exercises the pretrained model; it does
not reproduce their new sink-token pretraining. StreamingLLM's bespoke
StreamEval and streaming ARC dialogue setup are not bundled here.

## Long-stream perplexity

```bash
python -m rttt ppl --model EleutherAI/pythia-70m --corpus pg19 \
  --method streaming --sink-tokens 4 --cache-size 2048 \
  --max-tokens 400000 --log-every 1000 --output results/pg19-streaming.json

python -m rttt ppl --model EleutherAI/pythia-70m --corpus pg19 \
  --method linear --sink-tokens 4 --max-tokens 400000 \
  --log-every 1000 --output results/pg19-linear-sink.json

python -m rttt compare --corpus wikitext2 --max-tokens 512 \
  --cache-size 64 --output results/wikitext2.json

# Explicitly reset a learned-position TinyStories model every 512 inputs:
python -m rttt ppl --corpus tinystories --method linear \
  --reset-interval 512 --max-tokens 10000 --output results/tinystories-segmented.json

# Fully local data:
python -m rttt compare --corpus text --text-path sample.txt \
  --local-files-only --output results/local.json
```

PG19 reads the sorted official split manifest and retrieves books from the
original DeepMind storage, following the ordering in the
[historical dataset loader](https://huggingface.co/datasets/deepmind/pg19/blob/main/pg19.py).
It does not execute a remote dataset script. `--max-documents 1` selects the first
book. Other corpora use Hugging Face's streaming datasets. At most one book's
text/token IDs are held at a time by `ppl`; logits are discarded per step.

Document text plus `\n\n` is tokenized without special tokens by default.
The model state is retained across documents, and all adjacent token pairs,
including boundaries, are scored. `--separator ''` changes that convention.
This explicit concatenation differs from the original repository loop's handling
of document boundaries; record it when comparing numbers. `--prepend-bos` inserts
one BOS at the start of the entire stream. There is no automatic BOS per book.

`--max-tokens` counts next-token predictions, including `--warmup-tokens`.
Warmup predictions update state but are excluded from NLL. Perplexity is
`exp(total_nll / scored_tokens)`, never the average of individual perplexities.
Without `--reset-interval`, a run resets state only once at the start. Traces
contain cumulative and block mean NLL. `compare` buffers at most max_tokens+1
token IDs once to ensure identical inputs across variants; use `ppl` for very
large streams. Perplexity runs report loss, token counts, and attention state
bytes. Use `speed` for synchronized timing and peak CUDA memory measurements.

## H2O downstream tasks

```bash
python -m pip install -e '.[eval]'
python -m rttt lm-eval --model EleutherAI/pythia-70m \
  --method h2o --tasks h2o --cache-ratio 0.2 --output results/h2o-tasks.json

# Small integration check; this is not a statistically useful benchmark:
python -m rttt lm-eval --tasks piqa --num-fewshot 0 --limit 2 \
  --method linear --sink-tokens 1 --output results/piqa-smoke.json

python -m rttt lm-eval --tasks streamingllm --method streaming \
  --model EleutherAI/pythia-70m --sink-tokens 4 --cache-size 512 \
  --output results/streaming-tasks.json
```

The optional harness is pinned to 0.4.5, and its task implementations determine
prompts and metrics. `--num-fewshot` overrides the preset. Batch size is one;
every candidate/example resets state. Left truncation of overlong prompts is
reported in `rttt_diagnostics`. Each continuation token is scored once; rolling
likelihood uses disjoint scored blocks with a one-token conditioning overlap.

Dataset locations are updated to data-only repositories used by the current
upstream harness, including its [PIQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml)
and [MathQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mathqa/mathqa.yaml)
mirrors. The pinned prompts and metrics are retained. Actual dataset paths are
recorded in output diagnostics; preset tasks do not execute remote dataset scripts.

`--cache-ratio 0.2` is the **total** retained budget (minimum one token) based on
each tokenized, possibly truncated prompt. H2O defaults to half heavy hitters
and half recent. A fixed `--cache-size` is used when no ratio is supplied.
Here H2O eviction starts during sequential prefill. The paper also uses dense
prefill followed by cache selection in its system implementation, and its
released masking simulator has another prefill path. Those are not identical
experiments, even on the same tasks.

## HELM summarization

Use H2O's published HELM export/scoring workflow to produce the XSum or
CNN/DailyMail request JSONL with the exact prompts and request parameters:
[H2O HELM instructions](https://github.com/FMInference/H2O/tree/main/h2o_hf#text-summarization-with-helm).
Then replace its model-inference step with:

```bash
python -m rttt helm --model EleutherAI/pythia-70m --method h2o \
  --cache-ratio 0.2 --input xsum-requests.jsonl \
  --output results/xsum-replay.jsonl --limit 1000
```

Input lines contain `{"request": {"prompt": ..., "max_tokens": ..., "n": 1,
"temperature": ..., "top_p": ..., "stop": [...]}}`. The output preserves the
request and H2O's choices/logprobs/request_time schema. Greedy and nucleus
sampling, multiple return sequences, EOS, and multi-token stop strings are
supported. Log probabilities correspond to the selected generated tokens under
the original model distribution. `top_logprobs` is left empty; repetition
penalties and echo requests are rejected. Raw generated-token logs may include
the stop marker even though returned text excludes it.

Feed the resulting file through the original HELM workflow for official
summarization metrics. Replay alone does not compute ROUGE or claim a HELM score.
Using arbitrary prompts from the same datasets would change the benchmark.

## Efficiency measurements

```bash
python -m rttt speed --method h2o --cache-size 32 \
  --context-lengths 64,128,256,512 --decode-tokens 32 --repeats 3 \
  --output results/h2o-speed.json
```

The timing runner warms up, resets state for each repetition, and reports
prefill tokens/second, decode milliseconds/token, raw repetitions, medians,
persistent state bytes, and CUDA peak allocated bytes. It uses the same seeded
synthetic token stream for every method. Prefill is sequential and decode is
teacher-forced. These measurements characterize this reference implementation;
they do not reproduce H2O's FlexGen/DeepSpeed throughput or StreamingLLM's fused
system comparisons. Their sliding-window recomputation baseline is not bundled.
