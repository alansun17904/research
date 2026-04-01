# Embedding Initialization Experiment

This folder contains a small offline experiment for initializing a GPT-2-style language model from corpus bigram statistics.

## Idea

1. Tokenize a corpus with the GPT-2 tokenizer.
2. Estimate a smoothed log-bigram matrix on the most frequent `top_k_vocab` tokens in the corpus.
3. Compute a truncated SVD of that matrix.
4. Use the factorization to initialize separate input embedding and output unembedding tables in an untied GPT-2 language model.
5. Optionally freeze those token tables during training.
6. Compare against a randomly initialized model trained end-to-end from scratch.

For a bigram logit matrix `B \in R^{V x V}`, the code uses:

- `B ~= U diag(s) V^T`
- `E = U diag(s^{1/2})`
- `W_U = V diag(s^{1/2})`

so that `B ~= E W_U^T`.

## Files

- `experiment.py`: end-to-end experiment runner.
- `toy_corpus.txt`: bundled offline corpus for a quick sanity test.

## Example

Run a small sanity check:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --config-name tiny --train-steps 50 --top-k-vocab 128
```

Run the intended comparison where the bigram-initialized token tables stay frozen while the baseline trains all parameters:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --config-name tiny --train-steps 50 --top-k-vocab 128 --freeze-svd-token-tables
```

Run on the cached offline C4 subset instead of the toy corpus:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --use-c4-filter-small --max-documents 1000 --max-corpus-tokens 100000 --config-name tiny --top-k-vocab 256 --train-steps 50 --freeze-svd-token-tables
```

Run the streaming full-vocab variant, which accumulates sparse bigram statistics over the entire GPT-2 vocabulary and uses truncated randomized SVD:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --use-c4-filter-small --max-documents 1000 --max-corpus-tokens 100000 --config-name tiny --init-scheme streaming_full --train-steps 50 --freeze-svd-token-tables
```

Run the direct low-rank bigram variant, which learns factor matrices `V` and `U` by next-token prediction before initializing the transformer token tables:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --use-c4-filter-small --max-documents 1000 --max-corpus-tokens 100000 --config-name tiny --init-scheme factorized_bigram --bigram-train-steps 200 --bigram-batch-size 256 --bigram-learning-rate 1e-3 --train-steps 50 --freeze-svd-token-tables
```

Instantiate the default GPT-2 architecture and keep the same initialization logic:

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate ml
python experiment.py --config-name gpt2 --train-steps 10 --batch-size 1 --top-k-vocab 128
```

## Practical Note

The full GPT-2 vocabulary has 50,257 tokens, so a dense `V x V` bigram matrix is not practical to factorize directly in this prototype. The current code estimates and factorizes the active `top_k_vocab x top_k_vocab` block from tokens that actually appear in the corpus, then writes those vectors back into full GPT-2-sized embedding and unembedding tables.

The `streaming_full` mode avoids that dense matrix by accumulating only observed bigrams into a sparse matrix over the full GPT-2 vocabulary and running truncated randomized SVD on that sparse operator.
