# Bilevel Attention with an Exact Token Tape

## Motivation

Standard recurrent and linear-attention models compress the entire prefix into a fixed-size state. This is efficient, but information that is not preserved in that state cannot be recovered later. Full softmax attention avoids that compression by retaining contextual key/value vectors for every token, at the cost of a large KV cache.

The proposed bilevel-attention module separates these responsibilities:

- An **exact low-level tape** stores the original token sequence using approximately `log2(|V|)` bits per token. Token order supplies position, so the tape is a compact, lossless record of the input.
- A **large recurrent state** acts as contextual working memory. It can be lossy because the exact sequence remains available on the tape.
- A **small inner attention head** reads from the tape when information missing from the recurrent state is needed.

This is best understood as state-conditioned external-memory attention rather than simply an attention head nested inside another attention head.

## Proposed update

Let `x_i` be the token ID at position `i`, `S_(t-1)` the large state, and `d_r` the small inner-head dimension. The inner head performs a state-conditioned read:

```text
q_t = q(S_(t-1), x_t)

r_t = sum_{i <= t} softmax_i(
        q_t^T k(x_i, i) / sqrt(d_r)
      ) v(x_i, i)

S_t = GatedUpdate(S_(t-1), x_t, W_o r_t)
```

Because the attention weights depend on `S_(t-1)`, the tape read makes the recurrence nonlinear and multiplicative: the current state controls which past tokens alter the next state.

The keys and values do not need to be stored at the full state width. They may be generated from exact token IDs and procedural position features on demand, or cached at a very small width.

## Complexity

A single current-state query over a prefix of length `t` costs `O(t d_r)`. Processing a complete sequence recurrently therefore costs `O(T^2 d_r)`, not cubic time. It becomes `O(T^3 d_r)` only if the inner module unnecessarily recomputes full pairwise self-attention over every prefix at every step.

Peak storage can remain approximately:

```text
O(T log2(|V|))       exact token tape
+ O(dim(S))          recurrent working state
+ O(V d_r)           small embedding/projection tables
```

For example, a vocabulary of 50,000 tokens requires 16-bit token IDs, so one million tokens occupy roughly 2 MB. Caching BF16 keys and values with `d_r = 16` would require roughly 64 MB for the same sequence.

The important distinction is between **memory capacity** and **memory traffic**. The tape may fit easily in accelerator memory, but scanning it at every step still performs quadratic total work and repeatedly moves data. Computing keys and values from token IDs is especially attractive when the small vocabulary projection tables fit in cache and the entire read can use a fused, streaming softmax implementation.

## Main representational bottleneck

The tape is lossless, but access to it is not. One small attention result is a low-dimensional weighted sum. The architecture therefore separates three questions:

1. Is the original information still stored? Yes.
2. Can the controller address the correct occurrence? Not automatically.
3. Can one small read transmit everything needed for the update? Not necessarily.

If keys depend only on token identity, repeated occurrences of the same token are indistinguishable. Position features distinguish occurrences, but do not represent their original context. A very small head may also have difficulty selecting one location among a large number of similar distractors: as context length grows, the winning attention logit must increasingly separate itself from the alternatives.

## Recommended routed-retrieval variant

The most promising design uses the inner head as an **address router**, rather than expecting its weighted sum to contain the complete semantic answer:

1. Preserve the exact low-bit token tape.
2. Derive small keys from token identity, position, and inexpensive local context such as hashed n-grams or a tiny convolution.
3. Search over block summaries or landmarks and select a small number of relevant blocks or positions.
4. Retrieve exact token windows around those locations.
5. Re-encode only the retrieved windows at the large state width.
6. Apply the resulting contextual representation through a gated state update.

With blocks of `B` tokens and `k` selected blocks, a hierarchical search can reduce a global read from `O(T)` to approximately:

```text
O(T / B + kB)
```

The routing index may be compressed without compromising the losslessness of the underlying token tape. A fallback full scan can remain available when the router is uncertain.

## Parallelism tradeoff

If the inner query depends only on token-local inputs, all tape reads can be evaluated as a narrow causal-attention layer in parallel. However, the model then resembles a small ordinary attention layer feeding a recurrent state.

If the query depends on the previous large state, the module becomes a genuinely state-conditioned nonlinear recurrence. This is more expressive, but it prevents a fully parallel prefix scan because the query at time `t` is not known until `S_(t-1)` has been computed. Blockwise state updates are a practical compromise: perform parallel work within each block and update the global state between blocks.

## Initial evaluation plan

Useful ablations include:

- State-dependent versus token-only queries.
- Raw token and position features versus cheap local contextual features.
- Weighted-sum reads versus top-k address retrieval followed by window re-encoding.
- Inner widths such as 8, 16, 32, and 64 dimensions.
- One read versus multiple retrieval hops.
- Per-token global reads versus gated or block-level reads.
- Full scans versus hierarchical routing.

The model should be evaluated on repeated-entity disambiguation, associative recall, variable-length copying, long-range language modeling, and adversarial retrieval with many identical-token distractors. In addition to accuracy or perplexity, measurements should include bytes per stored token, prefill throughput, decode latency as context grows, attention entropy, router recall, and whether the recurrent state learns to ignore the tape.

## Summary

The module trades expensive contextual storage for inexpensive exact storage plus repeated computation. Its central promise is that the recurrent state need not preserve every detail because the raw sequence remains recoverable. Its central risks are state-dependent sequential training, repeated tape scans, ambiguous addressing, and the narrow communication channel between the tape and the large state. Hierarchical address retrieval followed by full-width re-encoding of a few exact windows offers the best balance of expressivity, storage, and computation.
