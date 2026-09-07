# `eap/core.py` Wiki

## What This Module Does

`eap/core.py` scores edges or nodes in a transformer computation graph based on how they influence the model's output. In a nutshell, we are trying to figure out:

> What model components most *causally influence* final logits / the model's performance?

### A Brief Introduction to Edge Attribution Patching (EAP)

$\def\R{\mathbb{R}}\def\E{\mathbf{E}}$

We can view any neural network through its computation graph where the nodes and edges are latent variables and the dependencies between them. For any model $M : X \to Y$, let $G = (V, E)$ be its computation graph. Also, let $L: Y\times Y \to \R^{\geq 0}$ be a loss function. For any given edge $e \in E$ and $(x,y) \in X\times Y$, we wish to estimate 

$$ L(M_{e \gets e'}(x), y) - L(M(x), y), $$   

where $M_e$ is the model with the edge $e$ fixed to be the constant function $f(x) = e'$ (essentially we are removing the edge between $e$'s source and target). Naively, we can estimate this by evaluating the model $O(|E|)$ times. But, this is expensive for large models. Instead, notice the "patched" model can be expressed as a function of two variables $M_e(x,e') = M_{e\gets e'}(x)$. EAP uses a first-order Taylor expansion to approximate this:

$$ L(M(x), y) \approx L(M_e(x), y) + \nabla_e L(M_e(x), y)(e' - e), $$

and so,

$$ L(M_{e \gets e'}(x), y) - L(M(x), y) \approx \nabla_e L(M_e(x), y)(e' - e). $$

this requires exactly two forward passes (one to compute $e$ and another to compute $e'$) and one backward pass (to compute $\nabla_e L(M_e(x), y)$).

*Throughout, we are implementing a variant of EAP (with integrated gradients). But for all intents and purposes, we can treat EAP as a general framework for edge attribution.*


## The Main Interaction Flow

`eap/core.py` computes attribution through two stages:

1. Build a graph of model components such as the input, attention heads, MLPs, and logits.
2. Run an attribution pass that assigns a numeric score to each edge or hooked node in that graph.

The exposed API is tight:

- `Graph.from_model(...)`
- `attribute(...)`

Most code uses the module like this:

```python
graph = Graph.from_model(model)
edge_scores = attribute(model, graph, dataloader, metric, patch_type="edge")
node_scores = attribute(model, graph, dataloader, metric, patch_type="node")
```

What happens behind the scenes:

1. `Graph.from_model(...)` reads the model shape and builds a fixed graph layout.
2. The dataloader provides batches of `(clean, corrupted, label)` examples.
3. `attribute(...)` compares clean and corrupted runs.
4. The code uses TransformerLens hooks to measure how activation differences flow through the model.
5. Each graph edge or hooked node gets a score, and the same scores are also returned as a NumPy vector.

## The Core Mental Model

Think of the module as combining two views of the same transformer:

- A **logical graph view**:
  nodes and edges that describe who can influence whom.
- A **hooked runtime view**:
  forward and backward hooks that capture activations and gradients during real model runs.

`Graph` connects these two views. It says:

- which nodes exist
- which edges are legal
- which hook names correspond to each node
- where each source and target lives inside the score tensor

## Main Concepts

### Nodes

The graph contains four node kinds:

- `input`: the embedding stream entering the model
- `attn`: one node per attention head, named like `a1.h3`
- `mlp`: one node per layer MLP, named like `m1`
- `logits`: the final output node

### Edges

Edges represent possible residual-stream influence from an earlier node to a later one. All edges are directed. Since the residual stream is by definition additive, there exists at least one edge from any node in layer $l$ to any node in layer $l + k$ for any $k > 0$. 

Edges into attention heads are special: they are tagged as `q`, `k`, or `v`, because attention reads the residual stream through three separate inputs.

For example, suppose we are trying to attribute the edge `a0.h1 -> a2.h3<q>`. Then, 
we build two hooks: 
- source hook: `blocks.0.attn.hook_result`, slice head 1
- target hook: `blocks.2.attn.hook_result`, slice head 3

These hooks are then combined into a single `score` matrix:
- row: `graph.forward_index(a0.h1)`
- col: `graph.backward_index(a2.h3, qkv="q")`
- final score: the averaged value of scores[row, col] after all examples and IG steps

If we instead score the node `a0.h1` itself, we use the same forward hook to capture its clean-corrupted activation difference and a backward hook on that node's output. In that mode, the score tensor is just a vector indexed by `graph.forward_index(...)`.

### Score Tensor

Internally, attribution is accumulated in one of two layouts:

- edge patching: a matrix with rows = source-side slots from the forward pass and columns = target-side slots from the backward pass
- node patching: a vector with one entry per hooked forward node (`graph.n_forward`)

Each edge knows which row and column belong to it, and each hooked node knows its forward index, so the final tensor can be converted back into either per-edge or per-node scores.

## How Attribution Works

`attribute(...)` currently runs one method: an Integrated Gradients-style attribution pass over either edges or nodes.

The flow is:

1. Run the corrupted and clean examples to capture the difference at the model input hook.
2. Record activation differences for candidate source nodes with forward hooks.
3. Interpolate from corrupted input activations toward clean input activations.
4. Backpropagate the user-provided `metric(...)` through either target hooks (`patch_type="edge"`) or node-output hooks (`patch_type="node"`).
5. Accumulate contributions into the score tensor.
6. Copy the relevant tensor entries back onto `graph.edge_list` or the hooked `NodeSpec`s.

The result is both:

- a returned NumPy vector of edge or node scores
- `edge.score` values stored on each `EdgeSpec` for edge patching
- `node.score` values stored on each hooked `NodeSpec` for node patching

## Important Pieces in the File

### `NodeSpec` and `EdgeSpec`

These small dataclasses hold graph metadata:

- names and node types
- layer/head information
- hook names
- edge endpoints
- optional node or edge score

### `Graph`

`Graph` is the structural core of the module. It:

- normalizes model config
- creates nodes and edges
- supports grouped key/value heads through `n_kv_heads`
- handles `parallel_attn_mlp` when deciding which nodes can feed an MLP
- precomputes lookup tables used during attribution

The most important helpers are:

- `forward_index(...)`: where a source node lives in the score matrix
- `prev_index(...)`: how far back a target is allowed to look
- `backward_index(...)`: where a target node lives in the score matrix

### Hook and Scoring Helpers

The runtime helpers follow the execution order:

- `tokenize_plus(...)`: convenience tokenization helper
- `_capture_input_endpoints(...)`: captures clean/corrupted input states and clean logits
- `make_hooks_and_matrices(...)`: allocates the temporary activation buffer and builds hook lists
- `_run_source_chunk_ig(...)`: runs one source chunk through the IG loop
- `_run_source_chunk_node_ig(...)`: runs the node-patching IG loop for one source chunk
- `get_scores_eap_ig(...)`: drives batching, chunking, and score accumulation
- `_assign_edge_scores(...)`: maps matrix entries back onto graph edges
- `_assign_node_scores(...)`: maps vector entries back onto hooked graph nodes

## Why Chunking Exists

Attribution can be memory-heavy, so the module can split the work:

- `chunk_mode="none"`: run all backward targets together
- `chunk_mode="layer"`: process one layer's backward hooks at a time
- `chunk_mode="fixed"`: process fixed-size chunks
- `source_chunk_size=...`: also limit how many forward-side sources are handled at once

This changes performance and memory use, but not the overall meaning of the scores.
For node patching, only `source_chunk_size=...` matters because there is no backward target matrix to chunk.

## Token Aggregation

The attribution API also supports `token_aggregation=...` to control how sequence positions are handled after batch-averaging:

- `all-tok`: keep every token position explicit during the source-gradient contraction
- `last-tok`: only keep the final token position (`pos = -1`) for both source activations and target gradients
- `avg-tok`: average source activations and target gradients across token positions before taking their product

`all-tok` is the default and matches the previous behavior. `last-tok` and `avg-tok` are additional approximations that reduce activation-buffer memory from `O(n_pos * n_sources * d_model)` to `O(n_sources * d_model)`.

### Natural-Language Benchmark

Using `benchmarks/benchmark_token_aggregation_natural.py` with `attn-only-2l`, `batch_size=2`, `ig_steps=2`, `repeats=3`, `chunk_mode="none"`, and 100 prompt pairs built from cached `datablations/c4-filter-small` documents truncated to 24 words on CPU:

| `token_aggregation` | Mean latency (s) | Latency stdev (s) | Activation buffer estimate (MB) | Relative buffer size |
| --- | ---: | ---: | ---: | ---: |
| `all-tok` | 41.89 | 0.75 | 1.855 | 1.00x |
| `last-tok` | 46.34 | 1.70 | 0.037 | 0.02x |
| `avg-tok` | 45.65 | 1.13 | 0.037 | 0.02x |

On this setup, `last-tok` and `avg-tok` reduced the activation workspace by about 50x versus `all-tok`, but they were slightly slower in wall-clock latency on CPU. The benchmark harness also records `rss_mb_delta`, but that metric was allocator-noisy, so the table above reports the stable activation-buffer estimate instead.

Under the same CPU setup with `token_aggregation="all-tok"`, changing `patch_type` from edges to nodes gives:

| `patch_type` | Mean latency (s) | Latency stdev (s) | Mean RSS delta (MB, noisy) | Activation buffer estimate (MB) | Score tensor estimate (KB) | Total attribution workspace estimate (MB) | Relative latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `edge` | 34.65 | 1.18 | 139.60 | 1.855 | 3.79 | 1.859 | 1.00x |
| `node` | 47.66 | 4.23 | 151.56 | 1.855 | 0.07 | 1.856 | 1.38x |

On this benchmark, node patching was about 38% slower than edge patching. The activation-buffer estimate stayed the same because both modes reuse the same forward-side activation workspace, which dominates memory here. Node patching does save score-tensor memory, but for this `attn-only-2l` graph that difference is only about `3.72 KB` (`969` edge-score slots versus `19` node-score slots), so the total estimated attribution workspace barely changes. The mean RSS delta was slightly higher for node patching, but that metric remained allocator-noisy across repeats.

The final edge-score vectors from this run are stored in `benchmarks/results/c4_filter_small_100_token_aggregation_24w/` as:

- `all-tok_edge_scores.npy`
- `last-tok_edge_scores.npy`
- `avg-tok_edge_scores.npy`

### Cross Rank Correlation

Pairwise Spearman rank correlation between the saved final edge-score vectors:

|  | `all-tok` | `last-tok` | `avg-tok` |
| --- | ---: | ---: | ---: |
| `all-tok` | 1.000 | 0.584 | 0.338 |
| `last-tok` | 0.584 | 1.000 | 0.180 |
| `avg-tok` | 0.338 | 0.180 | 1.000 |

The corresponding CSV and JSON summaries are stored alongside the vectors in `benchmarks/results/c4_filter_small_100_token_aggregation_24w/`.

## Inputs the Module Expects

The module assumes:

- a TransformerLens `HookedTransformer`
- hook surfaces such as attention result hooks, split Q/K/V input hooks, and MLP input hooks are enabled
- a dataloader that yields `(clean, corrupted, label)` batches
- a `metric(logits, clean_logits, input_lengths, label)` function that returns a scalar objective for backpropagation

## Assumptions

- The current implementation uses a batch-averaged attribution heuristic: source activation differences and target gradients are averaged over the batch dimension before their contraction.
- This lowers the size of the forward-side activation workspace by a factor of roughly `batch_size`, but it is only an approximation. It replaces the mean of per-example products with the product of batch means, so cross-example covariance terms are ignored.
- Token handling is configurable. `token_aggregation="all-tok"` keeps positions explicit, `token_aggregation="last-tok"` keeps only the final token position, and `token_aggregation="avg-tok"` averages over positions before contraction.
- `last-tok` and `avg-tok` are additional heuristics and will in general produce different scores from `all-tok`.
