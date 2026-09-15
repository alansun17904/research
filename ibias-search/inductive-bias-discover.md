# Experiment plan: sequence primitives across Transformer scale

## 1. Questions and scope

**Primary question:** which sequence families remain easy to learn as the frozen random Transformer core grows, and which become accessible only at larger sizes?

We retain the current scientific setting: binary next-bit prediction, a randomly initialized frozen core, and one shared trainable embedding/unembedding pair per support set. These experiments study the inductive bias of that architecture **under a specified initialization and adaptation protocol**. They do not measure the capabilities of pretrained language models.

Separate three changes that otherwise get conflated:

1. **Core scale:** increase the number of frozen parameters.
2. **Sequence scale:** increase sequence length and the distance over which a rule must operate.
3. **Adapter capacity:** increasing width also increases the number of trainable embedding/unembedding parameters. Include a control that holds this capacity fixed.

A primitive is an executable, length-parameterized generator or rule, such as repeating an arbitrary prefix or copying a bit at a fixed lag. A list of successful strings, an absolute-position mask, or a low loss on the discovery set is insufficient evidence of a persistent primitive.

This is a proposed protocol, not a record of completed experiments. Several settings below require implementation before submission; see Section 10.

## 2. Scaling recipe

Use a **GPT-style decoder ladder** with full causal multi-head attention, constant head dimension 64, and feed-forward width `4 * d_model`. The standard Transformer uses the 4:1 feed-forward expansion and 64-dimensional heads; these provide a recognizable starting geometry. [Attention Is All You Need, Sections 3.2–3.3 and Table 3](https://arxiv.org/html/1706.03762)

The two largest ladder entries use the depth/width pairs from the first two GPT-2 sizes: `(12, 768)` and `(24, 1024)`. The smaller entries are our explicit downscaled proxies, not published GPT-2 checkpoints. Use pre-normalization, a final normalization, and depth-scaled residual-output initialization. [GPT-2, Section 2.3 and Table 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Keep the following distinctions explicit:

- This is a binary, RoPE-based variant with untied input/output weights and no QKV biases. It is not an exact GPT-2 reproduction, and its parameter counts differ substantially from the usual vocabulary-heavy GPT model labels.
- A joint depth/width ladder alone cannot identify why a primitive changes. Add width-only, depth-only, and approximately parameter-matched comparisons.
- The primary optimization recipe uses standard parametrization and an equal learning-rate search for every size. Do not assume that a learning rate tuned on the smallest model transfers unchanged. μP is an established approach to width transfer, but its reference implementation requires coordinated initialization, readout, and optimizer changes; it also specifies a fixed-depth base/target relationship. A μP replication would be a separately specified follow-up, after checking its behavior with a frozen core. [Tensor Programs V](https://arxiv.org/abs/2203.03466), [official μP implementation](https://github.com/microsoft/mup)
- Chinchilla concerns compute-optimal training of language models, where both model size and training data scale. It does not establish a token budget for adapting `4d` parameters around a frozen core. Our primary comparison fixes the adaptation budget; a separate budget sensitivity study checks optimization limitations. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

## 3. Common model specification

Every model inherits this table unless a named control explicitly overrides a row.

| Hyperparameter | Planned value |
| --- | --- |
| Vocabulary / tokenization | Two tokens, `0` and `1`; no BOS, EOS, padding, or learned positional tokens |
| Prediction task | Input `x[:-1]`, target `x[1:]`; mean natural-log cross-entropy over all `n - 1` targets |
| Stack | Decoder-only; sequential attention and MLP residual branches |
| Attention | Full causal softmax attention; queries, keys, and values all have width `d`; no GQA/MQA |
| Head dimension | `d_head = 64`, with `heads = d / 64` |
| Attention scale | `1 / sqrt(d_head)`; no additional learned temperature |
| Positions | RoPE on all Q/K head coordinates, adjacent-pair rotation, base 10,000; no rotation of values |
| RoPE length handling | Generate/cache rotations through 2,048 positions for every model; no interpolation, NTK scaling, or learned length-specific parameters |
| MLP | `Linear(d, 4d) -> GELU -> Linear(4d, d)`; exact GELU, no gating |
| Normalization | Pre-LayerNorm before attention and MLP; final LayerNorm before unembedding; epsilon `1e-5` |
| Normalization parameters | Affine scale initialized to 1, offset to 0; both frozen |
| Biases | QKV: none. Attention output and both MLP linears: present, initialized to zero, frozen. Unembedding: none |
| Residual coefficients | Both additions have coefficient 1; depth correction is in initialization, not an extra forward multiplier |
| Dropout | 0 for embeddings, attention probabilities, MLP, and residual paths |
| Embedding scale / tying | No extra `sqrt(d)` multiplier; embedding and unembedding are independent and untied |
| Trainable parameters | Only `E` of shape `2 x d` and `U` of shape `d x 2`: `P_train = 4d` |
| Frozen parameters | All attention/MLP weights and biases, all LayerNorm parameters |
| Numerical policy | FP32 parameter storage and optimizer state; BF16 autocast for GPU matrix operations, FP32 loss and normalization reductions; no FP16 loss scaling; disable TF32 for the FP32 reference checks |
| Fit reset | Identical initial core, E/U, optimizer state, and minibatch RNG for each support-set evaluation |

### Initialization: primary recipe

Use independent zero-mean Gaussian draws with standard deviation `0.02` for E, U, QKV, and the MLP input projection. This follows the conventional GPT initialization scale; the untied binary readout is our extension. [GPT-2 reference implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)

For a model with `L` blocks, initialize the attention output projection and MLP output projection with standard deviation:

```text
0.02 / sqrt(2 * L)
```

There are two residual branches per block. This applies GPT-2's inverse-square-root correction in the number of residual layers. Do not also divide the forward residual contributions by this factor. [GPT-2, Section 2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Use deterministic RNG streams derived from `(core_seed, parameter_name)`, with a specified stable hash rather than Python's process-randomized `hash()`. Matching seeds across lengths must produce exactly the same weights. Matching seeds across widths labels a replicate; it does not imply nested or identical random features. Ablations that preserve a tensor's shape reuse its exact initialized value.

### Exact parameter accounting

For the architecture above, including biases and the final LayerNorm:

```text
P_block = 4*d*d + 2*d*ff_width + 6*d + ff_width
        = 12*d*d + 10*d                  when ff_width = 4*d
P_frozen = L * P_block + 2*d
P_train  = 4*d
P_total  = L * (12*d*d + 10*d) + 6*d
```

RoPE buffers are not parameters. Sequence length does not change the parameter count. Verify these formulas against the instantiated model before launching the sweep.

## 4. Models to test

### 4.1 Primary joint depth/width ladder

| ID | Layers `L` | Width `d` | Heads | Head dim | FF width | Frozen parameters | Trainable E/U | Total parameters | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T0 | 2 | 128 | 2 | 64 | 512 | 396,032 | 512 | 396,544 | Smallest serious reference |
| T1 | 4 | 256 | 4 | 64 | 1,024 | 3,156,480 | 1,024 | 3,157,504 | Discovery and shape anchor |
| T2 | 8 | 512 | 8 | 64 | 2,048 | 25,207,808 | 2,048 | 25,209,856 | Larger discovery model |
| T3 | 12 | 768 | 12 | 64 | 3,072 | 85,028,352 | 3,072 | 85,031,424 | GPT-2-small depth/width; primary confirmation |
| T4 | 24 | 1,024 | 16 | 64 | 4,096 | 302,237,696 | 4,096 | 302,241,792 | GPT-2-medium depth/width; later confirmation |

T0–T3 are the first complete scale comparison. T4 is a planned extension after resource profiling and freezing the rule bank, not a prerequisite for the initial result. All rows use the same architecture, initialization recipe, and adaptation-budget definition.

### 4.2 Width-only and depth-only controls

Run these on frozen rules at `n = 512` first. Each row uses the common specification, including depth-dependent initialization. Reuse W256/D4/T1 results when the entire configuration, data, and seeds match.

| ID | Layers | Width | Heads | Head dim | FF width | Total parameters | Trainable E/U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W128 | 4 | 128 | 2 | 64 | 512 | 792,320 | 512 |
| W256 = D4 = T1 | 4 | 256 | 4 | 64 | 1,024 | 3,157,504 | 1,024 |
| W512 | 4 | 512 | 8 | 64 | 2,048 | 12,606,464 | 2,048 |
| W1024 | 4 | 1,024 | 16 | 64 | 4,096 | 50,378,752 | 4,096 |
| D2 | 2 | 256 | 4 | 64 | 1,024 | 1,579,520 | 1,024 |
| D8 | 8 | 256 | 4 | 64 | 1,024 | 6,313,472 | 1,024 |
| D16 | 16 | 256 | 4 | 64 | 1,024 | 12,625,408 | 1,024 |

**Key shape comparison:** W512 versus D16 has approximately 12.6M parameters in both cases, differing by about 0.15%. Repeat this pair with the fixed-capacity adapters below; otherwise the wider model also has twice as many trainable parameters.

Width scaling here increases the number of heads while fixing head dimension. At T2, additionally test `(heads, head_dim) = (4, 128), (8, 64), (16, 32)` with `L=8`, `d=512`, and `ff_width=2048`. Parameter counts remain unchanged. This control changes both head decomposition and the RoPE spectrum within a head; interpret it accordingly.

### 4.3 Fixed trainable-capacity control

Repeat T0–T3, W512, and D16 with exactly **512 trainable adapter parameters**, initially at `n=512`.

- Set adapter dimension `r=128` at every width.
- Train `E0` of shape `2 x r` and `U0` of shape `r x 2`, initialized with Gaussian standard deviation `0.02`.
- Freeze independent expansion matrices `A` of shape `r x d` and `B` of shape `d x r`, with entries distributed as `Normal(0, 1/r)` where the second argument denotes **variance**.
- Use effective weights `E = E0 @ A` and `U = B @ U0`.
- Keep the original core and final LayerNorm. Optimize only E0/U0 using the same calibration procedure and fit budget as the corresponding native-adapter model.
- Count the additional `2rd` frozen projection parameters separately. Total stored parameters are `P_frozen + 2rd + 4r`; do not report the native model's total for this variant.

The expected marginal initialization variance of effective E/U matches the native recipe, but the induced correlations differ. This is a capacity diagnostic, not an equivalent random prior. Run the projected variant even at `d=r` so the projection construction is consistent across sizes.

### 4.4 Architecture and initialization controls

Use T1 and T2 on the first four frozen rules selected by discovery, at `n=512`. The selection must happen before held-out evaluation.

| Control | Exact change from the common model | Purpose |
| --- | --- | --- |
| Identity core | Remove all blocks; retain fixed final LayerNorm and native E/U; no position input. Total `6d`, trainable `4d` | Tests what token-to-next-token adapters alone can explain |
| Unigram / bigram | Laplace smoothing `alpha=1`; counts from exactly the same sampled support-example stream | Analytic local-prediction controls |
| No RoPE | Replace Q/K rotations with identity; all weights, dimensions, and counts unchanged | Tests dependence on positional structure |
| Local attention | Permit `0 <= query_position - key_position < 32`; all weights and counts unchanged | Tests dependence on direct long-range access; stacked layers can still communicate farther |
| Attention-only | Remove the MLP branch and its LayerNorm; retain original attention weights and their original initialization. Total `L*(4*d*d + 3*d) + 6*d` | Tests the contribution of tokenwise nonlinear transformations; explicitly not parameter matched |
| Fan-in initialization | Same architecture; E has variance 1, U variance `1/d`; QKV and MLP input weights variance `1/d`; attention output variance `1/(2Ld)`; MLP output variance `1/(2L*ff_width)`; biases and norms unchanged | Checks whether the result survives a different variance-preserving random prior |

Calibrate adapter learning rates independently for these controls with the same search allowance. For a stronger architecture-level claim, extend the relevant ablation and the fan-in replication to T0–T3. A result that depends on one initialization remains a result about that architecture–initialization pair.

## 5. Adaptation hyperparameters and fair budgets

### Primary optimizer recipe

| Hyperparameter | Value / selection rule |
| --- | --- |
| Optimizer | AdamW, E/U only |
| Betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Weight decay | `0.01`, decoupled, applied to both trainable matrices |
| Learning-rate groups | Same peak LR for the two trainable matrices; select separately per model variant using the grid below |
| Peak LR grid | `1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1` |
| Updates per fit | 128 |
| Warmup | First 8 updates, linearly from `peak_lr / 8` to `peak_lr` |
| Decay | Remaining 120 updates: cosine decay to `0.1 * peak_lr` on the final update |
| Gradient clipping | Global L2 norm of trainable gradients capped at 1.0, after accumulation |
| Sampling | Uniform with replacement from the fixed support set |
| Minibatch RNG | Seed 0, reset for every fit; identical sampled sequence indices for comparisons with the same batch and support |
| Loss aggregation | Mean over target tokens across the entire effective batch; gradient accumulation must preserve this mean |
| Early stopping / checkpoint choice | None; use the final update. Log losses at updates 0, 8, 32, 64, and 128 |
| Extra regularizers | No label smoothing, auxiliary losses, or adapter dropout |

**Calibration, before discovery:** evaluate each LR with core seeds `0,1,2` on three fixed calibration distributions: Bernoulli bits with `P(1)=0.3`; a symmetric first-order Markov chain with flip probability 0.1 and a fair initial bit; and repetition of a uniformly sampled 16-bit prefix. Use lengths 128 and 512, 64 support strings and 256 calibration-validation strings per distribution/length, with data seed 5000. Average held-out mean CE equally over distributions, lengths, and core seeds; choose the minimum, breaking ties toward the smaller LR. Discard nonfinite runs. Select one LR per model variant, shared across all later rules and lengths. The winning LR is an outcome of this fixed procedure, not an unspecified run-time choice.

These are development distributions and must be labeled as such; success on their rule types is not blind discovery. Archive their examples and latent generator identities. No final held-out data may influence LR selection. If a selected LR lies at a grid boundary, record the limitation; any grid extension requires a new calibration version before discovery. Do not silently carry the current CLI default LR of 0.2 into the sweep.

### Length and batch matrix

The primary budget holds optimizer updates fixed and approximately matches scored tokens per update. Every model uses each row; only physical microbatching may change.

| Sequence length `n` | Scored targets / example | Effective batch | Microbatch | Accumulation | Examples / fit | Scored tokens / fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 127 | 32 | 2 | 16 | 4,096 | 520,192 |
| 512 | 511 | 8 | 2 | 4 | 1,024 | 523,264 |
| 2,048 | 2,047 | 2 | 2 | 1 | 256 | 524,032 |

The small token-count difference is explicit; these budgets are not mathematically identical. Keep effective batches unchanged if memory forces microbatch 1, and double accumulation. Log the actual setting. Matching tokens across lengths changes the number of examples and support repetitions; therefore also run an **equal-example control** with batch 8 and 128 updates at every length, giving 1,024 examples per fit. Limit that secondary check to T0, T2, T3 and the first four frozen rules.

For optimization sensitivity, repeat those four rules on T0–T3 at `n=512` for 512 updates, with 32 warmup updates and the same fractional cosine schedule. This is a separate larger-budget result. A failed 128-update model that succeeds here indicates budget sensitivity, not an architectural impossibility.

Record wall time, effective tokens/s, peak allocated GPU memory, initial logit scale, per-layer activation RMS, gradient norms, and clipping frequency. Equal tokens do not mean equal FLOPs, particularly across sequence lengths. Do not use the full-training `6ND` approximation as an exact cost model for frozen-core adaptation; profile the actual forward/backward path.

## 6. Longer sequences and primitive definitions

Use `n=128` and `n=512` for the first discovery/validation matrix, then `n=2048` for length confirmation. Test two distinct notions of persistence:

1. **Across model size:** refit adapters at every model size on exactly the same frozen rule and support/test split.
2. **Across length:** fit at `n=128` or `n=512`, then evaluate the same adapted weights on fresh longer sequences without more updates. Also perform within-length adaptation at 2,048 to distinguish transfer failure from inability to fit at that length.

For each primitive, retain a fixed-difficulty version and, when meaningful, a growing-difficulty version. Examples: keep lag 16 fixed versus increase lag to `n/8`; keep a 16-bit motif fixed versus increase motif length with `n`. Merely adding more repetitions of a fixed motif is a different test from learning a longer motif.

Candidate rule categories to investigate include repetition, lagged copy/complement, finite-state transition constraints, and parity-like state tracking. They are hypotheses, not predicted discoveries. Store executable generators with explicit latent parameters and length mappings. A fixed `n`-bit mask or a fixed number of ones must be reformulated and revalidated before being called a length-independent primitive.

Report both all-target CE and performance on the rule's predictable target positions. For example, a repeated random prefix contains an initially unpredictable segment. Changing the fraction of such positions with length can improve mean loss without improving the underlying capability. Use the same predefined target mask for every model and control; the primary adaptation objective still scores all targets. Include a fair IID-bit negative control, whose expected optimal all-target CE is `log(2)` and accuracy is 0.5.

## 7. Discovery and held-out protocol

### Discovery settings

| Search hyperparameter | Value |
| --- | --- |
| Models / lengths | T0, T1, T2 at `n=128,512` initially |
| Discovery core seeds | `10,11,12` |
| Proposal RNG seed | `100 + core_seed`, matched across model configurations |
| Restarts / rounds | 2 / 32 |
| Initial seed / mutations | Random binary seed, no manual seed override / 3 mutations |
| Proposals per branch | 32 |
| Shortlist / low-transfer slots | 4 / 1 |
| Beam width | 2 |
| Novelty | Minimum normalized Levenshtein distance to a member; weight 0.05 |
| Ranking | Transfer gain plus novelty; no adapted-loss penalty (`beta=0`) |
| Acceptance gate | Every member: CE at most 0.65 nats and accuracy at least 0.60 |
| Trace | Enabled; archive every proposed sequence, including rejected additions |
| Fit budget | The primary 128-update recipe, reset for every evaluation |

A family can contain at most 36 members after 32 one-member growth rounds from a four-member seed group. A later 64-round expansion can reach 68, but constitutes a separately logged search budget. If random seed groups repeatedly fail, report that outcome; any structured-seed search is a separately labeled arm with its seed generator frozen in advance.

The initial matrix has 18 search runs. A conservative upper bound is `2 * [1 + 32 * 2 * (1 + 4)] = 642` adaptation fits per run, or **11,556 fits** overall, excluding calibration and final validation. Seed failures and unfilled beams reduce this. Use measured fit and proposal-scoring cost to estimate the campaign before submitting it. Python's current quadratic-time edit-distance loop also needs profiling at these lengths.

### Freeze one shared rule bank

Canonicalize the union of hypotheses from all discovery sizes. Freeze at most eight distinct executable rules using discovery-only evidence: decreasing best discovered family size, then increasing mean discovery loss, then canonical rule serialization as a deterministic tie-breaker. Preserve provenance from every model/seed that proposed a rule. Do not compare a separately selected set of “best” rules for each model size.

For shape/ablation studies capped at four rules, use the first four entries of this already frozen ordering. Record any hand-written extension of the rule language before final testing; the present rule templates do not yet express arbitrary lagged copy or state machines.

### Fresh data and cores

- Final core seeds: `100` through `109`, disjoint from calibration/discovery.
- Final data seeds: `1000,1001,1002`; each defines an independent split with 64 unique support strings and 1,024 unique test strings per rule/length.
- All model sizes and controls use the same split for a given rule, length, and data seed. Repeated seeds across sizes are replicate labels, not a claim of identical cores.
- Exclude the **union** of discovery and calibration examples across all models, seeds, and lengths. Where a rule has a latent generating prefix/motif, partition that latent identity too: the same motif repeated to a different length is not fresh evidence.
- For overlapping rule families, maintain an experiment-wide registry so a final test example cannot also enter another rule's training support. Freeze the registry and splits before any final fit.
- Finite domains require a separate treatment. A binary period-8 rule has at most 256 strings at a fixed length and cannot provide the requested split. Enumerate feasible domains, reduce splits explicitly when possible, and mark rules with no fresh domain as exhaustive diagnostics, not held-out successes. Do not duplicate samples to meet a nominal sample count.

At eight rules, four primary models, two initial lengths, ten core seeds, and three splits, the initial final-validation matrix contains **1,920 Transformer fits**, plus matched identity fits. Analytic controls require no optimizer runs. Calibration, additional lengths, and ablations are extra costs and must appear separately in the compute ledger.

## 8. What would count as persistence or emergence?

Use continuous performance curves first: per-sequence mean CE, predictable-position CE, accuracy, and paired loss advantage over each control, plotted against both frozen parameter count and trainable adapter count.

For a predefined summary label, call a rule **persistent over the tested primary range** if, at each of T0–T3:

- At least 8 of 10 core seeds pass the frozen CE/accuracy thresholds on at least 90% of test sequences for each of the three data splits.
- The mean held-out CE advantage over both identity and bigram controls is at least 0.02 nats per target, with a positive lower confidence bound after accounting for the planned rule/size comparisons.

These are operational criteria chosen for this study, not consequences of a scaling law. Report failed cells and uncertainty, not only the label. Bootstrap complete core-seed and data-split clusters, not individual correlated tokens; use 10,000 bootstrap replicates and Holm-adjust the planned superiority tests at familywise alpha 0.05. Treat the three split replicates as limited evidence about data variation and report their individual results as well.

Call a primitive **emergent within the tested range** only when a prespecified capability criterion is first met at a larger size and persists at subsequent tested sizes. Report the continuous trend and budget sensitivity; crossing a threshold alone is not evidence of a discontinuity or an intrinsic critical scale.

Reserve the stronger description **evidence of an architectural primitive** for behavior that also survives the fixed-adapter and alternate-initialization controls, transfers across length as specified, and changes in a mechanistically relevant architecture ablation. Even then, the claim is bounded by the architectures, initializations, lengths, and training budgets tested; a finite sweep cannot prove scale independence.

## 9. Execution order

| Stage | Experiments | Decision / artifact |
| --- | --- | --- |
| A: implementation and calibration | Check parameter counts and initialization; profile T0–T3; run the fixed LR grid and numerical checks | Frozen model manifests, selected LRs, memory/time estimates |
| B: discovery | T0–T2, lengths 128/512, three discovery cores, fixed search budget | Complete discovery archive and one frozen rule bank |
| C: primary size study | T0–T3 on every frozen rule, lengths 128/512, ten fresh cores and three splits; matched local controls | Rule-by-size performance matrix |
| D: explanation and length | Width/depth sweeps and ablations on the first four frozen rules at 512; fixed adapters; 512-update and equal-example controls; length transfer and within-length tests at 2,048 | Separate core capacity, adapter capacity, optimization, and length effects |
| E: larger confirmation | T4 on the first four frozen rules at 512/2,048 with the same ten final cores and three splits, after calibration and resource profiling | Extension of the measured scale range; no new rule selection from these tests |

Run all model experiments through SLURM on Kennel. Use `debug` only for short profiling/calibration jobs within its limits, and `standard` for ordinary independent fit/search jobs. Split work into resumable units before considering `long` or `sweep`. Keep code, manifests, and durable results under `/data/cl/u/awsun/rttt`; stage data-intensive inputs to node-local `/scratch` where available. Never use AFS in a job. Do not assign a GPU-hour total or promise that T4 fits a particular GPU until the allocated hardware and actual memory use have been measured.

## 10. Required implementation before these plans are runnable

The existing CLI is not yet an implementation of the complete specification above. In particular:

1. Add explicit initialization, final frozen LayerNorm, and recorded architecture options. Current constructors use mixed PyTorch defaults; increasing width/layers alone would not implement this recipe.
2. Add optimizer betas/epsilon, scheduler, clipping, microbatch accumulation, precision settings, and complete configuration serialization. Record actual rather than inferred training budgets.
3. Separate maximum RoPE capacity from data length, enabling evaluation at longer lengths without changing weights or refitting.
4. Add independent RNG streams, fixed-capacity adapters, and the named architecture/initialization controls.
5. Support a shared rule bank, latent-aware split/exclusion registry, cross-size validation, and per-position scoring. Current validation inherits the discovery model configuration; it must accept the explicitly selected test model while retaining the frozen rule and data split.
6. Preserve resetting while avoiding a full frozen-core deepcopy for every fit where possible. Frozen weights still participate in the gradient path to E; do not wrap the core in `no_grad()` or detach its activations.
7. Add resumable search/fit outputs and hardware-aware scoring batches. Profile proposal generation/novelty computation separately from model compute; optimize the same distance metric rather than silently changing it.

Every run manifest should contain its complete model/initialization/adapter configuration, optimizer and budget, rule/split identifiers, all seeds, code revision or source snapshot hash, dependency versions, hardware, observed resource use, and links to its frozen protocol and discovery provenance. The reporting/orchestration layer owns this metadata; computational modules need only receive their explicit inputs and return their measured results.
