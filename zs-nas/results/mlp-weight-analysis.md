# Read-only MLP weight inspection

All six MLPs and all 48 attention OV maps were inspected in CPU float64 from the original checkpoint. This inspection did not instantiate a model, evaluate GELU, process token inputs, run forward passes, or measure perplexity. The JSON companion contains norms, bias statistics, all key/value self-alignment quantiles, neuron examples, every MLP/head overlap, formulas, and unchanged hashes.

The weight summaries do **not** explain the MLP repetition ranking by themselves. In particular, every MLP is almost orthogonal to head 4.3 in this matrix summary (adjusted cosine between −0.0031 and +0.0040), so there is no evidence that repeating an MLP simply cancels or amplifies that head’s linear map.

| MLP | Input norm | Output norm | Raw map norm | LN-adjusted map norm | Adjusted trace |
|---|---:|---:|---:|---:|---:|
| 0 | 31.890 | 20.430 | 23.436 | 12.133 | 25.184 |
| 1 | 32.578 | 21.171 | 25.937 | 35.965 | -204.412 |
| 2 | 33.807 | 22.584 | 22.434 | 24.699 | -250.604 |
| 3 | 30.906 | 27.797 | 20.383 | 26.270 | -195.083 |
| 4 | 29.812 | 31.701 | 21.470 | 32.696 | -107.666 |
| 5 | 29.670 | 36.570 | 26.659 | 40.798 | -220.186 |

For neuron `i`, key/value self-alignment is `cos(W_in[i,:], W_out[:,i])`. It measures a diagonal feedback term in residual coordinates; it does not identify active neurons or a retrieval role.

| MLP | Self-cosine p05 | Median | p95 | Positive fraction |
|---|---:|---:|---:|---:|
| 0 | -0.017 | 0.074 | 0.157 | 91.2% |
| 1 | -0.567 | -0.284 | 0.075 | 9.9% |
| 2 | -0.682 | -0.410 | 0.108 | 9.5% |
| 3 | -0.664 | -0.264 | 0.249 | 24.3% |
| 4 | -0.596 | -0.126 | 0.303 | 35.7% |
| 5 | -0.490 | -0.167 | 0.275 | 16.7% |

The ungated surrogate is `A = W_out @ W_in`; the adjusted surrogate is `A @ diag(gamma) @ P`, using the original post-attention LayerNorm gain and `P = I − 11ᵀ/512`. This substitutes identity for GELU. The actual local derivative contains `W_out @ diag(GELU′(z)) @ W_in @ J_LN(x)`: input-dependent gates, variance and radial normalization matter. GELU derivatives can be negative or exceed one.

| MLP | Cosine with head 4.3 | Most aligned attention map | Most opposing attention map |
|---|---:|---|---|
| 0 | -0.0019 | 5.3 (+0.0296) | 2.6 (-0.0104) |
| 1 | -0.0023 | 2.6 (+0.0354) | 5.3 (-0.0824) |
| 2 | -0.0031 | 2.6 (+0.0401) | 5.3 (-0.1452) |
| 3 | 0.0006 | 2.6 (+0.0619) | 5.3 (-0.1527) |
| 4 | 0.0040 | 4.0 (+0.0400) | 5.6 (-0.0529) |
| 5 | 0.0033 | 5.3 (+0.0453) | 5.5 (-0.0826) |

Head5.3 is the strongest opposing map for MLPs 1–3 (cosines −0.0824, −0.1452, −0.1527), while MLP 5 is weakly aligned with it (+0.0453). MLP 1/MLP 2 have the largest MLP-pair alignment (+0.2497); MLP 2/MLP 3 follow (+0.2126). These are geometric overlaps, not measured output cancellation or learned chains.

Existing cluster results report perplexity 47.5756 for MLP 1 repeated twice and 847.1861 for MLP 5, against38.9507 baseline. MLP 5’s output-weight norm is 1.7273× MLP 1’s, but its adjusted ungated-map norm is only 1.1344×. Its identity-GELU affine output offset is 95.9504 versus 13.3114 (7.2081×). The offset includes LayerNorm shift and both MLP biases: `W_out @ (W_in @ beta + b_in) + b_out`. It is **not** an actual measured MLP output.

| MLP | Median affine preactivation offset | Positive offset fraction | Identity-GELU output offset norm |
|---|---:|---:|---:|
| 0 | -0.139 | 0.3% | 4.225 |
| 1 | -0.347 | 1.4% | 13.311 |
| 2 | -0.048 | 39.1% | 10.742 |
| 3 | -0.023 | 46.8% | 14.872 |
| 4 | 0.092 | 58.1% | 19.070 |
| 5 | 0.190 | 75.7% | 95.950 |

These offsets make different gating regimes plausible: MLP 1’s median is −0.3466, versus +0.1898 for MLP 5. Actual residual inputs can reverse that picture. We cannot infer why MLP 1 is least destructive, or attribute MLP 5’s loss increase to any one statistic without activation evidence.

Three bounded experiments are defensible as low-confidence probes, with the exact candidate structures also recorded in JSON:

1. **Condition MLP 4 with head 4.3.** Execute layers 0–3, one head-only application of 4.3, then layers 4–5; set MLP 4 count 2 and keep head counts 1. Compare with the identical sequence at MLP 4 count 1 and with ordinary traversal using head 4.3 count 2 plus MLP 4 count 2. This tests input/gating changes, not unsupported direct map cancellation.
2. **MLP 1 plus a later opposing map.** Use ordinary traversal with MLP 1 count 2 and head 5.3 count 2. Compare with each single change. The modest negative overlap suggests a counterbalance probe, while routing and later layers may defeat it; no gain is predicted.
3. **One extra pass through layers 1–2.** Use `[0]`, `[1,2]` repeated twice, `[3,4,5]`, with all within-layer counts 1. Their related ungated transformations may saturate or compound; this is a joint layer experiment, and also repeats attention and output biases.

Crucially, head and MLP feedback loops within the same layer remain parallel: `a ← a + C(x+a)` and `m ← m + M(x+m)`. Simply raising both counts cannot feed the repeated head into that same visit’s MLP recurrence. A preceding head-only block or another full-layer visit is necessary for that conditioning; the JSON format has no MLP-only sequence block.

All comparisons omit attention routing and token mixing, activation covariance, downstream nonlinearities, and changing recurrent inputs. Checks passed for tensor shapes, trace/key-value identities, gain/centering factorization, cosine bounds, and unchanged checkpoint/source hashes.

No performance forward passes or SSH were run. These weight-only hypotheses are unmeasured and do not assert functional labels or expected improvements.

## Separately scoped activation evidence

The parent supplied `results/mlp-activation-analysis.json`, a remote diagnostic on the first 2,048 validation tokens with zero scored targets. It reports zero hook-induced baseline logit error and frozen, unchanged parameter versions. This inspection only read that artifact; the table is activation evidence, not another weight statistic or a perplexity estimate.

| MLP | First delta RMS / layer-input RMS | Second / first RMS | Second–first cosine |
|---|---:|---:|---:|
| 0 | 8.311 | 0.974 | 0.246 |
| 1 | 0.561 | 0.765 | 0.297 |
| 2 | 0.676 | 0.752 | 0.161 |
| 3 | 0.742 | 0.846 | 0.178 |
| 4 | 0.929 | 1.157 | 0.256 |
| 5 | 3.240 | 1.570 | 0.914 |

On this window, MLP 5’s second delta grows by 57% and closely follows its first (cosine 0.914), whereas MLPs 1 and 2 have shrinking, less-aligned second deltas. This provides direct evidence of MLP 5 amplification here, consistent with its poor full-sample repeat result; it does not prove that amplification causes the complete loss change.

Given both sources of evidence, prioritize bounded total counts of **3 for MLP 1 and MLP 2** before further MLP 5 repeats. Their negative key/value self-alignment and contracting second applications motivate checking whether another iteration settles or rotates, although count 2 already worsens validation and convergence is unproven. Use the baseline or a fixed incumbent as the parent, hold its heads unchanged, and compare counts 1/2/3. The extra `[1,2]` block remains a separate interaction probe. These two additional hypothetical structures validate under the existing graph schema; no graph files were created.

All activation conclusions are limited to this single window. No new local forward pass, source edit, graph edit, or checkpoint change occurred.
