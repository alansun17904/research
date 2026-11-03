# Read-only attention-head weight inspection

All 48 heads were inspected from the original pinned Pythia-70M checkpoint; the four focal heads are compared below. No model was instantiated, no tokens or attention probabilities were computed, and no forward pass or performance evaluation ran. The JSON companion contains every head, all 1,128 pairwise OV cosines, rankings, formulas, and verified hashes.

Head 4.3 has the second-smallest raw OV Frobenius norm among all 48 heads; head 4.2 has the smallest. Their OV maps are moderately aligned, as are heads 4.3 and 4.5. These observations support testing repetition strength and interactions; they do not identify an induction, copy, or retrieval circuit.

| Head | Q Frobenius | K Frobenius | V Frobenius | Output Frobenius |
|---|---:|---:|---:|---:|
| 4.3 | 77.3275 | 7.0532 | 2.9010 | 2.4696 |
| 4.2 | 82.5056 | 5.3506 | 2.8830 | 2.4044 |
| 4.5 | 49.2677 | 6.4449 | 3.0670 | 2.9118 |
| 3.0 | 31.1987 | 4.1997 | 4.0624 | 4.2492 |
| Median of all 48 | 8.6275 | 6.4525 | 4.1266 | 3.9748 |

Ranks below are descending: 1 is largest; 48 is smallest. Traces are signed.

| Head | Raw OV norm (rank) | Raw trace | LN-adjusted norm (rank) | LN-adjusted trace |
|---|---:|---:|---:|---:|---:|
| 4.3 | 0.9378 (47/48) | 0.0666 | 1.9226 (44/48) | -0.0011 |
| 4.2 | 0.8576 (48/48) | -0.0710 | 1.7550 (46/48) | -0.2821 |
| 4.5 | 1.1627 (42/48) | 1.9491 | 2.3514 (35/48) | 3.9454 |
| 3.0 | 2.5913 (13/48) | -0.7710 | 4.0385 (9/48) | -1.2432 |

Using column-vector convention, raw `OV = O_h @ V_h`. The adjusted map is `OV @ diag(gamma) @ P`, where `gamma` is the original input-LayerNorm gain and `P = I - 11ᵀ/512` performs centering. LayerNorm shift and V bias produce the separate constant vector `O_h @ (V_h @ beta + b_V)`; its norm is 0.3326 for 4.3, 0.3702 for 4.2, 0.8060 for 4.5, and 0.9146 for 3.0. The shared attention output bias is excluded, matching extra head applications.

| Pair | Raw OV cosine | LN-adjusted OV cosine | Adjusted rank among 1,128 pairs |
|---|---:|---:|---:|
| 3.0 / 4.2 | 0.1039 | 0.1106 | 42 |
| 3.0 / 4.3 | 0.0935 | 0.1012 | 47 |
| 3.0 / 4.5 | 0.0311 | 0.0328 | 173 |
| 4.2 / 4.3 | 0.2963 | 0.2987 | 7 |
| 4.2 / 4.5 | 0.1877 | 0.1886 | 17 |
| 4.3 / 4.5 | 0.2739 | 0.2742 | 12 |

For head 4.3, heads 4.2 and 4.5 are its two most aligned peers. Its most negative adjusted cosine is only −0.0051 (with 2.2), so these matrices provide no strong opposing head to target for deletion. For scale, the largest adjusted cosine anywhere is 0.5343 (4.4/4.6); the smallest is −0.2868 (3.3/4.1).

The supported experimental hypotheses are limited:

- Sweep the repetition count of 4.3: its small value/output map makes modest per-pass changes plausible, conditional on comparable activations, attention, normalization scale, and the nonzero constant term. This is not a prediction that more repeats always help.
- Test 4.3 together with 4.2 or 4.5, but expect possible redundancy or overshoot as well as benefit. Their positive alignment supports shared linear directions, not additive perplexity gains.
- Compare a 3.0/4.3 composition as a less-aligned alternative (cosine 0.1012), while recognizing that different layers see different activation distributions. Validation must decide whether that difference is useful.

These summaries omit the input-dependent inverse standard deviation and Jacobian of LayerNorm, learned Q/K routing (including biases), rotary positions, causal masking, softmax probabilities, actual token/residual covariance, downstream layers and nonlinearities, and recurrent feedback. The Q norms of 4.2 and 4.3 rank first and third, but Q norms alone cannot establish attention sharpness or token roles. Traces and OV cosines are unweighted matrix summaries; they do not measure realized head-output alignment or loss causality.

Checks passed: 48 correctly sliced heads, 1,128 pairs, `trace(OV) == trace(VO)`, centering annihilates constant inputs, and cosine symmetry/unit diagonal/bounds. Checkpoint, config, model, evaluator, tests, requirements, and baseline-graph SHA-256 hashes are unchanged before and after inspection. Checkpoint SHA-256: `ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15`.

No activation/routing analysis or performance evaluation was performed. Weight-only statistics cannot establish induction/copy roles or predict gains.
