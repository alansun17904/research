# Discovering Inductive Biases Through Sequence Growth

## Goal

Given an architecture, discover sequence families on which **one shared embedding/unembedding pair, trained on some members, predicts unseen members well under a fixed training budget**. Keep the internal network frozen at random initialization, then describe each validated family in natural language.

## Setup

- Let **S** be the current sequence set and **φ = (E, U)** the trainable embedding/unembedding parameters.
- Use a **decoder-only transformer for next-token prediction**: inputs are `x[:-1]`, targets are `x[1:]`, and causal self-attention lets each position attend only to itself and earlier positions. Score all **n − 1** target bits.
- Define budget **B**: training examples, optimization steps, optimizer, and trainable parameters. Each decoder block uses causal self-attention with fixed rotary position embeddings (RoPE) on queries and keys, followed by a feed-forward network. The RoPE frequency base is 10,000; no positional vectors are added to token embeddings. Only E/U are trainable.
- Reset parameters for every capability evaluation. Search compute is separate from the adaptation budget.



## Candidate score

Starting from **φ₀**, train on **S** within budget **B** to obtain **φₛ**. Score a candidate **z** by its transfer gain:

$$
T_S(z)=\ell(z;\phi_0)-\ell(z;\phi_S).
$$

Use mean loss over scored tokens. Positive gain means training on the current set helps predict the candidate. For one small SGD step, this approximates the gradient inner product:

$$
T_S(z)\approx\eta\nabla_\phi\ell(z;\phi_0)^\top\nabla_\phi L_S(\phi_0).
$$

Thus, seek **high gradient similarity**, or low gradient distance. A practical ranking score also rewards novelty and low final loss:

$$
J_S(z)=T_S(z)-\beta\ell(z;\phi_S)+\lambda\operatorname{Novelty}(z,S).
$$

Start with normalized edit distance for novelty and reject duplicates. This score proposes candidates; it does not establish an inductive bias.

## Search loop

1. **Seed:** Initialize **S** with a seed sequence and a small population of mutations.
2. **Adapt:** Reset **E/U** and train one shared pair on **S** within **B**.
3. **Propose:** Generate a bounded batch of distinct candidates. For fixed-length binary sequences, use single-bit flips, copying a span over another span, multiple-bit flips, and random sequences; all proposals retain length **n**.
4. **Rank:** Retain promising, distinct candidates using **J**. Include some low-transfer candidates: complementary examples can be jointly learnable without strong initial gradient alignment.
5. **Verify additions:** For shortlisted additions, reset and train on the enlarged set under the same budget. Accept additions when the group meets predefined performance thresholds; inspect per-sequence performance so averages cannot hide failures.
6. **Repeat:** Update the set and candidate population. Preserve alternative branches and restart from multiple seeds.

Begin with mutation and selection. If needed, add gradient-guided token replacements using the loss difference between the two fixed models; evaluate every proposal as a discrete sequence.

## Search / clustering pseudocode

**Grow groups of sequences that one shared embedding/unembedding pair can learn together under a fixed budget.** Training on the current group suggests which sequences to add; retraining on the enlarged group checks that every member remains predictable. Each surviving group is a candidate family. Families can overlap and need not cover the sequence space.

The pseudocode below follows the current implementation in `inductive_bias/pipeline.py`. Sequences are binary strings of length **n = 16**; loss and accuracy score the **n − 1** next-bit predictions. The random transformer core and RoPE frequencies stay fixed throughout discovery, including across restarts. The **beam** is the small collection of alternative groups retained between rounds.

```text
Initialize frozen random decoder-only core θ with Q/K RoPE and initial E/U φ₀ once
families ← []; discovery_data ← ∅

FIT(S):
    Reset E/U to φ₀, and reset the optimizer and minibatch random seed
    Train ONE shared E/U pair on S for B.steps SGD steps
        Each step samples B.batch_size sequences uniformly with replacement
        Only E/U receive updates; θ and RoPE frequencies stay frozen
    Return the trained pair φ

PASSES(S, φ):
    Return true iff EVERY s in S has loss(s; φ) ≤ max_loss
                                    and accuracy(s; φ) ≥ min_accuracy

For each restart:
    S ← {seed sequence} ∪ small batch of distinct mutations
    discovery_data ← discovery_data ∪ S
    φ ← FIT(S)
    If not PASSES(S, φ): continue to the next restart
    beam ← [(S, empty proposal pool, metrics(S; φ))]

    For each search round:
        expanded ← []
        For each (S, proposal_pool, group_metrics) in beam:
            φ_S ← FIT(S)
            C ← bounded batch of proposals from S and proposal_pool
                Reuse promising prior proposals; fill with mutations/random draws
                Exclude members of S and duplicates within C
            discovery_data ← discovery_data ∪ C

            For each z in C:
                transfer(z) ← loss(z; φ₀) − loss(z; φ_S)
                novelty(z) ← min over s in S of edit_distance(z, s) / n
                J(z) ← transfer(z) − β · loss(z; φ_S) + λ · novelty(z)

            next_pool ← up to floor(proposal_count / 2) highest-J candidates in C
            Add (S, next_pool, group_metrics) to expanded
            shortlist ← highest-J candidates, reserving some slots for
                        lowest-transfer candidates among the remaining proposals

            For each z in shortlist:
                S_plus ← S ∪ {z}
                φ_plus ← FIT(S_plus)              # Same budget, reset from φ₀
                If PASSES(S_plus, φ_plus):
                    Add (S_plus, next_pool, metrics(S_plus; φ_plus)) to expanded
            Record proposal scores and each addition's verification result

        Deduplicate expanded branches by member set, keeping the lowest-loss copy
        beam ← best beam_width branches, preferring larger groups,
               then lower mean loss on their members

    Append the groups in beam to families

Return families ranked by size then mean loss, plus discovery_data and search trace
```

Keeping the parent group in `expanded` lets search continue after an unsuccessful proposal batch. Each shortlisted addition is verified separately against the same parent group; the beam preserves alternative accepted additions. Proposal pools may retain rejected candidates for later rounds.

Current defaults are **3 restarts**, **4 rounds**, up to **24 proposals per branch**, a shortlist of **3 highest-score candidates plus 1 lowest-transfer candidate**, and **2 retained branches**. Each restart begins with a seed and one mutation. Ranking uses **β = 0.25** and **λ = 0.05**. Every fit uses **100 SGD steps × 16 sampled sequences**, with learning rate **0.2**; every member must achieve loss **≤ 0.65 nats per target bit** and accuracy **≥ 0.60**. The transformer has one layer, two attention heads, width 16, and feed-forward width 32; only the 64 E/U parameters are trainable.

These groups establish shared learnability on discovery examples. To test the intended generalization claim, freeze a family rule and draw fresh support and test sequences **excluding all of** `discovery_data`**, including rejected proposals**. Train a fresh shared E/U pair on support under the same budget, evaluate every test member, and repeat across fresh random cores with predictor controls, as described below.

## Explain and validate

The grown set is a **candidate family**. All adaptively selected examples count as discovery data.

1. Give an LLM representative members, failures, and nearby rejected examples.
2. Request competing explanations with executable membership rules or generators, plus examples that distinguish them.
3. Freeze a proposed rule or generator and draw fresh support and test examples. Do not filter test examples by model success.
4. Repeat bounded adaptation and evaluation across fresh random-core initializations. Compare against simple predictors and alternative cores under matched conditions.



## Output and limits

For each validated family, report its description, generator or membership rule, training budget, held-out performance, variability across seeds, and failure cases.

This is a proposed heuristic search. It avoids exhaustive enumeration, but may miss disconnected families and need not find a unique organization. Conclusions depend on the proposal process, initialization, tokenization, and training protocol as well as the architecture.
