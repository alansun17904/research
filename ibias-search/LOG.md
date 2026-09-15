# 091526

I was reading Albert's blog today and one idea he discussed stuck with me: (1) in sublinear models, compression is an inductive bias; (2) when normalized by bytes-per-bit using SSMs as the "main" modules of H-net is significantly more data-efficient than using quadratic transformers. This implies that maybe SSMs or sublinear modules are perhaps doing something that other models cannot do...and that maybe COMPRESSION IS FUNDAMENTAL TO INTELLIGENCE.

One nice thing about Transformers is that they can essentially use compute (during training) optimally [that is, maximize GPU utilization]. For recurrent models, we need to work very hard to be compute-bound. I wonder if this if it is possible to verify (1) [above] that compression is an inductive bias of linear models [and conversely, this inductive bias is not present in quadratic models]. 

If this latter claim is *not* true, then this would mean that any transformer model that has been pretrained can be made to compress its context in a way that matches the efficiency of a sublinear model. 

Unrelated: For a given architecture, I wonder if there are concrete scaling recipes. 

# 091426

Throughout, we fix an initialization and only train the embedding/unembedding (E/U) [denote this as ϕ]. Then, we proceed with the following discovery algorithm:

0. We start with a seed set S [then repeat the following steps for N FAMILIES]
1. ϕ' <- FIT(S, ϕ)
2. C <- C ∪ { mutations of S }
3. shortlist <- {J(z) : z ∈ C, J(z) ≥ α}
4. for each element in the shortlist: if FIT(S ∪ {z}, ϕ') is still good then add z to S

EACH FAMILY DETERMINES A SUBTASK
WITHIN EACH FAMILY, EACH ROUND ADDS SOME EXAMPLES TO THIS SUBTASK

We can probably expand the complexity / efficiency of the search space by using language models to generate primitives. Concretely, the pipeline would be like 

0. Repeat the following steps for N primitives:
1. LM looks at proposed primitives and prior performance [proposes a new primitive, which includes both training and evaluation sets]
2. Train a model on the training set, evaluate and summarize results
3. If evaluation is good enough, append to set of primitives; otherwise append to set of rejected primitives

* This occurred to me when I was having dinner w/ Terry today: we want **architecture-intrinsic primitives**. That is, we want to find cells that persist across size and different scaling dimensions [for example, if I increase the width / depth / embedding dimension in fixed ratios]. 

# 091026

## Weekly Summary

1. Read MAML, iMAML, E2E, Test-time regression, mRNN, In-Place TTT, TTT-NTP, Decoupled Neural Interfaces
2. Went through E2E codebase

**More ideas:**

1. Do different variants of linear attention and softmax attention transformers have the same random capabilities? What are the right primitives
2. How does linear attention learn an induction head? Or more generally, what sort of "sequence primitives" does subquadratic attention learn? I wonder if we could construct a feedback loop like:
  1. We first assume that any seq-to-seq architecture operates on an "embedding space." That is, there are embedding and unembedding maps (Embed: v ∈ Σ → e ∈ R^d; Unembed: e ∈ R^d → v ∈ V).
  2. For a given neural architecture [which through parameterization gives a family of neural networks] A, we say that a partition of Σ* are the inductive biases of A if it satisfies:
    > I ∈ max_{i ∈ Σ*} Σ_{k ∈ i} E_θ L(A_{k, θ}*, k)
    >
    > where A_{k, θ}* is a neural network with architecture A parameterized with θ (fixed) and whose embedding and unembeddings have been trained to predict sequences defined by the task k. E_θ is the expectation taken over the random initialization θ.
  3. We then use a language model to provide natural language descriptions of k ∈ I.
3. Hybrid lookahead module with much smaller embedding dimension so that it can fit into HBM and then do speculative decoding to determine what to append to the linear attention.



# 090826

Attention sink detection over the entire model:

```
> Let Aij be the attention matrix where i->j, then sink_k = 1 / L Σ 1 / H Σ 1 / (T - k + 1) Σ_{i=k}^T A_ik
```

essentially we are measuring the average attention given to the token at position k by tokens at position j ≥ k.

1. We perform "test-time distillation" of softmax attention into linear attention. Specifically, denote my pre-trained model as `M` such that `xt = M(x<t)`. At inference time, we construct a model `L` [that warm starts from `M`, but instead of softmax attention, we use linear attention (this can either be done by directly removing softmax or using some more clever Delta rule / gating)]. Then, for a schedule of decay parameters `{αt}`, we predict the next token by

```
   > xt+1 = αtM(x<t) + (1 - αt)L(x<t) 
   > and then we learn to minimize LL(L(x<t), xt+1)
   > where LL is the CE loss.
```

All of the ideas also follow here, where we can do online, batched updates to L's parameters. As t ≫ 1, M's computation becomes more costly. Further, we yield more training data for L. Once t > τ (for some threshold τ), we then rapidly decay αt -> 0.

1. Construct a "memory pyramid" that is comprised of L levels. Each level has a resolution of r (embedding / token). Suppose that there are T tokens (either in the context or what we have already generated). At level 1 <= l <= L in the memory pyramid, we store N(l, T) embeddings [where for any fixed T, N(l, T) is a non-increasing function of l]. Each embedding has dimension D(l) where D(l) is a non-decreasing function of l.

Each layer of the memory pyramid interacts with the layer below it in the following way:

```
> At level 0, we append a KV using softmax attention over the previous KVs
> Given level l-1, at level l and at time T 
>   if N(l, T) > size(of KV-cache), then append to KV-cache an up-projection of the last token in level l-1.
>   else modify the last element of the current level's KV-cache using linear attention
```

After some more thought, I guess this would be equivalent to progressively applying sliding window attention across window sizes that are non-decreasing and on embeddings whose dimensions are non-increasing.

# 090626

- **Claim 1:** For vanilla linear attention and its variants, there doesn't exist a way to implement a `noop`.
- **Claim 2:** Linear attention is a type of KV-cache rentention / eviction policy.
- **Claim 2a:** Viewed in this lens, just as KV-cache eviction policies are compatible with any softmax Transformer so should linear attention. Specifically, we *should* be able to zero-shot transform any softmax Transformer into a linear one.

What if just like streaming LLM, when we linear softmax attention, we retain the first token? So, we maintain two states [the  token and the outer product state from linear attention]. Say that (k0, v0) is the sink token that we keep and St is KVB, then recurrence becomes:

```
> St+1 = St + vtkt^T
> zt+1 = zt + kt
> ot+1 = 1 / ( (exp(k0^Tqt) + exp(zt+1qt)) ) *
>         (v0 exp(k0^Tqt) + (1 / zt+1qt) * St+1kt (exp(zt+1qt)))
```

**Streaming LLM**: retain the first four tokens + sliding window  
**H2O**: sliding window + for each token in the cache, compute a score that is the sum of attention applied to this token, evict tokens that have smallest score [no normalization, there is a bias towards tokens that have been around longer]