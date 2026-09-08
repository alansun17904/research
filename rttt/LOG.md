# 090826

Attention sink detection over the entire model:

```
> Let Aij be the attention matrix where i->j, then sink_k = 1 / L Σ 1 / H Σ 1 / (T - k + 1) Σ_{i=k}^T A_ik
```

essentially we are measuring the average attention given to the token at position k by tokens at position j ≥ k.



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