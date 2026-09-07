# 090726



# 090626

- **Claim 1:** For vanilla linear attention and its variants, there doesn't exist a way to implement a `noop`.
- **Claim 2:** Linear attention is a type of KV-cache rentention / eviction policy. 
- **Claim 2a:** Viewed in this lens, just as KV-cache eviction policies are compatible with any softmax Transformer so should linear attention. Specifically, we *should* be able to zero-shot transform any softmax Transformer into a linear one.

What if just like streaming LLM, when we linear softmax attention, we retain the first token? So, we maintain two states [the <bos> token and the outer product state from linear attention]. Say that (k0, v0) is the <bos> token and St is KVB, then recurrence becomes:

    > St+1 = St + vtkt^T
    > zt+1 = zt + kt^Tqt
    > ot+1 = 1 / ( (exp(k0^Tqt) + exp(zt+1)) ) * 
    >         (v0 exp(k0^Tqt) + (1 / zt+1) * St+1kt (exp(zt+1)))

**Streaming LLM**: retain the first four tokens + sliding window
**H2O**: 