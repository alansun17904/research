# 090626

- **Claim 1:** For vanilla linear attention and its variants, there doesn't exist a way to implement a `noop`.

- **Claim 2:** Linear attention is a type of KV-cache rentention / eviction policy. 

- **Claim 2a:** Viewed in this lens, just as KV-cache eviction policies are compatible with any softmax Transformer so should linear attention. Specifically, we *should* be able to zero-shot transform any softmax Transformer into a linear one.

## Questions
1. What if just like streaming LLM, when we linear softmax attention, we retain the first token? So, we maintain two states [the <bos> token and the outer product state from linear attention]. 

    a. In this case, 
