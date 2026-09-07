"""Pretrained GPT-Neo/GPT-NeoX token loops; only the attention rule changes."""

from dataclasses import replace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import Attention, AttentionConfig

DEFAULT_MODEL = "roneneldan/TinyStories-1Layer-21M"


def rotate(x, positions, rotary_dim, base):
    """Vanilla GPT-NeoX RoPE, including its unrotated suffix; no position table."""
    frequencies = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim))
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    rotated, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, rest), dim=-1)


class AttentionModelRunner:
    """step([batch]) -> [batch, vocab]; reset() between independent sequences."""

    def __init__(self, pretrained_model, attention_config=None):
        self.hf_model = pretrained_model.eval()
        self.config = pretrained_model.config
        self.device, self.dtype = pretrained_model.device, pretrained_model.dtype
        self.max_sequence_length = self.config.max_position_embeddings if self.config.model_type == "gpt_neo" else None
        if self.config.model_type not in {"gpt_neo", "gpt_neox"}:
            raise ValueError("Supported architectures: GPT-Neo (TinyStories) and GPT-NeoX (Pythia)")
        if getattr(self.config, "rope_scaling", None):
            raise ValueError("Only vanilla RoPE is supported; rope_scaling must be unset")
        self.set_attention(attention_config or AttentionConfig())

    @property
    def state_bytes(self):
        return sum(policy.state_bytes for policy in self.policies)

    @property
    def cached_tokens(self):
        """Maximum number of explicit KV pairs per head across layers."""
        return max(policy.cached_tokens for policy in self.policies)

    def set_attention(self, config):
        self.attention_config = config
        if self.config.model_type == "gpt_neo":
            config = replace(config, score_scale=1.0)  # GPT-Neo does NOT divide QK by sqrt(d).
            kinds = self.config.attention_layers
        else:
            kinds = ["global"] * self.config.num_hidden_layers
        self.policies = []
        for kind in kinds:
            layer_config = config
            if kind == "local":
                if config.method != "full":
                    raise ValueError(f"Replacing local attention layers changes the experiment. Use {DEFAULT_MODEL}.")
                layer_config = replace(config, method="window", cache_size=self.config.window_size - 1)
            self.policies.append(Attention(layer_config))
        self.reset()

    def reset(self):
        self.tokens_seen = 0
        for policy in self.policies:
            policy.reset()

    @torch.inference_mode()
    def step(self, token_ids):
        token_ids = token_ids.to(self.device).reshape(-1)
        if self.max_sequence_length is not None and self.tokens_seen >= self.max_sequence_length:
            raise ValueError(
                f"Learned position limit ({self.max_sequence_length}) reached; reset or use GPT-NeoX."
            )
        if self.config.model_type == "gpt_neo":
            logits = self._neo_step(token_ids)
        else:
            logits = self._neox_step(token_ids)
        self.tokens_seen += 1
        return logits

    def _neo_step(self, ids):
        transformer = self.hf_model.transformer
        positions = torch.full_like(ids, self.tokens_seen)
        hidden = transformer.wte(ids) + transformer.wpe(positions)
        for block, policy in zip(transformer.h, self.policies):
            attn = block.attn.attention
            normed = block.ln_1(hidden)
            shape = (ids.shape[0], self.config.num_heads, -1)
            q = attn.q_proj(normed).reshape(shape)
            k = attn.k_proj(normed).reshape(shape)
            v = attn.v_proj(normed).reshape(shape)
            output = policy.step(q, k, v).reshape_as(hidden)
            hidden = hidden + attn.out_proj(output)
            hidden = hidden + block.mlp(block.ln_2(hidden))
        return self.hf_model.lm_head(transformer.ln_f(hidden))

    def _neox_step(self, ids):
        transformer = self.hf_model.gpt_neox
        hidden = transformer.embed_in(ids)
        heads = self.config.num_attention_heads
        dim = self.config.hidden_size // heads
        rotary_dim = int(dim * self.config.rotary_pct)
        base = self.config.rotary_emb_base

        def compact_rope(query, keys):
            positions = torch.arange(keys.shape[-2], device=keys.device)
            return (rotate(query, positions[-1], rotary_dim, base),
                    rotate(keys, positions, rotary_dim, base))

        for block, policy in zip(transformer.layers, self.policies):
            qkv = block.attention.query_key_value(block.input_layernorm(hidden))
            q, k, v = qkv.reshape(ids.shape[0], heads, 3 * dim).split(dim, dim=-1)
            if self.attention_config.method == "streaming":
                output = policy.step(q, k, v, score_transform=compact_rope)
            else:
                position = torch.tensor(self.tokens_seen, device=ids.device)
                q, k = rotate(q, position, rotary_dim, base), rotate(k, position, rotary_dim, base)
                output = policy.step(q, k, v)
            output = block.attention.dense(output.reshape_as(hidden))
            mlp_input = hidden if self.config.use_parallel_residual else hidden + output
            hidden = hidden + output + block.mlp(block.post_attention_layernorm(mlp_input))
        return self.hf_model.embed_out(transformer.final_layer_norm(hidden))

    @torch.inference_mode()
    def prefill(self, input_ids):
        """Consume [batch, sequence] tokens and return only the final [batch, vocab] logits."""
        for index in range(input_ids.shape[1]):
            logits = self.step(input_ids[:, index])
        return logits


def load_model(
    model_name=DEFAULT_MODEL, attention_config=None, device="auto", dtype="float32",
    cache_dir=None, local_files_only=False, revision="main",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs = dict(cache_dir=cache_dir, local_files_only=local_files_only, revision=revision, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), attn_implementation="eager", **kwargs
    ).to(device)
    return AttentionModelRunner(model, attention_config), tokenizer
