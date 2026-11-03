# Adapted from Hugging Face Transformers' GPT-NeoX implementation (Apache-2.0).
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. See reference/LICENSE.
"""Editable, full-sequence Pythia 70M inference; JSON semantics in AGENT.md."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from evaluate import MODEL, MODEL_REVISION

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "assets" / "pythia-70m"
WEIGHTS = {"embedding": "gpt_neox.embed_in", "final_norm": "gpt_neox.final_layer_norm",
           "output": "embed_out"}


def fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"Expected exactly these fields: {', '.join(names)}")


def count(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected an integer >= {minimum}, got {value!r}")


def validate_graph(graph):
    fields(graph, ("format", "checkpoint", "weights", "layers", "sequence"))
    if graph["format"] != "pythia-70m-v1":
        raise ValueError("Unsupported graph format")
    if graph["checkpoint"] != {"model": MODEL, "revision": MODEL_REVISION}:
        raise ValueError("This implementation requires the pinned Pythia 70M checkpoint")
    if graph["weights"] != WEIGHTS:
        raise ValueError("Unsupported embedding, final norm, or output weight identity")
    if type(graph["layers"]) is not list or len(graph["layers"]) != 6:
        raise ValueError("Expected six original layer definitions")
    for index, layer in enumerate(graph["layers"]):
        fields(layer, ("weights", "heads", "mlp_repeat"))
        if layer["weights"] != f"gpt_neox.layers.{index}":
            raise ValueError("Layer definitions must name their original checkpoint tensors")
        if type(layer["heads"]) is not list or len(layer["heads"]) != 8:
            raise ValueError("Each layer needs eight head repetition counts")
        for repetitions in layer["heads"]:
            count(repetitions, 0)
        count(layer["mlp_repeat"], 1)
    if type(graph["sequence"]) is not list or not graph["sequence"]:
        raise ValueError("Expected a nonempty sequence of layer or head blocks")
    for block in graph["sequence"]:
        if type(block) is not dict:
            raise ValueError("Expected a layer or head block")
        kind = "layers" if "layers" in block else "heads"
        fields(block, (kind, "repeat"))
        count(block["repeat"], 1)
        if type(block[kind]) is not list or not block[kind]:
            raise ValueError("A block must list at least one layer or head")
        for reference in block[kind]:
            if kind == "heads":
                if type(reference) is not list or len(reference) != 2:
                    raise ValueError("Head references must be [layer_index, head_index]")
                index, head = reference
                count(head, 0)
                if head >= 8:
                    raise ValueError("Head index must be in 0..7")
            else:
                index = reference
            count(index, 0)
            if index >= 6:
                raise ValueError("Layer index must be in 0..5")
    return graph


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_graph(path):
    return validate_graph(json.loads(Path(path).read_text(),
                                    object_pairs_hook=unique_fields,
                                    parse_constant=invalid_constant))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.heads = config.num_attention_heads
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Keep all original checkpoint buffers so strict loading accounts for every tensor.
        self.register_buffer("bias", torch.ones(config.max_position_embeddings,
                             config.max_position_embeddings, dtype=torch.bool).tril()[None, None])
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = nn.Module()
        rotary_dims = int(self.head_size * config.rotary_pct)
        self.rotary_emb.register_buffer("inv_freq", 1.0 / (
            config.rotary_emb_base ** (torch.arange(0, rotary_dims, 2).float() / rotary_dims)))
        # The old checkpoint rounds this buffer to FP16; Transformers regenerates it in FP32.
        self.register_buffer("_inv_freq", self.rotary_emb.inv_freq.clone(), persistent=False)

    def attend(self, qkv):
        query, key, value = qkv.chunk(3, dim=-1)
        positions = torch.arange(qkv.size(2), device=qkv.device, dtype=torch.float32)
        angles = positions[:, None] * self._inv_freq.float()[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cosine, sine = angles.cos().to(qkv.dtype), angles.sin().to(qkv.dtype)
        rotary_dims = cosine.size(-1)

        def rotate(x):
            rotary, rest = x[..., :rotary_dims], x[..., rotary_dims:]
            first, second = rotary.chunk(2, dim=-1)
            rotated = rotary * cosine + torch.cat((-second, first), dim=-1) * sine
            return torch.cat((rotated, rest), dim=-1)

        query, key = rotate(query), rotate(key)
        scores = (query @ key.transpose(-1, -2)) * self.head_size ** -0.5
        scores = scores.masked_fill(~self.bias[:, :, :qkv.size(2), :qkv.size(2)],
                                   torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return probabilities @ value

    def forward(self, normalized, repetitions):
        batch, length, _ = normalized.shape
        qkv = self.query_key_value(normalized).view(batch, length, self.heads,
                                                   3 * self.head_size).transpose(1, 2)
        heads = self.attend(qkv)
        active = heads.new_tensor([n > 0 for n in repetitions])[None, :, None, None]
        merged = (heads * active).transpose(1, 2).reshape(batch, length, -1)
        return self.dense(merged)

    def head(self, normalized, index):
        start = index * 3 * self.head_size
        qkv = F.linear(normalized, self.query_key_value.weight[start:start + 3 * self.head_size],
                       self.query_key_value.bias[start:start + 3 * self.head_size])
        values = self.attend(qkv[:, None]).squeeze(1)
        start = index * self.head_size
        return F.linear(values, self.dense.weight[:, start:start + self.head_size])


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(hidden)))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, hidden, specification):
        attention = self.attention(self.input_layernorm(hidden), specification["heads"])
        for head, repetitions in enumerate(specification["heads"]):
            for _ in range(max(0, repetitions - 1)):
                attention = attention + self.attention.head(
                    self.input_layernorm(hidden + attention), head)
        mlp = self.mlp(self.post_attention_layernorm(hidden))
        for _ in range(specification["mlp_repeat"] - 1):
            mlp = mlp + self.mlp(self.post_attention_layernorm(hidden + mlp))
        return mlp + attention + hidden


class Pythia(nn.Module):
    def __init__(self, config, graph):
        super().__init__()
        self.config = config
        self.graph = copy.deepcopy(validate_graph(graph))
        self.gpt_neox = nn.Module()
        self.gpt_neox.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gpt_neox.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.gpt_neox.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def device(self):
        return self.gpt_neox.embed_in.weight.device

    def forward(self, input_ids, labels=None, use_cache=False):
        if use_cache:
            raise ValueError("This full-sequence evaluator does not implement KV caching")
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.config.max_position_embeddings:
            raise ValueError("Expected unpadded [batch, tokens] within the model context limit")
        hidden = self.gpt_neox.embed_in(input_ids)
        for block in self.graph["sequence"]:
            for _ in range(block["repeat"]):
                if "layers" in block:
                    for index in block["layers"]:
                        hidden = self.gpt_neox.layers[index](hidden, self.graph["layers"][index])
                else:
                    for index, head in block["heads"]:
                        if self.graph["layers"][index]["heads"][head] > 0:
                            layer = self.gpt_neox.layers[index]
                            hidden = hidden + layer.attention.head(layer.input_layernorm(hidden), head)
        logits = self.embed_out(self.gpt_neox.final_layer_norm(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or input_ids.size(1) < 2:
                raise ValueError("Labels must match input_ids and include a next-token target")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def load_model(graph_path, checkpoint=CHECKPOINT):
    graph = read_graph(graph_path)
    checkpoint = Path(checkpoint)
    raw = json.loads((checkpoint / "config.json").read_text())
    expected = {"model_type": "gpt_neox", "hidden_size": 512, "intermediate_size": 2048,
                "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 50304,
                "max_position_embeddings": 2048, "rotary_pct": 0.25, "rotary_emb_base": 10000,
                "hidden_act": "gelu", "layer_norm_eps": 1e-5, "tie_word_embeddings": False,
                "use_parallel_residual": True}
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported checkpoint configuration; expected original Pythia 70M")
    if raw.get("hidden_dropout", 0) != 0 or raw.get("attention_dropout", 0) != 0 or not raw.get("attention_bias", True):
        raise ValueError("Expected zero dropout and biased attention projections")
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    if {key: provenance.get(key) for key in ("model", "revision")} != graph["checkpoint"]:
        raise ValueError("Checkpoint provenance does not match graph")
    for name in ("config.json", "model.safetensors"):
        with (checkpoint / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != provenance["files"][name]["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    model = Pythia(SimpleNamespace(**raw), graph)
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    return model.eval().requires_grad_(False)
