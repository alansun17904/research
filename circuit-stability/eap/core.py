from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utils import get_attention_mask


QKV = Optional[Literal["q", "k", "v"]]
TokenAggregation = Literal["all-tok", "last-tok", "avg-tok"]


@dataclass(frozen=True)
class NodeSpec:
    name: str
    kind: str
    layer: int
    out_hook: str
    in_hook: str
    head: Optional[int] = None
    kv_head: Optional[int] = None
    qkv_inputs: Optional[tuple[str, str, str]] = None


@dataclass
class EdgeSpec:
    name: str
    parent: int
    child: int
    qkv: QKV
    score: Optional[float] = None


class Graph:
    def __init__(self) -> None:
        self.cfg: Dict[str, int] = {}
        self.nodes: dict[str, NodeSpec] = {}
        self.node_list: list[NodeSpec] = []
        self.node_index: dict[str, int] = {}
        self.edges: dict[str, EdgeSpec] = {}
        self.edge_list: list[EdgeSpec] = []
        self.hooked_node_indices: tuple[int, ...] = ()
        self.backward_specs: tuple[tuple[str, int, slice | int, bool], ...] = ()
        self.edge_rows: np.ndarray = np.empty(0, dtype=np.int64)
        self.edge_cols: np.ndarray = np.empty(0, dtype=np.int64)
        self.n_forward = 0
        self.n_backward = 0

    @classmethod
    def from_model(
        cls,
        model_or_config: Union[
            HookedTransformer, HookedTransformerConfig, Dict[str, int]
        ],
    ) -> "Graph":
        graph = cls()
        graph.cfg = cls._normalize_cfg(model_or_config)
        graph._build()
        return graph

    @staticmethod
    def _normalize_cfg(
        model_or_config: Union[
            HookedTransformer, HookedTransformerConfig, Dict[str, int]
        ]
    ) -> Dict[str, int]:
        if isinstance(model_or_config, dict):
            cfg = dict(model_or_config)
        else:
            cfg_obj = getattr(model_or_config, "cfg", model_or_config)
            n_kv_heads = getattr(cfg_obj, "n_key_value_heads", None) or cfg_obj.n_heads
            cfg = {
                "n_layers": cfg_obj.n_layers,
                "n_heads": cfg_obj.n_heads,
                "parallel_attn_mlp": bool(cfg_obj.parallel_attn_mlp),
                "n_kv_heads": n_kv_heads,
            }
        return cfg

    def _add_node(self, node: NodeSpec) -> int:
        index = len(self.node_list)
        self.node_list.append(node)
        self.node_index[node.name] = index
        self.nodes[node.name] = node
        return index

    def _add_edge(self, parent_index: int, child_index: int, qkv: QKV = None) -> None:
        parent = self.node_list[parent_index]
        child = self.node_list[child_index]
        if child.kind == "attn" and qkv is None:
            raise ValueError("Edges into attention nodes must specify q, k, or v.")
        name = (
            f"{parent.name}->{child.name}"
            if qkv is None
            else f"{parent.name}->{child.name}<{qkv}>"
        )
        edge = EdgeSpec(name=name, parent=parent_index, child=child_index, qkv=qkv)
        self.edge_list.append(edge)
        self.edges[name] = edge

    def _build(self) -> None:
        input_index = self._add_node(
            NodeSpec(
                name="input",
                kind="input",
                layer=0,
                in_hook="",
                out_hook="hook_embed",
            )
        )
        residual_stream = [input_index]
        kv_group = self.cfg["n_heads"] // self.cfg["n_kv_heads"]

        for layer in range(self.cfg["n_layers"]):
            attn_indices = []
            for head in range(self.cfg["n_heads"]):
                attn_indices.append(
                    self._add_node(
                        NodeSpec(
                            name=f"a{layer}.h{head}",
                            kind="attn",
                            layer=layer,
                            in_hook=f"blocks.{layer}.hook_attn_in",
                            out_hook=f"blocks.{layer}.attn.hook_result",
                            head=head,
                            kv_head=head // kv_group,
                            qkv_inputs=tuple(
                                f"blocks.{layer}.hook_{letter}_input"
                                for letter in "qkv"
                            ),
                        )
                    )
                )

            mlp_index = self._add_node(
                NodeSpec(
                    name=f"m{layer}",
                    kind="mlp",
                    layer=layer,
                    in_hook=f"blocks.{layer}.hook_mlp_in",
                    out_hook=f"blocks.{layer}.hook_mlp_out",
                )
            )

            for source_index in residual_stream:
                for attn_index in attn_indices:
                    for letter in "qkv":
                        self._add_edge(source_index, attn_index, qkv=letter)

            if self.cfg["parallel_attn_mlp"]:
                for source_index in residual_stream:
                    self._add_edge(source_index, mlp_index)
                residual_stream.extend(attn_indices)
                residual_stream.append(mlp_index)
            else:
                residual_stream.extend(attn_indices)
                for source_index in residual_stream:
                    self._add_edge(source_index, mlp_index)
                residual_stream.append(mlp_index)

        logits_index = self._add_node(
            NodeSpec(
                name="logits",
                kind="logits",
                layer=self.cfg["n_layers"] - 1,
                in_hook=f"blocks.{self.cfg['n_layers'] - 1}.hook_resid_post",
                out_hook="",
            )
        )
        for source_index in residual_stream:
            self._add_edge(source_index, logits_index)

        self.n_forward = 1 + self.cfg["n_layers"] * (self.cfg["n_heads"] + 1)
        per_layer_backward = self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"] + 1
        self.n_backward = self.cfg["n_layers"] * per_layer_backward + 1
        self._finalize_tables()

    def _finalize_tables(self) -> None:
        hooked = [self.node_index["input"]]
        for layer in range(self.cfg["n_layers"]):
            for head in range(self.cfg["n_heads"]):
                hooked.append(self.node_index[f"a{layer}.h{head}"])
            hooked.append(self.node_index[f"m{layer}"])
        self.hooked_node_indices = tuple(hooked)

        backward_specs = []
        for layer in range(self.cfg["n_layers"]):
            attn = self.nodes[f"a{layer}.h0"]
            prev_attn = self.prev_index(attn)
            for letter_index, letter in enumerate("qkv"):
                backward_specs.append(
                    (
                        attn.qkv_inputs[letter_index],
                        prev_attn,
                        self.backward_index(attn, qkv=letter, attn_slice=True),
                        True,
                    )
                )
            mlp = self.nodes[f"m{layer}"]
            backward_specs.append(
                (
                    mlp.in_hook,
                    self.prev_index(mlp),
                    self.backward_index(mlp),
                    False,
                )
            )
        logits = self.nodes["logits"]
        backward_specs.append(
            (
                logits.in_hook,
                self.prev_index(logits),
                self.backward_index(logits),
                False,
            )
        )
        self.backward_specs = tuple(backward_specs)

        self.edge_rows = np.fromiter(
            (
                self.forward_index(self.node_list[edge.parent], attn_slice=False)
                for edge in self.edge_list
            ),
            dtype=np.int64,
            count=len(self.edge_list),
        )
        self.edge_cols = np.fromiter(
            (
                self.backward_index(
                    self.node_list[edge.child], qkv=edge.qkv, attn_slice=False
                )
                for edge in self.edge_list
            ),
            dtype=np.int64,
            count=len(self.edge_list),
        )

    def forward_index(self, node: NodeSpec, attn_slice: bool = True):
        if node.kind == "input":
            return 0
        if node.kind == "logits":
            raise ValueError("Logits does not have a forward index.")
        if node.kind == "mlp":
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        start = 1 + node.layer * (self.cfg["n_heads"] + 1)
        return (
            slice(start, start + self.cfg["n_heads"])
            if attn_slice
            else start + int(node.head)
        )

    def prev_index(self, node: NodeSpec) -> int:
        if node.kind == "input":
            return 0
        if node.kind == "logits":
            return self.n_forward
        start = 1 + node.layer * (self.cfg["n_heads"] + 1)
        if node.kind == "attn" or self.cfg["parallel_attn_mlp"]:
            return start
        return start + self.cfg["n_heads"]

    def backward_index(self, node: NodeSpec, qkv: QKV = None, attn_slice: bool = True):
        if node.kind == "input":
            raise ValueError("Input does not have a backward index.")
        if node.kind == "logits":
            return self.n_backward - 1

        per_layer = self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"]
        layer_offset = node.layer * per_layer
        if node.kind == "mlp":
            return layer_offset + self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"]

        if qkv not in {"q", "k", "v"}:
            raise ValueError("Attention backward indices require qkv.")

        if qkv == "q":
            start = layer_offset
            width = self.cfg["n_heads"]
            offset = int(node.head)
        elif qkv == "k":
            start = layer_offset + self.cfg["n_heads"]
            width = self.cfg["n_kv_heads"]
            offset = int(node.kv_head)
        else:
            start = layer_offset + self.cfg["n_heads"] + self.cfg["n_kv_heads"]
            width = self.cfg["n_kv_heads"]
            offset = int(node.kv_head)

        return slice(start, start + width) if attn_slice else start + offset

    def edge_vector(self) -> np.ndarray:
        output = np.empty(len(self.edge_list), dtype=np.float32)
        for i, edge in enumerate(self.edge_list):
            output[i] = 0.0 if edge.score is None else float(edge.score)
        return output

    def get_scores(self, nonzero: bool = False, sort: bool = True) -> torch.Tensor:
        values = [float(edge.score or 0.0) for edge in self.edge_list]
        if nonzero:
            values = [value for value in values if value != 0.0]
        scores = torch.tensor(values)
        return torch.sort(scores).values if sort and len(scores) else scores


def tokenize_plus(model: HookedTransformer, inputs: List[str]):
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side="left")
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos


def _model_device(model: HookedTransformer) -> torch.device:
    return torch.device(str(model.cfg.device))


def _hooked_nodes(graph: Graph) -> list[NodeSpec]:
    return [graph.node_list[index] for index in graph.hooked_node_indices]


def _iter_source_ranges(max_forward_index: int, source_chunk_size: Optional[int]):
    effective_source_chunk = (
        max_forward_index if source_chunk_size is None else source_chunk_size
    )
    if effective_source_chunk <= 0:
        raise ValueError("source_chunk_size must be positive when provided.")
    for source_start in range(0, max_forward_index, effective_source_chunk):
        source_end = min(source_start + effective_source_chunk, max_forward_index)
        yield source_start, source_end


def _chunked_backward_specs(
    graph: Graph,
    chunk_mode: Literal["none", "fixed", "layer"],
    chunk_size: Optional[int],
) -> list[tuple[tuple[str, int, slice | int, bool], ...]]:
    specs = list(graph.backward_specs)
    if chunk_mode == "none":
        return [tuple(specs)]
    if chunk_mode == "layer":
        chunks = []
        offset = 0
        for _layer in range(graph.cfg["n_layers"]):
            chunks.append(tuple(specs[offset : offset + 4]))
            offset += 4
        chunks.append(tuple(specs[offset:]))
        return [chunk for chunk in chunks if chunk]
    if chunk_mode == "fixed":
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be positive when chunk_mode='fixed'.")
        return [
            tuple(specs[start : start + chunk_size])
            for start in range(0, len(specs), chunk_size)
        ]
    raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")


def _aggregate_tokens(values: Tensor, token_aggregation: TokenAggregation) -> Tensor:
    if token_aggregation == "all-tok":
        return values
    if token_aggregation == "last-tok":
        return values[-1]
    if token_aggregation == "avg-tok":
        return values.mean(dim=0)
    raise ValueError(f"Unsupported token_aggregation: {token_aggregation}")


@contextmanager
def _frozen_parameters(model: HookedTransformer):
    params = list(model.parameters())
    original = [param.requires_grad for param in params]
    for param in params:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in zip(params, original):
            param.requires_grad_(requires_grad)


def _forward_capture_hook(
    activation_difference: Tensor,
    node: NodeSpec,
    sign: int,
    local_row_index: int,
    token_aggregation: TokenAggregation,
):
    def hook_fn(activations: Tensor, hook) -> Tensor:
        update = activations.detach()
        if node.kind == "attn":
            update = update[:, :, node.head, :]
        # Heuristic: average source activations over batch before accumulation.
        update = _aggregate_tokens(update.mean(dim=0), token_aggregation)
        if token_aggregation == "all-tok":
            activation_difference[:, local_row_index, :] += sign * update
        else:
            activation_difference[local_row_index, :] += sign * update
        return activations

    return hook_fn


def _capture_hooks_for_prefix(
    graph: Graph,
    activation_difference: Tensor,
    source_start: int,
    source_end: int,
    token_aggregation: TokenAggregation,
):
    fwd_hooks_corrupted = []
    fwd_hooks_clean = []
    for node in _hooked_nodes(graph):
        row_index = graph.forward_index(node, attn_slice=False)
        if row_index < source_start or row_index >= source_end:
            continue
        fwd_hooks_corrupted.append(
            (
                node.out_hook,
                _forward_capture_hook(
                    activation_difference,
                    node,
                    1,
                    row_index - source_start,
                    token_aggregation,
                ),
            )
        )
        fwd_hooks_clean.append(
            (
                node.out_hook,
                _forward_capture_hook(
                    activation_difference,
                    node,
                    -1,
                    row_index - source_start,
                    token_aggregation,
                ),
            )
        )
    return fwd_hooks_corrupted, fwd_hooks_clean


def _backward_accumulation_hook(
    activation_difference: Tensor,
    scores: Optional[Tensor],
    source_start: int,
    source_end: int,
    prev_index: int,
    target_index: slice | int,
    is_attn: bool,
    token_aggregation: TokenAggregation,
):
    def hook_fn(gradients, hook):
        if scores is None:
            return None
        local_end = min(prev_index, source_end) - source_start
        if local_end <= 0:
            return None
        grad_tensor = gradients[0] if isinstance(gradients, tuple) else gradients
        batch_size = grad_tensor.shape[0]
        grad = _aggregate_tokens(grad_tensor.detach().mean(dim=0), token_aggregation)
        if is_attn:
            if token_aggregation == "all-tok":
                prev = activation_difference[:, :local_end, :]
                contribution = batch_size * torch.einsum("pfd,phd->fh", prev, grad)
            else:
                prev = activation_difference[:local_end, :]
                contribution = batch_size * torch.einsum("fd,hd->fh", prev, grad)
            scores[source_start : source_start + local_end, target_index] += (
                contribution.to(device=scores.device, dtype=scores.dtype)
            )
        else:
            if token_aggregation == "all-tok":
                prev = activation_difference[:, :local_end, :]
                contribution = batch_size * torch.einsum("pfd,pd->f", prev, grad)
            else:
                prev = activation_difference[:local_end, :]
                contribution = batch_size * torch.einsum("fd,d->f", prev, grad)
            scores[source_start : source_start + local_end, target_index] += (
                contribution.to(device=scores.device, dtype=scores.dtype)
            )
        return None

    return hook_fn


def make_hooks_and_matrices(
    model: HookedTransformer,
    graph: Graph,
    n_pos: int,
    scores: Optional[Tensor],
    backward_specs: tuple[tuple[str, int, slice | int, bool], ...],
    source_start: int,
    source_end: int,
    token_aggregation: TokenAggregation,
):
    device = _model_device(model)
    activation_shape = (
        (n_pos, source_end - source_start, model.cfg.d_model)
        if token_aggregation == "all-tok"
        else (source_end - source_start, model.cfg.d_model)
    )
    activation_difference = torch.zeros(
        activation_shape,
        device=device,
        dtype=model.cfg.dtype,
    )

    fwd_hooks_corrupted, fwd_hooks_clean = _capture_hooks_for_prefix(
        graph=graph,
        activation_difference=activation_difference,
        source_start=source_start,
        source_end=source_end,
        token_aggregation=token_aggregation,
    )

    bwd_hooks = []
    for hook_name, prev_index, target_index, is_attn in backward_specs:
        bwd_hooks.append(
            (
                hook_name,
                _backward_accumulation_hook(
                    activation_difference=activation_difference,
                    scores=scores,
                    source_start=source_start,
                    source_end=source_end,
                    prev_index=prev_index,
                    target_index=target_index,
                    is_attn=is_attn,
                    token_aggregation=token_aggregation,
                ),
            )
        )

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def _capture_input_endpoints(
    model: HookedTransformer,
    graph: Graph,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    attention_mask: Tensor,
    batch_size: int,
    n_pos: int,
):
    model_device = _model_device(model)
    input_difference = torch.zeros(
        (batch_size, n_pos, model.cfg.d_model),
        device=model_device,
        dtype=model.cfg.dtype,
    )

    def input_capture_hook(sign: int):
        def hook_fn(activations: Tensor, hook) -> Tensor:
            input_difference.add_(sign * activations.detach())
            return activations

        return hook_fn

    with torch.inference_mode():
        with model.hooks(
            fwd_hooks=[(graph.nodes["input"].out_hook, input_capture_hook(1))]
        ):
            _ = model(corrupted_tokens, attention_mask=attention_mask)
        corrupted_input = input_difference.detach().clone()

        with model.hooks(
            fwd_hooks=[(graph.nodes["input"].out_hook, input_capture_hook(-1))]
        ):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)
        clean_input = corrupted_input - input_difference

    delta = clean_input - corrupted_input
    return clean_logits, corrupted_input, delta


def _run_source_chunk_ig(
    model: HookedTransformer,
    graph: Graph,
    scores: Tensor,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    attention_mask: Tensor,
    clean_logits: Tensor,
    input_lengths: Tensor,
    label: Tensor,
    metric: Callable[[Tensor], Tensor],
    backward_specs: tuple[tuple[str, int, slice | int, bool], ...],
    source_start: int,
    source_end: int,
    n_pos: int,
    steps: int,
    corrupted_input: Tensor,
    delta: Tensor,
    token_aggregation: TokenAggregation,
) -> None:
    (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), _ = make_hooks_and_matrices(
        model=model,
        graph=graph,
        n_pos=n_pos,
        scores=scores,
        backward_specs=backward_specs,
        source_start=source_start,
        source_end=source_end,
        token_aggregation=token_aggregation,
    )
    with torch.inference_mode():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)
        with model.hooks(fwd_hooks=fwd_hooks_clean):
            _ = model(clean_tokens, attention_mask=attention_mask)

    def interpolation_hook(step: int):
        alpha = step / steps

        def hook_fn(activations: Tensor, hook) -> Tensor:
            interpolated = corrupted_input + alpha * delta
            return interpolated.to(activations.device).requires_grad_(True)

        return hook_fn

    for step in range(1, steps + 1):
        with model.hooks(
            fwd_hooks=[(graph.nodes["input"].out_hook, interpolation_hook(step))],
            bwd_hooks=bwd_hooks,
        ):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()


def _assign_edge_scores(graph: Graph, scores: Tensor) -> np.ndarray:
    score_array = scores.float().cpu().numpy()
    edge_vector = np.empty(len(graph.edge_list), dtype=score_array.dtype)
    for index, edge in enumerate(graph.edge_list):
        edge_score = score_array[graph.edge_rows[index], graph.edge_cols[index]]
        edge.score = float(edge_score)
        edge_vector[index] = edge_score
    return edge_vector


def get_scores_eap_ig(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps: int = 30,
    quiet: bool = False,
    chunk_mode: Literal["none", "fixed", "layer"] = "none",
    chunk_size: Optional[int] = None,
    source_chunk_size: Optional[int] = None,
    token_aggregation: TokenAggregation = "all-tok",
) -> Tensor:
    if token_aggregation not in {"all-tok", "last-tok", "avg-tok"}:
        raise ValueError(
            "token_aggregation must be one of "
            "{'all-tok', 'last-tok', 'avg-tok'}, "
            f"got {token_aggregation}"
        )
    score_dtype = getattr(model.cfg, "dtype", torch.float32)
    model_device = _model_device(model)
    score_device = model_device if model_device.type != "cpu" else torch.device("cpu")
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=score_device, dtype=score_dtype
    )

    total_items = 0
    backward_chunks = _chunked_backward_specs(
        graph, chunk_mode=chunk_mode, chunk_size=chunk_size
    )
    iterator = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in iterator:
        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted
        batch_size = len(clean_tokens)
        total_items += batch_size

        clean_tokens = clean_tokens.to(model_device)
        corrupted_tokens = corrupted_tokens.to(model_device)
        attention_mask = attention_mask.to(model_device)
        clean_logits, corrupted_input, delta = _capture_input_endpoints(
            model=model,
            graph=graph,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            attention_mask=attention_mask,
            batch_size=batch_size,
            n_pos=n_pos,
        )

        with _frozen_parameters(model):
            for backward_specs in backward_chunks:
                max_forward_index = max(spec[1] for spec in backward_specs)
                for source_start, source_end in _iter_source_ranges(
                    max_forward_index=max_forward_index,
                    source_chunk_size=source_chunk_size,
                ):
                    _run_source_chunk_ig(
                        model=model,
                        graph=graph,
                        scores=scores,
                        clean_tokens=clean_tokens,
                        corrupted_tokens=corrupted_tokens,
                        attention_mask=attention_mask,
                        clean_logits=clean_logits,
                        input_lengths=input_lengths,
                        label=label,
                        metric=metric,
                        backward_specs=backward_specs,
                        source_start=source_start,
                        source_end=source_end,
                        n_pos=n_pos,
                        steps=steps,
                        corrupted_input=corrupted_input,
                        delta=delta,
                        token_aggregation=token_aggregation,
                    )

        model.zero_grad(set_to_none=True)

    if total_items == 0:
        return scores
    scores /= total_items
    scores /= steps
    return scores.cpu()


allowed_aggregations = {"sum", "mean", "l2"}
allowed_token_aggregations = {"all-tok", "last-tok", "avg-tok"}


def attribute(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    aggregation: Literal["sum", "mean", "l2"] = "sum",
    ig_steps: Optional[int] = 5,
    quiet: bool = False,
    chunk_mode: Literal["none", "fixed", "layer"] = "none",
    chunk_size: Optional[int] = None,
    source_chunk_size: Optional[int] = None,
    token_aggregation: TokenAggregation = "all-tok",
) -> np.ndarray:
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be one of {allowed_aggregations}, got {aggregation}"
        )
    if token_aggregation not in allowed_token_aggregations:
        raise ValueError(
            "token_aggregation must be one of "
            f"{allowed_token_aggregations}, got {token_aggregation}"
        )
    scores = get_scores_eap_ig(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric=metric,
        steps=ig_steps,
        quiet=quiet,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
        token_aggregation=token_aggregation,
    )

    if aggregation == "mean":
        scores = scores / model.cfg.d_model
    elif aggregation == "l2":
        scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    return _assign_edge_scores(graph, scores)


__all__ = ["Graph", "attribute"]
