"""Offline checks against the downloaded real checkpoint; run python -m unittest -v."""

import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import evaluate
import evaluate as evaluation
from model import CHECKPOINT, ROOT, load_model, read_graph, validate_graph


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.baseline = read_graph(ROOT / "graphs/baseline.json")
        cls.model = load_model(ROOT / "graphs/baseline.json")
        cls.reference, loading = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, local_files_only=True, attn_implementation="eager", output_loading_info=True
        )
        cls.reference.eval()
        if any(loading.values()):
            raise AssertionError(f"Official checkpoint mapping failed: {loading}")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        text = "The quick brown fox jumps over the lazy dog. A model predicts the next token. " * 40
        cls.ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :257]
        if cls.ids.size(1) != 257:
            raise AssertionError("Parity input should cover 257 rotary positions")

    def setUp(self):
        self.model.graph = copy.deepcopy(self.baseline)

    def test_baseline_parity_and_all_checkpoint_tensors(self):
        weights = load_file(CHECKPOINT / "model.safetensors")
        state = self.model.state_dict()
        self.assertEqual(set(weights), set(state))
        self.assertEqual(len(state), 94)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 70_426_624)
        for name, tensor in weights.items():
            self.assertTrue(torch.equal(state[name], tensor.to(state[name].dtype)), name)
        with torch.inference_mode():
            actual = self.model(self.ids, labels=self.ids).logits
            expected = self.reference(self.ids, use_cache=False).logits
        actual_nll = F.cross_entropy(actual[:, :-1].reshape(-1, actual.size(-1)),
                                     self.ids[:, 1:].reshape(-1), reduction="none")
        expected_nll = F.cross_entropy(expected[:, :-1].reshape(-1, expected.size(-1)),
                                       self.ids[:, 1:].reshape(-1), reduction="none")
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)
        torch.testing.assert_close(actual_nll, expected_nll, atol=1e-4, rtol=0)
        self.assertLessEqual(abs(actual_nll.mean().item() - expected_nll.mean().item()), 1e-5)
        print(json.dumps({"parity_tokens": self.ids.size(1), "mapped_tensors": len(state),
                          "unique_parameters": sum(p.numel() for p in self.model.parameters()),
                          "max_absolute_logit_error": (actual - expected).abs().max().item(),
                          "max_target_log_probability_error": (actual_nll - expected_nll).abs().max().item(),
                          "mean_nll_error": abs(actual_nll.mean().item() - expected_nll.mean().item()),
                          "official_mean_nll": expected_nll.mean().item(),
                          "local_mean_nll": actual_nll.mean().item()}), flush=True)

    def test_deleted_heads_and_shared_bias(self):
        layer = self.model.gpt_neox.layers[0]
        attention = layer.attention
        with torch.inference_mode():
            normalized = layer.input_layernorm(self.model.gpt_neox.embed_in(self.ids[:, :8]))
            qkv = attention.query_key_value(normalized).view(1, 8, 8, 192).transpose(1, 2)
            values = attention.attend(qkv)
            values[:, 0] = 0
            expected = attention.dense(values.transpose(1, 2).reshape(1, 8, 512))
            torch.testing.assert_close(attention(normalized, [0] + [1] * 7), expected)
            torch.testing.assert_close(attention(normalized, [0] * 8),
                                       attention.dense.bias.expand(1, 8, -1))

    def test_head_loop_recomputes_on_feedback(self):
        layer = self.model.gpt_neox.layers[0]
        spec = copy.deepcopy(self.baseline["layers"][0])
        spec["heads"][1] = 3
        with torch.inference_mode():
            hidden = self.model.gpt_neox.embed_in(self.ids[:, :8])
            normalized = layer.input_layernorm(hidden)
            delta = layer.attention(normalized, spec["heads"])
            initial_head = layer.attention.head(normalized, 1)
            qkv = F.linear(normalized, layer.attention.query_key_value.weight[192:384],
                           layer.attention.query_key_value.bias[192:384])
            expected_head = F.linear(layer.attention.attend(qkv[:, None]).squeeze(1),
                                     layer.attention.dense.weight[:, 64:128])
            torch.testing.assert_close(initial_head, expected_head)
            naive = layer(hidden, self.baseline["layers"][0]) + 2 * initial_head
            for _ in range(2):
                delta = delta + layer.attention.head(layer.input_layernorm(hidden + delta), 1)
            expected = layer.mlp(layer.post_attention_layernorm(hidden)) + delta + hidden
            actual = layer(hidden, spec)
            torch.testing.assert_close(actual, expected)
            self.assertFalse(torch.allclose(actual, naive, atol=1e-6, rtol=1e-6))

    def test_mlp_loop_recomputes_on_feedback(self):
        layer = self.model.gpt_neox.layers[0]
        spec = copy.deepcopy(self.baseline["layers"][0])
        spec["mlp_repeat"] = 3
        with torch.inference_mode():
            hidden = self.model.gpt_neox.embed_in(self.ids[:, :8])
            delta = layer.mlp(layer.post_attention_layernorm(hidden))
            for _ in range(2):
                delta = delta + layer.mlp(layer.post_attention_layernorm(hidden + delta))
            expected = delta + layer.attention(layer.input_layernorm(hidden), spec["heads"]) + hidden
            torch.testing.assert_close(layer(hidden, spec), expected)

    def test_subset_loops_and_duplicate_references_share_parameters(self):
        parameters = [id(p) for p in self.model.parameters()]
        self.model.graph["sequence"].append({"layers": [1, 3, 1], "repeat": 2})
        validate_graph(self.model.graph)
        calls = []
        hooks = [layer.register_forward_hook(lambda module, args, output, i=i: calls.append(i))
                 for i, layer in enumerate(self.model.gpt_neox.layers)]
        try:
            with torch.inference_mode():
                actual = self.model(self.ids[:, :8]).logits
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, list(range(6)) + [1, 3, 1, 1, 3, 1])
        self.model.graph["sequence"][-1] = {"layers": [1, 3, 1, 1, 3, 1], "repeat": 1}
        with torch.inference_mode():
            torch.testing.assert_close(actual, self.model(self.ids[:, :8]).logits, atol=0, rtol=0)
        self.assertEqual(parameters, [id(p) for p in self.model.parameters()])

    def test_modified_evaluation_and_causality(self):
        with torch.inference_mode():
            original = self.model(self.ids[:, :16]).logits
            changed_ids = self.ids[:, :16].clone()
            changed_ids[:, 8:] = 1
            changed = self.model(changed_ids).logits
        torch.testing.assert_close(original[:, :8], changed[:, :8], atol=0, rtol=0)
        self.model.graph = read_graph(ROOT / "graphs/example.json")
        result = evaluate(self.model, self.ids[:, :17])
        self.assertEqual(result["scored_tokens"], 16)
        self.assertEqual(result["input_tokens"], 17)
        with torch.inference_mode():
            output = self.model(self.ids[:, :17], labels=self.ids[:, :17])
        self.assertAlmostEqual(result["mean_nll_nats"], output.loss.item(), places=6)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertFalse(torch.allclose(original[:, :8], output.logits[:, :8]))
        with torch.inference_mode():
            modified_future = self.model(changed_ids).logits
        torch.testing.assert_close(output.logits[:, :8], modified_future[:, :8], atol=2e-4, rtol=0)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 2049, dtype=torch.long))

    def test_cross_layer_head_sequence(self):
        block = {"heads": [[1, 1], [2, 2]], "repeat": 3}
        self.model.graph["sequence"].append(block)
        validate_graph(self.model.graph)
        with torch.inference_mode():
            ids = self.ids[:, :8]
            hidden = self.model.gpt_neox.embed_in(ids)
            for index in range(6):
                hidden = self.model.gpt_neox.layers[index](hidden, self.model.graph["layers"][index])
            for _ in range(3):
                for index, head in block["heads"]:
                    layer = self.model.gpt_neox.layers[index]
                    hidden = hidden + layer.attention.head(layer.input_layernorm(hidden), head)
            expected = self.model.embed_out(self.model.gpt_neox.final_layer_norm(hidden))
            torch.testing.assert_close(self.model(ids).logits, expected, atol=0, rtol=0)
            self.model.graph["layers"][1]["heads"][1] = 0
            self.model.graph["sequence"][-1] = {"heads": [[1, 1]], "repeat": 3}
            deleted = self.model(ids).logits
            self.model.graph["sequence"].pop()
            torch.testing.assert_close(deleted, self.model(ids).logits, atol=0, rtol=0)

    def test_evaluator_weights_overlap_and_partial_tail_by_target_count(self):
        original_context = self.model.config.max_position_embeddings
        self.model.config.max_position_embeddings = 8
        try:
            ids = self.ids[:, :17]
            result = evaluate(self.model, ids)
            total = 0.0
            with torch.inference_mode():
                for start, end, offset in ((0, 8, 1), (4, 12, 4), (8, 16, 4), (12, 17, 4)):
                    logits = self.model(ids[:, start:end]).logits[:, offset - 1:-1]
                    total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                             ids[:, start + offset:end].reshape(-1),
                                             reduction="sum").item()
            self.assertEqual(result["scored_tokens"], 16)
            self.assertAlmostEqual(result["mean_nll_nats"], total / 16, places=6)
        finally:
            self.model.config.max_position_embeddings = original_context

    def test_graph_cli_uses_modified_model_without_loading_official_baseline(self):
        output = io.StringIO()
        argv = ["evaluate.py", "--graph", str(ROOT / "graphs/example.json"),
                "--max-tokens", "17", "--device", "cpu"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
             patch("datasets.load_dataset", return_value=[{"text": "Small synthetic CLI check. " * 20}]), \
             patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("baseline reloaded")):
            evaluation.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["scored_tokens"], 16)
        self.assertEqual(result["graph"], read_graph(ROOT / "graphs/example.json"))
        self.assertEqual(result["unique_parameters"], 70_426_624)

    def test_strict_mapping_and_checksums(self):
        state = self.model.state_dict()
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict({k: v for k, v in state.items() if k != "embed_out.weight"}, strict=True)
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict({**state, "unexpected.weight": torch.zeros(1)}, strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_bytes((CHECKPOINT / "config.json").read_bytes())
            provenance = json.loads((CHECKPOINT / "provenance.json").read_text())
            provenance["files"]["config.json"]["sha256"] = "incorrect"
            (directory / "provenance.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_model(ROOT / "graphs/baseline.json", directory)


class GraphChecks(unittest.TestCase):
    def test_reject_malformed_and_unsupported_graphs(self):
        baseline = read_graph(ROOT / "graphs/baseline.json")
        invalid = []
        for value in (-1, True, 1.5, "2", None):
            graph = copy.deepcopy(baseline)
            graph["layers"][0]["heads"][0] = value
            invalid.append(graph)
        for value in (0, -1, True, 1.5):
            for key in ("repeat", "mlp_repeat"):
                graph = copy.deepcopy(baseline)
                if key == "repeat":
                    graph["sequence"][0][key] = value
                else:
                    graph["layers"][0][key] = value
                invalid.append(graph)
        for indices in ([], [6], [-1], [False], [1.0]):
            graph = copy.deepcopy(baseline)
            graph["sequence"][0]["layers"] = indices
            invalid.append(graph)
        for heads in ([], [[0]], [[0, 0, 0]], [[0, 8]], [[6, 0]], [[0, True]], [[0, 1.0]]):
            graph = copy.deepcopy(baseline)
            graph["sequence"] = [{"heads": heads, "repeat": 1}]
            invalid.append(graph)
        graph = copy.deepcopy(baseline)
        graph["layers"][0]["heads"].pop()
        invalid.append(graph)
        invalid.extend([{}, [], {**baseline, "extra": 1}, {**baseline, "sequence": []},
                        {**baseline, "format": "other"}, {**baseline, "checkpoint": {}},
                        {**baseline, "weights": {}}])
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_graph(graph)
        graph = copy.deepcopy(baseline)
        graph["layers"][0]["heads"][0] = 10 ** 100
        graph["layers"][0]["mlp_repeat"] = 10 ** 100
        graph["sequence"][0]["repeat"] = 10 ** 100
        validate_graph(graph)

    def test_json_duplicate_fields_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.json"
            for text in ('{"format": 1, "format": 2}', '{"x": NaN}',
                         '{"x": Infinity}', '{"x": -Infinity}', '{broken'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_graph(path)


if __name__ == "__main__":
    unittest.main()
