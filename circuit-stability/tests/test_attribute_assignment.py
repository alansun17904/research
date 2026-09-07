import unittest

import numpy as np
import torch

from eap import Graph
from eap.core import _assign_edge_scores, _assign_node_scores


class AttributeAssignmentTest(unittest.TestCase):
    def test_edge_scores_follow_forward_backward_layout(self):
        graph = Graph.from_model(
            {
                "n_layers": 1,
                "n_heads": 2,
                "parallel_attn_mlp": False,
                "n_kv_heads": 2,
            }
        )
        scores = torch.arange(
            graph.n_forward * graph.n_backward, dtype=torch.float32
        ).reshape(graph.n_forward, graph.n_backward)

        edge_vector = _assign_edge_scores(graph, scores)

        first_edge = next(iter(graph.edges.values()))
        parent = graph.node_list[first_edge.parent]
        child = graph.node_list[first_edge.child]
        expected = scores[
            graph.forward_index(parent, attn_slice=False),
            graph.backward_index(child, qkv=first_edge.qkv, attn_slice=False),
        ].item()

        self.assertEqual(first_edge.score, expected)
        self.assertEqual(len(edge_vector), len(graph.edges))
        self.assertEqual(edge_vector[0], expected)

    def test_node_scores_follow_forward_layout(self):
        graph = Graph.from_model(
            {
                "n_layers": 1,
                "n_heads": 2,
                "parallel_attn_mlp": False,
                "n_kv_heads": 2,
            }
        )
        scores = torch.arange(graph.n_forward, dtype=torch.float32)

        node_vector = _assign_node_scores(graph, scores)

        input_node = graph.nodes["input"]
        attn_node = graph.nodes["a0.h1"]
        mlp_node = graph.nodes["m0"]
        logits_node = graph.nodes["logits"]

        self.assertEqual(input_node.score, scores[0].item())
        self.assertEqual(
            attn_node.score,
            scores[graph.forward_index(attn_node, attn_slice=False)].item(),
        )
        self.assertEqual(
            mlp_node.score,
            scores[graph.forward_index(mlp_node, attn_slice=False)].item(),
        )
        self.assertIsNone(logits_node.score)
        self.assertEqual(len(node_vector), graph.n_forward)
        self.assertTrue(np.array_equal(node_vector, graph.node_vector()))


if __name__ == "__main__":
    unittest.main()
