import unittest

import torch

from eap import Graph
from eap.core import _assign_edge_scores


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


if __name__ == "__main__":
    unittest.main()
