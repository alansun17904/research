import unittest

from eap import Graph


class GraphLayoutTest(unittest.TestCase):
    def test_counts_and_indices_with_grouped_kv(self):
        graph = Graph.from_model(
            {
                "n_layers": 2,
                "n_heads": 4,
                "parallel_attn_mlp": False,
                "n_kv_heads": 2,
            }
        )

        self.assertEqual(graph.n_forward, 11)
        self.assertEqual(graph.n_backward, 19)
        self.assertEqual(len(graph.edges), 110)

        attn = graph.nodes["a1.h3"]
        mlp = graph.nodes["m1"]
        logits = graph.nodes["logits"]

        self.assertEqual(graph.forward_index(attn, attn_slice=False), 9)
        self.assertEqual(graph.prev_index(attn), 6)
        self.assertEqual(graph.backward_index(attn, qkv="q", attn_slice=False), 11)
        self.assertEqual(graph.backward_index(attn, qkv="k", attn_slice=False), 13)
        self.assertEqual(graph.backward_index(attn, qkv="v", attn_slice=False), 15)
        self.assertEqual(graph.backward_index(mlp), 16)
        self.assertEqual(graph.backward_index(logits), 18)


if __name__ == "__main__":
    unittest.main()
