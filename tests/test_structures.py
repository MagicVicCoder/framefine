import itertools
import unittest

import torch

from framerefine.structures import maximum_tree, sample_tree, tree_log_prob


class TreeTests(unittest.TestCase):
    def test_three_rooted_trees_normalize_and_have_correct_gradients(self):
        # Complete graph on virtual root 0 and two real nodes has exactly 3 trees.
        scores = torch.tensor([[0.0, 0.2, -0.1], [0.2, 0.0, 0.7], [-0.1, 0.7, 0.0]], dtype=torch.double, requires_grad=True)
        trees = [[(0, 1), (0, 2)], [(0, 1), (1, 2)], [(0, 2), (1, 2)]]
        logp = torch.stack([tree_log_prob(scores, tree) for tree in trees])
        torch.testing.assert_close(logp.exp().sum(), torch.tensor(1.0, dtype=torch.double))
        energies = torch.stack([sum(scores[i, j] for i, j in tree) for tree in trees])
        torch.testing.assert_close(logp, energies.log_softmax(0))
        gradient = torch.autograd.grad(logp.exp().sum(), scores)[0]
        self.assertLess(float(gradient.abs().max()), 1e-10)
        self.assertEqual(set(maximum_tree(scores)), {(0, 1), (1, 2)})

    def test_wilson_samples_valid_trees_and_single_root_is_supported(self):
        generator = torch.Generator().manual_seed(3)
        scores = torch.zeros(5, 5)
        for _ in range(20):
            edges = sample_tree(scores, generator=generator)
            self.assertEqual(len(edges), 4)
            self.assertTrue(torch.isfinite(tree_log_prob(scores, edges)))
        self.assertEqual(sample_tree(torch.zeros(1, 1)), [])
        self.assertEqual(float(tree_log_prob(torch.zeros(1, 1), [])), 0.0)


if __name__ == "__main__":
    unittest.main()
