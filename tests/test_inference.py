import itertools
import unittest

import torch

from framerefine.inference import ForestDP


def brute(unary, edges, values, k, temperature=1.0):
    masks = []
    for chosen in itertools.combinations(range(len(unary)), k):
        mask = torch.zeros_like(unary)
        mask[list(chosen)] = 1
        masks.append(mask)
    masks = torch.stack(masks)
    energy = masks @ unary
    for (i, j), value in zip(edges, values):
        energy = energy + value * masks[:, i] * masks[:, j]
    scores = energy / temperature
    probabilities = scores.softmax(0)
    return masks, scores, probabilities


class ForestTests(unittest.TestCase):
    def test_partition_entropy_map_and_gradients_match_exhaustive_sets(self):
        unary = torch.tensor([0.2, -0.3, 0.8, 0.1], dtype=torch.double, requires_grad=True)
        values = torch.tensor([0.7, -0.9], dtype=torch.double, requires_grad=True)
        edges = [(0, 1), (1, 2)]
        dp = ForestDP(unary, edges, values, 2, temperature=0.7)
        masks, scores, probs = brute(unary, edges, values, 2, 0.7)
        expected_z = scores.logsumexp(0)
        torch.testing.assert_close(dp.log_partition, expected_z)
        torch.testing.assert_close(dp.entropy, -(probs * probs.log()).sum())
        torch.testing.assert_close(dp.energy(dp.mode()) / 0.7, scores.max())
        actual = torch.autograd.grad(dp.log_partition + dp.entropy, (unary, values), retain_graph=True)
        expected = torch.autograd.grad(expected_z - (probs * probs.log()).sum(), (unary, values))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
        generator = torch.Generator().manual_seed(9)
        for _ in range(30):
            self.assertEqual(int(dp.sample(generator=generator).sum()), 2)

    def test_empty_full_and_impossible_budgets(self):
        for n, k in [(0, 0), (3, 0), (3, 3)]:
            dp = ForestDP(torch.zeros(n), [], torch.zeros(0), k)
            self.assertEqual(int(dp.sample().sum()), k)
            self.assertAlmostEqual(float(dp.entropy), 0.0, places=6)
        with self.assertRaises(ValueError):
            ForestDP(torch.zeros(3), [], torch.zeros(0), 4)

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            ForestDP(torch.zeros(3), [(0, 1), (1, 2), (2, 0)], torch.zeros(3), 1)


if __name__ == "__main__":
    unittest.main()
