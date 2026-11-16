import itertools
import unittest

import torch

from framerefine.inference import ForestDP
from framerefine.policy import Branch, CompressionPolicy, ConditionalMaskDistribution


class ConditionalTests(unittest.TestCase):
    def test_context_gated_pair_recovers_a_three_way_energy(self):
        # Anchor 0 gates the interaction of residual nodes 1 and 2.
        branches = [Branch((c,), torch.tensor(0.0),
                           ForestDP(torch.zeros(4), [(0, 1)], torch.tensor([2.0 * c]), 3-c))
                    for c in (0, 1)]
        dist = ConditionalMaskDistribution(5, 3, (0,), branches, 1.0)
        scores = []
        for chosen in itertools.combinations(range(5), 3):
            mask = torch.zeros(5)
            mask[list(chosen)] = 1
            expected = 2 * mask[0] * mask[1] * mask[2]
            torch.testing.assert_close(dist.energy(mask), expected)
            scores.append(dist.log_prob(mask))
        torch.testing.assert_close(torch.stack(scores).exp().sum(), torch.tensor(1.0))
        self.assertEqual(dist.mode().nonzero().flatten().tolist(), [0, 1, 2])
        probs = torch.stack(scores).softmax(0)
        torch.testing.assert_close(dist.entropy, -(probs * probs.log()).sum())

    def test_actor_draw_has_finite_gradients_and_exact_budget_including_boundaries(self):
        torch.manual_seed(12)
        actor = CompressionPolicy(6, 4, hidden_dim=16, conditioning_nodes=2)
        features, query = torch.randn(6, 6), torch.randn(4)
        for k in (0, 2, 6):
            draw = actor.draw(features, query, k)
            mask = draw.distribution.sample()
            self.assertEqual(int(mask.sum()), k)
            self.assertEqual(int(draw.distribution.mode().sum()), k)
            objective = draw.distribution.log_prob(mask) + draw.log_structure_prob + 0.1 * draw.distribution.entropy
            actor.zero_grad()
            objective.backward()
            gradients = [p.grad for p in actor.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
