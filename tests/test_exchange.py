import itertools
import math
import unittest

import torch

from framerefine.exchange import ExchangeCube, face_differences, rao_blackwell_task_loss, third_difference


def all_pairings(items):
    if not items:
        yield ()
        return
    first = items[0]
    for j in range(1, len(items)):
        for remaining in all_pairings(items[1:j] + items[j+1:]):
            yield ((first, items[j]),) + remaining


class ExchangeTests(unittest.TestCase):
    def test_shared_eight_corners_supply_context_sensitive_faces(self):
        cube = ExchangeCube.from_seed(torch.tensor([1., 1., 1., 0., 0., 0.]), 3,
                                      generator=torch.Generator().manual_seed(4))
        masks = cube.masks()
        self.assertEqual(len({tuple(m.tolist()) for m in masks}), 8)
        self.assertTrue((masks.sum(1) == 3).all())
        utility = torch.tensor([0., 0., 0., 0., 0., 0., 0., 2.])
        faces = face_differences(utility, 3)
        torch.testing.assert_close(faces, torch.tensor([0., 2., 0., 2., 0., 2.]))
        self.assertEqual(float(third_difference(utility)), 2.0)

    def test_block_gradient_equals_global_gradient_after_averaging_blocks(self):
        # Independent complete enumeration of all 20 masks and 15 perfect matchings.
        n, k = 6, 3
        masks = []
        for chosen in itertools.combinations(range(n), k):
            mask = torch.zeros(n, dtype=torch.double)
            mask[list(chosen)] = 1
            masks.append(mask)
        masks = torch.stack(masks)
        lookup = {tuple(mask.tolist()): i for i, mask in enumerate(masks)}
        logits = torch.linspace(-0.7, 0.9, len(masks), dtype=torch.double, requires_grad=True)
        logp = logits.log_softmax(0)
        utilities = 2 * masks[:, 0] * masks[:, 2] * masks[:, 4] - masks[:, 1]
        exact_loss = -(logp.exp() * utilities).sum()
        expected = torch.autograd.grad(exact_loss, logits, retain_graph=True)[0]
        averaged = logits.sum() * 0
        block_mass = 0.0
        for pairs in all_pairings(tuple(range(n))):
            cube = ExchangeCube(n, (), pairs)
            indices = [lookup[tuple(mask.tolist())] for mask in cube.masks(dtype=torch.double)]
            probability = logp[indices].exp().sum().detach() / math.factorial(3)
            block_mass += float(probability)
            averaged = averaged + probability * rao_blackwell_task_loss(logp[indices], utilities[indices])
        self.assertAlmostEqual(block_mass, 1.0)
        actual = torch.autograd.grad(averaged, logits)[0]
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
