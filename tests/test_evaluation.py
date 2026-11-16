import unittest

import torch

from framerefine.data import toy_examples
from framerefine.evaluation import evaluate
from framerefine.policy import CompressionPolicy
from framerefine.readers import Budget, CachedReader, ToyReader


class EvaluationTests(unittest.TestCase):
    def test_eval_reports_accuracy_separately_from_utility_and_preserves_weights(self):
        torch.manual_seed(8)
        actor = CompressionPolicy(6, 4, hidden_dim=16)
        before = [p.detach().clone() for p in actor.parameters()]
        reader = CachedReader(ToyReader(), ":memory:", Budget(20, 60))
        report = evaluate(actor, reader, toy_examples(2, 8, 6, 4), 4)
        self.assertEqual(report["examples"], 2)
        self.assertFalse(report["results_are_real_video_experiments"])
        self.assertEqual(report["reader_identity"], ToyReader.identity)
        for method in ("selected", "uniform"):
            self.assertIn(report[method]["accuracy"], (0.0, 0.5, 1.0))
            self.assertEqual(report[method]["predictions_available"], 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, actor.parameters())))
        self.assertLessEqual(reader.budget.requests, 4)
        reader.close()


if __name__ == "__main__":
    unittest.main()
