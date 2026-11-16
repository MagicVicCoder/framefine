from pathlib import Path
from dataclasses import replace
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from framerefine.data import toy_examples
from framerefine.policy import CompressionPolicy
from framerefine.readers import Budget, CachedReader, ToyReader
from framerefine.training import TrainConfig, Trainer


class TrainingTests(unittest.TestCase):
    def test_optimizer_checkpoint_and_resume_preserve_the_real_actor(self):
        torch.manual_seed(2)
        examples = toy_examples(2, n=8, feature_dim=6, query_dim=4)
        policy = CompressionPolicy(6, 4, hidden_dim=16, conditioning_nodes=2)
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        config = TrainConfig(k=4, group_size=2, max_steps=3, probe_every=1,
                             relation_weight=0.2, learning_rate=0.001)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            trainer = Trainer(policy, reader, config)
            report = trainer.fit(examples, folder)
            self.assertEqual(report["completed_steps"], 3)
            self.assertEqual(report["stop_reason"], "max_steps")
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))
            self.assertTrue(Path(folder, "last.pt").is_file())
            current = policy.draw(examples[0].frame_features, examples[0].query_features,
                                  4, deterministic=True).distribution.mode()
            replacement = CompressionPolicy(**policy.settings)
            fresh_reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            other = Trainer(replacement, fresh_reader, config)
            other.load_checkpoint(Path(folder, "last.pt"))
            selected = replacement.draw(examples[0].frame_features, examples[0].query_features,
                                        4, deterministic=True).distribution.mode()
            torch.testing.assert_close(current, selected)
            self.assertEqual(other.step_index, 3)
            fresh_reader.close()
            reader.close()

    def test_exhausted_budget_stops_cleanly_without_fabricated_steps(self):
        policy = CompressionPolicy(6, 4, hidden_dim=16)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), ":memory:", Budget(0, 60))
            trainer = Trainer(policy, reader, TrainConfig(k=3, max_steps=2, group_size=2))
            report = trainer.fit(toy_examples(1, 8, 6, 4), folder)
            self.assertEqual(report["completed_steps"], 0)
            self.assertIn("budget", report["stop_reason"])
            self.assertTrue(Path(folder, "last.pt").is_file())
            reader.close()

    def test_resume_counts_old_elapsed_plus_new_startup_and_validates_semantics(self):
        config = TrainConfig(k=3, group_size=2, max_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            original = CachedReader(ToyReader(), ":memory:", Budget(1000, 100))
            trainer = Trainer(CompressionPolicy(6, 4, hidden_dim=16), original, config)
            original.budget.started = time.monotonic() - 10
            original.budget.requests = 5
            original.scoring_seconds, original.cache_hits = 2.0, 3
            checkpoint = Path(folder, "last.pt")
            trainer.save_checkpoint(checkpoint)
            fresh = CachedReader(ToyReader(), ":memory:", Budget(1000, 100))
            fresh.budget.started = time.monotonic() - 3
            fresh.budget.requests = 2
            fresh.scoring_seconds, fresh.cache_hits = 0.5, 1
            resumed = Trainer(CompressionPolicy(6, 4, hidden_dim=16), fresh, replace(config, max_steps=5))
            resumed.load_checkpoint(checkpoint)
            self.assertGreaterEqual(time.monotonic() - fresh.budget.started, 12.99)
            self.assertEqual(fresh.budget.requests, 7)
            self.assertEqual(fresh.scoring_seconds, 2.5)
            self.assertEqual(fresh.cache_hits, 4)
            mismatch = Trainer(CompressionPolicy(6, 4, hidden_dim=16), fresh,
                               replace(config, learning_rate=0.01))
            before = [p.detach().clone() for p in mismatch.policy.parameters()]
            with self.assertRaisesRegex(ValueError, "training"):
                mismatch.load_checkpoint(checkpoint)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, mismatch.policy.parameters())))
            fresh.scorer.identity = {**fresh.scorer.identity, "revision": "different"}
            with self.assertRaisesRegex(ValueError, "reader"):
                resumed.load_checkpoint(checkpoint)
            original.close()
            fresh.close()

    def test_resume_reproduces_the_next_optimizer_update(self):
        torch.manual_seed(19)
        sample = toy_examples(1, n=8, feature_dim=6, query_dim=4)[0]
        config = TrainConfig(k=4, group_size=2, max_steps=3, probe_every=1)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            trainer = Trainer(CompressionPolicy(6, 4, hidden_dim=16), reader, config)
            trainer.step(sample)
            path = Path(folder, "last.pt")
            trainer.save_checkpoint(path)
            expected = trainer.step(sample)
            fresh = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            resumed = Trainer(CompressionPolicy(6, 4, hidden_dim=16), fresh, config)
            resumed.load_checkpoint(path)
            actual = resumed.step(sample)
            self.assertAlmostEqual(expected["loss"], actual["loss"], places=6)
            for original, restored in zip(trainer.policy.parameters(), resumed.policy.parameters()):
                torch.testing.assert_close(original, restored, rtol=0, atol=0)
            reader.close()
            fresh.close()

    def test_reader_error_saves_completed_updates_and_periodic_checkpoint_survives_update_error(self):
        samples = toy_examples(1, n=8, feature_dim=6, query_dim=4)
        for fail_in_update in (False, True):
            with self.subTest(fail_in_update=fail_in_update), tempfile.TemporaryDirectory() as folder:
                reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
                config = TrainConfig(k=4, group_size=2, max_steps=3, probe_every=0, checkpoint_every=1)
                trainer = Trainer(CompressionPolicy(6, 4, hidden_dim=16), reader, config)
                target = trainer.optimizer if fail_in_update else reader
                name = "step" if fail_in_update else "score_details"
                original = getattr(target, name)
                calls = [0]

                def failing(*args, **kwargs):
                    calls[0] += 1
                    if calls[0] == 2:
                        raise RuntimeError("injected reader/update error")
                    return original(*args, **kwargs)

                with patch.object(target, name, side_effect=failing):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        trainer.fit(samples, folder)
                payload = torch.load(Path(folder, "last.pt"), weights_only=True)
                self.assertEqual(payload["step"], 1)
                reader.close()


if __name__ == "__main__":
    unittest.main()
