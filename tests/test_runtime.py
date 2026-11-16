from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from framerefine.annotations import read_raw_manifest
from framerefine.data import toy_examples, load_manifest
from framerefine.evaluation import evaluate
from framerefine.policy import CompressionPolicy
from framerefine.readers import Budget, BudgetExceeded, CachedReader, ToyReader
from framerefine.training import TrainConfig, Trainer


class RuntimeTests(unittest.TestCase):
    def test_validation_budget_is_charged_to_the_global_cap_and_shared_cache_is_reused(self):
        global_budget = Budget(2, 60)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), Path(folder, "scores.sqlite"), global_budget)
            validation = reader.fork(Budget(5, 60, parent=global_budget))
            sample = toy_examples(1, 8, 6, 4)[0]
            masks = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])
            validation.score(sample, masks)
            self.assertEqual(global_budget.requests, 2)
            reader.score(sample, masks)
            self.assertEqual(global_budget.requests, 2)
            with self.assertRaises(BudgetExceeded):
                reader.score(sample, torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]))
            validation.close()
            reader.close()

    def test_dense_evaluation_reports_accuracy_loss_in_absolute_points(self):
        actor = CompressionPolicy(6, 4, hidden_dim=16)
        reader = CachedReader(ToyReader(), ":memory:", Budget(30, 60))
        report = evaluate(actor, reader, toy_examples(2, 8, 6, 4), 4, include_dense=True,
                          accuracy_tolerance=0.03)
        self.assertEqual(report["dense"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["accuracy_drop_points"],
                               100 * (report["dense"]["accuracy"] - report["selected"]["accuracy"]))
        self.assertEqual(report["within_accuracy_tolerance"], report["accuracy_drop_points"] <= 3.0)
        self.assertEqual(report["retained_frame_fraction"], 0.5)
        reader.close()

    def test_unlabeled_manifest_is_usable_for_selection_but_not_training_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw.jsonl"
            raw.write_text(json.dumps(dict(id="q", frame_paths=["frame.jpg"], timestamps=[0.0],
                question="What?", options=["a", "b"])) + "\n")
            self.assertIsNone(read_raw_manifest(raw)[0]["answer_index"])
            sample = replace(toy_examples(1, 8, 6, 4)[0], answer_index=None)
            actor = CompressionPolicy(6, 4, hidden_dim=16)
            self.assertEqual(int(actor.draw(sample.frame_features, sample.query_features, 4).distribution.mode().sum()), 4)
            reader = CachedReader(ToyReader(), ":memory:", Budget(3, 60))
            with self.assertRaisesRegex(ValueError, "label"):
                reader.score(sample, torch.ones(1, 8))
            reader.close()

    def test_shuffled_training_resume_matches_uninterrupted_updates_and_rejects_new_data(self):
        samples = toy_examples(3, 8, 6, 4)
        cfg = TrainConfig(k=4, group_size=2, max_steps=2, probe_every=0, shuffle=True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(3)
            actor = CompressionPolicy(6, 4, hidden_dim=16)
            reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            trainer = Trainer(actor, reader, cfg)
            trainer.fit(samples, root / "first")
            checkpoint = root / "first" / "last.pt"
            trainer.config = replace(cfg, max_steps=4)
            trainer.fit(samples, root / "continued")
            fresh = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            resumed = Trainer(CompressionPolicy(6, 4, hidden_dim=16), fresh, replace(cfg, max_steps=4))
            resumed.load_checkpoint(checkpoint, examples=samples)
            resumed.fit(samples, root / "resumed")
            for expected, actual in zip(actor.parameters(), resumed.policy.parameters()):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "dataset"):
                resumed.load_checkpoint(checkpoint, examples=toy_examples(3, 8, 6, 4, seed=99))
            reader.close()
            fresh.close()

    def test_periodic_validation_saves_best_checkpoint_without_updating_from_validation(self):
        cfg = TrainConfig(k=4, group_size=2, max_steps=2, probe_every=0)
        train, heldout = toy_examples(2, 8, 6, 4), toy_examples(1, 8, 6, 4, seed=88)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), Path(folder, "scores.sqlite"), Budget(1000, 60))
            validation = reader.fork(Budget(100, 60, parent=reader.budget))
            trainer = Trainer(CompressionPolicy(6, 4, hidden_dim=16), reader, cfg)
            trainer.fit(train, folder, validation_examples=heldout, validation_reader=validation,
                        validate_every=1)
            self.assertTrue(Path(folder, "best.pt").is_file())
            rows = Path(folder, "validation.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertGreater(validation.budget.requests, 0)
            self.assertEqual(trainer.step_index, 2)
            with self.assertRaisesRegex(ValueError, "overlap"):
                trainer.fit(train, folder, validation_examples=train, validation_reader=validation,
                            validate_every=1)
            validation.close()
            reader.close()

    def test_resume_rejects_changed_validation_identity_before_any_new_updates(self):
        cfg = TrainConfig(k=4, group_size=2, max_steps=1, probe_every=0)
        train, heldout = toy_examples(1, 8, 6, 4), toy_examples(1, 8, 6, 4, seed=88)
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
            child = reader.fork(Budget(100, 60, parent=reader.budget))
            trainer = Trainer(CompressionPolicy(6, 4, hidden_dim=16), reader, cfg)
            trainer.fit(train, folder, heldout, child, validate_every=1)
            checkpoint = Path(folder, "last.pt")
            for data, cadence, tolerance in ((toy_examples(1, 8, 6, 4, seed=99), 1, .03),
                                              (heldout, 2, .03), (heldout, 1, .02),
                                              (None, 0, .03)):
                fresh = CachedReader(ToyReader(), ":memory:", Budget(1000, 60))
                validation = fresh.fork(Budget(100, 60, parent=fresh.budget))
                resumed = Trainer(CompressionPolicy(6, 4, hidden_dim=16), fresh, replace(cfg, max_steps=2))
                resumed.load_checkpoint(checkpoint, train)
                with self.assertRaisesRegex(ValueError, "validation.*identity"):
                    resumed.fit(train, Path(folder, "resume"), data, validation, cadence, tolerance)
                self.assertEqual(resumed.step_index, 1)
                validation.close()
                fresh.close()
            child.close()
            reader.close()


if __name__ == "__main__":
    unittest.main()
