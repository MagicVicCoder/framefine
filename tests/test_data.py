import json
from pathlib import Path
import tempfile
import unittest

import torch

from framerefine.data import CachedExample, FeatureDataset


class DatasetTests(unittest.TestCase):
    def test_manifest_is_lazy_and_fingerprint_tracks_question_and_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder, "manifest.jsonl")
            row = dict(id="q", feature_path="not-created-yet.pt", question="What?",
                       options=["a", "b"], answer_index=0, content_revision="video1")
            manifest.write_text(json.dumps(row) + "\n")
            dataset = FeatureDataset(manifest)
            self.assertEqual(len(dataset), 1)
            with self.assertRaises(FileNotFoundError):
                dataset[0]
            torch.save(dict(frame_features=torch.ones(3, 4), query_features=torch.zeros(2),
                            timestamps=torch.arange(3.)), Path(folder, "not-created-yet.pt"))
            self.assertEqual(dataset[0].frame_features.shape, (3, 4))
            row["question"] = "Changed?"
            manifest.write_text(json.dumps(row) + "\n")
            self.assertNotEqual(dataset.fingerprint, FeatureDataset(manifest).fingerprint)

    def test_nonfinite_features_and_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            CachedExample("x", torch.tensor([[float("nan")]]), torch.zeros(2), torch.zeros(1),
                          "q", ("a", "b"), 0, "rev")
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder, "manifest.jsonl")
            row = dict(id="x", feature_path="x.pt", question="q", options=["a", "b"],
                       answer_index=0, content_revision="r")
            manifest.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                FeatureDataset(manifest)


if __name__ == "__main__":
    unittest.main()
