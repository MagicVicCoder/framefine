import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from framerefine.data import FeatureDataset
from framerefine.preprocessing import prepare_manifest, uniform_indices


class ImageEncoderFixture:
    identity = {"model": "test-rgb", "revision": "1"}

    def __init__(self):
        self.image_calls, self.questions = 0, []

    def encode_frames(self, paths):
        self.image_calls += 1
        return torch.tensor([list(Image.open(path).convert("RGB").getpixel((0, 0))) for path in paths]).float()

    def encode_question(self, question):
        self.questions.append(question)
        return torch.tensor([len(question), 1.0])


class PreparationTests(unittest.TestCase):
    def test_explicit_source_video_identity_survives_different_frame_subsets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(3):
                Image.new("RGB", (8, 8), "red").save(root / f"{index}.png")
            rows = [dict(id=f"q{i}", frame_paths=[f"{i}.png", f"{i+1}.png"],
                         timestamps=[float(i), float(i+1)], question="What?", options=["a", "b"],
                         answer_index=0, metadata={"source_video": "same-original-video"}) for i in range(2)]
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            dataset = FeatureDataset(prepare_manifest(raw, root / "prepared", ImageEncoderFixture(), 2))
            self.assertEqual(dataset[0].metadata["source_video"], "same-original-video")
            from framerefine.training import Trainer
            self.assertTrue(Trainer._groups(dataset[:1])[1] & Trainer._groups(dataset[1:])[1])

    def test_uniform_sampling_is_unique_bounded_and_short_clip_aware(self):
        self.assertEqual(uniform_indices(10, 3, start=2, stop=8), [2, 4, 7])
        self.assertEqual(uniform_indices(3, 8), [0, 1, 2])
        with self.assertRaises(ValueError):
            uniform_indices(10, 3, start=8, stop=2)

    def test_frame_preparation_shares_clip_features_and_never_encodes_the_label(self):
        with tempfile.TemporaryDirectory(prefix="帧 data ") as folder:
            root = Path(folder)
            paths = []
            for i in range(6):
                path = root / f"frame {i}.png"
                Image.new("RGB", (12, 8), (i * 30, 0, 0)).save(path)
                paths.append(path.name)
            rows = [dict(id=f"q{i}", frame_paths=paths, timestamps=[0., 1., 2., 3., 4., 5.],
                         question=f"Question {i}", options=["correct text", "wrong text"], answer_index=i)
                    for i in range(2)]
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            encoder = ImageEncoderFixture()
            manifest = prepare_manifest(raw, root / "prepared", encoder, num_candidates=4)
            records = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
            self.assertEqual(records[0]["frame_feature_path"], records[1]["frame_feature_path"])
            self.assertEqual(encoder.image_calls, 1)
            self.assertEqual(encoder.questions, ["Question 0", "Question 1"])
            dataset = FeatureDataset(manifest, cache_size=2)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].frame_features.shape, (4, 3))
            torch.testing.assert_close(dataset[0].timestamps, torch.tensor([0., 1., 3., 5.]))
            self.assertTrue(all(Path(path).is_file() for path in dataset[1].frame_paths))
            self.assertNotEqual(dataset[0].query_features[0], 0)
            prepare_manifest(raw, root / "prepared", encoder, num_candidates=4)
            self.assertEqual(encoder.image_calls, 1)

    def test_modified_source_pixels_invalidate_prepared_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "one.png"
            Image.new("RGB", (8, 8), "red").save(image)
            raw = root / "raw.jsonl"
            row = dict(id="q", frame_paths=[image.name], timestamps=[2.0], question="Color?",
                       options=["red", "blue"], answer_index=0)
            raw.write_text(json.dumps(row) + "\n")
            encoder = ImageEncoderFixture()
            manifest = prepare_manifest(raw, root / "out", encoder, num_candidates=4)
            before = FeatureDataset(manifest)[0].content_revision
            Image.new("RGB", (8, 8), "blue").save(image)
            prepare_manifest(raw, root / "out", encoder, num_candidates=4)
            self.assertNotEqual(before, FeatureDataset(manifest)[0].content_revision)
            self.assertEqual(encoder.image_calls, 2)


if __name__ == "__main__":
    unittest.main()
