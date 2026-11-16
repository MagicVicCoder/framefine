from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from framerefine.data import CachedExample
from framerefine.readers import Budget, BudgetExceeded, CachedReader, QwenLocalReader, ReplicaReader, ToyReader


def example():
    return CachedExample("clip", torch.zeros(6, 4), torch.zeros(3), torch.arange(6.),
                         question="Which event?", options=("one", "two"), answer_index=0,
                         content_revision="pixels-v1", metadata={"toy_target": [0, 1, 2]})


class ReaderTests(unittest.TestCase):
    def test_cache_deduplicates_masks_and_invalidates_changed_label_or_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = CachedReader(ToyReader(), folder + "/scores.sqlite", Budget(20, 60))
            masks = torch.tensor([[1., 1., 1., 0., 0., 0.], [1., 1., 1., 0., 0., 0.]])
            values = reader.score(example(), masks)
            self.assertEqual(values[0], values[1])
            self.assertEqual(reader.budget.requests, 1)
            reader.score(example(), masks)
            self.assertEqual(reader.budget.requests, 1)
            reader.score(replace(example(), answer_index=1), masks)
            self.assertEqual(reader.budget.requests, 2)
            reader.score(replace(example(), content_revision="pixels-v2"), masks)
            self.assertEqual(reader.budget.requests, 3)
            reader.close()

    def test_query_cap_is_checked_before_evaluating_a_batch(self):
        reader = CachedReader(ToyReader(), ":memory:", Budget(1, 60))
        masks = torch.tensor([[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]])
        with self.assertRaises(BudgetExceeded):
            reader.score(example(), masks)
        self.assertEqual(reader.budget.requests, 0)
        reader.close()

    def test_expired_wall_time_is_checked_even_on_cache_hits(self):
        reader = CachedReader(ToyReader(), ":memory:", Budget(10, 60))
        masks = torch.tensor([[1., 1., 1., 0., 0., 0.]])
        reader.score(example(), masks)
        reader.budget.wall_seconds = 0
        with self.assertRaises(BudgetExceeded):
            reader.score(example(), masks)
        reader.close()

    def test_replica_results_preserve_mask_order(self):
        scorer = ReplicaReader([ToyReader(), ToyReader()])
        indices = [(0, 1, 2), (3, 4, 5), (0, 3, 5)]
        expected = [ToyReader().score_one(example(), item) for item in indices]
        self.assertEqual(scorer.score_many(example(), indices), expected)
        scorer.close()

    def test_cached_details_retain_predictions_without_extra_scoring(self):
        reader = CachedReader(ToyReader(), ":memory:", Budget(10, 60))
        masks = torch.tensor([[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]])
        reader.score(example(), masks)
        requests = reader.budget.requests
        details = reader.score_details(example(), masks)
        self.assertEqual([item.prediction for item in details], [0, 1])
        self.assertEqual(reader.budget.requests, requests)
        reader.close()

    def test_wall_time_is_checked_between_serial_calls_and_replica_chunk_calls(self):
        masks = torch.tensor([[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.],
                              [1., 0., 1., 0., 1., 0.]])
        for replica in (False, True):
            with self.subTest(replica=replica):
                clock = [0.0]

                class SlowReader(ToyReader):
                    calls = 0

                    def score_one(self, item, indices):
                        self.calls += 1
                        clock[0] += 0.035
                        return super().score_one(item, indices)

                scorer = SlowReader()
                wrapped = ReplicaReader([scorer]) if replica else scorer
                reader = CachedReader(wrapped, ":memory:", Budget(10, 0.025, started=0.0))
                try:
                    with patch("framerefine.readers.time.monotonic", side_effect=lambda: clock[0]):
                        with self.assertRaises(BudgetExceeded):
                            reader.score(example(), masks)
                    self.assertEqual(scorer.calls, 1)
                    self.assertEqual(reader.budget.requests, 3)  # conservatively reserved attempts
                finally:
                    reader.close()

    def test_qwen_frame_list_preserves_plain_paths_and_nominal_sample_fps(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            tokenizer = SimpleNamespace(encode=lambda letter, **kwargs: [ord(letter) - 65])

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return Inputs()

        reader = object.__new__(QwenLocalReader)
        reader.device, reader.fps, reader.max_pixels = "cpu", 1.25, 200704
        reader.logit_argument = "logits_to_keep"
        reader.processor = Processor()
        reader.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[[2., 0.]]]))
        with tempfile.TemporaryDirectory(prefix="帧 cache ") as folder:
            paths = tuple(str(Path(folder, f"帧 {i}.jpg").resolve()) for i in range(6))

            def vision_info(messages, **kwargs):
                video = messages[0]["content"][0]
                self.assertEqual(video["video"], [paths[1], paths[4]])
                self.assertEqual(video.get("sample_fps"), 1.25)
                return None, [object()], {"fps": [video["sample_fps"]]}

            reader.process_vision_info = vision_info
            result = reader.score_one(replace(example(), frame_paths=paths), (1, 4))
            self.assertEqual(result.prediction, 0)
            self.assertAlmostEqual(result.utility, 2.0)
            self.assertEqual(reader.processor.kwargs["fps"], [1.25])


if __name__ == "__main__":
    unittest.main()
