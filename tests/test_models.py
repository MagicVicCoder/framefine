from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn

from framerefine.data import CachedExample
from framerefine.models.common import option_token_ids, project_options, score_options
from framerefine.models.factory import build_reader
from framerefine.models.internvl import InternVLReader
from framerefine.models.llava_video import LlavaVideoReader
from framerefine.readers import Budget


class TokenizerFixture:
    unk_token_id = -1

    def encode(self, text, **kwargs):
        return [ord(text) - 65]

    def convert_tokens_to_ids(self, text):
        return 3 if text == "<IMG_CONTEXT>" else -1

    def __call__(self, text, **kwargs):
        self.prompt = text
        ids = torch.tensor([[3] * text.count("<IMG_CONTEXT>") + [2]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


class BaseFixture(nn.Module):
    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        self.inputs = inputs_embeds
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


class LanguageFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BaseFixture()
        self.embedding = nn.Embedding(4, 3)
        self.head = nn.Linear(3, 4, bias=True)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.head.weight.copy_(torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
            self.head.bias.zero_()

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


class ConversationFixture:
    roles = ("user", "assistant")

    def __init__(self):
        self.messages = []

    def append_message(self, role, text):
        self.messages.append((role, text))

    def get_prompt(self):
        return "\n".join(text or "" for role, text in self.messages)


class ModelTests(unittest.TestCase):
    def test_selected_output_projection_equals_full_logits_including_bias(self):
        torch.manual_seed(3)
        head = nn.Linear(5, 20)
        hidden = torch.randn(1, 7, 5)
        scores = project_options(head, hidden, [2, 9, 15])
        torch.testing.assert_close(scores, head(hidden)[0, -1, [2, 9, 15]])
        result = score_options(torch.tensor([1., 4., -2.]), 1)
        self.assertEqual(result.prediction, 1)
        self.assertAlmostEqual(result.utility, 4.0 - torch.logsumexp(torch.tensor([1., -2.]), 0).item(), places=6)
        self.assertEqual(option_token_ids(TokenizerFixture(), 2), [0, 1])

    def test_reader_registry_handles_toy_and_rejects_unknown_or_nonlocal_backbones(self):
        reader = build_reader({"family": "toy"}, Budget(5, 60), ":memory:")
        reader.close()
        with self.assertRaisesRegex(ValueError, "family"):
            build_reader({"family": "unknown"}, Budget(5, 60), ":memory:")
        for family in ("qwen2_5_vl", "internvl", "llava_video"):
            with self.subTest(family=family), self.assertRaises(ValueError):
                build_reader(dict(family=family, model_path="no-local-weights", revision=""), Budget(5, 60), ":memory:")

    def test_internvl_inserts_visual_tokens_and_reuses_independent_frame_features(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index in range(2):
                path = Path(folder, f"frame {index}.png")
                Image.new("RGB", (8, 8), "red").save(path)
                paths.append(str(path))
            example = CachedExample("clip", torch.zeros(2, 3), torch.ones(2), torch.tensor([1., 3.]),
                                    "Color?", ("red", "blue"), 0, "rev", tuple(paths))
            language = LanguageFixture()

            class Model:
                language_model = language
                num_image_token = 2
                conv_template = ConversationFixture()
                system_message = "system"
                calls = 0

                def extract_feature(self, pixels):
                    self.calls += 1
                    return torch.ones(len(pixels), 2, 3)

            reader = object.__new__(InternVLReader)
            reader.model, reader.tokenizer = Model(), TokenizerFixture()
            reader.device, reader.dtype, reader.image_size = "cpu", torch.float32, 8
            reader.vision_cache_frames, reader._vision_cache = 4, OrderedDict()
            result = reader.score_one(example, (0, 1))
            self.assertEqual(result.prediction, 0)
            self.assertAlmostEqual(result.utility, 8.0)
            self.assertIn("3.000", reader.tokenizer.prompt)
            self.assertEqual(reader.tokenizer.prompt.count("<IMG_CONTEXT>"), 4)
            self.assertTrue(torch.equal(language.model.inputs[0, :4], torch.ones(4, 3)))
            reader.score_one(example, (1,))
            self.assertEqual(reader.model.calls, 1)
            self.assertTrue(all(parameter.grad is None for parameter in language.parameters()))

    def test_llava_uses_video_modality_then_projects_only_option_logits(self):
        language = LanguageFixture()

        class Model:
            def get_model(self):
                return language.model

            def get_output_embeddings(self):
                return language.head

            def prepare_inputs_labels_for_multimodal(self, ids, positions, attention, past, labels,
                                                      images, modalities, image_sizes=None):
                self.images, self.modalities = images, modalities
                hidden = torch.ones(1, 3, 3)
                return None, None, torch.ones(1, 3, dtype=torch.long), None, hidden, None

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "帧 0.png")
            Image.new("RGB", (8, 8), "red").save(path)
            example = CachedExample("x", torch.ones(1, 3), torch.ones(2), torch.tensor([2.]),
                                    "Color?", ("red", "blue"), 0, "r", (str(path),))
            reader = object.__new__(LlavaVideoReader)
            reader.model, reader.tokenizer = Model(), TokenizerFixture()
            reader.device, reader.dtype = "cpu", torch.float32
            reader.conversation = ConversationFixture()
            reader.image_processor = SimpleNamespace(preprocess=lambda frames, **kwargs:
                {"pixel_values": torch.ones(len(frames), 3, 8, 8)})
            reader.image_token_index, reader.image_token = -200, "<image>"
            reader.tokenize_image = lambda *args, **kwargs: torch.tensor([2, -200, 2])
            result = reader.score_one(example, (0,))
            self.assertEqual(result.prediction, 0)
            self.assertAlmostEqual(result.utility, 6.0)
            self.assertEqual(reader.model.modalities, ["video"])
            self.assertEqual(reader.model.images[0].shape, (1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
