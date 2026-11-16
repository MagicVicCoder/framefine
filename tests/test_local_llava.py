"""Exercise local constructor dispatch, including upstream's known path trap."""

import builtins
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from framerefine.models.llava_video import LlavaVideoReader


@contextmanager
def local_upstream(forbid_conversation=False):
    names = ["transformers", "llava", "llava.model", "llava.model.llava_arch",
             "llava.model.language_model", "llava.model.language_model.llava_qwen",
             "llava.model.multimodal_encoder", "llava.model.multimodal_encoder.siglip_encoder",
             "llava.constants", "llava.mm_utils", "llava.conversation"]
    modules = {name: ModuleType(name) for name in names}
    state = SimpleNamespace(towers=[], prompts=[])
    architecture = modules["llava.model.llava_arch"]

    def original_builder(config, **kwargs):
        if Path(config.mm_vision_tower).exists():
            raise RuntimeError("upstream incorrectly selected CLIP for a local SigLIP directory")
        raise RuntimeError("unexpected remote vision tower")

    architecture.build_vision_tower = original_builder
    modules["llava.model"].llava_arch = architecture

    class Tower(nn.Module):
        def __init__(self, path, vision_tower_cfg, delay_load=False):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.is_loaded = not delay_load
            self.image_processor = object()
            state.towers.append((path, self))

        def load_model(self, device_map=None):
            self.is_loaded = True

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.tower = architecture.build_vision_tower(config, delay_load=getattr(config, "delay_load", False))

        @classmethod
        def from_pretrained(cls, path, config, **kwargs):
            return cls(config)

        def get_vision_tower(self):
            return self.tower

    modules["llava.model.language_model.llava_qwen"].LlavaQwenConfig = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(mm_vision_tower="remote"))
    modules["llava.model.language_model.llava_qwen"].LlavaQwenForCausalLM = Model
    modules["llava.model.multimodal_encoder.siglip_encoder"].SigLipVisionTower = Tower
    modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    modules["llava.constants"].IMAGE_TOKEN_INDEX = -200
    modules["llava.constants"].DEFAULT_IMAGE_TOKEN = "<image>"
    modules["llava.mm_utils"].tokenizer_image_token = lambda *args, **kwargs: torch.tensor([1, -200, 2])
    modules["llava.conversation"].conv_templates = {"qwen_1_5": object()}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if forbid_conversation and name == "llava.conversation":
            raise AssertionError("conversation import would initialize an unrelated remote tokenizer")
        return original_import(name, *args, **kwargs)

    with patch.dict(sys.modules, modules), patch("builtins.__import__", side_effect=guarded_import):
        yield state, architecture, original_builder


class LocalLlavaTests(unittest.TestCase):
    def test_existing_local_tower_uses_siglip_and_restores_upstream_dispatch(self):
        with tempfile.TemporaryDirectory() as folder, local_upstream() as (state, architecture, original):
            root = Path(folder)
            model, tower = root / "language", root / "siglip-local"
            model.mkdir()
            tower.mkdir()
            reader = LlavaVideoReader(str(model), "r1", "cpu", "float32", str(tower), "v1")
            self.assertEqual(state.towers[0][0], str(tower.resolve()))
            self.assertTrue(reader.model.tower.is_loaded)
            self.assertTrue(all(not p.requires_grad for p in reader.model.parameters()))
            self.assertIs(architecture.build_vision_tower, original)

    def test_local_reader_does_not_import_remote_tokenizer_conversation(self):
        with tempfile.TemporaryDirectory() as folder, local_upstream(True):
            root = Path(folder)
            model, tower = root / "language", root / "siglip-local"
            model.mkdir()
            tower.mkdir()
            LlavaVideoReader(str(model), "r1", "cpu", "float32", str(tower), "v1")


if __name__ == "__main__":
    unittest.main()
