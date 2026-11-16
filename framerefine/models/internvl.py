"""InternVL 2.5 local adapter with one image tile per frame and bounded ViT caching.

Matches the official <img><IMG_CONTEXT>...</img> video-frame convention.
The frozen base LM returns hidden states; only option vocabulary rows are
projected. This avoids allocating sequence-length x vocabulary logits.
"""

from collections import OrderedDict
import copy
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import (local_checkpoint, option_token_ids, project_options,
                     question_prompt, resolve_dtype, score_options)


class InternVLReader:
    def __init__(self, model_path, revision, device="cuda:0", dtype=None, vision_cache_frames=64):
        model_path = local_checkpoint(model_path, revision)
        if vision_cache_frames < 0:
            raise ValueError("vision_cache_frames must be nonnegative")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the internvl extra and local model custom-code files") from error
        self.device, self.dtype = device, resolve_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                                        local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                               local_files_only=True, torch_dtype=self.dtype,
                                               use_flash_attn=False, device_map={"": device})
        self.model.eval().requires_grad_(False)
        self.image_size = int(self.model.config.force_image_size or self.model.config.vision_config.image_size)
        if not hasattr(self.model.language_model, "model"):
            raise ValueError("InternVL adapter requires the InternVL2.5 base-LM model interface")
        self.vision_cache_frames, self._vision_cache = int(vision_cache_frames), OrderedDict()
        self.identity = dict(model=model_path, revision=revision, family="internvl2_5",
                             scoring="option-letter-logodds-v1", prompt="timestamped-one-tile-v1",
                             image_size=self.image_size, dtype=str(self.dtype))

    def _visual(self, paths):
        keys = [(str(Path(path).resolve()), Path(path).stat().st_mtime_ns, Path(path).stat().st_size)
                for path in paths]
        values, missing = {}, []
        for key in keys:
            if key in self._vision_cache:
                values[key] = self._vision_cache[key]
                self._vision_cache.move_to_end(key)
            elif key not in missing:
                missing.append(key)
        if missing:
            pixels = []
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            for key in missing:
                with Image.open(key[0]) as image:
                    image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
                    pixels.append((tensor - mean) / std)
            features = self.model.extract_feature(torch.stack(pixels).to(self.device, dtype=self.dtype))
            for key, feature in zip(missing, features):
                values[key] = feature.detach()
                if self.vision_cache_frames:
                    self._vision_cache[key] = values[key]
                    while len(self._vision_cache) > self.vision_cache_frames:
                        self._vision_cache.popitem(last=False)
        return torch.cat([values[key] for key in keys], dim=0)

    def score_one(self, example, indices):
        ids = option_token_ids(self.tokenizer, len(example.options))
        if indices and not example.frame_paths:
            raise ValueError("InternVL needs local candidate frame_paths")
        context_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if context_id is None or context_id == getattr(self.tokenizer, "unk_token_id", None):
            raise ValueError("tokenizer lacks InternVL image context tokens")
        self.model.img_context_token_id = context_id
        prefix = "".join(f"Frame {j + 1} at {float(example.timestamps[i]):.3f}s: <image>\n"
                         for j, i in enumerate(indices))
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prefix + question_prompt(example, indices))
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>"
        for _ in indices:
            prompt = prompt.replace("<image>", image_tokens, 1)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention = inputs["attention_mask"].to(self.device)
        with torch.inference_mode():
            language = self.model.language_model
            embeddings = language.get_input_embeddings()(input_ids).clone()
            if indices:
                visual = self._visual([example.frame_paths[i] for i in indices]).to(embeddings)
                selected = input_ids == context_id
                if int(selected.sum()) != len(visual):
                    raise ValueError("InternVL visual-token count does not match image features")
                embeddings[selected] = visual
            hidden = language.model(inputs_embeds=embeddings, attention_mask=attention,
                                    use_cache=False, return_dict=True).last_hidden_state
            return score_options(project_options(language.get_output_embeddings(), hidden, ids), example.answer_index)

    def close(self):
        self._vision_cache.clear()
