"""LLaVA-Video-7B-Qwen2 adapter for the official local LLaVA-NeXT package.

Load the language checkpoint and SigLIP tower from explicit local directories.
Use native multimodal preparation, then a last-state option-only projection.
"""

from contextlib import contextmanager
from pathlib import Path
import threading

from PIL import Image
import torch

from .common import (local_checkpoint, option_token_ids, project_options,
                     question_prompt, resolve_dtype, score_options)


_VISION_CONSTRUCTION_LOCK = threading.RLock()


@contextmanager
def _local_siglip_dispatch(architecture, tower_class, local_path):
    """Work around upstream dispatching existing SigLIP paths as CLIP.

    Only this exact snapshot is redirected, during serial model construction.
    Other tower requests keep their upstream behavior, and restoration is
    guaranteed even when loading fails. No source checkout is modified.
    """
    with _VISION_CONSTRUCTION_LOCK:
        original = architecture.build_vision_tower

        def build(config, **kwargs):
            requested = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
            if str(Path(requested).resolve()) == local_path:
                return tower_class(local_path, vision_tower_cfg=config, **kwargs)
            return original(config, **kwargs)

        architecture.build_vision_tower = build
        try:
            yield
        finally:
            architecture.build_vision_tower = original


def _qwen_prompt(question):
    # Native qwen_1_5 CHATML layout, rendered without importing conversation.py:
    # upstream conversation.py initializes an unrelated Meta-Llama tokenizer.
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")


class LlavaVideoReader:
    def __init__(self, model_path, revision, device="cuda:0", dtype=None,
                 vision_tower_path="", vision_revision="", conversation="qwen_1_5"):
        model_path = local_checkpoint(model_path, revision)
        vision_tower_path = local_checkpoint(vision_tower_path, vision_revision)
        if conversation != "qwen_1_5":
            raise ValueError("LLaVA-Video-Qwen2 requires conversation=qwen_1_5")
        try:
            from transformers import AutoTokenizer
            from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
            from llava.model import llava_arch
            from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.mm_utils import tokenizer_image_token
        except ImportError as error:
            raise RuntimeError("install the llava extra and the official local LLaVA-NeXT package") from error
        self.device, self.dtype = device, resolve_dtype(dtype, device)
        config = LlavaQwenConfig.from_pretrained(model_path, local_files_only=True)
        config.mm_vision_tower = vision_tower_path
        config.delay_load = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        with _local_siglip_dispatch(llava_arch, SigLipVisionTower, vision_tower_path):
            self.model = LlavaQwenForCausalLM.from_pretrained(model_path, config=config,
                local_files_only=True, device_map={"": device}, torch_dtype=self.dtype, attn_implementation="sdpa")
        tower = self.model.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model(device_map={"": device})
        tower.to(device=device, dtype=self.dtype)
        self.model.eval().requires_grad_(False)
        self.image_processor = tower.image_processor
        self.image_token_index, self.image_token = IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self.tokenize_image = tokenizer_image_token
        self.identity = dict(model=model_path, revision=revision, family="llava_video_qwen2",
                             vision_tower=vision_tower_path, vision_revision=vision_revision,
                             conversation=conversation, scoring="option-letter-logodds-v1",
                             prompt="timestamped-video-v1", vision_dispatch="local-siglip-v1", dtype=str(self.dtype))

    def score_one(self, example, indices):
        if len(indices) > 64:
            raise ValueError("LLaVA-Video-7B-Qwen2 supports at most 64 selected frames")
        ids = option_token_ids(self.tokenizer, len(example.options))
        if indices and not example.frame_paths:
            raise ValueError("LLaVA-Video needs local candidate frame_paths")
        question = (self.image_token + "\n" if indices else "") + question_prompt(example, indices)
        input_ids = self.tokenize_image(_qwen_prompt(question), self.tokenizer,
                                       self.image_token_index, return_tensors="pt").unsqueeze(0).to(self.device)
        images = None
        if indices:
            frames = []
            for index in indices:
                with Image.open(example.frame_paths[index]) as image:
                    frames.append(image.convert("RGB"))
            pixels = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            images = [pixels.to(device=self.device, dtype=self.dtype)]
        with torch.inference_mode():
            prepared = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids), None, None, images, ["video"], image_sizes=None)
            input_ids, positions, attention, _, embeddings, _ = prepared
            hidden = self.model.get_model()(input_ids=input_ids, position_ids=positions, attention_mask=attention,
                inputs_embeds=embeddings, use_cache=False, return_dict=True).last_hidden_state
            return score_options(project_options(self.model.get_output_embeddings(), hidden, ids), example.answer_index)
