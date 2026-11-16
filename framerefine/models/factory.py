"""Construct frozen reader replicas from TOML settings."""

from ..readers import CachedReader, QwenLocalReader, ReplicaReader, ToyReader
from .common import local_checkpoint


def build_reader(settings, budget, cache_path):
    family = settings.get("family")
    if family == "toy":
        return CachedReader(ToyReader(), cache_path, budget)
    if family not in ("qwen2_5_vl", "internvl", "llava_video"):
        raise ValueError("reader family must be toy, qwen2_5_vl, internvl or llava_video")
    path = local_checkpoint(settings.get("model_path"), settings.get("revision"))
    revision, dtype = settings["revision"], settings.get("dtype")
    devices = settings.get("devices", ["cuda:0"])
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("reader devices must be nonempty and distinct")
    readers = []
    try:
        for device in devices:
            budget.check()
            if family == "qwen2_5_vl":
                reader = QwenLocalReader(path, revision, device, settings.get("max_pixels", 200704),
                                        settings.get("fps", 1.0), dtype=dtype)
            elif family == "internvl":
                from .internvl import InternVLReader
                reader = InternVLReader(path, revision, device, dtype,
                                        settings.get("vision_cache_frames", 64))
            else:
                from .llava_video import LlavaVideoReader
                reader = LlavaVideoReader(path, revision, device, dtype,
                    settings.get("vision_tower_path", ""), settings.get("vision_revision", ""),
                    settings.get("conversation", "qwen_1_5"))
            readers.append(reader)
    except Exception:
        for reader in readers:
            if hasattr(reader, "close"):
                reader.close()
        raise
    scorer = readers[0] if len(readers) == 1 else ReplicaReader(readers)
    return CachedReader(scorer, cache_path, budget)
