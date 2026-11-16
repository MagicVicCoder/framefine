"""Cached actor features and answer-bearing reader metadata have separate fields."""

from dataclasses import dataclass, field
from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import torch


@dataclass
class CachedExample:
    example_id: str
    frame_features: torch.Tensor
    query_features: torch.Tensor
    timestamps: torch.Tensor
    question: str
    options: tuple
    answer_index: int | None
    content_revision: str
    frame_paths: tuple = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.frame_features)
        if self.frame_features.ndim != 2 or n < 1 or self.query_features.ndim != 1:
            raise ValueError("cached features must have shapes [N,D] and [Q]")
        if (not torch.isfinite(self.frame_features).all() or not torch.isfinite(self.query_features).all()
                or self.frame_features.shape[1] == 0 or not self.query_features.numel()):
            raise ValueError("cached features must be nonempty and finite")
        if self.timestamps.shape != (n,) or not torch.isfinite(self.timestamps).all():
            raise ValueError("one finite timestamp is required per candidate")
        if (self.timestamps[1:] < self.timestamps[:-1]).any():
            raise ValueError("candidate order must follow original timestamps")
        if not self.content_revision:
            raise ValueError("a stable content revision is required for score caching")
        if self.frame_paths and len(self.frame_paths) != n:
            raise ValueError("frame path count does not match candidate count")
        if len(set(self.frame_paths)) != len(self.frame_paths):
            raise ValueError("candidate paths must identify distinct frames")
        if (not 2 <= len(self.options) <= 26 or (self.answer_index is not None and
                (not isinstance(self.answer_index, int) or not 0 <= self.answer_index < len(self.options)))):
            raise ValueError("at least two options and a valid answer index are required")


class FeatureDataset(Sequence):
    """Lazy tensor-only feature loading, with a bounded file LRU cache.

    feature_path may contain all tensors, or only query_features when a shared
    frame_feature_path contains frame_features and timestamps. IDs must be unique.
    """
    def __init__(self, path, cache_size=128):
        self.manifest = Path(path).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8-sig").splitlines()
                     if line.strip()]
        ids = [str(row["id"]) for row in self.rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("manifest is empty or contains duplicate ids")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.cache_size, self._cache = cache_size, OrderedDict()
        self.fingerprint = hashlib.sha256(json.dumps(self.rows, sort_keys=True,
                                                     ensure_ascii=False).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def _load(self, relative_path):
        path = (self.manifest.parent / relative_path).resolve()
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        value = torch.load(path, map_location="cpu", weights_only=True)
        if self.cache_size:
            self._cache[path] = value
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        row = self.rows[index]
        query = self._load(row["feature_path"])
        frames = self._load(row["frame_feature_path"]) if "frame_feature_path" in row else query
        paths = tuple(str((self.manifest.parent / p).resolve()) for p in row.get("frame_paths", []))
        return CachedExample(str(row["id"]), frames["frame_features"].float(), query["query_features"].float(),
                             frames["timestamps"].float(), row["question"], tuple(row["options"]),
                             row.get("answer_index"), row["content_revision"], paths, row.get("metadata", {}))


def load_manifest(path, cache_size=128):
    """Backward-compatible sequence interface; tensors are now loaded lazily."""
    return FeatureDataset(path, cache_size)


def feature_encoder_identity(examples):
    """Read encoder provenance without eagerly loading dataset tensors."""
    metadata = ([row.get("metadata", {}) for row in examples.rows] if hasattr(examples, "rows")
                else [example.metadata for example in examples])
    identities = {json.dumps(item.get("feature_encoder"), sort_keys=True) for item in metadata}
    if len(identities) != 1:
        raise ValueError("a dataset must use one consistent feature encoder")
    return json.loads(identities.pop())


def toy_examples(count=4, n=8, feature_dim=8, query_dim=4, seed=7):
    generator = torch.Generator().manual_seed(seed)
    examples = []
    for index in range(count):
        frames = torch.randn(n, feature_dim, generator=generator)
        query = torch.randn(query_dim, generator=generator)
        target = torch.randperm(n, generator=generator)[:min(3, n)].tolist()
        frames[:, 0] = 0
        frames[target, 0] = 1
        revision = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
        examples.append(CachedExample(
            f"toy-{seed}-{index}", frames, query, torch.arange(n).float(),
            "Which event is supported by the selected frames?", ("target", "other"),
            0, revision, metadata={"toy_target": target}))
    return examples
