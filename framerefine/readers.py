"""Frozen scorers, replica fan-out, and cache/budget boundaries.

A request is one subset score, not necessarily one CUDA forward. All reserved
misses count toward the cap, including failed batches. Limits are checked before
each model call; an in-flight model call is not forcibly terminated.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3
import time

import torch


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ReaderScore:
    utility: float
    prediction: int | None = None


@dataclass
class Budget:
    max_requests: int
    wall_seconds: float
    requests: int = 0
    started: float = field(default_factory=time.monotonic)
    parent: "Budget | None" = None

    def check(self, additional=0):
        if self.parent is not None:
            self.parent.check(additional)
        if time.monotonic() - self.started >= self.wall_seconds:
            raise BudgetExceeded("training wall-time budget reached")
        if self.requests + additional > self.max_requests:
            raise BudgetExceeded("frozen-reader subset-score budget would be exceeded")

    def reserve(self, additional):
        self.check(additional)
        if self.parent is not None:
            self.parent.reserve(additional)
        self.requests += additional


class ToyReader:
    """Analytic testing fixture, never evidence of real video performance."""
    identity = {"model": "analytic-toy", "revision": "1", "scoring": "fixture-v1"}

    def score_one(self, example, indices):
        chosen = set(indices)
        target = set(example.metadata.get("toy_target", []))
        unary = 0.15 * sum(float(example.frame_features[i, 0]) for i in chosen)
        complete = bool(target and target.issubset(chosen))
        value = unary + (2.0 if complete else 0.0) - 1.0
        return ReaderScore(value if example.answer_index == 0 else -value, 0 if complete else 1)


class ReplicaReader:
    """Concurrent frozen replicas, normally one model instance per CUDA device.

    Each replica handles only one serial chunk at a time. Results retain request
    order. The wrapper is intended to be called serially by the actor process.
    """
    def __init__(self, readers):
        if not readers or any(reader.identity != readers[0].identity for reader in readers):
            raise ValueError("replicas must have identical model/processor/scoring identities")
        self.readers = list(readers)
        self.identity = readers[0].identity
        self.executor = ThreadPoolExecutor(max_workers=len(readers))

    def score_many(self, example, indices, before_score=None):
        chunks = [[] for _ in self.readers]
        for position, subset in enumerate(indices):
            chunks[position % len(chunks)].append((position, subset))

        def run(reader, chunk):
            result = []
            with torch.inference_mode():
                for position, subset in chunk:
                    if before_score is not None:
                        before_score()
                    result.append((position, reader.score_one(example, subset)))
            return result

        futures = [self.executor.submit(run, reader, chunk)
                   for reader, chunk in zip(self.readers, chunks) if chunk]
        # Drain all in-flight work before propagating a worker error. A caller
        # must never reuse a replica while a previous request is still running.
        wait(futures)
        result = [None] * len(indices)
        for future in futures:
            for position, value in future.result():
                result[position] = value
        return result

    def close(self):
        self.executor.shutdown(wait=True)


class CachedReader:
    def __init__(self, scorer, cache_path, budget, owns_scorer=True):
        self.scorer, self.budget = scorer, budget
        self.cache_path, self.owns_scorer, self.owns_db = str(cache_path), owns_scorer, True
        if str(cache_path) != ":memory:":
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value REAL NOT NULL, prediction INTEGER)")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(scores)")}
        if "prediction" not in columns:
            self.db.execute("ALTER TABLE scores ADD COLUMN prediction INTEGER")
        self.cache_hits = 0
        self.scoring_seconds = 0.0

    def _key(self, example, indices):
        payload = {"schema": 2, "reader": self.scorer.identity,
                   "id": example.example_id, "question": example.question,
                   "options": list(example.options), "answer_index": example.answer_index,
                   "content_revision": example.content_revision,
                   "frames": [(i, example.frame_paths[i] if example.frame_paths else None,
                               float(example.timestamps[i])) for i in indices]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def score(self, example, masks):
        return torch.tensor([item.utility for item in self.score_details(example, masks)], dtype=torch.float32)

    def score_details(self, example, masks):
        self.budget.check()
        if example.answer_index is None:
            raise ValueError("frozen-reader utilities require a labeled answer_index")
        masks = torch.as_tensor(masks).detach().cpu()
        if (masks.ndim != 2 or masks.shape[1] != len(example.frame_features)
                or not ((masks == 0) | (masks == 1)).all()):
            raise ValueError("reader masks must be binary [subsets, candidates]")
        keys, missing, known = [], {}, {}
        for mask in masks:
            indices = tuple(mask.nonzero().flatten().tolist())
            key = self._key(example, indices)
            keys.append(key)
            if key in known or key in missing:
                self.cache_hits += 1
                continue
            row = self.db.execute("SELECT value, prediction FROM scores WHERE key=?", (key,)).fetchone()
            if row is not None:
                known[key] = ReaderScore(float(row[0]), row[1])
                self.cache_hits += 1
            else:
                missing[key] = indices
        self.budget.check(len(missing))
        if missing:
            self.budget.reserve(len(missing))
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    if hasattr(self.scorer, "score_many"):
                        values = self.scorer.score_many(example, list(missing.values()),
                                                        before_score=self.budget.check)
                    else:
                        values = []
                        for indices in missing.values():
                            self.budget.check()
                            values.append(self.scorer.score_one(example, indices))
            finally:
                self.scoring_seconds += time.perf_counter() - started
            values = [v if isinstance(v, ReaderScore) else ReaderScore(float(v)) for v in values]
            if (len(values) != len(missing) or any(not math.isfinite(v.utility) for v in values)
                    or any(v.prediction is not None and (not isinstance(v.prediction, int)
                           or not 0 <= v.prediction < len(example.options)) for v in values)):
                raise ValueError("frozen reader returned missing or nonfinite utilities")
            for key, value in zip(missing, values):
                known[key] = value
                self.db.execute("INSERT OR REPLACE INTO scores (key,value,prediction) VALUES (?,?,?)",
                                (key, float(value.utility), value.prediction))
            self.db.commit()
        return [known[key] for key in keys]

    def fork(self, budget):
        child = CachedReader(self.scorer, self.cache_path, budget, owns_scorer=False)
        if self.cache_path == ":memory:":
            child.db.close()
            child.db, child.owns_db = self.db, False
        return child

    def close(self):
        if self.owns_db:
            self.db.close()
        if self.owns_scorer and hasattr(self.scorer, "close"):
            self.scorer.close()


class QwenLocalReader:
    """Optional Qwen2.5-VL scorer using existing LOCAL weights only.

    Uses video-frame-list input and one next-token option-letter logit margin.
    Actual timestamps are additionally written in the prompt; native video mRoPE
    uses the configured nominal FPS. This is not an arbitrary-timestamp mRoPE
    implementation. Keep this convention identical for dense/selected baselines.
    """
    def __init__(self, model_path, revision, device="cuda:0", max_pixels=200704, fps=1.0, dtype=None):
        from .models.common import local_checkpoint, resolve_dtype
        model_path = local_checkpoint(model_path, revision)
        if not math.isfinite(fps) or fps <= 0 or max_pixels < 1:
            raise ValueError("positive finite fps and max_pixels are required")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError("install the optional qwen dependencies; no model is downloaded automatically") from error
        self.device, self.fps, self.max_pixels = device, fps, max_pixels
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=resolve_dtype(dtype, device),
            device_map={"": device}, attn_implementation="sdpa").eval().requires_grad_(False)
        parameters = inspect.signature(self.model.forward).parameters
        self.logit_argument = next((name for name in ("logits_to_keep", "num_logits_to_keep")
                                    if name in parameters), None)
        if self.logit_argument is None:
            raise RuntimeError("this transformers version cannot request last-token-only logits")
        self.identity = dict(model=str(Path(model_path).resolve()), revision=revision,
                             scoring="option-letter-logodds-v1", prompt="timestamped-video-v1",
                             max_pixels=max_pixels, nominal_fps=fps,
                             dtype=str(resolve_dtype(dtype, device)), family="qwen2_5_vl")

    def score_one(self, example, indices):
        if len(example.options) > 26 or not example.frame_paths:
            raise ValueError("Qwen scorer requires frame paths and 2..26 answer options")
        letters = [chr(65 + i) for i in range(len(example.options))]
        token_ids = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError("option letters must each be a single token for this scorer")
        content = []
        if indices:
            content.append(dict(type="video", video=[str(Path(example.frame_paths[i]).resolve()) for i in indices],
                                sample_fps=self.fps, max_pixels=self.max_pixels))
        timestamps = ", ".join(f"{float(example.timestamps[i]):.3f}" for i in indices)
        options = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, example.options))
        prompt = (f"{example.question}\n{options}\nSelected frame timestamps in seconds, in order: "
                  f"{timestamps or 'none'}.\nAnswer with the option letter only.")
        content.append(dict(type="text", text=prompt))
        messages = [dict(role="user", content=content)]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, kwargs = self.process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt", **kwargs).to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, **{self.logit_argument: 1})
            scores = output.logits[0, -1, [ids[0] for ids in token_ids]].float()
            correct = scores[example.answer_index]
            others = torch.cat([scores[:example.answer_index], scores[example.answer_index+1:]])
            return ReaderScore(float(correct - others.logsumexp(0)), int(scores.argmax()))
