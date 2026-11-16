"""Local video/frame extraction, content-addressed caching and frozen CLIP features."""

import hashlib
import json
import math
from pathlib import Path
import re
import time

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from .annotations import read_raw_manifest, write_jsonl


def uniform_indices(total, count, start=0, stop=None):
    stop = total if stop is None else stop
    if total < 1 or count < 1 or not 0 <= start < stop <= total:
        raise ValueError("invalid frame count or temporal index bounds")
    count = min(int(count), stop - start)
    if count == 1:
        return [start]
    return [start + i * (stop - start - 1) // (count - 1) for i in range(count)]


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _frame_paths(row):
    if "frame_paths" in row:
        return [Path(item) for item in row["frame_paths"]]
    def order(path):
        return [(0, int(part)) if part.isdigit() else (1, part.lower())
                for part in re.split(r"(\d+)", path.name)]
    paths = [path for path in Path(row["frame_dir"]).iterdir()
             if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
    return sorted(paths, key=order)


def _extract(row, count):
    start, end = row.get("start", 0), row.get("end", float("inf"))
    if "video_path" not in row:
        paths = _frame_paths(row)
        stamps = row.get("timestamps", [i / row.get("fps", 1) for i in range(len(paths))])
        eligible = [i for i, stamp in enumerate(stamps) if start <= stamp < end]
        if not eligible:
            raise ValueError("no frames remain in the requested temporal interval")
        indices = [eligible[i] for i in uniform_indices(len(eligible), count)]
        images = []
        for index in indices:
            with Image.open(paths[index]) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        return images, torch.tensor([stamps[i] for i in indices], dtype=torch.float32)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("video decoding needs the video extra: pip install -e '.[video]'") from error
    capture = cv2.VideoCapture(row["video_path"])
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or total < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"cannot decode a finite video timeline: {row['video_path']}")
        lo = max(0, math.ceil(start * fps))
        hi = min(total, math.ceil(end * fps)) if math.isfinite(end) else total
        indices = uniform_indices(total, count, lo, hi)
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video decoding failed at original frame {index}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return images, torch.tensor([index / fps for index in indices], dtype=torch.float32)
    finally:
        capture.release()


class FrozenCLIPEncoder:
    """Shared frozen image/text encoder. No answer/options are sent to it."""
    def __init__(self, model_path, revision, device="cuda:0", batch_size=32):
        if not Path(model_path).is_dir() or not revision or batch_size < 1:
            raise ValueError("CLIP requires local weights, immutable revision and positive batch_size")
        try:
            from transformers import CLIPModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("install the features extra before preparing CLIP features") from error
        self.device, self.batch_size = device, batch_size
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval().requires_grad_(False)
        self.identity = dict(model=str(Path(model_path).resolve()), revision=revision,
                             encoder="clip-normalized-v1", text="question-only-truncate-v1")

    def encode_frames(self, paths):
        values = []
        with torch.inference_mode():
            for offset in range(0, len(paths), self.batch_size):
                images = []
                for path in paths[offset:offset + self.batch_size]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
                values.append(F.normalize(self.model.get_image_features(pixel_values=pixels).float(), dim=-1).cpu())
        return torch.cat(values)

    def encode_question(self, question):
        inputs = self.processor(text=[question], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return F.normalize(self.model.get_text_features(**inputs).float(), dim=-1)[0].cpu()


def prepare_manifest(raw_manifest, output_dir, encoder, num_candidates=32, max_side=448,
                     command_started_at=None, encoder_startup_seconds=0.0):
    if num_candidates < 1 or max_side < 1:
        raise ValueError("candidate count and cache image size must be positive")
    started = time.perf_counter()
    command_started_at = started if command_started_at is None else command_started_at
    rows, output = read_raw_manifest(raw_manifest), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources, clips, prepared = {}, {}, []
    for row in rows:
        source_paths = [Path(row["video_path"])] if "video_path" in row else _frame_paths(row)
        if not source_paths:
            raise ValueError("frame directory contains no images")
        hashes = []
        for path in source_paths:
            path = path.resolve()
            if path not in sources:
                sources[path] = _sha_file(path)
            hashes.append(sources[path])
        key = _digest(dict(source_hashes=hashes, timestamps=row.get("timestamps"), fps=row.get("fps"),
                           start=row.get("start"), end=row.get("end"), encoder=encoder.identity,
                           num_candidates=num_candidates, max_side=max_side, codec="jpeg95-v1"))
        if key not in clips:
            directory = output / "clips" / key
            metadata_path = directory / "clip.json"
            ready = None
            if metadata_path.exists():
                candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
                paths = [output / path for path in candidate["frame_paths"]]
                if (all(path.is_file() for path in paths) and (directory / "features.pt").is_file()
                        and candidate["pixel_hashes"] == [_sha_file(path) for path in paths]):
                    ready = candidate
            if ready is None:
                images, stamps = _extract(row, num_candidates)
                directory.mkdir(parents=True, exist_ok=True)
                paths = []
                for index, image in enumerate(images):
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    path = directory / f"{index:06d}.jpg"
                    image.save(path, quality=95)
                    paths.append(path)
                frame_features = torch.as_tensor(encoder.encode_frames(paths)).detach().float().cpu()
                if frame_features.ndim != 2 or len(frame_features) != len(paths) or not torch.isfinite(frame_features).all():
                    raise ValueError("feature encoder returned invalid frame features")
                torch.save(dict(frame_features=frame_features, timestamps=stamps), directory / "features.pt")
                pixel_hashes = [_sha_file(path) for path in paths]
                ready = dict(frame_paths=[path.relative_to(output).as_posix() for path in paths],
                             pixel_hashes=pixel_hashes,
                             content_revision=_digest(dict(pixels=pixel_hashes, timestamps=stamps.tolist())),
                             frame_feature_path=(directory / "features.pt").relative_to(output).as_posix())
                metadata_path.write_text(json.dumps(ready, indent=2), encoding="utf-8")
            clips[key] = ready
        clip = clips[key]
        question_key = _digest(dict(question=row["question"], encoder=encoder.identity))
        query_path = output / "questions" / (question_key + ".pt")
        if not query_path.exists():
            query_path.parent.mkdir(parents=True, exist_ok=True)
            query = torch.as_tensor(encoder.encode_question(row["question"])).detach().float().cpu()
            if query.ndim != 1 or not query.numel() or not torch.isfinite(query).all():
                raise ValueError("feature encoder returned invalid question features")
            torch.save(dict(query_features=query), query_path)
        prepared.append(dict(id=row["id"], question=row["question"], options=row["options"],
                             answer_index=row["answer_index"], content_revision=clip["content_revision"],
                             feature_path=query_path.relative_to(output).as_posix(),
                             frame_feature_path=clip["frame_feature_path"], frame_paths=clip["frame_paths"],
                             metadata={**row["metadata"], "feature_encoder": encoder.identity,
                                       "source_video": (row["metadata"].get("source_video")
                                           or row.get("video_path", row.get("frame_dir",
                                               _digest(row.get("frame_paths", []))))),
                                       "feature_revision": key + ":" + question_key}))
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, prepared)
    ended = time.perf_counter()
    report = dict(examples=len(rows), unique_clips=len(clips), encoder=encoder.identity,
                  preprocessing_seconds=ended - started,
                  encoder_startup_seconds=encoder_startup_seconds,
                  command_seconds=ended - command_started_at, num_candidates=num_candidates,
                  max_side=max_side, manifest=str(manifest))
    (output / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "preparation.history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
    return manifest
