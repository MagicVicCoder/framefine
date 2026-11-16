"""Convert local multiple-choice annotations into one explicit raw-video schema."""

import json
import math
from pathlib import Path
import re


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _label(value, original, cleaned):
    if isinstance(value, bool):
        raise ValueError("boolean answer labels are ambiguous")
    if isinstance(value, int) and 0 <= value < len(original):
        return value
    text = str(value).strip()
    if text in original:
        return original.index(text)
    if text in cleaned:
        return cleaned.index(text)
    match = re.fullmatch(r"\(?([A-Z])\)?[.)]?", text)
    if match and ord(match[1]) - 65 < len(original):
        return ord(match[1]) - 65
    raise ValueError(f"answer {value!r} does not identify one of the choices")


def _resolve(root, path):
    return str((Path(root) / str(path)).resolve())


def _canonical(row, root):
    original = row.get("options", row.get("candidates"))
    if not isinstance(original, (tuple, list)) or not 2 <= len(original) <= 26:
        raise ValueError("multiple-choice records require 2..26 options")
    original = [str(item).strip() for item in original]
    choices = [re.sub(rf"^(?:{chr(65+i)}[.)]|\({chr(65+i)}\))\s*", "", item).strip()
               for i, item in enumerate(original)]
    if any(not item for item in choices):
        raise ValueError("options must not be empty")
    value = row.get("answer_index", row.get("answer"))
    answer = None if value is None else _label(value, original, choices)
    result = dict(id=str(row["id"]), question=str(row["question"]).strip(), options=choices,
                  answer_index=answer, metadata=dict(row.get("metadata", {})))
    if not result["id"] or not result["question"]:
        raise ValueError("record id and question must not be empty")
    sources = [key for key in ("video_path", "frame_paths", "frame_dir") if row.get(key)]
    if len(sources) != 1:
        raise ValueError("supply exactly one of video_path, frame_paths, or frame_dir")
    source = sources[0]
    if source == "frame_paths":
        result[source] = [_resolve(root, item) for item in row[source]]
        if len(set(result[source])) != len(result[source]):
            raise ValueError("frame_paths must identify distinct real frames")
        if "timestamps" in row:
            stamps = [float(value) for value in row["timestamps"]]
            if (len(stamps) != len(result[source]) or any(not math.isfinite(t) for t in stamps)
                    or any(a > b for a, b in zip(stamps, stamps[1:]))):
                raise ValueError("timestamps must be finite, ordered and match frames")
            result["timestamps"] = stamps
    else:
        result[source] = _resolve(root, row[source])
    if "fps" in row:
        if not math.isfinite(float(row["fps"])) or float(row["fps"]) <= 0:
            raise ValueError("fps must be finite and positive")
        result["fps"] = float(row["fps"])
    if source != "video_path" and "timestamps" not in result and "fps" not in result:
        raise ValueError("frame sources need timestamps or their original fps")
    start, end = float(row.get("start", 0.0)), row.get("end")
    if not math.isfinite(start) or start < 0 or (end is not None and
            (not math.isfinite(float(end)) or float(end) <= start)):
        raise ValueError("invalid temporal start/end interval")
    result["start"] = start
    if end is not None:
        result["end"] = float(end)
    return result


def read_raw_manifest(path):
    path = Path(path).resolve()
    rows = [_canonical(json.loads(line), path.parent)
            for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("raw manifest is empty or contains duplicate ids")
    return rows


def convert_annotations(input_path, output_path, format="jsonl", video_root=None,
                        video_prefix="", task=None, fps=None):
    """Convert a single benchmark JSON file or a canonical raw JSONL manifest.

    video_root/video_prefix select the actual local directory for this task; no
    benchmark-specific download layout is silently guessed. Split membership is
    preserved from annotations and is never assigned by random question splitting.
    """
    source = Path(input_path).resolve()
    root = Path(video_root).resolve() if video_root is not None else source.parent
    task = task or source.stem
    if format == "jsonl":
        rows = read_raw_manifest(source)
    else:
        items = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(items, dict):
            items = items.get("data", items.get("annotations", items.get("videos")))
        if not isinstance(items, list):
            raise ValueError("benchmark JSON must contain a list of records")
        rows = []
        for position, item in enumerate(items):
            if format == "videomme":
                video_id = str(item.get("videoID", item.get("video_id", position)))
                video = item.get("video_path", video_id + ".mp4")
                questions = item.get("questions", [item])
            elif format in ("mvbench", "mlvu"):
                video = item.get("video_path", item.get("video"))
                if not video:
                    raise ValueError("record is missing video/video_path")
                video_id, questions = str(video), [item]
            else:
                raise ValueError(f"unknown annotation format: {format}")
            for question_index, question in enumerate(questions):
                row = dict(question)
                row["id"] = str(question.get("question_id", question.get("id",
                    f"{format}:{task}:{position}:{question_index}")))
                row["metadata"] = {**item.get("metadata", {}), **question.get("metadata", {}),
                                   "dataset": format, "task": task, "video_id": video_id}
                for key in ("split", "duration", "domain", "sub_category", "task_type"):
                    if key in question or key in item:
                        row["metadata"][key] = question.get(key, item.get(key))
                path = root / video_prefix / str(video)
                frame_source = item.get("data_type") == "frame" or path.is_dir()
                row["frame_dir" if frame_source else "video_path"] = str(path)
                if frame_source:
                    row["fps"] = fps if fps is not None else item.get("fps", 3.0 if task == "episodic_reasoning" else None)
                    if row["fps"] is None:
                        raise ValueError("frame-directory annotation needs --fps")
                canonical = _canonical(row, root)
                if canonical["answer_index"] is None:
                    raise ValueError("benchmark conversion requires answer labels")
                rows.append(canonical)
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("annotation set is empty or contains duplicate ids")
    write_jsonl(output_path, rows)
    return len(rows)
