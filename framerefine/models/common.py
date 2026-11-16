"""Shared frozen-reader scoring and local checkpoint validation."""

from pathlib import Path

import torch
from torch.nn import functional as F

from ..readers import ReaderScore


def local_checkpoint(model_path, revision):
    if not model_path or not Path(model_path).is_dir() or not revision or "REPLACE" in revision:
        raise ValueError("set an existing local model_path and immutable revision before reader use")
    return str(Path(model_path).resolve())


def resolve_dtype(name=None, device="cuda:0"):
    name = name or ("float32" if str(device).startswith("cpu") else "bfloat16")
    options = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in options:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    return options[name]


def option_token_ids(tokenizer, count):
    if not 2 <= count <= 26:
        raise ValueError("option-letter scoring requires 2..26 choices")
    tokens = [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(count)]
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != count:
        raise ValueError("each uppercase answer letter must have one distinct token")
    return [ids[0] for ids in tokens]


def score_options(scores, answer_index):
    scores = torch.as_tensor(scores).detach().float()
    if (scores.ndim != 1 or len(scores) < 2 or not torch.isfinite(scores).all()
            or not isinstance(answer_index, int) or not 0 <= answer_index < len(scores)):
        raise ValueError("finite option scores and a labeled answer_index are required")
    others = torch.cat([scores[:answer_index], scores[answer_index + 1:]])
    return ReaderScore(float(scores[answer_index] - others.logsumexp(0)), int(scores.argmax()))


def project_options(head, hidden, token_ids):
    """Project only the final hidden state onto option rows of a linear LM head."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not hasattr(head, "weight"):
        raise ValueError("single-example hidden states and a linear vocabulary head are required")
    bias = getattr(head, "bias", None)
    return F.linear(hidden[0, -1], head.weight[token_ids],
                    bias[token_ids] if bias is not None else None).float()


def question_prompt(example, indices):
    options = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(example.options))
    stamps = ", ".join(f"{float(example.timestamps[i]):.3f}" for i in indices)
    return (f"{example.question}\n{options}\nSelected frame timestamps in seconds, in order: "
            f"{stamps or 'none'}.\nAnswer with the option letter only.")
