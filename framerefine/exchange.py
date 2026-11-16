"""Equal-cardinality exchange blocks and their reusable interaction feedback."""

from dataclasses import dataclass
import itertools

import torch


@dataclass(frozen=True)
class ExchangeCube:
    n: int
    background: tuple
    pairs: tuple

    def __post_init__(self):
        indices = list(self.background) + [i for pair in self.pairs for i in pair]
        if (any(len(pair) != 2 for pair in self.pairs)
                or len(indices) != len(set(indices))
                or any(i < 0 or i >= self.n for i in indices)):
            raise ValueError("exchange indices must be distinct and within candidates")

    @property
    def dimensions(self):
        return len(self.pairs)

    @classmethod
    def from_seed(cls, mask, dimensions=3, generator=None):
        mask = torch.as_tensor(mask).detach().cpu()
        if mask.ndim != 1 or not ((mask == 0) | (mask == 1)).all() or dimensions < 0:
            raise ValueError("a binary seed mask and nonnegative dimension are required")
        selected = mask.nonzero().flatten().tolist()
        rejected = (mask == 0).nonzero().flatten().tolist()
        d = min(int(dimensions), len(selected), len(rejected))
        chosen = [selected[i] for i in torch.randperm(len(selected), generator=generator)[:d].tolist()]
        others = [rejected[i] for i in torch.randperm(len(rejected), generator=generator)[:d].tolist()]
        pairs = tuple(sorted(tuple(sorted(pair)) for pair in zip(chosen, others)))
        background = tuple(i for i in selected if i not in chosen)
        return cls(len(mask), background, pairs)

    def masks(self, device=None, dtype=torch.float32):
        result = torch.zeros(2 ** self.dimensions, self.n, device=device, dtype=dtype)
        if self.background:
            result[:, list(self.background)] = 1
        for index in range(len(result)):
            for bit, pair in enumerate(self.pairs):
                result[index, pair[(index >> bit) & 1]] = 1
        return result


def face_differences(values, dimensions):
    if values.shape != (2 ** dimensions,):
        raise ValueError("one value is required for each cube corner")
    faces = []
    for i, j in itertools.combinations(range(dimensions), 2):
        rest = [bit for bit in range(dimensions) if bit not in (i, j)]
        for assignment in range(2 ** len(rest)):
            base = sum(((assignment >> t) & 1) << bit for t, bit in enumerate(rest))
            faces.append(values[base | (1 << i) | (1 << j)] - values[base | (1 << i)]
                         - values[base | (1 << j)] + values[base])
    return torch.stack(faces) if faces else values.new_zeros(0)


def third_difference(values):
    if values.shape != (8,):
        raise ValueError("third differences require a 3D exchange cube")
    return sum(((-1) ** (3 - index.bit_count())) * value for index, value in enumerate(values))


def rao_blackwell_task_loss(full_log_probs, utilities, baseline=0.0):
    """Negative task-gradient surrogate for a uniformly constructed exchange block.

    Full policy scores (including their GLOBAL normalizer) are mandatory. The
    conditional weights and utilities must be detached. This function alone does
    not optimize the structure/entropy/relation objective of the full trainer.
    """
    if full_log_probs.ndim != 1 or full_log_probs.shape != utilities.shape:
        raise ValueError("matching vectors of full log probabilities and utilities required")
    weights = full_log_probs.detach().softmax(0)
    return -(weights * (utilities.detach() - torch.as_tensor(baseline).detach()) * full_log_probs).sum()
