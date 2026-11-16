"""Differentiable reference count-DP for arbitrary signed binary forests.

No global subset enumeration is used. Traceback samples exactly from the DP;
entropy is propagated as an expectation semiring, without autograd-of-autograd.
This correctness-oriented Python implementation is not a fused GPU kernel.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class _Entry:
    log_mass: Tensor
    mean_energy: Tensor  # Energy divided by temperature.
    best: float


@dataclass
class _Table:
    entries: dict
    leaf: int | None = None
    left: "_Table | None" = None
    right: "_Table | None" = None
    choices: dict = field(default_factory=dict)
    best_choices: dict = field(default_factory=dict)


def _join(left, right, interaction, k):
    groups = {}
    for (kl, zl), lv in left.entries.items():
        for (kr, zr), rv in right.entries.items():
            if kl + kr <= k:
                extra = interaction * zl * zr
                groups.setdefault((kl + kr, zl), []).append(
                    (lv.log_mass + rv.log_mass + extra,
                     lv.mean_energy + rv.mean_energy + extra,
                     lv.best + rv.best + float(extra.detach()),
                     (kl, zl), (kr, zr))
                )
    result = _Table({}, left=left, right=right)
    for key, candidates in groups.items():
        log_values = torch.stack([c[0] for c in candidates])
        means = torch.stack([c[1] for c in candidates])
        best_index = max(range(len(candidates)), key=lambda j: candidates[j][2])
        result.entries[key] = _Entry(
            log_values.logsumexp(0), (log_values.softmax(0) * means).sum(),
            candidates[best_index][2],
        )
        result.choices[key] = [(c[0], c[3], c[4]) for c in candidates]
        result.best_choices[key] = best_index
    return result


class ForestDP:
    """Exact distribution on k retained nodes of an undirected forest.

    Potentials are raw energy units; log probabilities divide them by temperature.
    Edges may be positive or negative. All structural indices are local to unary.
    """

    def __init__(self, unary: Tensor, edges, edge_values: Tensor, k: int,
                 temperature: float = 1.0):
        if unary.ndim != 1 or not unary.is_floating_point():
            raise ValueError("unary must be a floating point vector")
        self.unary = unary
        self.n = len(unary)
        self.k = int(k)
        self.temperature = float(temperature)
        self.edges = [tuple(map(int, edge)) for edge in edges]
        self.edge_values = torch.as_tensor(edge_values, dtype=unary.dtype, device=unary.device)
        if k != self.k or not 0 <= self.k <= self.n:
            raise ValueError("k must be an integer between 0 and the node count")
        if self.temperature <= 0 or not torch.isfinite(torch.tensor(self.temperature)):
            raise ValueError("temperature must be positive and finite")
        if self.edge_values.shape != (len(self.edges),):
            raise ValueError("one scalar potential is required per edge")
        if not torch.isfinite(unary).all() or not torch.isfinite(self.edge_values).all():
            raise ValueError("potentials must be finite")
        parent = list(range(self.n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        adjacency = [[] for _ in range(self.n)]
        for index, (i, j) in enumerate(self.edges):
            if not (0 <= i < self.n and 0 <= j < self.n) or i == j:
                raise ValueError("invalid forest edge")
            ri, rj = find(i), find(j)
            if ri == rj:
                raise ValueError("edges contain a cycle or duplicate")
            parent[ri] = rj
            adjacency[i].append((j, index))
            adjacency[j].append((i, index))

        zero = unary.sum() * 0 + self.edge_values.sum() * 0
        global_table = _Table({(0, 0): _Entry(zero, zero, 0.0)})
        seen = set()
        for root in range(self.n):
            if root in seen:
                continue
            order, predecessors, children = [], {root: -1}, {}
            stack = [root]
            while stack:
                node = stack.pop()
                seen.add(node)
                order.append(node)
                children[node] = []
                for neighbor, edge_index in adjacency[node]:
                    if neighbor != predecessors[node]:
                        predecessors[neighbor] = node
                        children[node].append((neighbor, edge_index))
                        stack.append(neighbor)
            tables = {}
            for node in reversed(order):
                entries = {(0, 0): _Entry(zero, zero, 0.0)}
                if self.k >= 1:
                    energy = unary[node] / self.temperature
                    entries[(1, 1)] = _Entry(energy, energy, float(energy.detach()))
                table = _Table(entries, leaf=node)
                for child, edge_index in children[node]:
                    table = _join(table, tables[child],
                                  self.edge_values[edge_index] / self.temperature, self.k)
                tables[node] = table
            global_table = _join(global_table, tables[root], zero, self.k)
        self._table = global_table
        self._key = (self.k, 0)
        entry = global_table.entries[self._key]
        self.log_partition = entry.log_mass
        self.mean_scaled_energy = entry.mean_energy
        self.entropy = entry.log_mass - entry.mean_energy
        self.max_scaled_energy = entry.best

    def energy(self, mask: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, dtype=self.unary.dtype, device=self.unary.device)
        if mask.shape != (self.n,):
            raise ValueError("mask shape does not match forest")
        value = (mask * self.unary).sum()
        for (i, j), edge_value in zip(self.edges, self.edge_values):
            value = value + edge_value * mask[i] * mask[j]
        return value

    def log_prob(self, mask: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, dtype=self.unary.dtype, device=self.unary.device)
        if mask.shape != (self.n,) or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask must be a binary vector matching the forest")
        if int(mask.sum()) != self.k:
            return self.unary.new_tensor(float("-inf"))
        return self.energy(mask) / self.temperature - self.log_partition

    def _trace(self, deterministic, generator=None):
        mask = self.unary.new_zeros(self.n)
        stack = [(self._table, self._key)]
        while stack:
            table, key = stack.pop()
            if table.left is None:
                if table.leaf is not None:
                    mask[table.leaf] = key[1]
                continue
            candidates = table.choices[key]
            if deterministic:
                index = table.best_choices[key]
            else:
                logits = torch.stack([c[0].detach().cpu() for c in candidates])
                index = int(torch.multinomial(logits.softmax(0), 1, generator=generator))
            _, left_key, right_key = candidates[index]
            stack.extend([(table.left, left_key), (table.right, right_key)])
        return mask

    def sample(self, generator=None):
        return self._trace(False, generator)

    def mode(self):
        return self._trace(True)
