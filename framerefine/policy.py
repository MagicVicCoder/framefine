"""Small cached-feature actor and a globally normalized conditional forest."""

from dataclasses import dataclass
import itertools

import torch
from torch import nn

from .inference import ForestDP
from .structures import maximum_tree, sample_tree, tree_log_prob


@dataclass
class Branch:
    bits: tuple
    gamma: torch.Tensor
    forest: ForestDP


class ConditionalMaskDistribution:
    def __init__(self, n, k, anchors, branches, temperature=1.0):
        self.n, self.k = int(n), int(k)
        self.anchors = tuple(anchors)
        self.remaining = tuple(i for i in range(n) if i not in self.anchors)
        self.branches = list(branches)
        self.temperature = float(temperature)
        if (not 0 <= self.k <= self.n or len(set(self.anchors)) != len(self.anchors)
                or any(i < 0 or i >= n for i in self.anchors) or not self.branches):
            raise ValueError("invalid conditional mask structure")
        expected = {bits for bits in itertools.product((0, 1), repeat=len(anchors))
                    if 0 <= k - sum(bits) <= len(self.remaining)}
        if {b.bits for b in self.branches} != expected or len(self.branches) != len(expected):
            raise ValueError("every feasible conditioning state must occur exactly once")
        for branch in self.branches:
            if (branch.forest.n != len(self.remaining)
                    or branch.forest.k != self.k - sum(branch.bits)
                    or branch.forest.temperature != self.temperature):
                raise ValueError("conditional branch has inconsistent count or temperature")
        self._lookup = {b.bits: b for b in self.branches}
        self._branch_logs = torch.stack([
            b.gamma / self.temperature + b.forest.log_partition for b in self.branches
        ])
        self.log_partition = self._branch_logs.logsumexp(0)
        means = torch.stack([b.gamma / self.temperature + b.forest.mean_scaled_energy
                             for b in self.branches])
        self.entropy = self.log_partition - (self._branch_logs.softmax(0) * means).sum()

    def _validate(self, mask):
        reference = self.branches[0].gamma
        mask = torch.as_tensor(mask, device=reference.device, dtype=reference.dtype)
        if mask.shape != (self.n,) or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask must be binary and match the candidate count")
        return mask

    def energy(self, mask):
        mask = self._validate(mask)
        bits = tuple(int(mask[i]) for i in self.anchors)
        if bits not in self._lookup:
            return mask.new_tensor(float("-inf"))
        branch = self._lookup[bits]
        return branch.gamma + branch.forest.energy(mask[list(self.remaining)])

    def log_prob(self, mask):
        mask = self._validate(mask)
        if int(mask.sum()) != self.k:
            return mask.new_tensor(float("-inf"))
        return self.energy(mask) / self.temperature - self.log_partition

    def _assemble(self, branch, local_mask):
        mask = local_mask.new_zeros(self.n)
        if self.anchors:
            mask[list(self.anchors)] = local_mask.new_tensor(branch.bits)
        mask[list(self.remaining)] = local_mask
        return mask

    def sample(self, generator=None):
        probabilities = self._branch_logs.detach().cpu().softmax(0)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        branch = self.branches[index]
        return self._assemble(branch, branch.forest.sample(generator))

    def mode(self):
        branch = max(self.branches, key=lambda b: float(b.gamma.detach()) / self.temperature
                     + b.forest.max_scaled_energy)
        return self._assemble(branch, branch.forest.mode())


@dataclass
class PolicyDraw:
    distribution: ConditionalMaskDistribution
    log_structure_prob: torch.Tensor


class CompressionPolicy(nn.Module):
    def __init__(self, feature_dim, query_dim, hidden_dim=32, conditioning_nodes=2,
                 temperature=0.7):
        super().__init__()
        if hidden_dim % 4 or hidden_dim < 4 or conditioning_nodes < 0 or temperature <= 0:
            raise ValueError("hidden dimension must be divisible by 4; check h and temperature")
        self.settings = dict(feature_dim=feature_dim, query_dim=query_dim,
                             hidden_dim=hidden_dim, conditioning_nodes=conditioning_nodes,
                             temperature=temperature)
        self.h, self.temperature = int(conditioning_nodes), float(temperature)
        self.frame = nn.Linear(feature_dim + 1, hidden_dim)
        self.query = nn.Linear(query_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2,
                                           dropout=0.0, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.anchor_head = nn.Linear(hidden_dim, 1)
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim * (self.h + 1) + self.h, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.unary = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.pair = nn.Sequential(nn.Linear(hidden_dim * 4 + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.root = nn.Linear(hidden_dim * 2, 1)
        self.gamma = nn.Linear(hidden_dim, 1)

    def draw(self, frame_features, query_features, k, timestamps=None,
             deterministic=False, generator=None):
        if frame_features.ndim != 2 or len(frame_features) < 1:
            raise ValueError("at least one candidate frame is required")
        n = len(frame_features)
        if self.h > n or k != int(k) or not 0 <= k <= n:
            raise ValueError("invalid conditioning count or retention budget")
        if query_features.shape != (self.settings["query_dim"],):
            raise ValueError("query feature shape is inconsistent")
        if timestamps is None:
            timestamps = torch.arange(n, dtype=frame_features.dtype, device=frame_features.device)
        timestamps = torch.as_tensor(timestamps, device=frame_features.device, dtype=frame_features.dtype)
        if timestamps.shape != (n,) or not torch.isfinite(timestamps).all():
            raise ValueError("invalid timestamps")
        if not torch.isfinite(frame_features).all() or not torch.isfinite(query_features).all():
            raise ValueError("cached features must be finite")
        time = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min()).clamp_min(1)
        query = self.query(query_features)
        hidden = self.encoder((self.frame(torch.cat([frame_features, time[:, None]], 1))
                               + query).unsqueeze(0)).squeeze(0)
        base = hidden.mean(0) + query
        logits = self.anchor_head(hidden).squeeze(-1)
        available = torch.ones(n, dtype=torch.bool, device=hidden.device)
        anchors, log_terms = [], []
        for _ in range(self.h):
            logp = logits.masked_fill(~available, float("-inf")).log_softmax(0)
            index = (int(logp.argmax()) if deterministic else
                     int(torch.multinomial(logp.detach().cpu().exp(), 1, generator=generator)))
            anchors.append(index)
            log_terms.append(logp[index])
            available = available.clone()
            available[index] = False
        remaining = [i for i in range(n) if i not in anchors]
        residual = hidden[remaining]
        m = len(remaining)
        pair_i, pair_j = torch.triu_indices(m, m, 1, device=hidden.device)
        branches = []
        for bits in itertools.product((0, 1), repeat=self.h):
            count = int(k) - sum(bits)
            if not 0 <= count <= m:
                continue
            condition_input = torch.cat([base, hidden[anchors].flatten(), hidden.new_tensor(bits)])
            context = self.condition(condition_input)
            repeated = context.expand(m, -1)
            node_inputs = torch.cat([residual, repeated], 1)
            unary = self.unary(node_inputs).squeeze(-1)
            pair_inputs = torch.cat([
                residual[pair_i], residual[pair_j], (residual[pair_i] - residual[pair_j]).abs(),
                context.expand(len(pair_i), -1),
                (time[remaining][pair_j] - time[remaining][pair_i])[:, None]], 1)
            pair_outputs = self.pair(pair_inputs)
            edge_matrix = hidden.new_zeros(m, m)
            edge_matrix[pair_i, pair_j] = pair_outputs[:, 0]
            edge_matrix[pair_j, pair_i] = pair_outputs[:, 0]
            tree_logits = hidden.new_zeros(m + 1, m + 1)
            scores = 3 * pair_outputs[:, 1].tanh()
            tree_logits[pair_i + 1, pair_j + 1] = scores
            tree_logits[pair_j + 1, pair_i + 1] = scores
            root_scores = 3 * self.root(node_inputs).squeeze(-1).tanh()
            tree_logits[0, 1:] = root_scores
            tree_logits[1:, 0] = root_scores
            tree = maximum_tree(tree_logits) if deterministic else sample_tree(tree_logits, generator)
            log_terms.append(tree_log_prob(tree_logits, tree))
            edges = [(i-1, j-1) for i, j in tree if i and j]
            values = (torch.stack([edge_matrix[i, j] for i, j in edges]) if edges
                      else unary.new_zeros(0))
            dp = ForestDP(unary, edges, values, count, self.temperature)
            branches.append(Branch(tuple(bits), self.gamma(context).squeeze(-1), dp))
        log_structure = sum(log_terms, hidden.sum() * 0)
        distribution = ConditionalMaskDistribution(n, int(k), tuple(anchors), branches, self.temperature)
        return PolicyDraw(distribution, log_structure)
