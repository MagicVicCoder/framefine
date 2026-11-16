"""Rooted-tree log probabilities and exact Wilson sampling.

The virtual root is node 0. Root attachments are part of the latent action;
removing them later produces a forest, not an additional probability marginal.
"""

import torch


def _weights(logits):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or len(logits) < 1:
        raise ValueError("tree logits must be a nonempty square matrix")
    if not torch.isfinite(logits).all():
        raise ValueError("tree logits must be finite")
    scores = (logits.double() + logits.double().T) / 2
    off = ~torch.eye(len(scores), dtype=torch.bool, device=scores.device)
    if len(scores) == 1:
        return scores, scores * 0
    scores = scores - scores[off].max().detach()
    weights = scores.exp().masked_fill(~off, 0)
    if (weights[off] <= 0).any():
        raise ValueError("tree logits are too extreme for finite-precision sampling")
    return scores, weights


def _validate_tree(n, edges):
    if len(edges) != n - 1:
        raise ValueError("a spanning tree must have n-1 edges")
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("invalid tree edge")
        ri, rj = find(i), find(j)
        if ri == rj:
            raise ValueError("tree has a cycle or duplicate edge")
        parent[ri] = rj


def tree_log_prob(logits, edges):
    scores, weights = _weights(logits)
    _validate_tree(len(scores), edges)
    if len(scores) == 1:
        return scores.sum() * 0
    laplacian = torch.diag(weights.sum(1)) - weights
    sign, log_z = torch.linalg.slogdet(laplacian[1:, 1:])
    if float(sign.detach()) != 1.0:
        raise ValueError("tree normalizer is numerically singular")
    return torch.stack([scores[i, j] for i, j in edges]).sum() - log_z


def sample_tree(logits, generator=None):
    """Weighted Wilson sampler; random walks and loop erasure run on CPU."""
    _, weights = _weights(logits)
    weights = weights.detach().cpu()
    n = len(weights)
    visited, edges = {0}, []
    for start in range(1, n):
        if start in visited:
            continue
        path, positions, node = [], {}, start
        steps = 0
        while node not in visited:
            if node in positions:
                path = path[:positions[node]]
                positions = {v: i for i, v in enumerate(path)}
            positions[node] = len(path)
            path.append(node)
            node = int(torch.multinomial(weights[node], 1, generator=generator))
            steps += 1
            if steps > 100000:
                raise RuntimeError("Wilson walk exceeded its limit; inspect graph logits")
        path.append(node)
        for left, right in zip(path, path[1:]):
            edges.append((min(left, right), max(left, right)))
            visited.add(left)
    return edges


def maximum_tree(logits):
    scores, _ = _weights(logits)
    n = len(scores)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = [(float(scores[i, j].detach()), i, j)
                  for i in range(n) for j in range(i + 1, n)]
    result = []
    for _, i, j in sorted(candidates, reverse=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            result.append((i, j))
    return result
