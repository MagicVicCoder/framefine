"""Held-out selected-vs-uniform comparison with explicit accuracy availability."""

import torch
import time

from .annotations import write_jsonl


def evaluate(policy, reader, examples, k, include_dense=False, accuracy_tolerance=0.03, details_path=None):
    if not examples:
        raise ValueError("evaluation requires examples")
    if not 0 <= accuracy_tolerance <= 1:
        raise ValueError("accuracy_tolerance must be an absolute fraction")
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    methods = ("selected", "uniform", "dense") if include_dense else ("selected", "uniform")
    totals = {method: dict(utility=0.0, correct=0, available=0) for method in methods}
    selected_count, candidate_count, policy_seconds, records = 0, 0, 0.0, []
    try:
        with torch.no_grad():
            for example in examples:
                started = time.perf_counter()
                draw = policy.draw(example.frame_features.to(device), example.query_features.to(device),
                                   k, example.timestamps.to(device), deterministic=True)
                selected = draw.distribution.mode()
                policy_seconds += time.perf_counter() - started
                selected_count += int(selected.sum())
                candidate_count += len(selected)
                uniform = torch.zeros_like(selected)
                if k:
                    uniform[torch.linspace(0, len(selected)-1, k, device=device).round().long()] = 1
                masks = [selected, uniform] + ([torch.ones_like(selected)] if include_dense else [])
                details = reader.score_details(example, torch.stack(masks))
                record = dict(id=example.example_id, answer_index=example.answer_index,
                              frame_indices=selected.nonzero().flatten().tolist(),
                              metadata=example.metadata)
                for method, item in zip(totals, details):
                    totals[method]["utility"] += item.utility
                    if item.prediction is not None:
                        totals[method]["available"] += 1
                        totals[method]["correct"] += int(item.prediction == example.answer_index)
                    record[method] = dict(prediction=item.prediction, utility=item.utility)
                if details_path is not None:
                    records.append(record)
    finally:
        policy.train(was_training)
    result = dict(examples=len(examples), reader_requests=reader.budget.requests,
                  reader_identity=reader.scorer.identity,
                  retained_frame_fraction=selected_count / candidate_count,
                  policy_seconds=policy_seconds,
                  results_are_real_video_experiments=reader.scorer.identity.get("model") != "analytic-toy")
    for method, total in totals.items():
        result[method] = dict(mean_utility=total["utility"] / len(examples),
                              predictions_available=total["available"],
                              accuracy=(total["correct"] / total["available"] if total["available"] else None))
    if include_dense:
        selected, dense = result["selected"], result["dense"]
        complete = selected["predictions_available"] == dense["predictions_available"] == len(examples)
        drop = dense["accuracy"] - selected["accuracy"] if complete else None
        result.update(accuracy_drop_points=100 * drop if drop is not None else None,
                      accuracy_tolerance_points=100 * accuracy_tolerance,
                      within_accuracy_tolerance=(drop <= accuracy_tolerance + 1e-12 if drop is not None else None))
    if details_path is not None:
        write_jsonl(details_path, records)
    return result
