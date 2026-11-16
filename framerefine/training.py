"""On-policy structured actor training with a detached shared relation bank."""

from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
import time

import torch

from .exchange import ExchangeCube, face_differences, third_difference
from .readers import BudgetExceeded
from .evaluation import evaluate
from .data import feature_encoder_identity


@dataclass
class TrainConfig:
    k: int = 8
    group_size: int = 8
    max_steps: int = 1000
    probe_every: int = 4
    cube_dimensions: int = 3
    relation_cubes: int = 1
    relation_weight: float = 0.2
    learning_rate: float = 0.0003
    grad_clip: float = 1.0
    seed: int = 17
    checkpoint_every: int = 100
    shuffle: bool = True

    def __post_init__(self):
        if (self.k < 0 or self.group_size < 2 or self.max_steps < 0 or self.probe_every < 0
                or not 0 <= self.cube_dimensions <= 3 or self.relation_cubes < 0
                or self.relation_weight < 0 or self.learning_rate <= 0 or self.grad_clip <= 0
                or self.checkpoint_every < 0):
            raise ValueError("invalid training configuration")


def _loo(values):
    return (values.sum() - values) / (len(values) - 1)


class Trainer:
    def __init__(self, policy, reader, config):
        self.policy, self.reader, self.config = policy, reader, config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step_index = 0
        self._update_in_progress = False
        self.dataset_fingerprint = None
        self.feature_encoder_identity = None
        self.best_accuracy = None
        self._epoch = None
        self._validation_reader = None
        self._restored_validation = None
        self.validation_identity = None
        self._validation_identity_bound = False

    @staticmethod
    def _fingerprint(examples):
        if hasattr(examples, "fingerprint"):
            return examples.fingerprint
        digest = hashlib.sha256()
        for item in examples:
            digest.update(json.dumps(dict(id=item.example_id, revision=item.content_revision,
                question=item.question, options=item.options, answer=item.answer_index,
                paths=item.frame_paths), sort_keys=True).encode())
            for tensor in (item.frame_features, item.query_features, item.timestamps):
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _example_index(self, size):
        epoch, offset = divmod(self.step_index, size)
        if not self.config.shuffle:
            return offset
        if self._epoch != (epoch, size):
            self._order = torch.randperm(size, generator=torch.Generator().manual_seed(self.config.seed + epoch)).tolist()
            self._epoch = (epoch, size)
        return self._order[offset]

    @staticmethod
    def _groups(examples):
        if hasattr(examples, "rows"):
            rows = examples.rows
        else:
            rows = [dict(id=item.example_id, metadata=item.metadata, content_revision=item.content_revision)
                    for item in examples]
        return ({str(row["id"]) for row in rows},
                {str(row.get("metadata", {}).get("source_video", row["content_revision"])) for row in rows})

    def _inputs(self, example):
        device = next(self.policy.parameters()).device
        return (example.frame_features.to(device), example.query_features.to(device),
                example.timestamps.to(device))

    def step(self, example):
        self.reader.budget.check()
        if example.answer_index is None:
            raise ValueError("training requires labeled examples")
        features, query, timestamps = self._inputs(example)
        cfg = self.config
        if cfg.k > len(features):
            raise ValueError("retention budget exceeds this example's candidate count")
        self.policy.train()
        bank, third_order = [], []
        probe = bool(cfg.probe_every and self.step_index % cfg.probe_every == 0)
        # This bank is sampled before, and independently of, the current structural
        # actions below. Its actor/reader sampling paths are not differentiated.
        if probe:
            for _ in range(cfg.relation_cubes):
                with torch.no_grad():
                    reference = self.policy.draw(features, query, cfg.k, timestamps,
                                                 generator=self.generator)
                    seed = reference.distribution.sample(self.generator)
                cube = ExchangeCube.from_seed(seed, cfg.cube_dimensions, self.generator)
                masks = cube.masks(device=features.device, dtype=features.dtype)
                utilities = self.reader.score(example, masks).to(features.device)
                targets = face_differences(utilities, cube.dimensions).detach()
                bank.append((cube, masks, targets))
                if cube.dimensions == 3:
                    third_order.append(float(third_difference(utilities)))

        draws, masks = [], []
        for _ in range(cfg.group_size):
            draw = self.policy.draw(features, query, cfg.k, timestamps, generator=self.generator)
            draws.append(draw)
            masks.append(draw.distribution.sample(self.generator))
        details = self.reader.score_details(example, torch.stack(masks))
        utilities = features.new_tensor([item.utility for item in details])
        predictions = [item.prediction for item in details]
        log_masks = torch.stack([draw.distribution.log_prob(mask) for draw, mask in zip(draws, masks)])
        log_structures = torch.stack([draw.log_structure_prob for draw in draws])
        entropy = torch.stack([draw.distribution.entropy for draw in draws])
        relations = []
        for draw in draws:
            errors = []
            for cube, feedback_masks, targets in bank:
                if len(targets):
                    energies = torch.stack([draw.distribution.energy(mask) for mask in feedback_masks])
                    errors.append((face_differences(energies, cube.dimensions) - targets).square().mean())
            relations.append(torch.stack(errors).mean() if errors else draw.distribution.log_partition * 0)
        relations = torch.stack(relations)
        tau = self.policy.temperature
        structural_returns = utilities + tau * entropy.detach() - cfg.relation_weight * relations.detach()
        mask_advantages = (utilities - _loo(utilities)).detach()
        structure_advantages = (structural_returns - _loo(structural_returns)).detach()
        loss = (-(mask_advantages * log_masks).mean()
                -(structure_advantages * log_structures).mean()
                -tau * entropy.mean() + cfg.relation_weight * relations.mean())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss; no optimizer step performed")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip,
                                                  error_if_nonfinite=True)
        self._update_in_progress = True
        self.optimizer.step()
        self.step_index += 1
        self._update_in_progress = False
        return dict(step=self.step_index, example_id=example.example_id,
                    loss=float(loss.detach()), utility=float(utilities.mean()),
                    sampled_accuracy=(sum(p == example.answer_index for p in predictions) / len(predictions)
                                      if all(p is not None for p in predictions) else None),
                    conditional_entropy=float(entropy.mean().detach()),
                    relation_loss=float(relations.mean().detach()),
                    third_order_signal=(sum(third_order) / len(third_order) if third_order else None),
                    gradient_norm=float(gradient), relation_step=probe,
                    reader_requests=self.reader.budget.requests, cache_hits=self.reader.cache_hits,
                    elapsed_seconds=time.monotonic() - self.reader.budget.started)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(actor_settings=self.policy.settings, actor=self.policy.state_dict(),
                       optimizer=self.optimizer.state_dict(), training_config=asdict(self.config),
                       step=self.step_index, generator_state=self.generator.get_state(),
                       torch_rng_state=torch.get_rng_state(),
                       dataset_fingerprint=self.dataset_fingerprint, best_accuracy=self.best_accuracy,
                       feature_encoder_identity=self.feature_encoder_identity,
                       validation_identity=self.validation_identity,
                       validation_state=(dict(requests=self._validation_reader.budget.requests,
                                              scoring_seconds=self._validation_reader.scoring_seconds,
                                              cache_hits=self._validation_reader.cache_hits)
                                         if self._validation_reader is not None else self._restored_validation),
                       reader_identity=self.reader.scorer.identity,
                       reader_requests=self.reader.budget.requests,
                       reader_scoring_seconds=self.reader.scoring_seconds,
                       reader_cache_hits=self.reader.cache_hits,
                       elapsed_seconds=time.monotonic() - self.reader.budget.started)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path, examples=None):
        payload = torch.load(path, map_location=next(self.policy.parameters()).device, weights_only=True)
        if payload["actor_settings"] != self.policy.settings:
            raise ValueError("checkpoint actor settings do not match")
        current = asdict(self.config)
        previous = payload["training_config"]
        operational = {"max_steps", "checkpoint_every"}
        if ({key: value for key, value in current.items() if key not in operational}
                != {key: value for key, value in previous.items() if key not in operational}
                or current["max_steps"] < previous["max_steps"]):
            raise ValueError("checkpoint training configuration differs; only step-limit extensions "
                             "and checkpoint cadence changes are permitted")
        if payload.get("reader_identity") != self.reader.scorer.identity:
            raise ValueError("checkpoint reader identity does not match")
        saved_fingerprint = payload.get("dataset_fingerprint")
        if examples is not None and saved_fingerprint is not None and self._fingerprint(examples) != saved_fingerprint:
            raise ValueError("checkpoint dataset identity does not match")
        # Validate experiment semantics before mutating actor or optimizer state.
        self.policy.load_state_dict(payload["actor"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step_index = int(payload["step"])
        self.dataset_fingerprint = saved_fingerprint
        self.feature_encoder_identity = payload.get("feature_encoder_identity")
        self.best_accuracy = payload.get("best_accuracy")
        self.validation_identity = payload.get("validation_identity")
        self._validation_identity_bound = True
        self._restored_validation = payload.get("validation_state")
        self.generator.set_state(payload["generator_state"].cpu())
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        self.reader.budget.requests += int(payload["reader_requests"])
        self.reader.scoring_seconds += float(payload.get("reader_scoring_seconds", 0.0))
        self.reader.cache_hits += int(payload.get("reader_cache_hits", 0))
        now = time.monotonic()
        already_elapsed = now - self.reader.budget.started
        self.reader.budget.started = now - (already_elapsed + payload["elapsed_seconds"])

    def fit(self, examples, output_dir, validation_examples=None, validation_reader=None,
            validate_every=0, accuracy_tolerance=0.03):
        if not examples:
            raise ValueError("training requires at least one example")
        fingerprint = self._fingerprint(examples)
        if self.dataset_fingerprint is not None and fingerprint != self.dataset_fingerprint:
            raise ValueError("training dataset changed across checkpoint resume")
        self.dataset_fingerprint = fingerprint
        self.feature_encoder_identity = feature_encoder_identity(examples)
        if validate_every < 0:
            raise ValueError("validate_every must be nonnegative")
        identity = None
        if validate_every and validation_examples and validation_reader is not None:
            identity = dict(dataset=self._fingerprint(validation_examples), every=validate_every,
                            reader=validation_reader.scorer.identity, k=self.config.k,
                            accuracy_tolerance=accuracy_tolerance, methods=["selected", "uniform", "dense"])
        if validate_every:
            if not validation_examples or validation_reader is None:
                raise ValueError("periodic validation needs examples and a reader")
            train_ids, train_groups = self._groups(examples)
            valid_ids, valid_groups = self._groups(validation_examples)
            if train_ids & valid_ids or train_groups & valid_groups:
                raise ValueError("training and validation examples/videos overlap")
            if feature_encoder_identity(validation_examples) != self.feature_encoder_identity:
                raise ValueError("training and validation feature encoder identities differ")
        if self._validation_identity_bound and identity != self.validation_identity:
            raise ValueError("validation dataset/configuration identity changed across resume")
        if validate_every:
            self._validation_reader = validation_reader
            if self._restored_validation:
                validation_reader.budget.requests += self._restored_validation["requests"]
                validation_reader.scoring_seconds += self._restored_validation["scoring_seconds"]
                validation_reader.cache_hits += self._restored_validation["cache_hits"]
                self._restored_validation = None
        self.validation_identity = identity
        self._validation_identity_bound = True
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        reason = "max_steps"
        with (directory / "metrics.jsonl").open("a" if self.step_index else "w", encoding="utf-8") as log:
            try:
                while self.step_index < self.config.max_steps:
                    metrics = self.step(examples[self._example_index(len(examples))])
                    log.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                    log.flush()
                    if validate_every and self.step_index % validate_every == 0:
                        validation = evaluate(self.policy, validation_reader, validation_examples, self.config.k,
                                              include_dense=True, accuracy_tolerance=accuracy_tolerance)
                        validation["step"] = self.step_index
                        with (directory / "validation.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(validation) + "\n")
                        accuracy = validation["selected"]["accuracy"]
                        if accuracy is not None and (self.best_accuracy is None or accuracy > self.best_accuracy):
                            self.best_accuracy = accuracy
                            self.save_checkpoint(directory / "best.pt")
                    if (self.config.checkpoint_every and
                            (self.step_index == 1 or self.step_index % self.config.checkpoint_every == 0)):
                        self.save_checkpoint(directory / "last.pt")
            except BudgetExceeded as error:
                reason = str(error)
            except (Exception, KeyboardInterrupt):
                # Reader/loss failures happen before the optimizer mutates state.
                # A failed/interrupted optimizer may be partly updated: preserve
                # the preceding periodic checkpoint instead of overwriting it.
                if not self._update_in_progress:
                    self.save_checkpoint(directory / "last.pt")
                raise
        self.save_checkpoint(directory / "last.pt")
        elapsed = time.monotonic() - self.reader.budget.started
        report = dict(completed_steps=self.step_index, stop_reason=reason,
                      dataset_fingerprint=self.dataset_fingerprint, best_validation_accuracy=self.best_accuracy,
                      validation_requests=(validation_reader.budget.requests if validation_reader is not None else 0),
                      reader_requests=self.reader.budget.requests, cache_hits=self.reader.cache_hits,
                      elapsed_seconds=elapsed,
                      scoring_seconds=self.reader.scoring_seconds,
                      effective_subset_scores_per_second=(self.reader.budget.requests / elapsed if elapsed > 0 else None),
                      reader_identity=self.reader.scorer.identity,
                      results_are_real_video_experiments=self.reader.scorer.identity.get("model") != "analytic-toy")
        (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
