"""Local research commands; checkpoints and datasets are never downloaded implicitly."""

import argparse
import json
from pathlib import Path
import time
try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

import torch

from .annotations import convert_annotations, write_jsonl
from .cost import CostPlan
from .data import feature_encoder_identity, load_manifest, toy_examples
from .evaluation import evaluate
from .models.factory import build_reader
from .policy import CompressionPolicy
from .preprocessing import FrozenCLIPEncoder, prepare_manifest
from .readers import Budget, CachedReader, ToyReader
from .training import TrainConfig, Trainer


def _read_config(path):
    with open(path, "rb") as stream:
        return tomllib.load(stream)


def _cost(config, aggregate_rate=None):
    training, budget = config["training"], config["budget"]
    validation = config.get("validation", {})
    plan = CostPlan(training["max_steps"], training["group_size"], training["probe_every"],
                    training["cube_dimensions"], budget["wall_hours"],
                    budget.get("reserve_fraction", 0.2), training.get("relation_cubes", 1),
                    validation.get("every", 0), validation.get("max_examples", 0))
    return plan.report(aggregate_rate)


def _prepare(raw_manifest, config_path, output):
    started = time.perf_counter()
    config = _read_config(config_path)
    encoder_started = time.perf_counter()
    encoder = FrozenCLIPEncoder(**config["features"])
    startup = time.perf_counter() - encoder_started
    return prepare_manifest(raw_manifest, output, encoder, command_started_at=started,
                            encoder_startup_seconds=startup, **config.get("preparation", {}))


def _select(checkpoint, manifest, output, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    examples = load_manifest(manifest)
    _check_features(payload, examples)
    actor = CompressionPolicy(**payload["actor_settings"]).to(device).eval()
    actor.load_state_dict(payload["actor"])
    k = payload["training_config"]["k"]
    rows = []
    with torch.no_grad():
        for example in examples:
            draw = actor.draw(example.frame_features.to(device), example.query_features.to(device),
                              k, example.timestamps.to(device), deterministic=True)
            indices = draw.distribution.mode().nonzero().flatten().tolist()
            if indices and not example.frame_paths:
                raise ValueError("selection output requires local frame_paths")
            for index in indices:
                if not Path(example.frame_paths[index]).is_file():
                    raise FileNotFoundError(f"selected candidate frame is missing: {example.frame_paths[index]}")
            rows.append(dict(id=example.example_id, frame_indices=indices,
                             frame_paths=[example.frame_paths[i] for i in indices],
                             timestamps=[float(example.timestamps[i]) for i in indices]))
    write_jsonl(output, rows)
    print(str(Path(output).resolve()))


def _check_features(payload, examples):
    expected = payload.get("feature_encoder_identity")
    if expected is not None and feature_encoder_identity(examples) != expected:
        raise ValueError("checkpoint and dataset feature encoder identities differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="analytic CPU smoke test, not a video experiment")
    demo.add_argument("--steps", type=int, default=6)
    demo.add_argument("--output", default="outputs/demo")
    cost = commands.add_parser("cost", help="score-query upper bounds and throughput scenarios")
    cost.add_argument("--config", default="configs/probe.toml")
    cost.add_argument("--aggregate-rate", type=float)
    imported = commands.add_parser("import-data", help="convert local MCQ annotations to raw JSONL")
    imported.add_argument("--annotations", required=True)
    imported.add_argument("--format", choices=("jsonl", "videomme", "mvbench", "mlvu"), default="jsonl")
    imported.add_argument("--video-root")
    imported.add_argument("--video-prefix", default="")
    imported.add_argument("--task")
    imported.add_argument("--fps", type=float)
    imported.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="extract candidates and cache frozen CLIP features")
    prepare.add_argument("--raw-manifest", required=True)
    prepare.add_argument("--config", default="configs/features.toml")
    prepare.add_argument("--output", required=True)
    compress = commands.add_parser("compress", help="select frames from raw videos, including unlabeled inputs")
    compress.add_argument("--checkpoint", required=True)
    compress.add_argument("--raw-manifest", required=True)
    compress.add_argument("--features-config", default="configs/features.toml")
    compress.add_argument("--cache-dir", required=True)
    compress.add_argument("--output", required=True)
    compress.add_argument("--device", default="cpu")
    train = commands.add_parser("train", help="train only the compressor using frozen reader rewards")
    train.add_argument("--config", default="configs/probe.toml")
    train.add_argument("--reader-config", help="replace [reader] with a backbone-specific local config")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", default="outputs/probe")
    train.add_argument("--resume")
    train.add_argument("--limit-steps", type=int)
    train.add_argument("--validation-manifest")
    train.add_argument("--validation-every", type=int)
    train.add_argument("--validation-max-examples", type=int)
    validation = commands.add_parser("evaluate", help="held-out selected, uniform and dense accuracy")
    validation.add_argument("--config", default="configs/probe.toml")
    validation.add_argument("--reader-config")
    validation.add_argument("--checkpoint", required=True)
    validation.add_argument("--manifest", required=True)
    validation.add_argument("--output", default="outputs/evaluation.json")
    validation.add_argument("--skip-dense", action="store_true")
    validation.add_argument("--accuracy-tolerance", type=float, default=0.03)
    select = commands.add_parser("select", help="deterministic compression of prepared features")
    select.add_argument("--checkpoint", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument("--output", default="outputs/selected.jsonl")
    select.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "cost":
        print(json.dumps(_cost(_read_config(args.config), args.aggregate_rate), indent=2))
        return
    if args.command == "import-data":
        count = convert_annotations(args.annotations, args.output, args.format, args.video_root,
                                    args.video_prefix, args.task, args.fps)
        print(json.dumps(dict(output=str(Path(args.output).resolve()), examples=count)))
        return
    if args.command == "prepare":
        print(str(_prepare(args.raw_manifest, args.config, args.output)))
        return
    if args.command == "compress":
        started = time.perf_counter()
        manifest = _prepare(args.raw_manifest, args.features_config, args.cache_dir)
        _select(args.checkpoint, manifest, args.output, args.device)
        timing = dict(end_to_end_seconds=time.perf_counter() - started,
                      preparation=json.loads((Path(args.cache_dir) / "preparation.json").read_text(encoding="utf-8")))
        Path(args.output).with_suffix(".timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        with Path(args.output).with_suffix(".timing.history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(timing) + "\n")
        return
    if args.command == "select":
        _select(args.checkpoint, args.manifest, args.output, args.device)
        return
    if args.command == "demo":
        torch.manual_seed(17)
        examples = toy_examples(4, 8, 8, 4)
        actor = CompressionPolicy(8, 4, hidden_dim=16, conditioning_nodes=2, temperature=0.2)
        reader = CachedReader(ToyReader(), ":memory:", Budget(10000, 120))
        trainer = Trainer(actor, reader, TrainConfig(k=4, group_size=2, max_steps=args.steps, probe_every=2))
        try:
            print(json.dumps(trainer.fit(examples, args.output), indent=2))
        finally:
            reader.close()
        return
    config = _read_config(args.config)
    if args.reader_config:
        config["reader"] = _read_config(args.reader_config)["reader"]
    if args.command == "train" and not args.resume:
        directory = Path(args.output)
        if any((directory / name).exists() for name in ("last.pt", "metrics.jsonl", "summary.json")):
            raise ValueError("output contains a prior run; use --resume or choose another output directory")
    # Includes dataset/model loading. In-flight loads/calls cannot be forcibly killed.
    budget = Budget(config["budget"]["max_requests"], config["budget"]["wall_hours"] * 3600)
    examples = load_manifest(args.manifest, config.get("data", {}).get("cache_size", 128))
    if args.command == "evaluate":
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        _check_features(payload, examples)
        actor = CompressionPolicy(**payload["actor_settings"]).to(config.get("actor_device", "cpu"))
        actor.load_state_dict(payload["actor"])
        reader = build_reader(config["reader"], budget, Path(args.output).with_suffix(".sqlite"))
        try:
            report = evaluate(actor, reader, examples, payload["training_config"]["k"],
                              include_dense=not args.skip_dense, accuracy_tolerance=args.accuracy_tolerance,
                              details_path=Path(args.output).with_suffix(".predictions.jsonl"))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            reader.close()
        return
    if args.limit_steps is not None:
        config["training"]["max_steps"] = args.limit_steps
    training = TrainConfig(**config["training"])
    torch.manual_seed(training.seed)
    actor = CompressionPolicy(examples[0].frame_features.shape[1], examples[0].query_features.shape[0],
                              **config["policy"]).to(config.get("actor_device", "cpu"))
    settings = config.get("validation", {})
    validation_examples, validation_reader = None, None
    cadence = 0
    if args.validation_manifest:
        cadence = args.validation_every if args.validation_every is not None else settings.get("every", 250)
        limit = (args.validation_max_examples if args.validation_max_examples is not None
                 else settings.get("max_examples", 32))
        if cadence < 1 or limit < 1:
            raise ValueError("validation cadence and maximum example count must be positive")
        validation_examples = load_manifest(args.validation_manifest)[:limit]
    elif args.validation_every:
        raise ValueError("--validation-every requires --validation-manifest")
    reader = build_reader(config["reader"], budget, Path(args.output) / "scores.sqlite")
    trainer = Trainer(actor, reader, training)
    try:
        if args.resume:
            trainer.load_checkpoint(args.resume, examples)
        if validation_examples is not None:
            child = Budget(settings.get("max_requests", budget.max_requests), budget.wall_seconds, parent=budget)
            validation_reader = reader.fork(child)
        report = trainer.fit(examples, args.output, validation_examples, validation_reader, cadence,
                             settings.get("accuracy_tolerance", 0.03))
        print(json.dumps(report, indent=2))
    finally:
        if validation_reader is not None:
            validation_reader.close()
        reader.close()


if __name__ == "__main__":
    main()
