from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from framerefine.cli import main
from framerefine.cost import CostPlan
from test_preprocessing import ImageEncoderFixture


class CLITests(unittest.TestCase):
    def test_forecast_includes_configured_dense_validation_queries(self):
        plan = CostPlan(1000, 8, 4, 3, 2, validation_every=250, validation_examples=32)
        self.assertEqual(plan.training_score_requests, 10000)
        self.assertEqual(plan.validation_score_requests, 384)
        self.assertEqual(plan.score_requests, 10384)

    def test_import_prepare_train_evaluate_select_and_raw_compress(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            for index in range(4):
                Image.new("RGB", (12, 8), (index * 50, 0, 0)).save(root / f"{index}.png")
            row = dict(id="question", frame_paths=[f"{i}.png" for i in range(4)],
                       timestamps=[0., 1., 2., 3.], question="Which color?",
                       options=["red", "blue"], answer_index=0)
            source = root / "source.jsonl"
            source.write_text(json.dumps(row) + "\n")
            raw, prepared = root / "raw.jsonl", root / "prepared"
            config = root / "features.toml"
            config.write_text('[features]\nmodel_path="local"\nrevision="test"\n[preparation]\nnum_candidates=4\n')
            main(["import-data", "--annotations", str(source), "--format", "jsonl", "--output", str(raw)])
            with patch("framerefine.cli.FrozenCLIPEncoder", return_value=ImageEncoderFixture()):
                main(["prepare", "--raw-manifest", str(raw), "--config", str(config), "--output", str(prepared)])
            training = root / "train.toml"
            training.write_text('[policy]\nhidden_dim=16\nconditioning_nodes=1\n'
                '[training]\nk=2\ngroup_size=2\nmax_steps=2\nprobe_every=0\n'
                '[reader]\nfamily="toy"\n[budget]\nwall_hours=0.1\nmax_requests=100\n')
            manifest, run = prepared / "manifest.jsonl", root / "run"
            main(["train", "--config", str(training), "--manifest", str(manifest), "--output", str(run)])
            with self.assertRaisesRegex(ValueError, "resume"):
                main(["train", "--config", str(training), "--manifest", str(manifest), "--output", str(run)])
            checkpoint = run / "last.pt"
            report = root / "evaluation.json"
            main(["evaluate", "--config", str(training), "--checkpoint", str(checkpoint),
                  "--manifest", str(manifest), "--output", str(report)])
            self.assertIn("dense", json.loads(report.read_text()))
            selected = root / "selected.jsonl"
            main(["select", "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(selected)])
            result = json.loads(selected.read_text())
            self.assertEqual(len(result["frame_indices"]), 2)
            self.assertEqual(len(result["frame_paths"]), 2)
            row.pop("answer_index")
            source.write_text(json.dumps(row) + "\n")
            with patch("framerefine.cli.FrozenCLIPEncoder", return_value=ImageEncoderFixture()):
                main(["compress", "--checkpoint", str(checkpoint), "--raw-manifest", str(source),
                      "--features-config", str(config), "--cache-dir", str(root / "inference-cache"),
                      "--output", str(selected)])
            self.assertEqual(len(json.loads(selected.read_text())["frame_indices"]), 2)
            Path(result["frame_paths"][0]).unlink()
            previous_output = selected.read_bytes()
            with self.assertRaises(FileNotFoundError):
                main(["select", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                      "--output", str(selected)])
            self.assertEqual(selected.read_bytes(), previous_output)
            self.assertGreater(json.loads(selected.with_suffix(".timing.json").read_text())["end_to_end_seconds"], 0)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            records[0]["metadata"]["feature_encoder"]["revision"] = "incompatible-encoder"
            manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "feature encoder"):
                main(["select", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                      "--output", str(selected)])

    def test_preparation_cost_includes_encoder_startup_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            Image.new("RGB", (8, 8), "red").save(root / "frame.png")
            raw, config = root / "raw.jsonl", root / "features.toml"
            raw.write_text(json.dumps(dict(id="q", frame_paths=["frame.png"], timestamps=[0.],
                question="What?", options=["a", "b"])) + "\n")
            config.write_text('[features]\nmodel_path="local"\nrevision="test"\n')

            def delayed_encoder(**kwargs):
                time.sleep(.02)
                return ImageEncoderFixture()

            destination = root / "prepared"
            with patch("framerefine.cli.FrozenCLIPEncoder", side_effect=delayed_encoder):
                for _ in range(2):
                    main(["prepare", "--raw-manifest", str(raw), "--config", str(config), "--output", str(destination)])
            report = json.loads((destination / "preparation.json").read_text())
            self.assertGreaterEqual(report["encoder_startup_seconds"], .02)
            self.assertGreaterEqual(report["command_seconds"], report["encoder_startup_seconds"])
            self.assertEqual(len((destination / "preparation.history.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
