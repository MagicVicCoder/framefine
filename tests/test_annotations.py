import json
from pathlib import Path
import tempfile
import unittest

from framerefine.annotations import convert_annotations, read_raw_manifest


class AnnotationTests(unittest.TestCase):
    def test_videomme_expands_questions_and_normalizes_letter_options(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder, "video.json"), Path(folder, "raw.jsonl")
            source.write_text(json.dumps([dict(videoID="clip", duration="short", questions=[
                dict(question_id="q1", question="What?", options=["A. cat", "B. dog"], answer="B"),
                dict(question_id="q2", question="Where?", options=["A. left", "B. right"], answer="A")])]))
            self.assertEqual(convert_annotations(source, output, "videomme", video_root=folder), 2)
            rows = read_raw_manifest(output)
            self.assertEqual(rows[0]["options"], ["cat", "dog"])
            self.assertEqual(rows[0]["answer_index"], 1)
            self.assertEqual(rows[0]["video_path"], str(Path(folder, "clip.mp4").resolve()))
            self.assertEqual(rows[0]["metadata"]["video_id"], rows[1]["metadata"]["video_id"])

    def test_mvbench_and_mlvu_preserve_text_labels_and_clip_boundaries(self):
        for family in ("mvbench", "mlvu"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                source, output = Path(folder, "task.json"), Path(folder, "raw.jsonl")
                source.write_text(json.dumps([dict(video="x.mp4", question="Action?",
                    candidates=["walking", "running"], answer="running", start=1.5, end=3.0)]))
                convert_annotations(source, output, family, video_root=folder)
                row = read_raw_manifest(output)[0]
                self.assertEqual(row["answer_index"], 1)
                self.assertEqual((row["start"], row["end"]), (1.5, 3.0))

    def test_invalid_label_and_duplicate_ids_fail_before_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder, "raw.jsonl"), Path(folder, "output.jsonl")
            row = dict(id="q", frame_paths=["0.jpg"], timestamps=[0.0],
                       question="What?", options=["a", "b"], answer_index=9)
            source.write_text(json.dumps(row) + "\n")
            output.write_text("keep existing output")
            with self.assertRaises(ValueError):
                convert_annotations(source, output, "jsonl")
            self.assertEqual(output.read_text(), "keep existing output")
            row["answer_index"] = 0
            source.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_raw_manifest(source)


if __name__ == "__main__":
    unittest.main()
