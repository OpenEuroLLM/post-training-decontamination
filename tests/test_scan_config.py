import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from datasets import Dataset

from decontamination.cli.util import _cast_value
from decontamination.ds.config import (
    BenchmarkConfig,
    DatasetConfig,
    expand_benchmarks,
    expand_datasets,
)
from decontamination.ds.search import _prompt_text, read_dataset, search


class ScanConfigTest(unittest.TestCase):
    def test_cartesian_expansion(self):
        benchmark = BenchmarkConfig(
            "example", None, "example/repo", ["en", "fr"], ["test", "validation"],
            None, ["prompt"], is_chat=True,
        )
        units = expand_benchmarks([benchmark])
        self.assertEqual([(unit.subset, unit.split) for unit in units], [
            ("en", "test"), ("en", "validation"), ("fr", "test"), ("fr", "validation"),
        ])
        self.assertTrue(all(unit.is_chat for unit in units))
        self.assertEqual(benchmark.subset, ["en", "fr"])

    def test_dataset_expansion_and_cli_types(self):
        hint = DatasetConfig.__annotations__["subset"]
        self.assertEqual(_cast_value(hint, "default"), "default")
        self.assertEqual(_cast_value(hint, ["de", "fr"]), ["de", "fr"])
        config = DatasetConfig("example", None, "example/repo", ["de", "fr"], "train")
        self.assertEqual([unit.subset for unit in expand_datasets([config])], ["de", "fr"])

    def test_chat_detection_and_override(self):
        conversation = '[{"role":"user","content":"Question"}, {"role":"assistant","content":"Answer"}]'
        self.assertEqual(_prompt_text(conversation, None), "Question")
        self.assertEqual(_prompt_text(conversation, False), conversation)
        self.assertEqual(_prompt_text("Plain prompt", None), "Plain prompt")
        with self.assertRaises(ValueError):
            _prompt_text("Plain prompt", True)

    @patch("decontamination.ds.search.load_dataset")
    def test_chat_loading(self, loader):
        loader.return_value = Dataset.from_list([{
            "conversation": [
                {"role": "system", "content": [{"type": "text", "text": "System"}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "First"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
                {"role": "user", "content": [{"type": "text", "text": "Second"}]},
            ],
        }])
        rows = read_dataset("example/repo", "fr", "test", query_fields=["conversation"], is_chat=True)
        self.assertEqual(rows[0]["conversation"], "First\nSecond")

    @patch("decontamination.ds.search.ngram_match", return_value=([0], [], {}))
    @patch("decontamination.ds.search.read_dataset")
    @patch("decontamination.ds.search.create_es_connection")
    def test_expanded_search_resumes_only_at_same_hit_limit(self, connect, load, match):
        connect.return_value = Mock()
        connect.return_value.count.return_value = {"count": 1}
        load.return_value = Dataset.from_dict({"question": ["fixture"]})
        dataset = DatasetConfig("local", None, "example/data", ["a", "b"], "train")
        benchmark = BenchmarkConfig(
            "fixture", None, "example/eval", ["en"], ["test", "validation"],
            None, ["question"],
        )
        with TemporaryDirectory() as directory:
            for hit_limit, calls in [(10, 4), (10, 4), (25, 8)]:
                search([dataset], [benchmark], top_k=hit_limit, output_dir=Path(directory))
                self.assertEqual(match.call_count, calls)
