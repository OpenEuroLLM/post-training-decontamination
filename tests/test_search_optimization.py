import unittest
from unittest.mock import Mock, patch

from decontamination.ds.search import _search_hits, ngram_match
from decontamination.es.run import run
from decontamination.es.util import resolve_image_arch, resolve_jvm_cpus
from decontamination.util import Architecture


class SearchOptimizationTest(unittest.TestCase):
    def test_batches_preserve_request_order_and_size(self):
        es = Mock()
        es.msearch.side_effect = [
            {"responses": [{"hits": {"hits": [1]}}, {"hits": {"hits": [2]}}]},
            {"responses": [{"hits": {"hits": [3]}}]},
        ]
        self.assertEqual(list(_search_hits(es, "index", ["a", "b", "c"], 10, 2)),
                         [("a", [1]), ("b", [2]), ("c", [3])])
        self.assertEqual(es.msearch.call_count, 2)
        self.assertNotIn("max_concurrent_searches", es.msearch.call_args_list[0].kwargs)
        requests = es.msearch.call_args_list[0].kwargs["searches"]
        self.assertEqual([body["size"] for body in requests[1::2]], [10, 10])

    def test_concurrency_is_capped_by_batch_length(self):
        es = Mock()
        hit = {"hits": {"hits": []}}
        es.msearch.side_effect = [{"responses": [hit, hit]}, {"responses": [hit]}]
        list(_search_hits(es, "index", ["a", "b", "c"], 10, 2, 8))
        self.assertEqual(
            [call.kwargs["max_concurrent_searches"] for call in es.msearch.call_args_list], [2, 1],
        )

    @patch("decontamination.ds.search.spacy.load")
    def test_cross_prompt_batches_preserve_scores_and_field_boundaries(self, load):
        import spacy
        load.return_value = spacy.blank("en")
        rows = [
            {"a": "alpha beta", "b": "beta gamma"},
            {"a": None, "b": "skip"},
            {"a": "", "b": "alpha delta"},
            {"a": "beta gamma", "b": "delta gamma"},
        ]
        es = Mock()

        def msearch(*, index, searches, max_concurrent_searches):
            hits = []
            for body in searches[1::2]:
                phrase = body["query"]["bool"]["filter"][0]["match_phrase"]["text"]
                hit = {"_id": phrase, "_source": {"text": phrase, "original_id": ord(phrase[0])}}
                hits.append({"hits": {"hits": [hit]}})
            return {"responses": hits}

        es.msearch.side_effect = msearch
        baseline = ngram_match(es, "index", rows, ["a", "b"], 2, 10, 3, 20)
        baseline_calls = es.msearch.call_count
        es.reset_mock()
        grouped = ngram_match(es, "index", rows, ["a", "b"], 2, 10, 3, 20, 3)
        self.assertEqual(grouped, baseline)
        self.assertLess(es.msearch.call_count, baseline_calls)
        parallel = ngram_match(es, "index", rows, ["a", "b"], 2, 10, 3, 20, 3, requests_in_flight=2)
        self.assertEqual(parallel, baseline)

    @patch("decontamination.ds.search.spacy.load")
    def test_summary_only_matching_preserves_scores(self, load):
        import spacy

        load.return_value = spacy.blank("en")
        es = Mock()
        es.search.return_value = {
            "hits": {"hits": [
                {"_id": "0", "_source": {"text": "one two three", "original_id": 7}},
            ]},
        }
        arguments = (es, "index", [{"text": "one two three"}], ["text"], 2, 10)
        scores, _, row_scores = ngram_match(*arguments)
        self.assertEqual(
            ngram_match(*arguments, collect_details=False), (scores, [], row_scores)
        )

    @patch("decontamination.ds.search.spacy.load")
    def test_pair_scores_deduplicate_original_ids_and_keep_benchmark_positions(self, load):
        import spacy

        load.return_value = spacy.blank("en")
        es = Mock()
        es.search.return_value = {
            "hits": {"hits": [
                {"_id": "a", "_source": {"original_id": 7}},
                {"_id": "b", "_source": {"original_id": 7}},
            ]},
        }
        callback = Mock()
        rows = [{"text": "one two"}, {"text": None}, {"text": "three four"}]
        ngram_match(es, "index", rows, ["text"], 2, 10, prompt_batch_size=3,
                    collect_details=False, on_match_scores=callback)
        self.assertEqual([call.args for call in callback.call_args_list],
                         [(0, {7: 1.0}), (2, {7: 1.0})])

    def test_partial_search_fails(self):
        es = Mock()
        es.msearch.return_value = {"responses": [{"error": "failure"}]}
        with self.assertRaises(RuntimeError):
            list(_search_hits(es, "index", ["a"], 10, 2))

    def test_affinity_override_and_architecture(self):
        with patch("decontamination.es.util.os.sched_getaffinity", return_value={1, 3, 5}):
            self.assertEqual(resolve_jvm_cpus(None), 3)
            self.assertEqual(resolve_jvm_cpus(8), 8)
        with patch.object(Architecture, "from_platform", return_value=Architecture.x86_64):
            self.assertEqual(resolve_image_arch("elastic-{auto-arch}.sif"), "elastic-amd64.sif")
        self.assertEqual(resolve_image_arch("custom.sif"), "custom.sif")

    @patch("decontamination.es.run.run_command_streaming", return_value=(0, ""))
    @patch.dict("os.environ", {"ES_JAVA_OPTS": "-Xmx8g"})
    def test_run_passes_jvm_settings_to_apptainer(self, execute):
        run("custom.sif", mounts={}, jvm_cpus=8)
        execute.assert_called_once_with([
            "apptainer", "run", "--env", "ES_JAVA_OPTS=-Xmx8g -XX:ActiveProcessorCount=8", "custom.sif",
        ])
