import json
import logging
import os
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import batched, islice
from pathlib import Path

import pandas as pd
import spacy
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from decontamination.util import bold, TypedDataFrame
from elasticsearch import Elasticsearch

from .config import BenchmarkConfig, DatasetConfig, Result, expand_benchmarks, expand_datasets
from .util import build_default_index_name, create_es_connection

log = logging.getLogger(__name__)


def get_ngram_mapping(
    doc: spacy.tokens.Doc,
    n: int,
) -> dict[str, list[int]]:
    ngram_docs = [doc[i : i + n] for i in range(len(doc) - n + 1)]
    return {
        ngram_doc.text: [token.i for token in ngram_doc] for ngram_doc in ngram_docs
    }


def _search_hits(
    es, index_name, ngrams, size, batch_size, max_concurrent_searches=None,
    executor=None, requests_in_flight=1,
):
    def query(ngram):
        return {"bool": {"filter": [{"match_phrase": {"text": ngram}}]}}

    def fetch(group):
        if batch_size == 1:
            responses = [es.search(
                index=index_name, search_type="query_then_fetch",
                rest_total_hits_as_int=True, size=size, query=query(group[0]),
            )]
        else:
            searches = []
            for ngram in group:
                searches.extend((
                    {"search_type": "query_then_fetch"},
                    {"size": size, "query": query(ngram)},
                ))
            options = {}
            if max_concurrent_searches is not None:
                options["max_concurrent_searches"] = min(max_concurrent_searches, len(group))
            responses = es.msearch(index=index_name, searches=searches, **options)["responses"]
        return responses

    groups = iter(batched(ngrams, batch_size))
    pending = deque()
    if executor is not None:
        pending.extend((group, executor.submit(fetch, group)) for group in islice(groups, requests_in_flight))

    while pending or executor is None:
        if executor is None:
            group = next(groups, None)
            if group is None:
                break
            future = None
        else:
            group, future = pending.popleft()
        responses = future.result() if future is not None else fetch(group)
        if executor is not None:
            following = next(groups, None)
            if following is not None:
                pending.append((following, executor.submit(fetch, following)))
        for ngram, response in zip(group, responses, strict=True):
            if "error" in response or response.get("timed_out") or response.get("_shards", {}).get("failed"):
                raise RuntimeError(f"Incomplete Elasticsearch search: {response}")
            yield ngram, response["hits"]["hits"]


def _query_groups(rows, fields, model, ngram_size, prompt_batch_size):
    for group in batched(rows, prompt_batch_size):
        prepared = []
        for row in group:
            strings = [row[field] for field in fields]
            if any(text is None for text in strings):
                continue
            queries = []
            for text in strings:
                doc = model.make_doc(text)
                queries.append(([token.text for token in doc], get_ngram_mapping(doc, ngram_size)))
            prepared.append((strings, queries))
        yield prepared


def ngram_match(
    es: Elasticsearch,
    index_name: str,
    query_dataset: Dataset,
    fields: list[str],
    ngram_size: int,
    search_size: int,
    search_batch_size: int = 1,
    max_concurrent_searches: int | None = None,
    prompt_batch_size: int = 1,
    requests_in_flight: int = 1,
    collect_details: bool = True,
) -> tuple[list[float], list[dict], dict[int, float]]:
    if requests_in_flight < 1:
        raise ValueError("requests_in_flight must be positive")
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be positive")
    spacy_model = spacy.load("en_core_web_lg")

    match_scores = []
    output_data = []
    max_train_match_scores = {}

    def search_hits(ngrams):
        return _search_hits(
            es, index_name, ngrams, search_size, search_batch_size, max_concurrent_searches,
            executor, requests_in_flight,
        )

    pool = ThreadPoolExecutor(max_workers=requests_in_flight) if requests_in_flight > 1 else nullcontext()
    with pool as executor:
        rows = tqdm(query_dataset, desc="Running n-gram match search on benchmark queries")
        for group in _query_groups(rows, fields, spacy_model, ngram_size, prompt_batch_size):
            group_hits = None
            if prompt_batch_size > 1:
                group_hits = search_hits(ngram for _, queries in group for _, mapping in queries for ngram in mapping)
            for query_strings, queries in group:
                query_string_match_scores = []
                query_string_match_tokens = defaultdict(list)
                matching_doc_ids = set()
                doc_id_source_mapping = {}
                for query_string_tokens, ngram_mapping in queries:
                    # We compute the match score for each query string for ngram matches as follows:
                    # For each token in the query string, we retrieve the training documents that
                    # contain ngrams from the query string the token belongs to. Then we compute the
                    # match score as the ratio of the tokens in the query string that match that training document.
                    query_string_length = len(query_string_tokens)
                    train_doc_matches = defaultdict(set)
                    hits_for_query = (
                        islice(group_hits, len(ngram_mapping)) if group_hits is not None
                        else search_hits(ngram_mapping)
                    )
                    for ngram, hits in hits_for_query:
                        tokens = ngram_mapping[ngram]
                        for hit_info in hits:
                            doc_id = hit_info["_id"]
                            doc = hit_info["_source"]
                            train_doc_matches[doc_id].update(tokens)
                            matching_doc_ids.add(doc_id)
                            doc_id_source_mapping[doc_id] = doc

                    query_string_match_scores.append(
                        {
                            doc_id: len(matching_tokens) / query_string_length
                            for doc_id, matching_tokens in train_doc_matches.items()
                        }
                    )
                    if collect_details:
                        for doc_id, matching_tokens in train_doc_matches.items():
                            query_string_match_tokens[doc_id].append(
                                [query_string_tokens[t] for t in matching_tokens]
                            )

                if matching_doc_ids:
                    aggregated_match_scores = {
                        doc_id: sum(score.get(doc_id, 0.0) for score in query_string_match_scores)
                        / len(query_strings)
                        for doc_id in matching_doc_ids
                    }
                    match_score = max(aggregated_match_scores.values())
                    match_scores.append(match_score)
                    for doc_id, score in aggregated_match_scores.items():
                        original_id = doc_id_source_mapping[doc_id]["original_id"]
                        max_train_match_scores[original_id] = max(
                            score, max_train_match_scores.get(original_id, 0.0)
                        )
                    if collect_details:
                        sorted_matches = sorted(
                            aggregated_match_scores.items(), key=lambda item: item[1], reverse=True
                        )
                        output_data.append({
                            "query": query_strings,
                            "matches": [
                                {
                                    "doc_id": doc_id,
                                    "source": doc_id_source_mapping[doc_id],
                                    "matching_tokens": query_string_match_tokens[doc_id],
                                    "score": score,
                                }
                                for doc_id, score in sorted_matches
                            ],
                            "score": match_score,
                        })
                else:
                    match_scores.append(0)
            if group_hits is not None:
                assert next(group_hits, None) is None
    return match_scores, output_data, max_train_match_scores


def _prompt_text(value, is_chat):
    if value is None or is_chat is False:
        return value
    if isinstance(value, str):
        if is_chat is None and not value.lstrip().startswith("["):
            return value
        try:
            chat = json.loads(value)
        except json.JSONDecodeError:
            if is_chat is None:
                return value
            raise
    else:
        chat = value
    if not isinstance(chat, list) or not all(isinstance(turn, dict) and "role" in turn for turn in chat):
        if is_chat is None:
            return value
        raise ValueError("Expected a chat with role/content messages")
    prompts = []
    for turn in chat:
        if turn["role"] != "user":
            continue
        content = turn.get("content")
        if isinstance(content, list):
            content = " ".join(
                block.get("text") or ""
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if isinstance(content, str) and content.strip():
            prompts.append(content)
    return "\n".join(prompts) or None


def read_dataset(
    name: str,
    subset: str | None,
    split: str,
    data_files: list[str] | None = None,
    query_fields: list[str] | None = None,
    is_chat: bool | None = None,
    revision: str | None = None,
) -> Dataset:
    log.info("Loading dataset: %s, subset: %s, split: %s", name, subset, split)
    if data_files and revision:
        snapshot = Path(snapshot_download(name, repo_type="dataset", revision=revision,
                                          allow_patterns=data_files))
        files = [str(file) for pattern in data_files for file in sorted(snapshot.glob(pattern))]
        formats = {".json": "json", ".jsonl": "json", ".parquet": "parquet", ".csv": "csv"}
        builders = {formats[Path(file).suffix] for file in files}
        if len(builders) != 1:
            raise ValueError(f"Expected files of one format for {name}: {data_files}")
        ds = load_dataset(builders.pop(), data_files={split: files}, split=split)
    else:
        ds = load_dataset(name, subset, split=split, data_files=data_files, revision=revision)
    log.info("Loaded dataset: %s", ds)
    log.info("\tColumns: %s", ds.column_names)
    log.info("\tNumber of examples: %d", len(ds))

    query_fields = ["question"] if query_fields is None else query_fields
    ds = ds.select_columns(query_fields)
    if is_chat is False:
        return ds
    return Dataset.from_dict({
        field: [_prompt_text(value, is_chat) for value in ds[field]]
        for field in query_fields
    })


def search(
    datasets: list[DatasetConfig],
    benchmarks: list[BenchmarkConfig],
    ngram_size: int = 8,
    match_threshold: float = 0.5,
    top_k: int = 10,
    output_dir: Path = Path("outputs"),
    search_batch_size: int = 1,
    max_concurrent_searches: int | None = None,
    prompt_batch_size: int = 1,
    requests_in_flight: int = 1,
) -> None:
    """
    Search for benchmark queries in the indexed datasets and save results to a CSV file.

    Args:
        datasets: A list of DatasetConfig objects specifying the datasets to search.
        benchmarks: A list of BenchmarkConfig objects specifying the benchmarks to search for.
        ngram_size: The size of n-grams to use for matching (default: 8).
        match_threshold: The threshold for considering a match as contamination (default: 0.5).
        top_k: The number of top matches to retrieve for each query (default: 10).
        output_dir: The directory to save the results CSV file (default: "outputs").
        search_batch_size: N-gram queries per request; 1 uses individual searches.
        max_concurrent_searches: Concurrent searches per batch. None uses the Elasticsearch default.
        prompt_batch_size: Number of prompt rows to group for search requests.
        requests_in_flight: Maximum number of concurrent HTTP search requests.
    """
    if ngram_size < 1 or top_k < 1:
        raise ValueError("ngram_size and top_k must be positive")
    if requests_in_flight < 1:
        raise ValueError("requests_in_flight must be positive")
    if search_batch_size < 1:
        raise ValueError("search_batch_size must be positive")
    if max_concurrent_searches is not None and max_concurrent_searches < 1:
        raise ValueError("max_concurrent_searches must be positive")
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be positive")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_results = (
        TypedDataFrame(pd.read_csv(output_dir / "results.csv"), Result)
        if (output_dir / "results.csv").exists()
        else TypedDataFrame.new(Result)
    )

    es = create_es_connection(connections_per_node=max(10, requests_in_flight))

    datasets = expand_datasets(datasets)
    benchmarks = expand_benchmarks(benchmarks)
    for dataset in datasets:
        log.info(
            bold("Processing dataset: %s (subset: %s, split: %s)..."),
            dataset.path,
            dataset.subset,
            dataset.split,
        )
        if not dataset.index_name:
            dataset.index_name = build_default_index_name(
                dataset.path, dataset.subset, dataset.split
            )
        log.info("- Using index name: %s", dataset.index_name)
        dataset_num_docs = es.count(index=dataset.index_name)["count"]
        log.info("- Number of docs: %d", dataset_num_docs)
        log.info("-" * 60)

        for benchmark in benchmarks:
            log.info(
                f"Processing benchmark: {bold('%s')} (subset: %s, split: %s)...",
                benchmark.path,
                benchmark.subset,
                benchmark.split,
            )

            result = Result(
                dataset_path=dataset.path,
                dataset_subset=dataset.subset,
                dataset_split=dataset.split,
                dataset_query_filter=dataset.query_filter,
                dataset_query_field=dataset.query_field,
                benchmark_path=benchmark.path,
                benchmark_subset=benchmark.subset,
                benchmark_split=benchmark.split,
                benchmark_query_fields=benchmark.query_fields,
                benchmark_is_chat=benchmark.is_chat,
                benchmark_revision=benchmark.revision,
                dataset_num_docs=dataset_num_docs,
                dataset_num_contaminated_docs=0,
                dataset_contamination_fraction=0.0,
                benchmark_num_docs=0,
                benchmark_num_contaminated_docs=0,
                benchmark_contamination_fraction=0.0,
                ngram_size=ngram_size,
                match_threshold=match_threshold,
                top_k=top_k,
                contaminated_ids={},
            )

            if any(result.is_same_setup(other) for other in df_results):
                log.info(
                    "Results already exist for this dataset and benchmark; skipping..."
                )
                log.info("-" * 60)
                continue

            benchmark_dataset = read_dataset(
                name=benchmark.path,
                subset=benchmark.subset,
                split=benchmark.split,
                data_files=benchmark.data_files,
                query_fields=benchmark.query_fields,
                is_chat=benchmark.is_chat,
                revision=benchmark.revision,
            )
            result.benchmark_num_docs = len(benchmark_dataset)

            match_scores, _, train_id_scores = ngram_match(
                es,
                dataset.index_name,
                benchmark_dataset,
                benchmark.query_fields,
                ngram_size,
                top_k,
                search_batch_size=search_batch_size,
                max_concurrent_searches=max_concurrent_searches,
                prompt_batch_size=prompt_batch_size,
                requests_in_flight=requests_in_flight,
                collect_details=False,
            )
            dataset_contaminated_docs = list(
                _id for _id, score in train_id_scores.items() if score > match_threshold
            )
            result.dataset_num_contaminated_docs = len(dataset_contaminated_docs)
            result.dataset_contamination_fraction = (
                len(dataset_contaminated_docs) / dataset_num_docs if dataset_num_docs else 0.0
            )
            result.contaminated_ids = {id: set() for id in dataset_contaminated_docs}
            result.benchmark_num_contaminated_docs = len(
                [s for s in match_scores if s > match_threshold]
            )
            result.benchmark_contamination_fraction = (
                result.benchmark_num_contaminated_docs / result.benchmark_num_docs
                if result.benchmark_num_docs else 0.0
            )

            # Print resutlts in a table format with two columns: Parameter and Value
            log.info("Results:")
            log.info("-" * 57)
            for param, value in {
                "# docs (dataset)": result.dataset_num_docs,
                "# contaminated docs (dataset)": result.dataset_num_contaminated_docs,
                "contamination rate (dataset)": result.dataset_contamination_fraction,
                "# docs (benchmark)": result.benchmark_num_docs,
                "# contaminated docs (benchmark)": result.benchmark_num_contaminated_docs,
                "contamination rate (benchmark)": result.benchmark_contamination_fraction,
            }.items():
                log.info(f"| {str(param):<34} | {str(value):<16} |")
            log.info("-" * 57)
            log.info("-" * 60)

            df_results.add(result)
            df_results.df.to_csv(output_dir / "results.csv", index=False)

        if dataset != datasets[-1]:
            log.info("-" * 80)
