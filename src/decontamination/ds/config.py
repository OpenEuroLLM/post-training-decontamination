from ast import literal_eval
from dataclasses import dataclass, replace
from itertools import product

import pandas as pd

from decontamination.util import TypedRow


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset to be indexed and searched in Elasticsearch.

    Args:
        display_name: A human-readable name for the dataset (optional).
        description: A description of the dataset (optional).
        path: The path to the dataset.
        subset: A subset, a list of subsets, or null.
        split: The split of the dataset (e.g., "train", "test").
        messages_field: The field in the dataset that contains the text to be
                        indexed and searched (default: "messages").
        query_filter: A tuple of (field, value) to filter the dataset for the
                      benchmark queries (default: ("role", "user")).
        query_field: The field in the dataset that contains the text to be
                     used for querying (default: "content").
        index_name: The name of the Elasticsearch index to use for this dataset
                    (optional; if not provided, a default name will be generated
                    based on the path, subset, and split).
    """

    display_name: str | None
    description: str | None
    path: str
    subset: str | list[str | None] | None
    split: str
    messages_field: str = "messages"
    query_filter: tuple[str, str] = ("role", "user")
    query_field: str = "content"
    index_name: str | None = None


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark to be searched for contamination.

    Args:
        display_name: A human-readable name for the benchmark.
        description: A description of the benchmark (optional).
        path: The path to the benchmark dataset.
        subset: A subset, a list of subsets, or null.
        split: A split or a list of splits.
        data_files: The data files in the benchmark dataset to use for querying.
        query_fields: The fields in the benchmark dataset to use for querying.
        is_chat: Extract user messages when true, use raw fields when false,
                 or infer from the field value when null.
        revision: Optional Hugging Face dataset revision.
    """

    display_name: str
    description: str | None
    path: str
    subset: str | list[str | None] | None
    split: str | list[str]
    data_files: list[str] | None
    query_fields: list[str]
    is_chat: bool | None = None
    revision: str | None = None


def _choices(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(v is not None and (not isinstance(v, str) or not v.strip()) for v in values):
        raise ValueError(f"Expected nonempty subset or split values, got {value!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate subset or split values: {value!r}")
    return values


def expand_datasets(datasets: list[DatasetConfig]) -> list[DatasetConfig]:
    expanded = []
    for dataset in datasets:
        subsets = _choices(dataset.subset)
        if len(subsets) > 1 and dataset.index_name:
            raise ValueError("An explicit index_name requires a single dataset subset")
        expanded.extend(replace(dataset, subset=subset) for subset in subsets)
    return expanded


def expand_benchmarks(benchmarks: list[BenchmarkConfig]) -> list[BenchmarkConfig]:
    expanded = []
    for benchmark in benchmarks:
        splits = _choices(benchmark.split)
        if None in splits:
            raise ValueError("Benchmark splits must be strings")
        expanded.extend(
            replace(benchmark, subset=subset, split=split)
            for subset, split in product(_choices(benchmark.subset), splits)
        )
    return expanded


@dataclass
class Result(TypedRow["Result"]):
    """
    A result of a contamination search.

    Args:
        dataset_path: The path to the dataset that was searched.
        dataset_subset: The subset of the dataset that was searched (if any).
        dataset_split: The split of the dataset that was searched.
        dataset_query_filter: The query filter used for the dataset.
        dataset_query_field: The query field used for the dataset.
        benchmark_path: The path to the benchmark that was searched.
        benchmark_subset: The subset of the benchmark that was searched (if any).
        benchmark_split: The split of the benchmark that was searched.
        benchmark_query_fields: The query fields used for the benchmark.
        dataset_num_docs: The total number of documents in the dataset.
        dataset_num_contaminated_docs: The number of contaminated documents found in
                                      the dataset.
        dataset_contamination_fraction: The fraction of contaminated documents in
                                      the dataset.
        benchmark_num_docs: The total number of documents in the benchmark.
        benchmark_num_contaminated_docs: The number of contaminated documents found
                                        in the benchmark.
        benchmark_contamination_fraction: The fraction of contaminated documents in
                                        the benchmark.
        ngram_size: The n-gram size used for matching.
        match_threshold: The similarity threshold used for matching.
        contaminated_ids: A dictionary mapping document IDs in the dataset to sets
                         of matching document IDs in the benchmark.
    """

    dataset_path: str
    dataset_subset: str | None
    dataset_split: str
    dataset_query_filter: tuple[str, str]
    dataset_query_field: str

    benchmark_path: str
    benchmark_subset: str | None
    benchmark_split: str
    benchmark_query_fields: list[str]

    dataset_num_docs: int
    dataset_num_contaminated_docs: int
    dataset_contamination_fraction: float

    benchmark_num_docs: int
    benchmark_num_contaminated_docs: int
    benchmark_contamination_fraction: float

    ngram_size: int
    match_threshold: float

    contaminated_ids: dict[int, set[int]]
    benchmark_is_chat: bool | None = None
    benchmark_revision: str | None = None
    top_k: int | None = None

    def from_row(row: pd.Series) -> "Result":
        dataset_query_filter = row.pop("dataset_query_filter")
        dataset_query_filter = (
            literal_eval(dataset_query_filter)
            if isinstance(dataset_query_filter, str)
            else dataset_query_filter
        )
        benchmark_query_fields = row.pop("benchmark_query_fields")
        benchmark_query_fields = (
            literal_eval(benchmark_query_fields)
            if isinstance(benchmark_query_fields, str)
            else benchmark_query_fields
        )
        contaminated_ids = row.pop("contaminated_ids")
        contaminated_ids = (
            literal_eval(contaminated_ids)
            if isinstance(contaminated_ids, str)
            else contaminated_ids
        )
        values = row.where(row.notna(), None).to_dict()
        return Result(
            dataset_query_filter=dataset_query_filter,
            benchmark_query_fields=benchmark_query_fields,
            contaminated_ids=contaminated_ids,
            **values,
        )

    def is_same_setup(self, other: "Result") -> bool:
        """
        Check if this result is from the same dataset and benchmark setup as another result.

        Args:
            other: The other result to compare against.

        Returns:
            True if the results are from the same setup, False otherwise.
        """
        return (
            self.dataset_path == other.dataset_path
            and self.dataset_subset == other.dataset_subset
            and self.dataset_split == other.dataset_split
            and self.dataset_query_filter == other.dataset_query_filter
            and self.dataset_query_field == other.dataset_query_field
            and self.benchmark_path == other.benchmark_path
            and self.benchmark_subset == other.benchmark_subset
            and self.benchmark_split == other.benchmark_split
            and self.benchmark_query_fields == other.benchmark_query_fields
            and self.benchmark_is_chat == other.benchmark_is_chat
            and self.benchmark_revision == other.benchmark_revision
            and self.top_k == other.top_k
            and self.ngram_size == other.ngram_size
            and self.match_threshold == other.match_threshold
        )
