# Decontamination

An n-gram-based decontamination tool for LLM post-training datasets.

--------------------

# TL;DR

```bash
uv pip install https://github.com/OpenEuroLLM/post-training-decontamination.git

uv run es build         # Build an Elasticsearch apptainer image
uv run es prepare       # Prepare the image mounts and single-node configuration
uv run es run           # Run the Elasticsearch container

uv run ds index         # Index datasets defined in `configs/datasets.yaml`
uv run ds search        # Find overlap with benchmarks defined in `configs/benchmarks.yaml`
uv run ds remove [--huggingface-id <huggingface-id>]  # Decontaminate and (optionally) push to HF
```

--------------------

# Installation

```bash
uv pip install https://github.com/OpenEuroLLM/post-training-decontamination.git
```

# CLI

## Configuration

All commands can be configured using either:
1. command arguments. Check
  - `uv run es [subcommand] --help`
  - `uv run ds [subcommand] --help`
2. a YAML config file with the `--config <config_file>` argument. Check
  - `configs/es/build.yaml`
  - `configs/es/prepare.yaml`
  - `configs/es/run.yaml`
  - `configs/ds/index.yaml`
  - `configs/ds/search.yaml`
  - `configs/ds/remove.yaml`

## Autocompletion

To enable autocompletion for the CLI, run the following commands:
```bash
uv run es --install-completion [bash|zsh|fish]
uv run ds --install-completion [bash|zsh|fish]
```

# Usage

Run the commands below from the repository root so the CLI can find the default config files.

## 1. Run `elasticsearch` container

```bash
uv run es build         # Build an Elasticsearch apptainer image
uv run es prepare       # Prepare the image mounts and single-node configuration
uv run es run           # Run the Elasticsearch container
```

Elasticsearch is used as a search engine to efficiently index the training datasets and perform n-gram search for overlap with evaluation benchmarks.
Set `jvm_cpus` in `configs/es/run.yaml` to override the processor count, which defaults to CPU affinity.
Image names in the Elasticsearch configs can use `{auto-arch}` for the OCI architecture suffix.

## 2. Index datasets into Elasticsearch

```bash
uv run ds index
```

The datasets to be indexed are specified in `configs/datasets.yaml`.
`subset` accepts a name, a list of names, or `null`. Each selected subset is indexed and filtered separately.
Set `index_name` only for an entry with a single subset.

```yaml
- ...
  path: <dataset_path>
  subset: <subset>
  split: <split>
  ...
```

## 3. Find contaminated samples

```bash
uv run ds search
```

The benchmarks to search are specified in `configs/benchmarks.yaml`.
`subset` and `split` each accept a scalar or a list. The search runs every subset and split combination.
Set `revision` to pin the Hub version. Use `is_chat: true` to extract user turns,
`false` to search raw fields, or `null` to detect chat records automatically.

```yaml
- ...
  path: <benchmark_path>
  subset:
    - en
    - fr
  split: test
  query_fields:
    - prompt
  is_chat: false
  revision: <commit_sha>
```

The search results are saved to `outputs/results.csv`, with a row for each dataset and benchmark slice.
The default policy uses 8-grams and up to 10 hits per n-gram. A row is flagged when shared
n-grams cover strictly more than half the benchmark prompt's tokens.

In `configs/ds/search.yaml`, `search_batch_size` controls n-gram queries per HTTP request
(use 1 for individual requests), `prompt_batch_size` controls how many prompts can share a batch,
and `requests_in_flight` limits concurrent HTTP requests. Set `max_concurrent_searches` to
cap concurrent sub-searches per batch, or leave it `null` to use Elasticsearch's default.

## 4. Remove contaminated samples

```bash
uv run ds remove [--huggingface-id <huggingface-id>]
```

The decontaminated dataset is saved to
- `outputs/<dataset_name>-decontaminated/` locally, and
- `<huggingface-id>/<dataset_name>-decontaminated/` on Hugging Face (if `--huggingface-id` is provided).

If you changed the search output directory, set `results_path` in `configs/ds/remove.yaml` to its `results.csv`.

--------------------

# Command abbreviations

- `es` -> `elasticsearch`
- `ds` -> `datasets`

--------------------

# Acknowledgements

The decontamination scripts in this project are adapted from [allenai/open-instruct](https://github.com/allenai/open-instruct), AllenAI's post-training codebase. See their [Contamination checks](https://github.com/allenai/open-instruct#contamination-checks) section for the original implementation.

--------------------

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
