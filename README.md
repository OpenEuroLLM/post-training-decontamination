# Decontamination

An n-gram-based decontamination tool for LLM post-training datasets.

--------------------

# TL;DR

```bash
uv pip install https://github.com/OpenEuroLLM/decontamination.git

es build         # Build an Elasticsearch apptainer image
es prepare       # Prepare the image mounts and single-node configuration
es run           # Run the Elasticsearch container

ds index         # Index datasets defined in `configs/datasets.yaml`
ds search        # Find overlap with benchmarks defined in `configs/benchmarks.yaml`
ds remove [--huggingface-id <huggingface-id>]  # Decontaminate and (optionally) push to HF
```

--------------------

# Installation

```bash
uv pip install https://github.com/OpenEuroLLM/decontamination.git
```

# CLI

## Configuration

All commands can be configured using either:
1. command arguments. Check
  - `es [subcommand] --help`
  - `ds [subcommand] --help`
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
es --install-completion [bash|zsh|fish]
ds --install-completion [bash|zsh|fish]
```

# Usage

## 1. Run `elasticsearch` container

```bash
es build         # Build an Elasticsearch apptainer image
es prepare       # Prepare the image mounts and single-node configuration
es run           # Run the Elasticsearch container
```

Elasticsearch indexes training text and retrieves candidates for the n-gram overlap check.
Set `jvm_cpus` in `configs/es/run.yaml` to override the JVM processor count.
An omitted or null value uses the CPUs available to the current process.
Image names in the build, prepare, and run configs can include `{auto-arch}`.
The CLI substitutes the OCI architecture (for example, `amd64` or `arm64`).

## 2. Index datasets into Elasticsearch

```bash
ds index
```

The datasets to index are specified in `configs/datasets.yaml`. Use a single
subset, a list of subsets, or `null` when no subset is needed. Each subset is
indexed and removed separately. Set `index_name` only on entries with a single subset.

```yaml
- display_name: Example training dataset
  description: Training conversations
  path: example/training-data
  subset:
    - en
    - fr
  split: train
  messages_field: messages
  query_filter:
    - role
    - user
  query_field: content
```

## 3. Find contaminated samples

```bash
ds search
```

Configure benchmarks in `configs/benchmarks.yaml`. Both `subset` and `split`
accept scalars or YAML lists. The CLI searches every selected subset against
every selected split. When `revision` is set, it loads the specified Hub revision.

```yaml
- display_name: Example benchmark
  description: Multilingual evaluation conversations
  path: example/benchmark
  subset:
    - en
    - fr
  split:
    - validation
    - test
  data_files: null
  query_fields:
    - messages
  is_chat: true
  revision: <dataset_commit_sha>
```

`is_chat: true` extracts and joins user turns, including text blocks in
multimodal messages. Use `false` to search the raw fields. Omit it or use
`null` to detect message lists, including JSON-encoded lists. The ComparIA
and LMArena configs scan raw source prompts.

The matcher retrieves up to `top_k` hits per n-gram. It flags a training row
when shared n-grams cover strictly more than `match_threshold` of the benchmark
prompt's tokens. The defaults in `configs/ds/search.yaml` are 8-grams,
a threshold of 0.5, and 10 hits per n-gram.

| Setting | Meaning | Checked-in value |
|---|---|---:|
| `search_batch_size` | N-gram queries per Elasticsearch request. Use 1 for individual requests. | 64 |
| `max_concurrent_searches` | Maximum sub-searches per batch, capped at the batch length. Use `null` for Elasticsearch's default. | `null` |
| `prompt_batch_size` | Number of prompt rows that can share a batch. | 1 |
| `requests_in_flight` | Maximum concurrent HTTP search requests. | 1 |

The search writes `outputs/results.csv` with the resolved dataset and
benchmark subset and split. Use a fresh output directory for a separate
decontamination policy or when migrating older results. Point `ds remove`
at the corresponding results file.

## 4. Remove contaminated samples

```bash
ds remove [--huggingface-id <huggingface-id>]
```

The decontaminated dataset is saved to 
- `outputs/<dataset_name>-decontaminated/` locally, and
- `<huggingface-id>/<dataset_name>-decontaminated/` on Hugging Face (if `--huggingface-id` is provided).

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
