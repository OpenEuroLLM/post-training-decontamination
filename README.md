# Decontamination

Find and remove benchmark overlap in post-training datasets using Elasticsearch and n-gram matching.

## Runbook

Run these commands from the repository root. You need Python 3.14+, uv, and Apptainer.

1. Install the dependencies.

   ```bash
   uv sync --frozen
   ```

2. Edit [datasets.yaml](configs/datasets.yaml) and [benchmarks.yaml](configs/benchmarks.yaml) to select the training data and evaluation slices.

3. Build and start Elasticsearch. Leave it running in this terminal.

   ```bash
   uv run es build
   uv run es prepare
   uv run es run
   ```

4. In a second terminal, index the datasets, run the search, and apply the exclusions.

   ```bash
   uv run ds index
   uv run ds search
   uv run ds remove
   ```

   Review `outputs/results.csv` for counts by dataset and benchmark slice. Filtered datasets are written under `outputs/`. To publish the filtered data, pass `--huggingface-id <namespace>` to `ds remove`.

## Configuration

Every subcommand accepts `--config <file>` and lists its options with `--help`. Defaults live in [configs/ds](configs/ds/) and [configs/es](configs/es/).

| File | What to set |
|---|---|
| [datasets.yaml](configs/datasets.yaml) | Set `path`, `subset`, and `split`. A subset can be a scalar, a list, or `null`. Each selected subset is indexed and filtered separately. Set an explicit `index_name` only for a single subset. |
| [benchmarks.yaml](configs/benchmarks.yaml) | Set `path`, `subset`, `split`, and `query_fields`. Lists of subsets and splits expand into every combination. Set `revision` to pin a Hub revision. Use `is_chat: true` to extract user turns, `false` to search raw text, or `null` to detect chat records automatically. |
| [search.yaml](configs/ds/search.yaml) | The default policy uses 8-grams, up to 10 hits per n-gram, and flags rows at benchmark token coverage strictly above 0.5. Set a separate `output_dir` for each policy and point removal at its `results_path`. |
| [run.yaml](configs/es/run.yaml) | Set `jvm_cpus` to override the JVM processor count. The default uses process CPU affinity. Elasticsearch image names can use `{auto-arch}` for the OCI architecture suffix. |

Elasticsearch search controls in `configs/ds/search.yaml`:

| Setting | Default | Controls |
|---|---:|---|
| `search_batch_size` | 64 | N-gram queries per HTTP request. Use 1 for individual requests. |
| `max_concurrent_searches` | `null` | Concurrent sub-searches per batch. Null uses the Elasticsearch default. |
| `prompt_batch_size` | 1 | Prompt rows that can share an HTTP batch. |
| `requests_in_flight` | 1 | Concurrent HTTP search requests. |

The matcher is adapted from [Ai2 Open Instruct](https://github.com/allenai/open-instruct/tree/main/decontamination). This project is licensed under [Apache 2.0](LICENSE).
