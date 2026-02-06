# dspy_rag

Folder-based RAG tool with a fullscreen TUI built on DSPy. It indexes a local folder, retrieves the most relevant chunks for a question, and answers with grounded citations.

## Requirements

- Python 3.10+
- OpenAI-compatible API endpoint
- API key

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For local embeddings:

```bash
pip install -e '.[local]'
```

For fastembed embeddings (offline-capable):

```bash
pip install -e '.[fastembed]'
```

## Usage

```bash
dspy_rag /path/to/folder \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY
```

You can also use module mode:

```bash
python -m dspy_rag /path/to/folder --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY
```

## Custom GPT (File + Tools)

Use a single knowledge file (or a folder) and enable tool use via a ReAct agent:

```bash
dspy_custom_gpt /path/to/knowledge.txt \\
  --api-base https://api.openai.com/v1 \\
  --api-key $OPENAI_API_KEY
```

If you pass a folder, it indexes the folder like `dspy_rag` but answers using tools.

Available tools:
- `search_knowledge`: retrieves passages from the local knowledge base for grounded answers.
- `calculate`: evaluates math expressions.

Options:
- `--max-iters <N>`: max tool steps for the agent (default: 6).
- `--no-rag`: disable embeddings and use keyword-based local text search.

No-RAG example (no embeddings at all):

```bash
dspy_custom_gpt /path/to/knowledge.txt \\
  --api-base https://openrouter.ai/api/v1 \\
  --api-key $OPENROUTER_API_KEY \\
  --model openrouter/openai/gpt-4o-mini \\
  --no-rag
```

## CLI Options

- `--api-base <URL>`: OpenAI-compatible base URL (required)
- `--api-key <KEY>`: API key (required)
- `--model <MODEL>`: LLM model (default: `gpt-4o-mini`)
- `--embed-model <MODEL>`: embedding model (default: `text-embedding-3-small`)
- `--topk <N>`: number of retrieved chunks (default: 5)
- `--chunk-size <N>`: chunk size in characters (default: 1200)
- `--chunk-overlap <N>`: overlap in characters (default: 200)
- `--index-dir <PATH>`: index directory (default: `<folder>/.rag_index`)
- `--max-file-size <BYTES>`: max file size to index (default: 3000000)
- `--reindex`: rebuild index
- `--verbose`: verbose logs

Model names without a provider prefix are automatically normalized to `openai/<model>`.

Local embedding models (no Hugging Face network)
- You can pass a local model path or a pip-installed package resource:
  - `--embed-model local:/path/to/model`
  - `--embed-model /path/to/model`
  - `--embed-model pkg:my_model_pkg` or `pkg:my_model_pkg/subdir`
- This uses `sentence-transformers` and forces offline mode (`HF_HUB_OFFLINE=1`).
- Install locally: `pip install sentence-transformers` and your model package (pip).

Fastembed (offline, bundled in pip if provided)
- Use `--embed-model fastembed` to load fastembed's default model.
- Or `--embed-model fastembed:MODEL_NAME` to select a specific fastembed model.
- Install: `pip install fastembed` or `pip install -e '.[fastembed]'`.

## Environment Variables

CLI arguments take precedence over environment variables.

- `RAG_API_BASE`
- `RAG_API_KEY`
- `RAG_MODEL`
- `RAG_EMBED_MODEL`
- `SSL_CERT_FILE` (CA bundle path used for TLS verification)

When `SSL_CERT_FILE` is set, the tool propagates it to HTTP client env vars while keeping TLS verification enabled.

## TUI Keybindings

- `F1`: Help
- `F2`: Send message
- `Ctrl+C`: Exit
- `Enter`: New line in input

## Slash Commands

- `/help`: show help
- `/exit`: exit the TUI
- `/reindex`: rebuild the index
- `/reload`: load index from disk if present
- `/clear`: clear the chat log
- `/stats`: show index stats and settings
- `/sources`: reprint last sources

## Notes

- Binary files are skipped via a null-byte check.
- Large files over `--max-file-size` are skipped.
- Default excluded directories: `.git`, `.hg`, `.svn`, `node_modules`, `dist`, `build`, `.venv`, `venv`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.idea`, `.vscode`.

## Minimal Test

```bash
pytest -q
```

## Troubleshooting

- Missing API base or key: pass `--api-base`/`--api-key` or set `RAG_API_BASE`/`RAG_API_KEY`.
- Empty index: confirm your folder contains text files and is not excluded.
