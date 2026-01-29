from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

#
# Lazy-import heavy modules inside main() so `-h` works even if deps
# are not installed yet.
#


def normalize_lm_model(model: str, api_base: str | None) -> str:
    if "/" not in model:
        model = f"openai/{model}"
    if api_base and "openrouter.ai" in api_base and not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    return model


def normalize_embed_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("api_base must be a valid URL")
    path = parsed.path or ""
    if path in ("", "/"):
        path = "/v1"
    normalized = urlunparse(
        (parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment)
    )
    return normalized.rstrip("/")


def load_env_fallback(args: argparse.Namespace) -> None:
    if not args.api_base:
        args.api_base = os.environ.get("RAG_API_BASE")
    if not args.api_key:
        args.api_key = os.environ.get("RAG_API_KEY")
    if not args.model:
        args.model = os.environ.get("RAG_MODEL")
    if not args.embed_model:
        args.embed_model = os.environ.get("RAG_EMBED_MODEL")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspy_rag",
        description="Folder-based RAG tool with a TUI.",
    )
    parser.add_argument("folder", help="Folder to index")
    parser.add_argument("--api-base", dest="api_base", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", dest="api_key", help="API key")
    parser.add_argument("--model", dest="model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument(
        "--embed-model",
        dest="embed_model",
        default="text-embedding-3-small",
        help="Embedding model",
    )
    parser.add_argument("--topk", dest="topk", type=int, default=5)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=200)
    parser.add_argument("--index-dir", dest="index_dir")
    parser.add_argument("--max-file-size", dest="max_file_size", type=int, default=3_000_000)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.api_base or not args.api_key:
        raise ValueError(
            "Missing api_base/api_key. Use --api-base/--api-key or set RAG_API_BASE/RAG_API_KEY."
        )
    args.api_base = normalize_api_base(args.api_base)
    args.model = normalize_lm_model(args.model, args.api_base)
    args.embed_model = normalize_embed_model(args.embed_model)

    folder = Path(args.folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise ValueError("folder must be an existing directory")
    args.folder = str(folder)

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_fallback(args)

    try:
        validate_args(args)
    except Exception as exc:
        parser.error(str(exc))

    root_dir = Path(args.folder).resolve()
    index_dir = (
        Path(args.index_dir).expanduser().resolve()
        if args.index_dir
        else root_dir / ".rag_index"
    )

    # Lazy imports here to avoid importing DSPy when only requesting --help.
    from .rag import IndexManager, RAGAppState, configure_ssl_from_env
    from .tui import FolderRAGTUI

    configure_ssl_from_env()

    index_manager = IndexManager(
        root_dir=root_dir,
        index_dir=index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_file_size=args.max_file_size,
        embed_model=args.embed_model,
        api_base=args.api_base,
        api_key=args.api_key,
        topk=args.topk,
    )

    app_state = RAGAppState(
        index_manager=index_manager,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        topk=args.topk,
    )

    if args.verbose:
        print(f"Index dir: {index_dir}")

    try:
        app_state.setup_lm()
        print("Preparing index...")
        app_state.ensure_index(reindex=args.reindex, verbose=args.verbose)
        print("Index ready. Launching TUI...")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    tui = FolderRAGTUI(app_state)
    tui.run()


if __name__ == "__main__":
    main()
