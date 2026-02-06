from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from .cli import (
    load_env_fallback,
    normalize_api_base,
    normalize_embed_model,
    normalize_lm_model,
)
from .no_rag import NoRAGAppState
from .rag import IndexManager, RAGAppState, ToolingRAG, configure_ssl_from_env, format_exception_chain
from .tui import FolderRAGTUI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspy_custom_gpt",
        description="Custom GPT TUI with a file-based knowledge base and tool use.",
    )
    parser.add_argument("source", help="Text file or folder containing the knowledge base")
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
    parser.add_argument("--max-iters", dest="max_iters", type=int, default=6)
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable embeddings and use keyword-based local search instead.",
    )
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_source(args: argparse.Namespace) -> tuple[Path, list[Path] | None, Path]:
    source = Path(args.source).expanduser()
    if source.exists() and source.is_file():
        root_dir = source.parent.resolve()
        include_paths = [source.resolve()]
        if args.index_dir:
            index_dir = Path(args.index_dir).expanduser().resolve()
        else:
            index_dir = Path(f"{source}.rag_index").resolve()
        return root_dir, include_paths, index_dir
    if source.exists() and source.is_dir():
        root_dir = source.resolve()
        include_paths = None
        index_dir = (
            Path(args.index_dir).expanduser().resolve()
            if args.index_dir
            else root_dir / ".rag_index"
        )
        return root_dir, include_paths, index_dir
    raise ValueError("source must be an existing text file or directory")


def validate_args(args: argparse.Namespace) -> tuple[Path, list[Path] | None, Path]:
    if not args.api_base or not args.api_key:
        raise ValueError(
            "Missing api_base/api_key. Use --api-base/--api-key or set RAG_API_BASE/RAG_API_KEY."
        )
    args.api_base = normalize_api_base(args.api_base)
    args.model = normalize_lm_model(args.model, args.api_base)
    if not args.no_rag:
        args.embed_model = normalize_embed_model(args.embed_model)

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return _resolve_source(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_fallback(args)

    try:
        root_dir, include_paths, index_dir = validate_args(args)
    except Exception as exc:
        parser.error(str(exc))

    configure_ssl_from_env()

    if args.no_rag:
        app_state = NoRAGAppState(
            root_dir=root_dir,
            include_paths=include_paths,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_file_size=args.max_file_size,
            topk=args.topk,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            max_iters=args.max_iters,
        )
    else:
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
            include_paths=include_paths,
        )

        rag_factory = partial(ToolingRAG, max_iters=args.max_iters)

        app_state = RAGAppState(
            index_manager=index_manager,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            topk=args.topk,
            rag_factory=rag_factory,
        )

    if args.verbose:
        if args.no_rag:
            print("No-RAG mode active (embeddings disabled).")
        else:
            print(f"Index dir: {index_dir}")

    try:
        app_state.setup_lm()
        print("Preparing index...")
        app_state.ensure_index(reindex=args.reindex, verbose=args.verbose)
        print("Index ready. Launching TUI...")
    except Exception as exc:
        print(f"Error:\n{format_exception_chain(exc)}", file=sys.stderr)
        sys.exit(1)

    tui = FolderRAGTUI(app_state)
    tui.run()


if __name__ == "__main__":
    main()
