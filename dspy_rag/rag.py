from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


import dspy

from .indexing import (
    IndexStats,
    build_index_payload,
    build_fingerprint_from_dir,
    load_config,
    load_fingerprint,
    load_metadata,
    load_stats,
    save_config,
    save_fingerprint,
    save_metadata,
    save_stats,
)


@dataclass
class Source:
    number: int
    rel_path: str
    chunk_index: int
    start_line: int | None
    end_line: int | None
    snippet: str


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source]


class GenerateAnswer(dspy.Signature):
    """Answer questions using only the provided context with citations."""

    context: list[str] = dspy.InputField(
        desc="Retrieved passages labeled with [n]. Use them to answer."
    )
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Grounded answer that cites sources like [1]. If context is insufficient, say what is missing."
    )


class FolderRAG:
    def __init__(self, embeddings, metadata: list[dict], topk: int):
        self.embeddings = embeddings
        self.metadata = metadata
        self.topk = topk
        self.generator = dspy.Predict(GenerateAnswer)

    def answer(self, question: str) -> AnswerResult:
        try:
            retrieval = self.embeddings(question)
        except Exception as exc:
            raise RuntimeError("Retrieval failed during embedding search") from exc
        passages: list[str] = list(getattr(retrieval, "passages", []) or [])
        indices: list[int] = list(getattr(retrieval, "indices", []) or [])

        if not passages:
            return AnswerResult(answer="No relevant context found.", sources=[])

        pairs = list(zip(indices, passages))[: self.topk]
        numbered_context = [f"[{idx + 1}] {text}" for idx, (_, text) in enumerate(pairs)]
        try:
            prediction = self.generator(context=numbered_context, question=question)
        except Exception as exc:
            raise RuntimeError("Generation failed during LLM call") from exc
        raw_answer = prediction.answer.strip() if getattr(prediction, "answer", None) else ""

        sources: list[Source] = []
        for local_idx, (corpus_idx, passage) in enumerate(pairs, start=1):
            meta = self.metadata[corpus_idx]
            snippet = passage
            snippet = snippet.replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            sources.append(
                Source(
                    number=local_idx,
                    rel_path=meta["rel_path"],
                    chunk_index=meta["chunk_index"],
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    snippet=snippet,
                )
            )

        if not raw_answer:
            return AnswerResult(answer="", sources=sources)

        if "[" not in raw_answer:
            cite_list = " ".join(f"[{source.number}]" for source in sources)
            raw_answer = f"{raw_answer}\n\nSources: {cite_list}".strip()

        return AnswerResult(answer=raw_answer, sources=sources)


class IndexManager:
    def __init__(
        self,
        root_dir: Path,
        index_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        max_file_size: int,
        embed_model: str,
        api_base: str,
        api_key: str,
        topk: int,
    ):
        self.root_dir = root_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self.embed_model = embed_model
        self.api_base = api_base
        self.api_key = api_key
        self.topk = topk

        self._metadata: list[dict] = []
        self._embeddings = None
        self._stats: IndexStats | None = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            raise RuntimeError("Index not loaded")
        self._embeddings.k = self.topk
        return self._embeddings

    @property
    def metadata(self) -> list[dict]:
        return self._metadata

    @property
    def stats(self) -> IndexStats | None:
        return self._stats

    def _config_payload(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_file_size": self.max_file_size,
            "embed_model": self.embed_model,
        }

    def _embeddings_dir(self) -> Path:
        return self.index_dir / "embeddings"

    def _index_exists(self) -> bool:
        return (
            self._embeddings_dir().exists()
            and (self.index_dir / "metadata.json").exists()
            and (self.index_dir / "fingerprint.json").exists()
            and (self.index_dir / "config.json").exists()
        )

    def _load_saved(self) -> None:
        metadata = load_metadata(self.index_dir)
        self._metadata = [meta.__dict__ for meta in metadata]
        self._stats = load_stats(self.index_dir)
        embedder = build_embedder(self.embed_model, self.api_base, self.api_key)
        self._embeddings = dspy.Embeddings.from_saved(self._embeddings_dir(), embedder)
        self._embeddings.k = self.topk

    def load(self) -> bool:
        if not self._index_exists():
            return False
        config = load_config(self.index_dir)
        if config != self._config_payload():
            return False
        self._load_saved()
        return True

    def needs_reindex(self) -> bool:
        if not self._index_exists():
            return True
        config = load_config(self.index_dir)
        if config != self._config_payload():
            return True
        try:
            existing = load_fingerprint(self.index_dir)
        except Exception:
            return True
        current = self._current_fingerprint()
        return existing.get("sha256") != current.get("sha256")

    def _current_fingerprint(self) -> dict:
        return build_fingerprint_from_dir(self.root_dir, self.max_file_size)

    def build(self, verbose: bool = False) -> IndexPayload:
        try:
            payload = build_index_payload(
                self.root_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_file_size=self.max_file_size,
            )
        except Exception as exc:
            raise RuntimeError("Indexing failed during file scan/chunking") from exc
        if not payload.corpus:
            raise RuntimeError("No indexable text files found.")

        try:
            embedder = build_embedder(self.embed_model, self.api_base, self.api_key)
            brute_force_threshold = resolve_bruteforce_threshold()
            embeddings = dspy.Embeddings(
                payload.corpus,
                embedder,
                k=self.topk,
                brute_force_threshold=brute_force_threshold,
            )
        except Exception as exc:
            raise RuntimeError("Indexing failed during embedding generation") from exc
        self.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            embeddings.save(self._embeddings_dir())
        except Exception as exc:
            raise RuntimeError("Indexing failed while saving embeddings") from exc

        try:
            save_metadata(self.index_dir, payload.metadata)
            save_fingerprint(self.index_dir, payload.fingerprint)
            save_config(self.index_dir, self._config_payload())
            save_stats(self.index_dir, payload.stats)
        except Exception as exc:
            raise RuntimeError("Indexing failed while saving metadata") from exc

        self._metadata = [meta.__dict__ for meta in payload.metadata]
        self._embeddings = embeddings
        self._stats = payload.stats
        if verbose:
            print(
                f"Indexed {payload.stats.files_indexed} files into {payload.stats.chunks} chunks."
            )
        return payload


class RAGAppState:
    def __init__(
        self,
        index_manager: IndexManager,
        model: str,
        api_base: str,
        api_key: str,
        topk: int,
    ):
        self.index_manager = index_manager
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.topk = topk
        self._rag: FolderRAG | None = None

    def setup_lm(self) -> None:
        configure_ssl_from_env()
        enable_debug_logging()
        dspy.configure(lm=dspy.LM(self.model, api_base=self.api_base, api_key=self.api_key))

    def ensure_index(self, reindex: bool = False, verbose: bool = False) -> None:
        if reindex or self.index_manager.needs_reindex():
            self.index_manager.build(verbose=verbose)
        else:
            loaded = self.index_manager.load()
            if not loaded:
                self.index_manager.build(verbose=verbose)
        self._rag = FolderRAG(self.index_manager.embeddings, self.index_manager.metadata, self.topk)

    def reload(self) -> bool:
        loaded = self.index_manager.load()
        if loaded:
            self._rag = FolderRAG(self.index_manager.embeddings, self.index_manager.metadata, self.topk)
        return loaded

    def reindex(self, verbose: bool = False) -> None:
        self.index_manager.build(verbose=verbose)
        self._rag = FolderRAG(self.index_manager.embeddings, self.index_manager.metadata, self.topk)

    def answer(self, question: str) -> AnswerResult:
        if self._rag is None:
            raise RuntimeError("RAG not initialized")
        return self._rag.answer(question)

    def stats_payload(self) -> dict:
        stats = self.index_manager.stats
        return {
            "files": stats.files_indexed if stats else 0,
            "chunks": stats.chunks if stats else 0,
            "bytes": stats.total_bytes if stats else 0,
            "root_dir": str(self.index_manager.root_dir),
            "index_dir": str(self.index_manager.index_dir),
            "chunk_size": self.index_manager.chunk_size,
            "chunk_overlap": self.index_manager.chunk_overlap,
            "topk": self.topk,
            "model": self.model,
            "embed_model": self.index_manager.embed_model,
        }


def build_embedder(embed_model: str, api_base: str, api_key: str):
    configure_ssl_from_env()
    enable_debug_logging()
    return dspy.Embedder(
        embed_model,
        api_base=api_base,
        api_key=api_key,
        encoding_format="float",
    )


def resolve_bruteforce_threshold() -> int:
    try:
        __import__("faiss")
        return 20000
    except Exception:
        return 1_000_000_000


def configure_ssl_from_env() -> None:
    cert_path = os.environ.get("SSL_CERT_FILE")
    if not cert_path:
        return
    if not Path(cert_path).exists():
        raise RuntimeError(f"SSL_CERT_FILE points to missing file: {cert_path}")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    os.environ.setdefault("OPENAI_CA_BUNDLE", cert_path)


def enable_debug_logging() -> None:
    try:
        import litellm
    except Exception:
        return
    try:
        litellm.set_verbose(True)
        litellm.suppress_debug_info = False
    except Exception:
        return


def format_exception_chain(exc: BaseException, max_depth: int = 6) -> str:
    lines = []
    current: BaseException | None = exc
    depth = 0
    while current and depth < max_depth:
        message = str(current).strip()
        label = current.__class__.__name__
        if message:
            lines.append(f"{label}: {message}")
        else:
            lines.append(label)
        current = current.__cause__ or current.__context__
        depth += 1
    return "\nCaused by: ".join(lines)
