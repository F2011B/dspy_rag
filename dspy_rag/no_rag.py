from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import dspy

from .indexing import chunk_text, iter_explicit_files, iter_files, read_text_file
from .rag import AnswerResult, Source, configure_ssl_from_env, enable_debug_logging


@dataclass
class KnowledgeChunk:
    text: str
    rel_path: str
    chunk_index: int
    start_line: int
    end_line: int


@dataclass
class NoRAGStats:
    files_indexed: int
    chunks: int
    total_bytes: int
    build_seconds: float


class NoRAGToolingAnswer(dspy.Signature):
    """Answer using local text search and tools.

    Use search_knowledge for factual questions and cite sources like [1].
    Use calculate for math.
    Reply in German.
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Antwort auf Deutsch. Nutze Quellenzitate [n] bei Fakten.")


class NoRAGKnowledgeBase:
    def __init__(
        self,
        root_dir: Path,
        include_paths: list[Path] | None,
        max_file_size: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.root_dir = root_dir.resolve()
        self.include_paths = [p.resolve() for p in include_paths] if include_paths else None
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[KnowledgeChunk] = []
        self.stats = NoRAGStats(files_indexed=0, chunks=0, total_bytes=0, build_seconds=0.0)

    @staticmethod
    def _line_positions(text: str) -> list[int]:
        return [idx for idx, char in enumerate(text) if char == "\n"]

    @staticmethod
    def _line_for_pos(newlines: list[int], pos: int) -> int:
        lo, hi = 0, len(newlines)
        while lo < hi:
            mid = (lo + hi) // 2
            if newlines[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]{2,}", text.lower())

    def rebuild(self) -> None:
        start = time.time()
        chunks: list[KnowledgeChunk] = []
        files = 0
        total_bytes = 0

        records_iter = (
            iter_explicit_files(self.include_paths, self.root_dir, self.max_file_size)
            if self.include_paths
            else iter_files(self.root_dir, self.max_file_size)
        )

        for record in records_iter:
            files += 1
            total_bytes += record.size
            try:
                text = read_text_file(record.path)
            except Exception:
                continue
            if not text.strip():
                continue
            newlines = self._line_positions(text)
            for idx, (part, start_char, end_char) in enumerate(
                chunk_text(text, self.chunk_size, self.chunk_overlap)
            ):
                start_line = self._line_for_pos(newlines, start_char)
                end_line = self._line_for_pos(newlines, max(start_char, end_char - 1))
                chunks.append(
                    KnowledgeChunk(
                        text=part,
                        rel_path=record.rel_path,
                        chunk_index=idx,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        if not chunks:
            raise RuntimeError("No indexable text files found for no-rag mode.")

        self.chunks = chunks
        self.stats = NoRAGStats(
            files_indexed=files,
            chunks=len(chunks),
            total_bytes=total_bytes,
            build_seconds=time.time() - start,
        )

    def search(self, query: str, topk: int) -> list[tuple[KnowledgeChunk, float]]:
        terms = list(dict.fromkeys(self._tokenize(query)))
        scored: list[tuple[KnowledgeChunk, float]] = []

        for chunk in self.chunks:
            haystack = chunk.text.lower()
            score = 0.0
            for term in terms:
                hits = haystack.count(term)
                if hits:
                    score += 1.0 + min(hits, 3) * 0.25
                if term in chunk.rel_path.lower():
                    score += 0.2
            if score > 0:
                scored.append((chunk, score))

        if not scored:
            return [(chunk, 0.0) for chunk in self.chunks[:topk]]

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topk]


class NoRAGToolAgent:
    def __init__(self, kb: NoRAGKnowledgeBase, topk: int, max_iters: int = 6):
        self.kb = kb
        self.topk = topk
        self._last_sources: list[Source] = []
        self.agent = dspy.ReAct(
            signature=NoRAGToolingAnswer,
            tools=[self.search_knowledge, self.calculate],
            max_iters=max_iters,
        )

    def search_knowledge(self, query: str) -> str:
        """Search the local knowledge text via keyword matching and return numbered passages."""
        matches = self.kb.search(query, self.topk)
        if not matches:
            self._last_sources = []
            return "No relevant context found."

        lines: list[str] = []
        sources: list[Source] = []
        for idx, (chunk, _score) in enumerate(matches, start=1):
            lines.append(f"[{idx}] {chunk.text}")
            snippet = chunk.text.replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            sources.append(
                Source(
                    number=idx,
                    rel_path=chunk.rel_path,
                    chunk_index=chunk.chunk_index,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    snippet=snippet,
                )
            )

        self._last_sources = sources
        return "\n".join(lines)

    def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions."""
        try:
            interpreter = dspy.PythonInterpreter()
            return str(interpreter.execute(expression))
        except Exception as exc:
            return f"Error: {exc}"

    def answer(self, question: str) -> AnswerResult:
        self._last_sources = []
        try:
            pred = self.agent(question=question)
        except Exception as exc:
            raise RuntimeError("No-RAG tooling agent failed during LLM call") from exc

        text = pred.answer.strip() if getattr(pred, "answer", None) else ""
        sources = list(self._last_sources)

        if not text:
            return AnswerResult(answer="", sources=sources)

        if sources and "[" not in text:
            citations = " ".join(f"[{s.number}]" for s in sources)
            text = f"{text}\n\nSources: {citations}".strip()

        return AnswerResult(answer=text, sources=sources)


class NoRAGAppState:
    def __init__(
        self,
        root_dir: Path,
        include_paths: list[Path] | None,
        chunk_size: int,
        chunk_overlap: int,
        max_file_size: int,
        topk: int,
        model: str,
        api_base: str,
        api_key: str,
        max_iters: int,
    ):
        self.root_dir = root_dir.resolve()
        self.include_paths = [p.resolve() for p in include_paths] if include_paths else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self.topk = topk
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_iters = max_iters

        self.kb = NoRAGKnowledgeBase(
            root_dir=self.root_dir,
            include_paths=self.include_paths,
            max_file_size=max_file_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._agent: NoRAGToolAgent | None = None

    def setup_lm(self) -> None:
        configure_ssl_from_env()
        enable_debug_logging()
        dspy.configure(lm=dspy.LM(self.model, api_base=self.api_base, api_key=self.api_key))

    def ensure_index(self, reindex: bool = False, verbose: bool = False) -> None:
        if reindex or not self.kb.chunks:
            self.kb.rebuild()
            if verbose:
                print(f"Built no-rag search index with {self.kb.stats.chunks} chunks.")
        self._agent = NoRAGToolAgent(self.kb, topk=self.topk, max_iters=self.max_iters)

    def reload(self) -> bool:
        self.kb.rebuild()
        self._agent = NoRAGToolAgent(self.kb, topk=self.topk, max_iters=self.max_iters)
        return True

    def reindex(self, verbose: bool = False) -> None:
        self.kb.rebuild()
        self._agent = NoRAGToolAgent(self.kb, topk=self.topk, max_iters=self.max_iters)
        if verbose:
            print(f"Built no-rag search index with {self.kb.stats.chunks} chunks.")

    def answer(self, question: str) -> AnswerResult:
        if self._agent is None:
            raise RuntimeError("No-RAG app not initialized")
        return self._agent.answer(question)

    def stats_payload(self) -> dict:
        stats = self.kb.stats
        payload = {
            "files": stats.files_indexed,
            "chunks": stats.chunks,
            "bytes": stats.total_bytes,
            "root_dir": str(self.root_dir),
            "index_dir": "(disabled in --no-rag mode)",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "topk": self.topk,
            "model": self.model,
            "embed_model": "disabled",
            "mode": "no-rag",
        }
        if self.include_paths:
            payload["include_paths"] = [
                str(path.relative_to(self.root_dir)) if path.is_relative_to(self.root_dir) else path.name
                for path in self.include_paths
            ]
        return payload
