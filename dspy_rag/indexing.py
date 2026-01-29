from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "dist",
    "build",
    ".rag_index",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".scala",
    ".kt",
    ".swift",
    ".php",
    ".pl",
    ".lua",
    ".dart",
    ".r",
    ".jl",
    ".tex",
    ".adoc",
    ".org",
    ".xml",
    ".svg",
    ".env",
    ".gitignore",
    ".dockerfile",
    ".ipynb",
}

TEXT_FILENAMES = {
    "dockerfile",
    "makefile",
    "cmakelists.txt",
    "requirements.txt",
    "pyproject.toml",
    "readme",
    "license",
}


@dataclass
class FileRecord:
    path: Path
    rel_path: str
    size: int
    mtime: float


@dataclass
class ChunkMeta:
    rel_path: str
    chunk_index: int
    start_char: int
    end_char: int
    start_line: int | None
    end_line: int | None


@dataclass
class IndexStats:
    files_indexed: int
    chunks: int
    total_bytes: int
    build_seconds: float


@dataclass
class IndexPayload:
    corpus: list[str]
    metadata: list[ChunkMeta]
    fingerprint: dict
    stats: IndexStats


def is_text_filename(path: Path) -> bool:
    name = path.name.lower()
    if name in TEXT_FILENAMES:
        return True
    if name.endswith(".md"):
        return True
    if "." in name:
        ext = path.suffix.lower()
        return ext in TEXT_EXTENSIONS
    return False


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
        return b"\x00" in chunk
    except OSError:
        return True


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def iter_files(root_dir: Path, max_file_size: int) -> Iterable[FileRecord]:
    root_dir = root_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > max_file_size:
                continue
            if not is_text_filename(path):
                continue
            if is_binary_file(path):
                continue
            rel_path = str(path.relative_to(root_dir))
            yield FileRecord(path=path, rel_path=rel_path, size=stat.st_size, mtime=stat.st_mtime)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - chunk_overlap
    text_len = len(text)
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= text_len:
            break
        start += step
    return chunks


def _line_positions(text: str) -> list[int]:
    return [idx for idx, char in enumerate(text) if char == "\n"]


def _line_for_pos(newlines: list[int], pos: int) -> int:
    # bisect_right without import for speed/clarity
    lo, hi = 0, len(newlines)
    while lo < hi:
        mid = (lo + hi) // 2
        if newlines[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo + 1


def build_index_payload(
    root_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_file_size: int,
) -> IndexPayload:
    start_time = time.time()
    corpus: list[str] = []
    metadata: list[ChunkMeta] = []
    file_records: list[FileRecord] = []
    total_bytes = 0

    for record in iter_files(root_dir, max_file_size=max_file_size):
        file_records.append(record)
        total_bytes += record.size
        try:
            text = read_text_file(record.path)
        except Exception:
            continue
        if not text.strip():
            continue
        newlines = _line_positions(text)
        for idx, (chunk, start, end) in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            start_line = _line_for_pos(newlines, start)
            end_line = _line_for_pos(newlines, max(start, end - 1))
            metadata.append(
                ChunkMeta(
                    rel_path=record.rel_path,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            corpus.append(chunk)

    fingerprint = build_fingerprint(file_records)
    stats = IndexStats(
        files_indexed=len(file_records),
        chunks=len(corpus),
        total_bytes=total_bytes,
        build_seconds=time.time() - start_time,
    )
    return IndexPayload(corpus=corpus, metadata=metadata, fingerprint=fingerprint, stats=stats)


def build_fingerprint(records: list[FileRecord]) -> dict:
    files = [
        {
            "path": record.rel_path,
            "size": record.size,
            "mtime": record.mtime,
        }
        for record in sorted(records, key=lambda r: r.rel_path)
    ]
    payload = {"files": files}
    digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["sha256"] = digest
    return payload


def build_fingerprint_from_dir(root_dir: Path, max_file_size: int) -> dict:
    records = list(iter_files(root_dir, max_file_size=max_file_size))
    return build_fingerprint(records)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(index_dir: Path, metadata: list[ChunkMeta]) -> None:
    payload = [meta.__dict__ for meta in metadata]
    save_json(index_dir / "metadata.json", payload)


def load_metadata(index_dir: Path) -> list[ChunkMeta]:
    data = load_json(index_dir / "metadata.json")
    return [ChunkMeta(**item) for item in data]


def save_config(index_dir: Path, config: dict) -> None:
    save_json(index_dir / "config.json", config)


def load_config(index_dir: Path) -> dict:
    return load_json(index_dir / "config.json")


def save_fingerprint(index_dir: Path, fingerprint: dict) -> None:
    save_json(index_dir / "fingerprint.json", fingerprint)


def load_fingerprint(index_dir: Path) -> dict:
    return load_json(index_dir / "fingerprint.json")


def save_stats(index_dir: Path, stats: IndexStats) -> None:
    save_json(index_dir / "stats.json", stats.__dict__)


def load_stats(index_dir: Path) -> IndexStats:
    data = load_json(index_dir / "stats.json")
    return IndexStats(**data)
