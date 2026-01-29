from __future__ import annotations

import importlib
import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class LocalEmbedder:
    model_path: str
    batch_size: int = 64

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        # Enforce offline mode to avoid any HF network calls.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._model = SentenceTransformer(self.model_path)

    def __call__(self, inputs: Iterable[str]):
        texts = list(inputs)
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def resolve_local_model_path(spec: str) -> str:
    if spec.startswith("local:"):
        return spec[len("local:") :]
    if spec.startswith("pkg:"):
        value = spec[len("pkg:") :]
        if "/" in value:
            pkg, subpath = value.split("/", 1)
        else:
            pkg, subpath = value, ""
        package = importlib.import_module(pkg)
        base = importlib.resources.files(package)
        path = base.joinpath(subpath) if subpath else base
        return str(path)
    return spec


def is_local_model_spec(spec: str) -> bool:
    if spec.startswith("local:") or spec.startswith("pkg:"):
        return True
    return Path(spec).expanduser().exists()


def build_local_embedder(spec: str) -> LocalEmbedder:
    model_path = resolve_local_model_path(spec)
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Local embedding model path not found: {model_path}")
    return LocalEmbedder(str(path))


@dataclass
class FastEmbedder:
    model_name: str | None = None
    batch_size: int = 64

    def __post_init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "fastembed is required for fastembed embeddings. "
                "Install with: pip install fastembed"
            ) from exc

        # Enforce offline mode to avoid network calls (models must be preinstalled).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if self.model_name:
            self._model = TextEmbedding(model_name=self.model_name)
        else:
            self._model = TextEmbedding()

    def __call__(self, inputs: Iterable[str]):
        texts = list(inputs)
        if not texts:
            return []
        vectors = list(self._model.embed(texts, batch_size=self.batch_size))
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in vectors]


def is_fastembed_spec(spec: str) -> bool:
    return spec == "fastembed" or spec.startswith("fastembed:")


def build_fastembedder(spec: str) -> FastEmbedder:
    if spec == "fastembed":
        return FastEmbedder()
    model_name = spec.split(":", 1)[1] if ":" in spec else spec
    return FastEmbedder(model_name=model_name)
