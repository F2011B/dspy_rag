from pathlib import Path

from dspy_rag.indexing import chunk_text, is_text_filename


def test_chunk_text_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    # Expected slices: 0-10, 8-18, 16-26
    assert chunks[0][0] == "abcdefghij"
    assert chunks[1][0] == "ijklmnopqr"
    assert chunks[2][0] == "qrstuvwxyz"


def test_is_text_filename():
    assert is_text_filename(Path("README.md"))
    assert is_text_filename(Path("script.py"))
    assert not is_text_filename(Path("image.png"))
