import pytest
from app.ingestion import calculate_sha256, build_chunks, load_text_pages

def test_deterministic_sha256():
    data = b"Enterprise standard operating procedure test string."
    assert calculate_sha256(data) == calculate_sha256(data)
    assert calculate_sha256(data) != calculate_sha256(b"Different payload")

def test_chunking_constraints():
    pages = [("A" * 1500, 1)]
    chunks = build_chunks(pages, doc_id="test_id", filename="test.txt")
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == idx
        assert c.metadata["doc_id"] == "test_id"
        assert len(c.page_content) <= 1000

def test_unsupported_file_extension():
    with pytest.raises(ValueError):
        load_text_pages(b"data", "test.xlsx")