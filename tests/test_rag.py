"""Offline tests: fake embeddings + fake LLM, so no API key or internet is needed."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag.ingest import file_fingerprint, ingest_pdf, load_pdf, split_documents
from rag.qa import answer_question, format_docs


@pytest.fixture()
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 700, "Refund policy: customers can request a refund within 30 days.")
    c.showPage()
    c.drawString(72, 700, "Support hours are 9 AM to 6 PM, Monday to Friday.")
    c.showPage()
    c.save()
    return path


def test_load_pdf_adds_page_numbers(sample_pdf):
    docs = load_pdf(sample_pdf)
    assert [d.metadata["page_number"] for d in docs] == [1, 2]
    assert "refund" in docs[0].page_content.lower()


def test_split_respects_chunk_size(sample_pdf):
    docs = load_pdf(sample_pdf)
    chunks = split_documents(docs, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > len(docs)
    assert all(len(c.page_content) <= 30 for c in chunks)


def test_ingest_then_reuse_does_not_duplicate(sample_pdf, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    store, reused = ingest_pdf(sample_pdf, emb, persist_dir=str(tmp_path / "chroma"))
    first_count = store._collection.count()
    assert not reused and first_count > 0

    store2, reused2 = ingest_pdf(sample_pdf, emb, persist_dir=str(tmp_path / "chroma"))
    assert reused2 and store2._collection.count() == first_count


def test_fingerprint_is_stable(sample_pdf):
    assert file_fingerprint(sample_pdf) == file_fingerprint(sample_pdf)


def test_answer_returns_text_and_sources(sample_pdf, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    store, _ = ingest_pdf(sample_pdf, emb, persist_dir=str(tmp_path / "chroma"))
    llm = FakeListChatModel(responses=["Refunds are allowed within 30 days (p. 1)."])
    result = answer_question(store, llm, "What is the refund policy?", k=2)
    assert "30 days" in result["answer"]
    assert len(result["sources"]) == 2
    assert all(s["page"] in (1, 2) for s in result["sources"])


def test_format_docs_includes_page_label(sample_pdf):
    text = format_docs(load_pdf(sample_pdf))
    assert "[Page 1]" in text and "[Page 2]" in text
