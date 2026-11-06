"""Load a PDF, split it into chunks, embed the chunks and store them in ChromaDB."""
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings


def file_fingerprint(path: str | Path) -> str:
    """Short hash of the file contents, used as the Chroma collection name."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"doc-{digest[:16]}"


def load_pdf(path: str | Path) -> list[Document]:
    """Read a PDF page by page into LangChain Documents.

    Each page keeps its 1-based `page_number` so answers can cite sources.
    Pages without extractable text (e.g. scanned images) are skipped.
    """
    name = Path(path).name
    docs = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": name, "page_number": number})
            )
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_pdf(
    path: str | Path,
    embeddings,
    persist_dir: str | None = None,
) -> tuple[Chroma, bool]:
    """Index a PDF and return (vector_store, reused).

    The collection is named after the file hash, so uploading the same PDF again
    re-uses the stored vectors instead of paying for embeddings a second time.
    """
    persist_dir = persist_dir or settings.chroma_dir
    store = Chroma(
        collection_name=file_fingerprint(path),
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if store._collection.count() > 0:
        return store, True

    docs = load_pdf(path)
    if not docs:
        raise ValueError(
            "No text found in this PDF. It may be a scanned image; run OCR on it first."
        )
    chunks = split_documents(docs)
    store.add_documents(chunks)
    return store, False
