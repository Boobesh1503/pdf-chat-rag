"""Retrieve the most relevant chunks and ask the LLM to answer from them only."""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import settings

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about a PDF document.\n"
    "Use ONLY the context below. If the answer is not in the context, reply exactly: "
    "\"I couldn't find that in the document.\" Do not guess.\n"
    "Keep answers short and clear, and mention page numbers like (p. 3) when you use them."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page_number', '?')}]\n{d.page_content}" for d in docs
    )


def answer_question(store, llm, question: str, k: int | None = None) -> dict:
    """Return {"answer": str, "sources": [{"page": int, "snippet": str}, ...]}."""
    docs = store.as_retriever(search_kwargs={"k": k or settings.top_k}).invoke(question)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    sources = [
        {
            "page": d.metadata.get("page_number"),
            "snippet": d.page_content[:300].replace("\n", " "),
        }
        for d in docs
    ]
    return {"answer": answer.strip(), "sources": sources}
