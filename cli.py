"""Command-line chat with a PDF.

    python cli.py path/to/file.pdf
    python cli.py path/to/file.pdf -q "What is the notice period?"
"""
import argparse

from rag.ingest import ingest_pdf
from rag.providers import get_embeddings, get_llm
from rag.qa import answer_question


def show(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    pages = sorted({s["page"] for s in result["sources"] if s["page"]})
    if pages:
        print("Sources: pages " + ", ".join(map(str, pages)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF (RAG)")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    args = parser.parse_args()

    store, reused = ingest_pdf(args.pdf, get_embeddings())
    print("Loaded existing index." if reused else "Indexed the PDF.")
    llm = get_llm()

    if args.question:
        show(answer_question(store, llm, args.question))
        return

    print("Ask a question (type 'exit' to quit).")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        show(answer_question(store, llm, question))


if __name__ == "__main__":
    main()
