"""Create the embedding model and chat model for the selected provider.

Imports are lazy so you only need the package for the provider you actually use.
"""
from .config import settings


def get_embeddings(provider: str | None = None):
    provider = (provider or settings.provider).lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=256)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'openai'.")


def get_llm(provider: str | None = None):
    provider = (provider or settings.provider).lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        primary = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            temperature=0,
            max_retries=2,
            timeout=30,
        )
        if settings.gemini_fallback_model:
            backup = ChatGoogleGenerativeAI(
                model=settings.gemini_fallback_model,
                temperature=0,
                max_retries=2,
                timeout=30,
            )
            return primary.with_fallbacks([backup])
        return primary
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_chat_model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'openai'.")