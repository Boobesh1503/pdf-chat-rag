"""Central configuration, read from environment variables / a local .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # "gemini" (free tier available), "openai", or "fake" (offline, for tests only)
    provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-3.5-flash")
    gemini_fallback_model: str = os.getenv("GEMINI_FALLBACK_MODEL", "models/gemini-3.1-flash-lite")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    chroma_dir: str = os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma"))
    upload_dir: str = os.getenv("UPLOAD_DIR", str(ROOT / "data" / "uploads"))


settings = Settings()
