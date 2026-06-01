from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://keygateway.arshnivlabs.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
