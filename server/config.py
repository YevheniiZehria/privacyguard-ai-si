"""Central configuration loaded from `.env` (never hardcode secrets)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    return value.strip() if value and value.strip() else default


TESSERACT_CMD = _env("TESSERACT_CMD")
OCR_LANGS = _env("OCR_LANGS", "ron+eng")
VAULT_SECRET_HEX = _env("VAULT_SECRET")
REDACTION_MODE = _env("REDACTION_MODE", "blur").lower()     # blur | black
OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
WORK_DIR = OUTPUT_DIR / "work"

LOCAL_SLM_BASE_URL = _env("LOCAL_SLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_SLM_MODEL = _env("LOCAL_SLM_MODEL", "qwen2.5:3b")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
