"""Secure local state vault.

All raw OCR text, word boxes and PII values live ONLY in this process' memory.
MCP tools hand out opaque identifiers (``doc_*``, ``det_*``, ``res_*``) and
aggregate metadata; nothing in this module is ever serialized back to the LLM.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import uuid
from dataclasses import dataclass, field

from server import config


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float          # 0..100
    page: int            # 1-based
    line_key: tuple      # (page, block, paragraph, line) — groups words on the same text line

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class Page:
    number: int
    base_image_path: str   # deskewed, original-resolution colour image used for redaction
    width: int
    height: int


@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    pages: list[Page]
    words: list[Word]
    avg_confidence: float


@dataclass
class Entity:
    type: str                       # CNP, PERSON_NAME, ADDRESS, ID_SERIES, IBAN, PHONE, EMAIL, ...
    value: str                      # raw value — stays in the vault
    word_indices: list[int]
    confidence: float
    source: str = "regex"           # regex | context | spacy | slm


@dataclass
class DetectionRecord:
    detection_id: str
    doc_id: str
    entities: list[Entity]
    boxes: list[tuple[int, int, int, int, int]]   # (page, x, y, w, h) merged per line
    entity_hashes: dict[str, set[str]] = field(default_factory=dict)  # type -> {hmac}


@dataclass
class RedactionRecord:
    result_id: str
    doc_id: str
    detection_id: str
    output_paths: list[str]
    boxes_drawn: int
    padding_px: int = 4
    residual_types: list[str] = field(default_factory=list)   # local diagnostics only, never returned


class Vault:
    """Thread-safe in-memory store. One instance per MCP server process."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._lock = threading.RLock()
        if secret is None:
            secret = bytes.fromhex(config.VAULT_SECRET_HEX) if config.VAULT_SECRET_HEX else secrets.token_bytes(32)
        self._secret = secret
        self._docs: dict[str, DocumentRecord] = {}
        self._detections: dict[str, DetectionRecord] = {}
        self._results: dict[str, RedactionRecord] = {}

    # ---------- ids ----------
    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    # ---------- hashing ----------
    def hmac_hex(self, message: str) -> str:
        return hmac.new(self._secret, message.encode("utf-8"), hashlib.sha256).hexdigest()

    # ---------- documents ----------
    def put_document(self, rec: DocumentRecord) -> None:
        with self._lock:
            self._docs[rec.doc_id] = rec

    def get_document(self, doc_id: str) -> DocumentRecord:
        with self._lock:
            if doc_id not in self._docs:
                raise KeyError(f"Unknown doc_id '{doc_id}'. Call ocr_document first.")
            return self._docs[doc_id]

    def list_doc_ids(self) -> list[str]:
        with self._lock:
            return list(self._docs)

    # ---------- detections ----------
    def put_detection(self, rec: DetectionRecord) -> None:
        with self._lock:
            self._detections[rec.detection_id] = rec

    def get_detection(self, detection_id: str) -> DetectionRecord:
        with self._lock:
            if detection_id not in self._detections:
                raise KeyError(f"Unknown detection_id '{detection_id}'. Call detect_pii first.")
            return self._detections[detection_id]

    # ---------- redactions ----------
    def put_result(self, rec: RedactionRecord) -> None:
        with self._lock:
            self._results[rec.result_id] = rec

    def get_result(self, result_id: str) -> RedactionRecord:
        with self._lock:
            if result_id not in self._results:
                raise KeyError(f"Unknown result_id '{result_id}'. Call redact_image first.")
            return self._results[result_id]

    def clear(self) -> None:
        with self._lock:
            self._docs.clear()
            self._detections.clear()
            self._results.clear()
