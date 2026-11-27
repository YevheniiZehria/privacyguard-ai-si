"""OCR engine: OpenCV preprocessing + Tesseract word-level extraction.

Text and boxes are returned to the caller (the MCP server) which stores them in the
vault; they are never sent to the LLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytesseract

from server import config
from server.image_preprocessing import load_bgr, preprocess, save_png
from server.pdf_reader import render_pdf_pages
from server.vault import Page, Word

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_WORD_CONF = 30.0
LOW_CONF_FALLBACK = 60.0


@dataclass
class OcrResult:
    pages: list[Page]
    words: list[Word]
    avg_confidence: float


def _tesseract_words(img: np.ndarray, page_no: int, scale: float, langs: str, psm: int) -> list[Word]:
    data = pytesseract.image_to_data(img, lang=langs, config=f"--oem 1 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < MIN_WORD_CONF:
            continue
        words.append(Word(
            text=text,
            x=int(round(data["left"][i] / scale)),
            y=int(round(data["top"][i] / scale)),
            w=max(1, int(round(data["width"][i] / scale))),
            h=max(1, int(round(data["height"][i] / scale))),
            conf=conf,
            page=page_no,
            line_key=(page_no, data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        ))
    return words


def _avg(words: list[Word]) -> float:
    return float(np.mean([w.conf for w in words])) if words else 0.0


def ocr_image(bgr: np.ndarray, page_no: int = 1, langs: str | None = None, psm: int = 6) -> tuple[np.ndarray, list[Word]]:
    """Preprocess + OCR one image. Returns (redaction base image, words in base coords)."""
    langs = langs or config.OCR_LANGS
    pre = preprocess(bgr)
    words = _tesseract_words(pre.ocr_binary, page_no, pre.scale, langs, psm)
    if _avg(words) < LOW_CONF_FALLBACK:
        alt = _tesseract_words(pre.ocr_gray, page_no, pre.scale, langs, psm)
        if _avg(alt) * max(len(alt), 1) > _avg(words) * max(len(words), 1):
            words = alt
    return pre.base_bgr, words


def ocr_file(file_path: str, doc_id: str, work_dir: Path | None = None) -> OcrResult:
    """OCR an image or PDF. Saves deskewed page images to ``work_dir`` for later redaction."""
    work_dir = Path(work_dir or config.WORK_DIR)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        sources = render_pdf_pages(file_path, str(work_dir), doc_id)
    elif ext in IMAGE_EXTS:
        sources = [file_path]
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use PDF, PNG, JPG, TIFF or BMP.")

    pages: list[Page] = []
    words: list[Word] = []
    for page_no, src in enumerate(sources, start=1):
        base, page_words = ocr_image(load_bgr(src), page_no=page_no)
        base_path = str(work_dir / f"{doc_id}_p{page_no}.png")
        save_png(base_path, base)
        h, w = base.shape[:2]
        pages.append(Page(number=page_no, base_image_path=base_path, width=w, height=h))
        words.extend(page_words)
    return OcrResult(pages=pages, words=words, avg_confidence=round(_avg(words), 2))
