"""OCR engine: OpenCV preprocessing + Tesseract word-level extraction.

Text and boxes are returned to the caller (the MCP server) which stores them in the
vault; they are never sent to the LLM.
"""
from __future__ import annotations

import os
import re
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
        # MRZ strings ('<<<<') get very low Tesseract confidence; keep them so they can be redacted
        is_mrz_like = text.count("<") >= 2 and len(text) >= 10
        if not text or (conf < MIN_WORD_CONF and not is_mrz_like):
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


def ocr_image(bgr: np.ndarray, page_no: int = 1, langs: str | None = None, psm: int = 6,
              deskew: bool = True) -> tuple[np.ndarray, list[Word]]:
    """Preprocess + OCR one image. Returns (redaction base image, words in base coords)."""
    langs = langs or config.OCR_LANGS
    pre = preprocess(bgr, deskew=deskew)
    words = _tesseract_words(pre.ocr_binary, page_no, pre.scale, langs, psm)
    if _avg(words) < LOW_CONF_FALLBACK:
        alt = _tesseract_words(pre.ocr_gray, page_no, pre.scale, langs, psm)
        if _avg(alt) * max(len(alt), 1) > _avg(words) * max(len(words), 1):
            words = alt
    foreign = foreign_langs()
    if foreign and foreign != langs and needs_foreign_pass(words):
        second = _tesseract_words(pre.ocr_binary, page_no, pre.scale, foreign, psm)
        words = merge_passes(words, second)
    return pre.base_bgr, words


# ----------------------------------------------------------------------------- Cyrillic / foreign second pass
# Cyrillic read with ron+eng comes back as Latin look-alikes in capitals: МАР'ЯНА -> "MAP'AHA", ІВАНІВНА -> "IBAHIBHA"
LOOKALIKE_TOKEN = re.compile(r"^[ABEHKMOPCTXYI'’]{4,}$")
FOREIGN_MRZ = re.compile(r"^(?:P|I|ID|A|C)[<A-Z]?(?!ROU)[A-Z]{3}[A-Z<]{5,}")
_FOREIGN: str | None = None


def foreign_langs() -> str:
    """OCR_LANGS_FOREIGN restricted to installed traineddata (+ukr when available); '' if none is usable."""
    global _FOREIGN
    if _FOREIGN is None:
        try:
            have = set(pytesseract.get_languages(config=""))
        except Exception:
            have = set()
        wanted = config.OCR_LANGS_FOREIGN.split("+") + ["ukr"]
        langs = [l for l in dict.fromkeys(wanted) if l in have]
        _FOREIGN = "+".join(langs) if any(l in ("rus", "ukr", "bul", "srp") for l in langs) else ""
    return _FOREIGN


def needs_foreign_pass(words: list[Word]) -> bool:
    texts = [w.text.strip(",.;:|()[]") for w in words]
    lookalikes = sum(1 for t in texts if LOOKALIKE_TOKEN.match(t))
    mrz = any(FOREIGN_MRZ.match(t.replace(" ", "")) for t in texts if t.count("<") >= 2)
    return lookalikes >= 2 or mrz


def merge_passes(first: list[Word], second: list[Word]) -> list[Word]:
    """Per OCR line of the second pass: keep it where it reads better than the first pass (or where the first
    pass read nothing), otherwise keep the first pass. Words stay in base-image coordinates."""
    groups: dict[tuple, list[Word]] = {}
    for w in second:
        groups.setdefault(w.line_key, []).append(w)
    result = list(first)
    for key, ws in groups.items():
        x0, x1 = min(w.x for w in ws), max(w.x + w.w for w in ws)
        y0, y1 = min(w.y for w in ws), max(w.y + w.h for w in ws)
        overlap = [w for w in result if w.x < x1 and w.x + w.w > x0
                   and min(w.y + w.h, y1) - max(w.y, y0) >= 0.5 * min(w.h, y1 - y0)]
        new_conf = _avg(ws)
        if overlap and new_conf <= _avg(overlap) + 5:
            continue
        drop = {id(w) for w in overlap}
        result = [w for w in result if id(w) not in drop]
        result += [Word(w.text, w.x, w.y, w.w, w.h, w.conf, w.page, (w.page, 1000 + key[1], key[2], key[3]))
                   for w in ws]
    return result


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
