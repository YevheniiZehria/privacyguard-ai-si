"""Render PDF pages to 300 DPI images (scanned PDFs are then OCR'd like images)."""
from __future__ import annotations

import os

import pymupdf

DPI = 300


def render_pdf_pages(pdf_path: str, out_dir: str, stem: str) -> list[str]:
    """Render every page of ``pdf_path`` to PNG and return the image paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=DPI)
            path = os.path.join(out_dir, f"{stem}_src_p{index + 1}.png")
            pix.save(path)
            paths.append(path)
    return paths
