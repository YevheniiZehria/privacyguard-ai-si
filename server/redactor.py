"""Irreversible redaction: solid black filled rectangles drawn on the page images."""
from __future__ import annotations

from pathlib import Path

import cv2

from server.image_preprocessing import load_bgr, save_png
from server.vault import Page


def redact_pages(pages: list[Page], boxes: list[tuple[int, int, int, int, int]], doc_id: str,
                 output_dir: Path, padding_px: int = 4) -> tuple[list[str], int]:
    """Draw (0,0,0) filled boxes. Returns (output paths, boxes drawn)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    drawn = 0
    for page in pages:
        img = load_bgr(page.base_image_path)
        h, w = img.shape[:2]
        for (pg, x, y, bw, bh) in boxes:
            if pg != page.number:
                continue
            x0, y0 = max(0, x - padding_px), max(0, y - padding_px)
            x1, y1 = min(w - 1, x + bw + padding_px), min(h - 1, y + bh + padding_px)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
            drawn += 1
        suffix = "" if page.number == 1 else f"_p{page.number}"
        out = output_dir / f"redacted_{doc_id}{suffix}.png"
        save_png(str(out), img)
        paths.append(str(out))
    return paths, drawn
