"""Irreversible redaction of PII regions on the page images.

Default mode "blur": every region is destroyed in three steps so the text cannot be read or reconstructed:
  1. random noise is mixed into the region (defeats de-pixelation attacks that brute-force known fonts),
  2. coarse pixelation (block = ~half the region height, so one block spans several glyph strokes),
  3. a wide Gaussian blur over the pixelated region.
Mode "black" (REDACTION_MODE=black in .env, or mode="black") draws solid black rectangles as before.
Every box gets a margin (effective_padding: REDACTION_PADDING_RATIO x box height, at least padding_px).
Both modes keep the output path / return values the pipeline expects.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from server import config
from server.image_preprocessing import load_bgr, save_png
from server.vault import Page

DEFAULT_MODE = config.REDACTION_MODE


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def effective_padding(box_h: int, padding_px: int, long_side: int) -> int:
    """Margin added around a box: at least padding_px, else REDACTION_PADDING_RATIO x box height, capped at
    1.5% of the image's long side so big boxes (portraits) don't swallow the page. Scales with resolution,
    so a 3000 px phone photo and a 600 px scan get the same relative coverage."""
    ratio_pad = int(round(config.REDACTION_PADDING_RATIO * box_h))
    cap = max(padding_px, int(0.015 * long_side))
    return max(padding_px, config.REDACTION_MIN_PADDING_PX, min(ratio_pad, cap))


def blur_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, rng: np.random.Generator) -> None:
    """Noise + pixelate + Gaussian-blur img[y0:y1, x0:x1] in place.

    Block size and kernel scale with BOTH the region height and the image resolution, so text is equally
    unreadable on a high-res phone photo and on a low-res scan."""
    roi = img[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        img[y0:y1, x0:x1] = 0
        return
    noise = rng.integers(-60, 61, size=roi.shape, dtype=np.int16)
    noisy = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    long_side = max(img.shape[:2])
    block = max(6, h // 2, long_side // 250)
    small = cv2.resize(noisy, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # kernel wider than a block so no block edges survive (sharp edges are what OCR hallucinates glyphs from)
    k = _odd(max(15, block * 3))
    img[y0:y1, x0:x1] = cv2.GaussianBlur(pix, (k, k), 0, borderType=cv2.BORDER_REPLICATE)


def redact_pages(pages: list[Page], boxes: list[tuple[int, int, int, int, int]], doc_id: str,
                 output_dir: Path, padding_px: int = 4, mode: str | None = None) -> tuple[list[str], int]:
    """Redact every box on its page. Returns (output paths, boxes redacted)."""
    mode = (mode or DEFAULT_MODE).lower()
    if mode not in ("blur", "black"):
        raise ValueError("mode must be 'blur' or 'black'")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    paths: list[str] = []
    drawn = 0
    for page in pages:
        img = load_bgr(page.base_image_path)
        h, w = img.shape[:2]
        for (pg, x, y, bw, bh) in boxes:
            if pg != page.number:
                continue
            pad = effective_padding(bh, padding_px, max(h, w))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad + 1), min(h, y + bh + pad + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            if mode == "black":
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=-1)
            else:
                blur_region(img, x0, y0, x1, y1, rng)
            drawn += 1
        suffix = "" if page.number == 1 else f"_p{page.number}"
        out = output_dir / f"redacted_{doc_id}{suffix}.png"
        save_png(str(out), img)
        paths.append(str(out))
    return paths, drawn
