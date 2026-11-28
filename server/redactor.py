"""Irreversible redaction of PII regions on the page images.

Default mode "blur": every region is destroyed in three steps so the text cannot be read or reconstructed:
  1. random noise is mixed into the region (defeats de-pixelation attacks that brute-force known fonts),
  2. coarse pixelation (one cell spans several glyph strokes),
  3. a wide Gaussian blur over the pixelated region,
followed by a residual-legibility guard that re-runs the region with a coarser grid while any stroke-scale
detail survives.

All three steps are ADAPTIVE: cell size, kernel and noise amplitude scale with the text height (unpadded box
height), the image resolution (long side vs. an A4-at-~140-dpi reference) and the local contrast, so a 4K phone
photo, a large header and a 600 px scan are equally unreadable.

``blur_strength`` ("low" | "medium" | "high" | "maximum" or 1..10, default "high" = 7) tunes how far above the
safety floor the parameters go. Even level 1 keeps the floor (cell >= 0.35 x text height), so no level leaves
text legible. Mode "black" (REDACTION_MODE=black in .env, or mode="black") draws solid black rectangles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from server import config
from server.image_preprocessing import load_bgr, save_png
from server.vault import Page

DEFAULT_MODE = config.REDACTION_MODE
STRENGTH_LEVELS = {"low": 3, "medium": 5, "high": 7, "maximum": 10}
DEFAULT_LEVEL = STRENGTH_LEVELS["high"]
RESIDUAL_HP_THRESHOLD = 2.5      # std of stroke-scale high-pass (gray levels) above which a region is re-blurred
REFERENCE_LONG_SIDE = 1600       # ~A4 at 140 dpi


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 else n + 1


def resolve_strength(value: str | int | float | None) -> int:
    """'low'|'medium'|'high'|'maximum' or 1..10 (int or numeric string) -> level 1..10."""
    if value is None:
        return DEFAULT_LEVEL
    if isinstance(value, bool):
        raise ValueError("blur_strength must be low|medium|high|maximum or an integer 1..10")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in STRENGTH_LEVELS:
            return STRENGTH_LEVELS[v]
        if v in ("med", "max"):
            return STRENGTH_LEVELS["medium" if v == "med" else "maximum"]
        try:
            value = float(v)
        except ValueError:
            raise ValueError("blur_strength must be low|medium|high|maximum or an integer 1..10") from None
    if isinstance(value, (int, float)) and float(value).is_integer() and 1 <= value <= 10:
        return int(value)
    raise ValueError("blur_strength must be low|medium|high|maximum or an integer 1..10")


def strength_name(level: int) -> str:
    """Canonical name for a level (exact preset name, else the numeric level as a string)."""
    for name, lv in STRENGTH_LEVELS.items():
        if lv == level:
            return name
    return str(level)


def pad_scale_for(level: int) -> float:
    return 1.5 if level >= 10 else 1.25 if level >= 8 else 1.0


@dataclass(frozen=True)
class BlurProfile:
    level: int
    block: int
    kernel: int
    noise: int
    passes: int
    pad_scale: float


def local_contrast(roi: np.ndarray) -> float:
    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return 0.0
    lo, hi = np.percentile(gray, (5, 95))
    return float(max(0.0, min(1.0, (hi - lo) / 255.0)))


def profile(level: int, roi: np.ndarray | None, text_h: int, long_side: int) -> BlurProfile:
    """Blur parameters for one region. Grows monotonically with level, text height, resolution and contrast."""
    level = resolve_strength(level)
    res = min(4.0, max(0.5, long_side / REFERENCE_LONG_SIDE))
    contrast = local_contrast(roi) if roi is not None else 1.0
    text_h = max(1, int(text_h))
    block = max(int(round(6 * res)),
                math.ceil(text_h * (0.35 + 0.05 * level) * (1 + 0.25 * contrast)),
                long_side // (300 - 15 * level))
    kernel = _odd(max(15, block * (2 + 0.25 * level)))
    noise = int(round((30 + 6 * level) * (0.6 + 0.6 * contrast)))
    return BlurProfile(level, block, kernel, noise, 2 if level >= 9 else 1, pad_scale_for(level))


def residual_detail(roi: np.ndarray, text_h: int) -> float:
    """Stroke-scale high-frequency energy left in a region (std of gray - blurred gray)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) if roi.ndim == 3 else roi.astype(np.float32)
    sigma = max(1.0, text_h / 8.0)
    return float((gray - cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)).std())


def _destroy(roi: np.ndarray, block: int, kernel: int, noise: int, rng: np.random.Generator) -> np.ndarray:
    h, w = roi.shape[:2]
    jitter = rng.integers(-noise, noise + 1, size=roi.shape, dtype=np.int16)
    noisy = np.clip(roi.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    sw, sh = max(1, w // block), max(1, h // block)
    small = cv2.resize(noisy, (sw, sh), interpolation=cv2.INTER_AREA)
    # The wide blur (kernel wider than a cell, so no cell edges survive — sharp edges are what OCR hallucinates
    # glyphs from) runs on the cell grid: same result as blurring the pixelated image, without O(kernel) cost
    # per full-resolution pixel on big boxes.
    ks = _odd(max(3, round(kernel / block)))
    small = cv2.GaussianBlur(small, (ks, ks), 0, borderType=cv2.BORDER_REPLICATE)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = _odd(min(max(3, block), 151))           # smooth the interpolation creases at full resolution
    return cv2.GaussianBlur(up, (k, k), 0, borderType=cv2.BORDER_REPLICATE)


def blur_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, rng: np.random.Generator,
                level: int | str = DEFAULT_LEVEL, text_h: int | None = None) -> BlurProfile | None:
    """Noise + pixelate + Gaussian-blur img[y0:y1, x0:x1] in place, then re-run while stroke detail survives.

    text_h: height of the text inside the region (the unpadded box). Defaults to the region height."""
    roi = img[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        img[y0:y1, x0:x1] = 0
        return None
    th = int(text_h or h)
    prof = profile(resolve_strength(level), roi, th, max(img.shape[:2]))
    out = roi.copy()
    for _ in range(prof.passes):
        out = _destroy(out, prof.block, prof.kernel, prof.noise, rng)
    block, kernel = prof.block, prof.kernel
    for _ in range(2):                       # residual-legibility guard
        if residual_detail(out, th) <= RESIDUAL_HP_THRESHOLD:
            break
        block, kernel = block * 2, _odd(kernel * 2)
        out = _destroy(out, block, kernel, prof.noise, rng)
    img[y0:y1, x0:x1] = out
    return prof


def effective_padding(box_h: int, padding_px: int, long_side: int, scale: float = 1.0) -> int:
    """Margin added around a box: at least padding_px, else REDACTION_PADDING_RATIO x box height (x scale), capped
    at 1.5% of the image's long side (x scale) so big boxes (portraits) don't swallow the page. Scales with
    resolution, so a 3000 px phone photo and a 600 px scan get the same relative coverage."""
    ratio_pad = int(round(config.REDACTION_PADDING_RATIO * box_h * scale))
    cap = max(padding_px, int(0.015 * long_side * scale))
    return max(padding_px, config.REDACTION_MIN_PADDING_PX, min(ratio_pad, cap))


class Redactor:
    """Configured redaction pass: mode, blur strength and padding (fixed px floor + ratio multiplier)."""

    def __init__(self, mode: str | None = None, blur_strength: str | int | None = "high", padding_px: int = 4,
                 padding_scale: float | None = None) -> None:
        self.mode = (mode or DEFAULT_MODE).lower()
        if self.mode not in ("blur", "black"):
            raise ValueError("mode must be 'blur' or 'black'")
        if not 0 <= int(padding_px) <= 50:
            raise ValueError("padding_px must be between 0 and 50")
        self.level = resolve_strength(blur_strength)
        self.padding_px = int(padding_px)
        self.padding_scale = float(padding_scale) if padding_scale is not None else pad_scale_for(self.level)
        if not 0.5 <= self.padding_scale <= 4:
            raise ValueError("padding_scale must be between 0.5 and 4")

    def padding(self, box_h: int, long_side: int) -> int:
        return effective_padding(box_h, self.padding_px, long_side, self.padding_scale)

    def redact_image(self, img: np.ndarray, boxes, rng: np.random.Generator) -> int:
        """Redact (x, y, w, h) boxes on one image in place. Returns how many were drawn."""
        h, w = img.shape[:2]
        drawn = 0
        for (x, y, bw, bh) in boxes:
            pad = self.padding(bh, max(h, w))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad + 1), min(h, y + bh + pad + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            if self.mode == "black":
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=-1)
            else:
                blur_region(img, x0, y0, x1, y1, rng, self.level, text_h=bh)
            drawn += 1
        return drawn

    def redact_pages(self, pages: list[Page], boxes: list[tuple[int, int, int, int, int]], doc_id: str,
                     output_dir: Path) -> tuple[list[str], int]:
        """Redact every box on its page. Returns (output paths, boxes redacted)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng()
        paths: list[str] = []
        drawn = 0
        for page in pages:
            img = load_bgr(page.base_image_path)
            drawn += self.redact_image(img, [(x, y, bw, bh) for (pg, x, y, bw, bh) in boxes if pg == page.number],
                                       rng)
            suffix = "" if page.number == 1 else f"_p{page.number}"
            out = output_dir / f"redacted_{doc_id}{suffix}.png"
            save_png(str(out), img)
            paths.append(str(out))
        return paths, drawn


def redact_pages(pages: list[Page], boxes: list[tuple[int, int, int, int, int]], doc_id: str,
                 output_dir: Path, padding_px: int = 4, mode: str | None = None,
                 blur_strength: str | int | None = "high", padding_scale: float | None = None
                 ) -> tuple[list[str], int]:
    """Backward-compatible wrapper around Redactor."""
    return Redactor(mode, blur_strength, padding_px, padding_scale).redact_pages(pages, boxes, doc_id, output_dir)
