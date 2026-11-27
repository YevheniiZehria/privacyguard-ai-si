"""OpenCV preprocessing for noisy / slightly rotated scans.

Pipeline: grayscale -> deskew (projection-profile search) -> optional upscale ->
denoise -> adaptive threshold.  The deskewed colour image at ORIGINAL resolution is
kept as the redaction base so that OCR boxes align exactly with what gets redacted.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_OCR_WIDTH = 1500


@dataclass
class Preprocessed:
    base_bgr: np.ndarray     # deskewed colour image, original resolution (redaction base)
    ocr_binary: np.ndarray   # binarized image for OCR (possibly upscaled)
    ocr_gray: np.ndarray     # denoised grayscale for OCR fallback (same scale as ocr_binary)
    scale: float             # ocr image size / base image size
    angle: float             # rotation applied (degrees, counter-clockwise positive)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """Find the rotation that maximises the variance of horizontal projection profiles."""
    h, w = gray.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    sh, sw = binary.shape
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    # coarse then fine search
    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, step)
    best = max(fine, key=score)
    return float(best)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def load_bgr(path: str) -> np.ndarray:
    # np.fromfile + imdecode handles Windows paths with spaces / diacritics
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_png(path: str, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Cannot encode image for {path}")
    buf.tofile(path)


def preprocess(bgr: np.ndarray, deskew: bool = True) -> Preprocessed:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray) if deskew else 0.0
    if abs(angle) >= 0.3:
        bgr = rotate(bgr, angle)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        angle = 0.0

    h, w = gray.shape[:2]
    scale = 2.0 if w < MIN_OCR_WIDTH else 1.0
    ocr_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else gray
    ocr_gray = cv2.fastNlMeansDenoising(ocr_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(ocr_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Preprocessed(base_bgr=bgr, ocr_binary=binary, ocr_gray=ocr_gray, scale=scale, angle=angle)
