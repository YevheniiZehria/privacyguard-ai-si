"""OpenCV preprocessing for noisy / slightly rotated scans.

Pipeline: perspective dewarp (photos of a sheet / card on a background) -> grayscale -> deskew (projection-profile search) -> optional upscale ->
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
    dewarped: bool = False   # a photographed page was perspective-flattened first


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


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """(tl, tr, br, bl) from 4 unordered points."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_document_quad(bgr: np.ndarray) -> np.ndarray | None:
    """Corners of a photographed sheet / card on a contrasting background, or None.

    Accepted only for a convex 4-corner contour covering 25-98% of the image that is visibly NOT already a
    rectangle aligned with the image (a scan's own border, a full-bleed page): those are left to deskew."""
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    area = gray.shape[0] * gray.shape[1]
    edges = cv2.Canny(gray, 40, 120)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    for mask in (cv2.dilate(edges, np.ones((5, 5), np.uint8)), thr, 255 - thr):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(c)
            if not 0.25 * area <= a <= 0.98 * area:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _order_corners(approx) / scale
            tl, tr, br, bl = quad
            angles = []
            for p, q, r in ((bl, tl, tr), (tl, tr, br), (tr, br, bl), (br, bl, tl)):
                v1, v2 = p - q, r - q
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
            left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
            perspective = max(abs(a - 90) for a in angles) > 3 or \
                abs(top - bottom) / max(top, bottom) > 0.04 or abs(left - right) / max(left, right) > 0.04
            if perspective:
                return quad
            return None                                # a clean rectangle: deskew is enough
    return None


def dewarp(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral to a flat, top-down rectangle (keeps the source resolution)."""
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(bgr: np.ndarray, deskew: bool = True) -> Preprocessed:
    dewarped = False
    if deskew:                                    # photographed at an angle: flatten the page first
        from server import config
        quad = find_document_quad(bgr) if config.DEWARP_ENABLED else None
        if quad is not None:
            bgr, dewarped = dewarp(bgr, quad), True
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
    return Preprocessed(base_bgr=bgr, ocr_binary=binary, ocr_gray=ocr_gray, scale=scale, angle=angle,
                        dewarped=dewarped)
