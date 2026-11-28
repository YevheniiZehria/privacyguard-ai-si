"""Fill a blank form with values from an ID profile.

Case A — native PDF form (AcroForm widgets): widget names / tooltips are matched to profile keys and filled.
         Values a standard PDF font cannot show (ș ț ă â î ...) are stamped with a TrueType font over the widget.
Case B — flat form (scanned / flat PDF / image): Romanian labels are found in the OCR words; for each label the
         writable area is located (underline, dotted line, "____" token, empty box, or empty space after a
         "Label:"); values are drawn onto a copy of the deskewed page with a diacritics-capable TTF, auto-sized
         so long addresses never overflow.

Everything here runs inside the server; callers outside only learn which field KEYS were filled.
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server.image_preprocessing import load_bgr, save_png
from server.pii_detector import build_lines, fz, strip_diacritics
from server.vault import DocumentRecord, Word

EXTRA_KEYS = ("phone", "email", "date", "purpose")
DIACRITICS = "ăâîșțĂÂÎȘȚ"
MIN_FONT_PX = 9
BLANK_TOKEN = re.compile(r"^[_.…\-–—·:]{3,}[,;.]?$")
BLANK_TAIL = re.compile(r"[_.…·]{3,}")


# ----------------------------------------------------------------------------- fonts
FONT_CANDIDATES = [
    "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/Library/Fonts/Arial.ttf",
    "arial.ttf", "segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
]


def _covers_diacritics(font: ImageFont.FreeTypeFont) -> bool:
    try:
        notdef = font.getmask("\uffff").getbbox()
        return all(font.getmask(c).getbbox() and font.getmask(c).getbbox() != notdef for c in DIACRITICS)
    except Exception:
        return False


_FONT_PATH: str | None = None


def resolve_font() -> str:
    """Path of a TrueType font with Romanian diacritics: FORM_FONT_PATH, DejaVu/Liberation, Arial, Segoe UI."""
    global _FONT_PATH
    if _FONT_PATH:
        return _FONT_PATH
    env = os.getenv("FORM_FONT_PATH", "").strip()
    for cand in ([env] if env else []) + FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(cand, 32)
        except OSError:
            continue
        if _covers_diacritics(f):
            _FONT_PATH = f.path if isinstance(getattr(f, "path", None), str) else cand
            return _FONT_PATH
    raise RuntimeError("No TrueType font with Romanian diacritics found; set FORM_FONT_PATH in .env")


# ----------------------------------------------------------------------------- labels -> keys
def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in parts), re.I)


# Matched on the diacritic-free, lower-cased line text. Longer (more specific) matches win over shorter ones.
SLOT_LABELS: list[tuple[str, re.Pattern]] = [
    ("full_name", _rx(fz("subsemnat") + r"(?:ul|a)?(?:\s*/\s*a|\s*\(\s*a\s*\))?",
                      fz("nume") + r"(?:le)?\s*(?:si|/|,)\s*" + fz("prenume") + r"(?:le)?",
                      r"nume\s+complet")),
    ("first_name", _rx(r"\b" + fz("prenume") + r"(?:le)?\b")),
    ("last_name", _rx(r"\b" + fz("nume") + r"(?:le)?\b")),
    ("cnp", _rx(r"\bc\.?\s?n\.?\s?p\b\.?", r"cod\s+numeric\s+personal")),
    ("birth_date", _rx(fz("nascut") + r"(?:a|\(a\)|/a)?\s+la\s+data\s+de", r"data\s+" + fz("nasterii"),
                       r"data\s+de\s+" + fz("nastere"))),
    ("birth_place", _rx(fz("nascut") + r"(?:a|\(a\)|/a)?\s+(?:in|în)\b", r"loc(?:ul)?\s+(?:de\s+)?" + fz("nast") + r"er(?:ii|e)")),
    ("address_full", _rx(r"(?:cu\s+)?" + fz("domiciliat") + r"(?:a|ul|\(a\)|/a)?\s+(?:in|în)\b",
                         r"(?:cu\s+)?" + fz("domiciliul") + r"(?:\s+(?:in|în))?\b", r"\badresa\b(?:\s+de\s+domiciliu)?")),
    ("city", _rx(r"\blocalitatea\b", r"\bmunicipiul\b", r"\boras(?:ul)?\b", r"\bcomuna\b")),
    ("street", _rx(r"\bstr(?:ada)?\b\.?")),
    ("nr", _rx(r"\bnr\b\.?", r"\bnum[aă]r(?:ul)?\b")),
    ("building", _rx(r"\bbl(?:oc)?\b\.?", r"\bb[i1|]\.")),          # OCR: "Bl." -> "BI." (only with the dot)
    ("staircase", _rx(r"\bsc(?:ara)?\b\.?", r"\bse\.")),            # OCR: "Sc." -> "Se." (only with the dot)
    ("floor", _rx(r"\bet(?:aj)?\b\.?")),
    ("apartment", _rx(r"\bap(?:artament)?\b\.?")),
    ("county", _rx(r"\bjud(?:e[tț]ul|e[tț])?\b\.?")),
    ("sector", _rx(r"\bsector(?:ul)?\b")),
    ("id_series", _rx(fz("posesor") + r"(?:ul|oare|/oare)?\s+(?:al|a)\s+(?:c\.?\s?i\.?|b\.?\s?i\.?|cartii\s+de\s+identitate|"
                      r"buletinului)(?:\s+seria)?", r"\bc\.?\s?i\.?\s+seria\b", r"\bseria\b")),
    ("issuing_authority", _rx(fz("eliberat") + r"(?:a|ă)?\s+de\b", r"\bemis(?:a|ă)?\s+de\b")),
    ("issue_date", _rx(r"\bla\s+data\s+de\b")),
    ("expiry_date", _rx(fz("valabil") + r"(?:a|ă)?\s+pana\s+la\b")),
    ("nationality", _rx(r"\bcetatenia?\b", r"\bcetatenie\b")),
    ("sex", _rx(r"\bsex(?:ul)?\b")),
    ("phone", _rx(r"\btel(?:efon)?\b\.?", r"\bmobil\b")),
    ("email", _rx(r"\be-?\s?mail\b")),
    ("purpose", _rx(r"\bscopul\b(?:\s+cererii)?")),
    ("date", _rx(r"\bdata\b(?!\s+(?:" + fz("nasterii") + r"|de\b|emiterii|eliberarii))")),
]
ADDRESS_PART_KEYS = {"street", "nr", "street_no", "building", "staircase", "floor", "apartment", "county", "sector"}
# labels that unambiguously mean "the address is split into components" ("nr." alone may be an ID number)
ADDRESS_SPLIT_KEYS = {"street", "building", "staircase", "floor", "apartment"}

WIDGET_SYNONYMS: dict[str, list[str]] = {
    "full_name": ["numesiprenume", "numeleprenumele", "numeleşiprenumele", "numecomplet", "fullname", "subsemnatul",
                  "subsemnata", "solicitant", "titular", "numeprenume"],
    "last_name": ["nume", "numele", "lastname", "surname", "numefamilie"],
    "first_name": ["prenume", "prenumele", "firstname", "givenname"],
    "cnp": ["cnp", "codnumericpersonal"],
    "id_series": ["seria", "serie", "seriaci", "serieci", "seriabuletin"],
    "id_number": ["numarci", "nrci", "numarbuletin", "nrbuletin", "numaract"],
    "id_series_number": ["seriasinumar", "seriesinumar", "seriasinr", "actidentitate", "ci", "buletin"],
    "address_full": ["adresa", "domiciliu", "domiciliul", "adresadomiciliu", "address", "adresacompleta"],
    "street": ["strada", "str", "street"], "street_no": ["nrstrada", "numarstrada", "numar", "nr"],
    "building": ["bloc", "bl"], "staircase": ["scara", "sc"], "floor": ["etaj", "et"],
    "apartment": ["apartament", "ap"], "county": ["judet", "judetul", "county"],
    "city": ["localitate", "localitatea", "oras", "municipiu", "city"], "sector": ["sector"],
    "birth_date": ["datanasterii", "datanastere", "birthdate", "dob"],
    "birth_place": ["loculnasterii", "locnastere", "birthplace"],
    "issue_date": ["dataeliberarii", "dataemiterii", "eliberatladata"],
    "expiry_date": ["valabilpanala", "dataexpirarii", "valabilitate"],
    "issuing_authority": ["eliberatde", "emisde", "emitent", "spclep"],
    "nationality": ["cetatenie", "cetatenia", "nationalitate"], "sex": ["sex", "gen"],
    "phone": ["telefon", "tel", "mobil", "phone"], "email": ["email", "mail", "postaelectronica"],
    "date": ["data", "datacurenta", "date"], "purpose": ["scopul", "scop", "motiv", "scopulcererii"],
}


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_diacritics(text or "").lower())


def match_widget(name: str, label: str = "") -> str | None:
    """Profile key for an AcroForm field name / tooltip: exact synonym, then contains, then fuzzy."""
    cands = [c for c in (norm_key(name), norm_key(label)) if c]
    for c in cands:
        for key, syns in WIDGET_SYNONYMS.items():
            if c in syns:
                return key
    best, best_len = None, 0
    for c in cands:
        for key, syns in WIDGET_SYNONYMS.items():
            for s in syns:
                if len(s) >= 4 and s in c and len(s) > best_len:
                    best, best_len = key, len(s)
    if best:
        return best
    score, best = 0.0, None
    for c in cands:
        for key, syns in WIDGET_SYNONYMS.items():
            for s in syns:
                r = difflib.SequenceMatcher(None, c, s).ratio()
                if r > score:
                    score, best = r, key
    return best if score >= 0.8 else None


# ----------------------------------------------------------------------------- result types
@dataclass
class Placement:
    key: str
    page: int
    box: tuple[int, int, int, int]          # drawn text bbox (x0, y0, x1, y1) in page pixels


@dataclass
class FillOutput:
    pages: list[str]
    pdf_path: str | None
    fields_filled: list[str]
    unfilled_fields: list[str]
    placements: list[Placement] = field(default_factory=list)   # local diagnostics / tests only
    label_boxes: list[tuple[int, int, int, int, int]] = field(default_factory=list)


@dataclass
class Segment:
    x0: int
    x1: int
    y: int                                  # top of the line / the writing baseline
    top: int | None = None                  # top of an enclosing box, when the blank is a box
    used: bool = False


@dataclass
class Slot:
    key: str
    page: int
    x0: int
    x1: int
    baseline: int
    text_h: int
    label_box: tuple[int, int, int, int]
    top: int | None = None


# ----------------------------------------------------------------------------- values
def value_for(key: str, profile: dict[str, str], extras: dict[str, str]) -> str | None:
    if key in extras and extras[key]:
        return extras[key]
    if key in profile:
        return profile[key]
    if key == "full_name" and profile.get("last_name") and profile.get("first_name"):
        return f"{profile['last_name']} {profile['first_name']}"
    if key == "county" and profile.get("sector"):
        return f"București, Sector {profile['sector']}"
    return None


# ----------------------------------------------------------------------------- Case B: flat forms
def _is_blank_token(text: str) -> bool:
    return bool(BLANK_TOKEN.match(text.strip()))


def _binary_inv(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def _thin_rules(binv: np.ndarray, text_h: int, min_len: int) -> np.ndarray:
    """Mask of long, thin horizontal rules: solid underlines / "____" runs, and dotted lines.

    Letters must never merge into a "rule": solid rules are found on the raw binary (only 1-2 px gaps bridged —
    glyphs are further apart), dotted lines only by joining TINY components (dots), never letters."""
    horiz = lambda k: cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))   # noqa: E731
    solid = cv2.morphologyEx(cv2.morphologyEx(binv, cv2.MORPH_CLOSE, horiz(3)), cv2.MORPH_OPEN, horiz(min_len))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    tiny = np.zeros(n, dtype=bool)
    tiny[1:] = (stats[1:, cv2.CC_STAT_HEIGHT] <= max(3, text_h // 4)) & \
               (stats[1:, cv2.CC_STAT_WIDTH] <= max(3, int(text_h * 0.4)))
    dots = np.where(tiny[labels], 255, 0).astype(np.uint8)
    dotted = cv2.morphologyEx(cv2.morphologyEx(dots, cv2.MORPH_CLOSE, horiz(max(3, text_h // 3))),
                              cv2.MORPH_OPEN, horiz(min_len))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid | dotted, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_HEIGHT] <= max(5, int(text_h * 0.3))
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def estimate_text_h(binv: np.ndarray) -> int:
    """Typical glyph height from connected components (no OCR needed)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    h, w = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH]
    glyphs = h[(h >= 6) & (h <= binv.shape[0] // 4) & (w <= h * 3)]
    return int(np.percentile(glyphs, 90)) if len(glyphs) >= 5 else max(12, binv.shape[0] // 60)


def find_blank_segments(img: np.ndarray, words: list[Word], text_h: int) -> list[Segment]:
    """Underlines / dotted lines / '____' runs / empty boxes on one page."""
    binv = _binary_inv(img)
    long_rules = _thin_rules(binv, text_h, max(40, text_h * 3))   # survive the word erase (OCR boxes include them)
    for w in words:                           # erase real text so glyph baselines cannot merge into "lines"
        if not _is_blank_token(w.text) and not BLANK_TAIL.search(w.text):
            binv[max(0, w.y - 2):w.y + w.h + 2, max(0, w.x - 2):w.x + w.w + 2] = 0
    binv |= long_rules
    min_len = max(30, int(text_h * 1.6))
    segs: list[Segment] = []
    contours, _ = cv2.findContours(_thin_rules(binv, text_h, min_len), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_len:
            segs.append(Segment(x, x + w, y))
    # OCR'd blank tokens ("______", "......") — Tesseract sometimes reads them when morphology misses dots
    for w in words:
        t = w.text.strip()
        if _is_blank_token(t):
            segs.append(Segment(w.x, w.x + w.w, w.y + w.h))
        else:
            m = BLANK_TAIL.search(t)
            if m and m.start() > 0:           # "CNP:______" glued into one token -> blank part only
                frac = m.start() / len(t)
                segs.append(Segment(w.x + int(w.w * frac), w.x + w.w, w.y + w.h))
    segs.sort(key=lambda s: (s.y, s.x0))
    merged: list[Segment] = []
    for s in segs:                            # merge duplicates (morphology + OCR token of the same blank)
        dup = next((m for m in merged if abs(m.y - s.y) <= text_h * 0.6 and s.x0 < m.x1 and s.x1 > m.x0), None)
        if dup:
            dup.x0, dup.x1, dup.y = min(dup.x0, s.x0), max(dup.x1, s.x1), max(dup.y, s.y)
        else:
            merged.append(s)
    # empty boxes: two stacked horizontal lines of the same extent joined by vertical edges at both ends
    # -> write above the bottom one (two consecutive underlined rows have no vertical edges)
    def edge(x: int, y0: int, y1: int) -> bool:
        cols = binv[y0 + 2:max(y0 + 3, y1 - 1), max(0, x - 3):x + 4]
        return cols.size > 0 and (cols.max(axis=1) > 0).mean() >= 0.7

    for s in merged:
        above = next((m for m in merged if m is not s and text_h * 0.8 <= s.y - m.y <= text_h * 3
                      and abs(m.x0 - s.x0) <= text_h * 0.5 and abs(m.x1 - s.x1) <= text_h * 0.5
                      and edge(s.x0, m.y, s.y) and edge(s.x1 - 1, m.y, s.y)), None)
        if above:
            s.top = above.y
            above.used = True                 # the top edge of a box is not a writing line
    return [s for s in merged if not s.used]


def form_words(img: np.ndarray, page_no: int) -> list[Word]:
    """OCR a form page with its rule lines erased: long underlines / dotted lines make Tesseract drop whole text
    lines (it reads them as line art), which would hide exactly the labels the filler needs."""
    from server.ocr_engine import ocr_image
    binv = _binary_inv(img)
    text_h = estimate_text_h(binv)
    mask = cv2.dilate(_thin_rules(binv, text_h, max(40, text_h * 3)),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))
    clean = img.copy()
    clean[mask > 0] = 255
    return ocr_image(clean, page_no=page_no, deskew=False)[1]


def _label_matches(text: str) -> list[tuple[int, int, str]]:
    """(start, end, key) label matches on one line; overlaps resolved by preferring the longer match."""
    found = []
    for key, rx in SLOT_LABELS:
        for m in rx.finditer(text):
            found.append((m.start(), m.end(), key))
    found.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    chosen: list[tuple[int, int, str]] = []
    for f in found:
        if all(f[1] <= c[0] or f[0] >= c[1] for c in chosen):
            chosen.append(f)
    return sorted(chosen)


def locate_slots(words: list[Word], page_no: int, img: np.ndarray) -> tuple[list[Slot], list[tuple], list[Segment]]:
    """Writable slots on one page (words = that page's OCR words), the label boxes (never to be overdrawn) and
    the blank segments found (unclaimed ones are spare lines for wrapping long values)."""
    if not words:
        return [], [], []
    real_ids = [i for i, w in enumerate(words) if not _is_blank_token(w.text)]
    real = [words[i] for i in real_ids]
    # 75th percentile: short lower-case words ("in", "nr.") would make the median too small for the values
    text_h = int(np.percentile([w.h for w in real], 75)) if real else 20
    segs = find_blank_segments(img, words, text_h)
    # a word box that runs into its blank ("Sc.____" read as one wide word) ends where the blank starts
    words = list(words)
    for i in real_ids:
        w = words[i]
        s = next((s for s in segs if w.y <= s.y <= w.y + w.h + text_h * 0.3
                  and w.x + 0.3 * w.w < s.x0 < w.x + w.w < s.x1 + 2), None)
        if s:
            words[i] = replace(w, w=max(1, s.x0 - w.x))
    real = [words[i] for i in real_ids]
    lines = build_lines(words)
    page_w = img.shape[1]
    slots: list[Slot] = []
    labels_out: list[tuple] = []
    page_has_parts = False
    raw: list[tuple] = []                     # (line, start, end, key, label_box, word_ids)
    for line in lines:
        low = line.norm.lower()
        for s, e, key in _label_matches(low):
            ids = [line.word_idx[k] for k, (a, b) in enumerate(line.spans) if a < e and b > s]
            if not ids:
                continue
            # a label glued to its blank ("CNP:______"): the label box ends where the blank begins
            x0 = min(words[i].x for i in ids)
            x1 = 0
            for i in ids:
                w = words[i]
                m = BLANK_TAIL.search(w.text)
                x1 = max(x1, w.x + int(w.w * m.start() / len(w.text)) if m and m.start() > 0 else w.x + w.w)
            y0 = min(words[i].y for i in ids)
            y1 = max(words[i].y + words[i].h for i in ids)
            raw.append((line, s, e, key, (x0, y0, x1, y1), ids))
            page_has_parts |= key in ADDRESS_SPLIT_KEYS
    for line in lines:                        # resolve context-dependent labels per line, left to right
        items = sorted((r for r in raw if r[0] is line), key=lambda r: r[4][0])
        prev_key = None
        line_keys = {r[3] for r in items}
        for idx, (_, _s, _e, key, lb, ids) in enumerate(items):
            if key == "nr":
                key = "id_number" if prev_key in ("id_series", "id_number") else (
                    None if prev_key == "phone" else "street_no")
            elif key == "address_full" and (page_has_parts or line_keys & ADDRESS_SPLIT_KEYS):
                key = "city"
            labels_out.append((page_no, *lb))
            if key is None:
                continue
            prev_key = key
            nxt_x = items[idx + 1][4][0] if idx + 1 < len(items) else page_w - 10
            own = set(ids)
            others = [words[i].x for i in real_ids if i not in own and lb[1] - text_h < words[i].y < lb[3]
                      and lb[2] - 2 <= words[i].x < nxt_x and not BLANK_TAIL.search(words[i].text)]
            limit = min([nxt_x] + others)
            slot = _slot_for(key, page_no, lb, limit, segs, text_h, line.text.rstrip().endswith(":") or
                             line.norm.lower()[:_e + 2].rstrip().endswith(":"))
            if slot:
                slots.append(slot)
    return slots, labels_out, segs


def _slot_for(key, page_no, lb, limit, segs, text_h, colon) -> Slot | None:
    lx0, ly0, lx1, ly1 = lb
    gap = max(4, text_h // 4)
    # (a) blank on the same line, right of the label, before the next label / word
    same = [s for s in segs if ly0 - text_h * 0.2 <= s.y <= ly1 + text_h * 0.8 and s.x1 > lx1 + gap
            and s.x0 < limit - gap]
    if same:
        s = min(same, key=lambda s: s.x0)
        x0, x1 = max(s.x0, lx1 + gap), min(s.x1, limit - gap)
        if x1 - x0 >= text_h * 1.2:
            return Slot(key, page_no, x0, x1, s.y, text_h, lb, s.top)
    # (c) blank directly beneath a column label
    below = [s for s in segs if not s.used and ly1 < s.y <= ly1 + text_h * 2.6
             and min(s.x1, lx1) - max(s.x0, lx0) >= 0.5 * (lx1 - lx0)]
    if below:
        s = min(below, key=lambda s: s.y)
        s.used = True
        return Slot(key, page_no, s.x0 + gap, s.x1 - gap, s.y, text_h, lb, s.top)
    # (b) "Label:" followed by empty space
    if colon and limit - lx1 >= text_h * 4:
        return Slot(key, page_no, lx1 + gap * 2, int(limit - gap), ly1, text_h, lb)
    return None


def _ink(img: np.ndarray, lb) -> tuple[int, int, int]:
    x0, y0, x1, y1 = lb
    roi = img[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
    if not len(roi):
        return (20, 20, 20)
    gray = roi.mean(axis=1)
    dark = roi[gray <= np.percentile(gray, 10)]
    b, g, r = (int(v) for v in dark.mean(axis=0))
    if (r + g + b) / 3 > 90:                  # never render light-grey "ink"
        r, g, b = (int(v * 0.4) for v in (r, g, b))
    return (r, g, b)


def _fit(text: str, font_path: str, px: int, width: int) -> tuple[ImageFont.FreeTypeFont, int]:
    size = px
    font = ImageFont.truetype(font_path, size)
    while font.getlength(text) > width and size > MIN_FONT_PX:
        size = max(MIN_FONT_PX, int(size * 0.92))
        font = ImageFont.truetype(font_path, size)
        if size == MIN_FONT_PX:
            break
    return font, size


def render_slots(img: np.ndarray, slots: list[Slot], values: dict[int, str], label_boxes: list[tuple],
                 page_no: int, spare: list[Segment]) -> tuple[np.ndarray, list[Placement]]:
    font_path = resolve_font()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    placements: list[Placement] = []
    labels = [lb[1:] for lb in label_boxes if lb[0] == page_no]

    def hits_label(bb) -> bool:
        return any(bb[0] < l[2] and bb[2] > l[0] and bb[1] < l[3] and bb[3] > l[1] for l in labels)

    for i, slot in enumerate(slots):
        text = values.get(i)
        if not text:
            continue
        ink = _ink(img, slot.label_box)
        px = int(slot.text_h * 1.15)
        if slot.top is not None:
            px = min(px, int((slot.baseline - slot.top) * 0.8))
        width = slot.x1 - slot.x0
        start_px = px
        font, size = _fit(text, font_path, px, width)
        parts = [(text, slot)]
        if font.getlength(text) > width or size < start_px * 0.55:
            nxt = next((s for s in spare if not s.used and slot.baseline < s.y <= slot.baseline + slot.text_h * 2.6
                        and min(s.x1, slot.x1) - max(s.x0, slot.x0) > 0.5 * width), None)
            if nxt and "," in text:           # wrap at the comma nearest the middle onto the blank line below
                cuts = [m.start() for m in re.finditer(",", text)]
                cut = min(cuts, key=lambda c: abs(c - len(text) / 2))
                nxt.used = True
                second = Slot(slot.key, page_no, max(nxt.x0, slot.x0), nxt.x1, nxt.y, slot.text_h, slot.label_box)
                parts = [(text[:cut + 1].strip(), slot), (text[cut + 1:].strip(), second)]
        for part, sl in parts:
            w = sl.x1 - sl.x0
            font, size = _fit(part, font_path, px, w)
            base_y = sl.baseline - max(2, int(sl.text_h * 0.15))
            x = sl.x0 + 2
            bb = draw.textbbox((x, base_y), part, font=font, anchor="ls")
            while hits_label(bb) and size > MIN_FONT_PX:          # never overwrite printed text
                size -= 1
                font = ImageFont.truetype(font_path, size)
                bb = draw.textbbox((x, base_y), part, font=font, anchor="ls")
            if hits_label(bb):
                continue
            draw.text((x, base_y), part, font=font, fill=ink, anchor="ls")
            placements.append(Placement(sl.key, page_no, tuple(int(v) for v in bb)))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), placements


def _fill_flat(doc: DocumentRecord, profile: dict, extras: dict, out_dir: Path, stem: str,
               want_pdf: bool) -> FillOutput:
    pages_out: list[str] = []
    filled: list[str] = []
    unfilled: list[str] = []
    placements: list[Placement] = []
    label_boxes: list[tuple] = []
    for page in doc.pages:
        img = load_bgr(page.base_image_path)
        words = form_words(img, page.number)
        slots, labels, segs = locate_slots(words, page.number, img)
        label_boxes += labels
        has_number_slot = any(s.key == "id_number" for s in slots)
        values: dict[int, str] = {}
        for i, s in enumerate(slots):
            key = "id_series_number" if s.key == "id_series" and not has_number_slot else s.key
            v = value_for(key, profile, extras)
            if v:
                values[i] = v
        spare = [s for s in segs if not s.used
                 and not any(abs(s.y - sl.baseline) < 3 and s.x0 <= sl.x0 <= s.x1 for sl in slots)]
        img, placed = render_slots(img, slots, values, labels, page.number, spare)
        placements += placed
        done = {p.key for p in placed}
        for s in slots:
            (filled if s.key in done else unfilled).append(s.key)
        suffix = "" if page.number == 1 else f"_p{page.number}"
        out = out_dir / f"{stem}{suffix}.png"
        save_png(str(out), img)
        pages_out.append(str(out))
    pdf = None
    if want_pdf or len(pages_out) > 1:
        pdf = str(out_dir / f"{stem}.pdf")
        ims = [Image.open(p).convert("RGB") for p in pages_out]
        ims[0].save(pdf, "PDF", save_all=True, append_images=ims[1:], resolution=300)
    return FillOutput(pages_out, pdf, list(dict.fromkeys(filled)),
                      [k for k in dict.fromkeys(unfilled) if k not in filled], placements, label_boxes)


# ----------------------------------------------------------------------------- Case A: AcroForm PDFs
def _winansi_ok(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def pdf_has_widgets(path: str) -> bool:
    if not str(path).lower().endswith(".pdf"):
        return False
    import pymupdf
    with pymupdf.open(path) as d:
        return any(True for p in d for _ in (p.widgets() or []))


def _fill_acroform(src_pdf: str, profile: dict, extras: dict, out_dir: Path, stem: str) -> FillOutput:
    import pymupdf
    font_path = resolve_font()
    font = pymupdf.Font(fontfile=font_path)
    filled, unfilled = [], []
    pdf_path = out_dir / f"{stem}.pdf"
    with pymupdf.open(src_pdf) as d:
        for page in d:
            stamps = []
            for w in list(page.widgets() or []):
                if w.field_type not in (pymupdf.PDF_WIDGET_TYPE_TEXT, pymupdf.PDF_WIDGET_TYPE_COMBOBOX):
                    continue
                key = match_widget(w.field_name or "", w.field_label or "")
                if not key:
                    continue
                v = value_for(key, profile, extras)
                if not v:
                    unfilled.append(key)
                    continue
                if _winansi_ok(v):
                    w.field_value = v
                    w.update()
                else:                         # ș/ț etc.: stamp with a TTF so the glyphs really render
                    stamps.append((pymupdf.Rect(w.rect), v))
                    page.delete_widget(w)
                filled.append(key)
            for rect, v in stamps:
                size = min(rect.height * 0.7, 12)
                while font.text_length(v, fontsize=size) > rect.width - 4 and size > 5:
                    size -= 0.5
                page.insert_text((rect.x0 + 2, rect.y1 - rect.height * 0.28), v, fontsize=size,
                                 fontname="pgfill", fontfile=font_path, color=(0.08, 0.08, 0.08))
        d.save(str(pdf_path))
        pages = []
        for i, page in enumerate(d):
            out = out_dir / f"{stem}{'' if i == 0 else f'_p{i + 1}'}.png"
            page.get_pixmap(dpi=300).save(str(out))
            pages.append(str(out))
    return FillOutput(pages, str(pdf_path), list(dict.fromkeys(filled)),
                      [k for k in dict.fromkeys(unfilled) if k not in filled])


# ----------------------------------------------------------------------------- entry point
def fill_document(form: DocumentRecord, profile: dict[str, str], extras: dict[str, str] | None,
                  out_dir: Path, stem: str) -> FillOutput:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = {k: unicodedata.normalize("NFC", str(v)).strip() for k, v in (extras or {}).items() if v}
    profile = {k: unicodedata.normalize("NFC", v) for k, v in profile.items()}
    if pdf_has_widgets(form.source_path):
        return _fill_acroform(form.source_path, profile, extras, out_dir, stem)
    return _fill_flat(form, profile, extras, out_dir, stem, want_pdf=str(form.source_path).lower().endswith(".pdf"))
