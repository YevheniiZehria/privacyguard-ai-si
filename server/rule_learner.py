"""Privacy-safe active learning from manual blur boxes.

When the user blurs something the detector missed, the nearest *label* left of / above the box ("Titular cont:",
"Reprezentant", "Beneficiar") is turned into a reusable rule: normalized label words, where the value sits relative
to it (right / below), the coarse SHAPE of the value (ALPHA_WORDS, DIGITS_9_13, DATE_LIKE, ...) and rounded
relative coordinates. The value itself is never stored — it is read only in memory to classify its shape and then
discarded, and the label crop is OCR'd with the box masked out.

Rules live in one JSON file (LEARNED_RULES_PATH, default <project>/learned_rules.json, git-ignored) shared by every
folder, the MCP server and the CLI agent. Every rule is validated on write AND on load: no digit run longer than 4,
no '@', no IBAN / CNP / phone shapes, no known first/last names, letters-only labels of at most 4 words.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import string
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from server import config
from server.universal_detector import DATE_ANY, LABELS, build_cells, norm
from server.vault import Word

SHAPE_KIND = {"ALPHA_WORDS": "name", "DATE_LIKE": "birth_date", "ALNUM_CODE": "id", "DIGITS_1_4": "id",
              "DIGITS_5_8": "id", "DIGITS_9_13": "id", "DIGITS_14_PLUS": "id", "MIXED": "address",
              "UNKNOWN": "address"}
FORBIDDEN = [re.compile(r"\d{5,}"), re.compile("@"), re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,}", re.I),
             re.compile(r"(?<!\d)[1-9]\d{12}(?!\d)"), re.compile(r"(?:\+?\d[\s.\-]?){9,}")]
LABEL_OK = re.compile(r"^[^\W\d_](?:[^\W\d_]|['\s])*$")
JUNK_WORDS = {"a", "i", "si", "and", "de", "of", "la", "le", "the"}


@dataclass
class LearnedRule:
    id: str
    label: str
    relation: str             # right | below
    shape: str
    x_rel: float
    y_rel: float
    w_rel: float
    h_rel: float
    hits: int = 1
    created: float = 0.0

    @property
    def kind(self) -> str:
        return SHAPE_KIND.get(self.shape, "address")

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "relation": self.relation, "shape": self.shape,
                "hits": self.hits, "created": self.created}


def _new_id() -> str:                          # letters only: ids can never look like a number
    return "r_" + "".join(secrets.choice(string.ascii_lowercase) for _ in range(8))


def _known_names() -> set[str]:
    from server.pii_detector import RO_GIVEN_NAMES, RO_SURNAMES
    return {n.lower() for n in RO_GIVEN_NAMES | RO_SURNAMES}


def validate(rule: LearnedRule) -> bool:
    """True only for rules that carry structure, never a value."""
    if rule.relation not in ("right", "below") or rule.shape not in SHAPE_KIND:
        return False
    if not re.fullmatch(r"r_[a-z]{8}", rule.id):
        return False
    label = rule.label
    if not (2 <= len(label) <= 40) or not LABEL_OK.match(label) or len(label.split()) > 4:
        return False
    blob = " ".join((rule.id, rule.label, rule.relation, rule.shape))      # the text fields (numbers are coords / time)
    if any(p.search(blob) for p in FORBIDDEN):
        return False
    if any(w in _known_names() for w in label.split() if len(w) >= 3):
        return False
    if any(re.search(r"(?:escu|eanu|ovici|evici|iuc)$", w) for w in label.split()):   # Romanian surname shapes
        return False
    if all(w in JUNK_WORDS for w in label.split()):
        return False
    return all(0.0 <= v <= 1.0 for v in (rule.x_rel, rule.y_rel, rule.w_rel, rule.h_rel))


def value_shape(text: str) -> str:
    t = text.strip()
    if not t:
        return "UNKNOWN"
    if DATE_ANY.search(t):
        return "DATE_LIKE"
    compact = re.sub(r"[\s\-./]", "", t)
    if compact.isdigit():
        n = len(compact)
        return "DIGITS_1_4" if n <= 4 else "DIGITS_5_8" if n <= 8 else "DIGITS_9_13" if n <= 13 else "DIGITS_14_PLUS"
    if re.fullmatch(r"[A-Za-z0-9]+", compact) and re.search(r"\d", compact):
        return "ALNUM_CODE"
    if re.fullmatch(r"[^\W\d_]+(?:[\s,.'\-][^\W\d_]+)*[.,]?", t):
        return "ALPHA_WORDS"
    return "MIXED"


def clean_label(text: str) -> str:
    """Normalized label: bilingual 'Прізвище/ Surname' -> 'surname', 'Titular cont:' -> 'titular cont'."""
    t = norm(text).split("/")[-1]
    t = re.sub(r"[^\w\s']", " ", t)
    words = [w for w in t.split() if not re.search(r"[\d_]", w)]
    while words and len(words[0]) == 1:          # OCR junk before the label
        words.pop(0)
    return " ".join(words[-4:])


def find_label(words: list[Word], box: tuple[int, int, int, int]) -> tuple[str, str] | None:
    """(label text, relation) for the nearest label left of (same band) or above the box. Words inside the box
    are ignored, so the value can never become part of the label."""
    bx, by, bw, bh = box
    outside = [w for w in words if not (w.x < bx + bw and w.x + w.w > bx and w.y < by + bh and w.y + w.h > by)]
    if not outside:
        return None
    rows, med_h = build_cells(outside)
    med_h = med_h or bh
    best = None
    for row in rows:                             # left: a cell ending just before the box, vertically overlapping
        for c in row:
            ov = min(c.y1, by + bh) - max(c.y0, by)
            gap = bx - c.x1
            if ov >= 0.4 * min(c.h, bh) and -2 <= gap <= 8 * med_h:
                if best is None or gap < best[0]:
                    best = (gap, c.text, "right")
    if best:
        return best[1], best[2]
    for row in reversed(rows):                   # above: the nearest cell over the box's left part
        for c in row:
            gap = by - c.y1
            if -2 <= gap <= 2.5 * max(med_h, c.h) and c.x0 < bx + bw and c.x1 > bx - med_h and \
                    abs(c.x0 - bx) <= max(bw, 6 * med_h):
                return c.text, "below"
    return None


def learn_from_box(label_words: list[Word], box: tuple[int, int, int, int], value_text: str,
                   page_w: int, page_h: int) -> LearnedRule | None:
    """Build (but do not store) a rule. value_text is used only to classify the shape and is then dropped."""
    found = find_label(label_words, box)
    if not found:
        return None
    label = clean_label(found[0])
    if not label:
        return None
    bx, by, bw, bh = box
    rule = LearnedRule(_new_id(), label, found[1], value_shape(value_text),
                       round(bx / page_w, 2), round(by / page_h, 2), round(min(1.0, bw / page_w), 2),
                       round(min(1.0, bh / page_h), 2), 1, round(time.time(), 0))
    return rule if validate(rule) else None


class RuleStore:
    """Thread-safe JSON store; patterns() is re-read when the file changes (other processes may learn too)."""

    _shared: dict[str, "RuleStore"] = {}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._mtime = None
        self._rules: list[LearnedRule] = []
        self._compiled: list[tuple[LearnedRule, re.Pattern]] = []

    @classmethod
    def shared(cls) -> "RuleStore":
        path = str(Path(os.getenv("LEARNED_RULES_PATH", "").strip() or config.LEARNED_RULES_PATH))
        if path not in cls._shared:
            cls._shared[path] = cls(path)
        return cls._shared[path]

    def _load(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._rules, self._compiled, self._mtime = [], [], None
            return
        if mtime == self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8")).get("rules", [])
        except (OSError, ValueError):
            raw = []
        rules = []
        for r in raw:
            try:
                rule = LearnedRule(**{k: r[k] for k in LearnedRule.__dataclass_fields__ if k in r})
            except TypeError:
                continue
            if validate(rule):                       # a hand-edited / tampered file never injects values
                rules.append(rule)
        self._rules = rules
        self._compiled = [(r, re.compile(r"(?<![^\W_])" + r"\s*".join(re.escape(w) for w in r.label.split())
                                         + r"(?![^\W_])")) for r in rules]
        self._mtime = mtime

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "rules": [asdict(r) for r in self._rules]}, indent=1,
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = None
        self._load()

    def rules(self) -> list[LearnedRule]:
        with self._lock:
            self._load()
            return list(self._rules)

    def patterns(self) -> list[tuple[LearnedRule, re.Pattern]]:
        with self._lock:
            self._load()
            return list(self._compiled)

    def add(self, rule: LearnedRule) -> LearnedRule:
        if not validate(rule):
            raise ValueError("rule rejected by the privacy validator")
        with self._lock:
            self._load()
            same = next((r for r in self._rules if r.label == rule.label and r.relation == rule.relation), None)
            if same:
                same.hits += 1
                if same.shape == "UNKNOWN":
                    same.shape = rule.shape
                rule = same
            else:
                self._rules.append(rule)
            self._save()
            return rule

    def delete(self, rule_id: str) -> bool:
        with self._lock:
            self._load()
            n = len(self._rules)
            self._rules = [r for r in self._rules if r.id != rule_id]
            if len(self._rules) == n:
                return False
            self._save()
            return True

    def clear(self) -> int:
        with self._lock:
            self._load()
            n = len(self._rules)
            self._rules = []
            self._save()
            return n


def is_builtin_label(label: str) -> bool:
    """True when the universal lexicon already knows this label (learning it adds nothing)."""
    return any(rx.search(label) for _k, rx, _s in LABELS)
