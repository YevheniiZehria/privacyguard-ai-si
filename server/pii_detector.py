"""Hybrid PII detector for Romanian documents.

Deterministic layer (regex + checksums + label context) runs always; spaCy NER and a
LOCAL small language model (Ollama) are optional add-ons. Every entity is mapped back
to OCR word indices so it can be redacted.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

from server.vault import Entity, Word

# ----------------------------------------------------------------------------
# Validators
# ----------------------------------------------------------------------------
CNP_KEY = "279146358279"


def cnp_is_valid(cnp: str) -> bool:
    """Official Romanian CNP check: S AA LL ZZ JJ NNN C with control key 279146358279."""
    if not re.fullmatch(r"[1-9]\d{12}", cnp):
        return False
    total = sum(int(d) * int(k) for d, k in zip(cnp[:12], CNP_KEY))
    control = total % 11
    control = 1 if control == 10 else control
    if control != int(cnp[12]):
        return False
    century = {"1": 1900, "2": 1900, "3": 1800, "4": 1800, "5": 2000, "6": 2000,
               "7": 1900, "8": 1900, "9": 1900}[cnp[0]]
    try:
        date(century + int(cnp[1:3]), int(cnp[3:5]), int(cnp[5:7]))
    except ValueError:
        return False
    county = int(cnp[7:9])
    return 1 <= county <= 52 or county in (70, 80, 81, 82, 83)


def cnp_control_digit(first12: str) -> str:
    total = sum(int(d) * int(k) for d, k in zip(first12, CNP_KEY))
    c = total % 11
    return "1" if c == 10 else str(c)


def iban_is_valid(iban: str) -> bool:
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def strip_diacritics(text: str) -> str:
    text = text.replace("ş", "s").replace("Ş", "S").replace("ţ", "t").replace("Ţ", "T")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


OCR_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                               "i": "1", "Z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "T": "7"})

RO_ID_SERIES = {
    "AB", "AR", "AS", "AX", "AZ", "BC", "BH", "BR", "BT", "BV", "BZ", "CJ", "CL", "CT", "CV", "CZ", "DB",
    "DD", "DP", "DR", "DT", "DX", "GG", "GL", "GZ", "HD", "HR", "IF", "IL", "IZ", "KL", "KS", "KT", "KV",
    "KX", "MH", "MM", "MS", "MX", "MZ", "NT", "OT", "PH", "PX", "RD", "RK", "RR", "RT", "RX", "SB", "SM",
    "SV", "SX", "SZ", "TC", "TM", "TR", "TZ", "VN", "VS", "XB", "XC", "XH", "XM", "XR", "XT", "XV", "XZ",
    "ZC", "ZH", "ZL", "ZR",
}

NAME_LABELS = [
    r"nume\s*(?:si|și|/)?\s*prenume(?:le)?", r"numele\s+si\s+prenumele", r"nume(?:le)?", r"prenume(?:le)?",
    r"subsemnat(?:ul|a)", r"titular(?:ul)?", r"client(?:ul)?", r"beneficiar(?:ul)?", r"cumparator(?:ul)?",
    r"vanzator(?:ul)?", r"locatar(?:ul)?", r"proprietar(?:ul)?", r"solicitant(?:ul)?", r"angajat(?:ul)?",
    r"domnul", r"doamna", r"dl\.?", r"dna\.?", r"last\s*name", r"first\s*name", r"nom", r"prenom",
    r"semnatura", r"reprezentant(?:ul)?", r"intocmit", r"student(?:ul|a)?",
]
ADDRESS_LABELS = [r"domiciliu(?:l)?", r"domiciliat(?:a)?\s+(?:in|în)", r"adresa(?:\s+de\s+\w+)?",
                  r"adresse", r"address", r"sediul", r"resedinta", r"locul\s+consumului", r"loc\s+consum"]
ADDRESS_START = r"(?:str\.|strada|bd\.|b-dul|bulevardul|calea|aleea|sos\.|soseaua|piata|splaiul|intrarea)"
ADDRESS_CONT = re.compile(r"\b(nr|bl|sc|et|ap|jud|judet|mun|municipiul|oras|or|sat|com|comuna|sector|cod)\b\.?", re.I)
STOP_WORDS = re.compile(r"\b(cnp|domiciliat|domiciliu|nascut|nascuta|seria|legitimat|posesor|cu|in|în|"
                        r"data|tel|telefon|email|e-mail|iban|adresa|identificat|act)\b", re.I)
NAME_TOKEN = re.compile(r"^[A-ZĂÂÎȘŞȚŢ][A-Za-zĂÂÎȘŞȚŢăâîșşțţ\-]{1,}$")
LABEL_WORD = re.compile(r"^(nume|prenume|nom|prenom|last|first|name|si|și|/)$", re.I)
LABEL_ONLY_REST = re.compile(r"[\s/:.\-]*(?:(?:nume|prenume|nom|prenom|last|first|name|si|adresse|address|"
                             r"adresa|domiciliu)[\s/:.\-]*)*", re.I)
ADDR_EXCLUDE = re.compile(r"\b(CNP|seria|emis[aă]?|valabil\w*|telefon|tel|e-?mail|IBAN)\b", re.I)

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+(?:\.\s[A-Za-z0-9._%+\-]+)?@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # OCR often reads '@' as 'Q', '©' or '(a)'; require a common TLD to stay precise
    "EMAIL_OCR": re.compile(r"[A-Za-z0-9._%+\-]{2,}(?:Q|©|®|\(a\))[A-Za-z0-9\-]+\.(?:com|ro|net|org|eu|info)\b"),
    "IBAN": re.compile(r"\bRO\s?\d{2}\s?[A-Z]{4}(?:\s?[A-Z0-9]{4}){4}\b"),
    "PHONE": re.compile(r"(?<![\d])(?:\+40|0040|0)[\s.\-]?(?:7\d{2}|[23]\d{1,2})(?:[\s.\-]?\d{2,3}){2,3}(?![\d])"),
    "CNP": re.compile(r"(?<!\d)[1-9]\d{12}(?!\d)"),
    "ID_SERIES_LABELED": re.compile(r"\bSERIA\s*[:.]?\s*([A-Z]{2})\s*[,;]?\s*(?:NR\.?|NUMAR(?:UL)?|N[O0]\.?)?\s*[:.]?\s*(\d{6})\b", re.I),
    "ID_SERIES_BARE": re.compile(r"\b([A-Z]{2})\s?(\d{6})\b"),
    "BIRTH_DATE": re.compile(r"(?:data\s+na[sș]terii|n[aă]scut[aă]?\s+(?:la|în|in)?\s*(?:data\s+de)?|date\s+of\s+birth)\s*[:.]?\s*"
                             r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})", re.I),
}


# ----------------------------------------------------------------------------
# Line model: words grouped by OCR line, with char offsets -> word index mapping
# ----------------------------------------------------------------------------
@dataclass
class Line:
    text: str
    norm: str                         # diacritic-stripped text (same length mapping not guaranteed -> we keep per-word)
    word_idx: list[int]
    spans: list[tuple[int, int]]      # char span of each word inside `text`


def build_lines(words: list[Word]) -> list[Line]:
    groups: dict[tuple, list[int]] = {}
    for i, w in enumerate(words):
        groups.setdefault(w.line_key, []).append(i)
    ordered = sorted(groups.values(), key=lambda idx: (words[idx[0]].page,
                                                        min(words[i].y for i in idx),
                                                        min(words[i].x for i in idx)))
    lines: list[Line] = []
    for idx in ordered:
        idx.sort(key=lambda i: words[i].x)
        parts, spans, pos = [], [], 0
        for i in idx:
            t = words[i].text
            spans.append((pos, pos + len(t)))
            parts.append(t)
            pos += len(t) + 1
        text = " ".join(parts)
        lines.append(Line(text=text, norm=strip_diacritics(text), word_idx=idx, spans=spans))
    return lines


def _words_in_span(line: Line, start: int, end: int) -> list[int]:
    return [line.word_idx[k] for k, (s, e) in enumerate(line.spans) if s < end and e > start]


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class PIIDetector:
    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp = None
        if use_spacy:
            try:  # optional dependency
                import spacy  # type: ignore
                self._nlp = spacy.load("ro_core_news_sm")
            except Exception:
                self._nlp = None
        self._name_label_re = re.compile(r"^\s*(?:" + "|".join(NAME_LABELS) + r")\b\s*[:.\-]?\s*", re.I)
        self._addr_label_re = re.compile(r"(?:^|\b)(?:" + "|".join(ADDRESS_LABELS) + r")\b\s*[:.\-]?\s*", re.I)
        self._addr_start_re = re.compile(r"\b" + ADDRESS_START, re.I)

    # ---------------------------------------------------------------- public
    def detect(self, words: list[Word], use_local_slm: bool = False,
               slm_base_url: str | None = None, slm_model: str | None = None) -> list[Entity]:
        lines = build_lines(words)
        entities: list[Entity] = []
        entities += self._regex_entities(lines, words)
        entities += self._name_entities(lines, words)
        entities += self._address_entities(lines, words)
        entities += self._mrz_entities(lines)
        if self._nlp is not None:
            entities += self._spacy_entities(lines)
        if use_local_slm and slm_base_url and slm_model:
            entities += self._slm_entities(lines, words, slm_base_url, slm_model)
        entities = self._propagate_names(entities, lines, words)
        return self._dedupe(entities)

    # ---------------------------------------------------------------- regex
    def _regex_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        out: list[Entity] = []
        doc_norm = " ".join(l.norm for l in lines).upper()
        id_context = any(k in doc_norm for k in ("CARTE DE IDENTITATE", "CARTE IDENTITATE", "BULETIN", "SERIA",
                                                  "IDENTITY CARD", " C.I.", "ACT DE IDENTITATE"))
        for line in lines:
            t = line.norm
            for m in PATTERNS["EMAIL"].finditer(t):
                out.append(Entity("EMAIL", m.group(0), _words_in_span(line, *m.span()), 0.95))
            if "@" not in t:
                for m in PATTERNS["EMAIL_OCR"].finditer(t):
                    out.append(Entity("EMAIL", re.sub(r"Q|©|®|\(a\)", "@", m.group(0), count=1),
                                      _words_in_span(line, *m.span()), 0.8))
            for m in PATTERNS["IBAN"].finditer(t.upper()):
                val = m.group(0).replace(" ", "")
                out.append(Entity("IBAN", val, _words_in_span(line, *m.span()), 0.99 if iban_is_valid(val) else 0.7))
            for m in PATTERNS["CNP"].finditer(t):
                if cnp_is_valid(m.group(0)):
                    out.append(Entity("CNP", m.group(0), _words_in_span(line, *m.span()), 0.99))
            for m in PATTERNS["PHONE"].finditer(t):
                digits = re.sub(r"\D", "", m.group(0))
                if 9 <= len(digits) <= 13 and not (len(digits) == 13 and cnp_is_valid(digits)):
                    out.append(Entity("PHONE", m.group(0), _words_in_span(line, *m.span()), 0.85))
            for m in PATTERNS["ID_SERIES_LABELED"].finditer(t):
                out.append(Entity("ID_SERIES", (m.group(1) + m.group(2)).upper(),
                                  _words_in_span(line, m.start(1), m.end(2)), 0.97, "context"))
            if id_context:
                for m in PATTERNS["ID_SERIES_BARE"].finditer(t):
                    if m.group(1) in RO_ID_SERIES:
                        out.append(Entity("ID_SERIES", m.group(1) + m.group(2),
                                          _words_in_span(line, *m.span()), 0.8))
            for m in PATTERNS["BIRTH_DATE"].finditer(t):
                out.append(Entity("BIRTH_DATE", m.group(1), _words_in_span(line, *m.span(1)), 0.9, "context"))
            # Lenient CNP: 13 noisy chars right after a CNP label (OCR confusions O->0, l->1 ...)
            for m in re.finditer(r"\bCNP\b\s*[:.]?\s*(\S{12,15})", t, re.I):
                raw = m.group(1).strip(".,;:")
                fixed = raw.translate(OCR_DIGIT_FIX)
                fixed = re.sub(r"\D", "", fixed)
                if len(fixed) == 13 and not any(e.type == "CNP" and e.value == fixed for e in out):
                    conf = 0.95 if cnp_is_valid(fixed) else 0.6
                    out.append(Entity("CNP", fixed, _words_in_span(line, m.start(1), m.start(1) + len(m.group(1))),
                                      conf, "context"))
        return out

    # ---------------------------------------------------------------- names
    def _take_name_tokens(self, line: Line, start_char: int, words: list[Word]) -> list[int]:
        picked: list[int] = []
        for k, (s, e) in enumerate(line.spans):
            if s < start_char:
                continue
            tok = words[line.word_idx[k]].text.strip(",.;:")
            if STOP_WORDS.fullmatch(strip_diacritics(tok)) or (picked and words[line.word_idx[k]].text.startswith("(")):
                break
            if LABEL_WORD.match(strip_diacritics(tok)) and not picked:
                continue
            if NAME_TOKEN.match(tok) or (picked and re.match(r"^l[a-zăâîșşțţ]{2,}$", tok)):  # OCR: I -> l
                picked.append(line.word_idx[k])
                if words[line.word_idx[k]].text.endswith(","):
                    break
                if len(picked) >= 5:
                    break
            else:   # names must directly follow the label (avoids grabbing e.g. bank codes later on the line)
                break
        return picked

    def _name_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        out: list[Entity] = []
        for li, line in enumerate(lines):
            for m in re.finditer(r"(?:^|(?<=[\s/]))(?:" + "|".join(NAME_LABELS) + r")\b\s*[:.\-]?\s*",
                                 line.norm, re.I):
                label = m.group(0).strip().lower()
                if label.startswith("semnatura") and not line.norm[m.end():].strip():
                    continue
                idx = self._take_name_tokens(line, m.end(), words)
                # label alone on its line (ID-card layout) -> value is on the next line
                label_starts_line = not line.norm[:m.start()].strip(" /:.-")
                if not idx and label_starts_line and LABEL_ONLY_REST.fullmatch(line.norm[m.end():]) \
                        and li + 1 < len(lines):
                    nxt = lines[li + 1]
                    if not self._name_label_re.match(nxt.norm):
                        idx = self._take_name_tokens(nxt, 0, words)
                if idx:
                    value = " ".join(words[i].text.strip(",.;:") for i in idx)
                    kind = "surname" if re.match(r"^(nume(le)?|nom|last\s*name)\b", label) and "prenume" not in label \
                        else "given" if re.match(r"^(prenume(le)?|prenom|first\s*name)\b", label) else "full"
                    out.append(Entity("PERSON_NAME", value, idx, 0.9, f"context:{kind}"))
        return out

    # ---------------------------------------------------------------- addresses
    def _address_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        out: list[Entity] = []
        for li, line in enumerate(lines):
            m = self._addr_label_re.search(line.norm)
            start = None
            if m:
                start = m.end()
            else:
                m2 = self._addr_start_re.search(line.norm)
                if m2:
                    start = m2.start()
            if start is None:
                continue
            rest = line.norm[start:]
            if m and LABEL_ONLY_REST.fullmatch(rest):
                # label alone on its line (ID-card layout): the address is on the next line(s)
                if li + 1 < len(lines) and not ADDR_EXCLUDE.search(lines[li + 1].norm) \
                        and not self._name_label_re.match(lines[li + 1].norm):
                    idx = list(lines[li + 1].word_idx)
                    if li + 2 < len(lines) and ADDRESS_CONT.search(lines[li + 2].norm) \
                            and not self._name_label_re.match(lines[li + 2].norm) \
                            and not ADDR_EXCLUDE.search(lines[li + 2].norm):
                        idx += lines[li + 2].word_idx
                    out.append(Entity("ADDRESS", " ".join(words[i].text for i in idx), idx, 0.85, "context"))
                continue
            stop =re.search(r"\b(CNP|tel(?:efon)?|e-?mail|IBAN|seria|legitimat)\b", rest, re.I)
            end = start + (stop.start() if stop else len(rest))
            idx = _words_in_span(line, start, end)
            # continuation line (ID cards wrap the address over two lines)
            if li + 1 < len(lines):
                nxt = lines[li + 1]
                if ADDRESS_CONT.search(nxt.norm) and not self._name_label_re.match(nxt.norm) \
                        and not ADDR_EXCLUDE.search(nxt.norm) and "@" not in nxt.norm and len(nxt.word_idx) <= 10:
                    idx += nxt.word_idx
            if not idx and li + 1 < len(lines) and m:  # label alone on line
                idx = list(lines[li + 1].word_idx)
            if idx:
                value = " ".join(words[i].text for i in idx)
                if len(value) >= 6:
                    out.append(Entity("ADDRESS", value, idx, 0.85, "context"))
        return out

    # ---------------------------------------------------------------- MRZ
    def _mrz_entities(self, lines: list[Line]) -> list[Entity]:
        out = []
        for line in lines:
            compact = line.text.replace(" ", "")
            clean_ratio = len(re.findall(r"[A-Z0-9<«]", compact.upper())) / max(len(compact), 1)
            if compact.count("<") + compact.count("«") >= 2 and len(compact) >= 20 and clean_ratio >= 0.85:
                out.append(Entity("MRZ", compact, list(line.word_idx), 0.9))
        return out

    # ---------------------------------------------------------------- spaCy (optional)
    def _spacy_entities(self, lines: list[Line]) -> list[Entity]:
        out = []
        for line in lines:
            doc = self._nlp(line.text)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "PER"):
                    idx = _words_in_span(line, ent.start_char, ent.end_char)
                    if idx:
                        out.append(Entity("PERSON_NAME", ent.text, idx, 0.7, "spacy"))
        return out

    # ---------------------------------------------------------------- local SLM (optional)
    @staticmethod
    def _is_local(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost")

    def _slm_entities(self, lines: list[Line], words: list[Word], base_url: str, model: str) -> list[Entity]:
        # Privacy guard: raw text may only ever go to a model running on THIS machine.
        if not self._is_local(base_url):
            return []
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key="local", timeout=60)
            text = "\n".join(l.text for l in lines)
            prompt = ("Extract every person full name and postal address from this Romanian document. "
                      'Answer ONLY JSON: {"names": [...], "addresses": [...]} using exact substrings.\n\n' + text)
            resp = client.chat.completions.create(model=model, temperature=0,
                                                  messages=[{"role": "user", "content": prompt}])
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content[content.find("{"): content.rfind("}") + 1])
        except Exception:
            return []
        out = []
        for etype, key in (("PERSON_NAME", "names"), ("ADDRESS", "addresses")):
            for value in data.get(key, []) or []:
                if not isinstance(value, str) or len(value) < 3:
                    continue
                for line in lines:
                    pos = line.text.find(value)
                    if pos >= 0:
                        idx = _words_in_span(line, pos, pos + len(value))
                        if idx:
                            out.append(Entity(etype, value, idx, 0.75, "slm"))
        return out

    # ---------------------------------------------------------------- propagation & dedupe
    def _propagate_names(self, entities: list[Entity], lines: list[Line], words: list[Word]) -> list[Entity]:
        """Redact every other occurrence of a detected name token (signatures, headers...)."""
        tokens = {strip_diacritics(words[i].text.strip(",.;:")).upper()
                  for e in entities if e.type == "PERSON_NAME" for i in e.word_indices}
        tokens = {t for t in tokens if len(t) >= 3}
        if not tokens:
            return entities
        covered = {i for e in entities for i in e.word_indices}
        extra: list[Entity] = []
        for line in lines:
            run: list[int] = []
            for i in line.word_idx + [-1]:
                tok = strip_diacritics(words[i].text.strip(",.;:")).upper() if i >= 0 else ""
                if i >= 0 and tok in tokens and i not in covered:
                    run.append(i)
                else:
                    if run:
                        extra.append(Entity("PERSON_NAME", " ".join(words[j].text for j in run), run, 0.75, "propagated"))
                    run = []
        return entities + extra

    @staticmethod
    def _dedupe(entities: list[Entity]) -> list[Entity]:
        # Keep higher-confidence entity when two entities of the same type cover the same words.
        entities = sorted(entities, key=lambda e: -e.confidence)
        kept: list[Entity] = []
        for e in entities:
            s = set(e.word_indices)
            if any(k.type == e.type and s <= set(k.word_indices) for k in kept):
                continue
            kept.append(e)
        return kept


# ----------------------------------------------------------------------------
# Box merging
# ----------------------------------------------------------------------------
def entity_boxes(entities: list[Entity], words: list[Word], gap_factor: float = 1.5) -> list[tuple[int, int, int, int, int]]:
    """Union word boxes per (entity, line); merge horizontally adjacent boxes on the same line."""
    per_line: dict[tuple, list[list[int]]] = {}
    for e in entities:
        by_line: dict[tuple, list[int]] = {}
        for i in sorted(set(e.word_indices)):
            by_line.setdefault(words[i].line_key, []).append(i)
        for key, idx in by_line.items():
            xs = [words[i].x for i in idx]
            ys = [words[i].y for i in idx]
            x2 = [words[i].x + words[i].w for i in idx]
            y2 = [words[i].y + words[i].h for i in idx]
            per_line.setdefault(key, []).append([min(xs), min(ys), max(x2), max(y2)])
    boxes: list[tuple[int, int, int, int, int]] = []
    for key, rects in per_line.items():
        rects.sort()
        merged: list[list[int]] = []
        for r in rects:
            if merged:
                m = merged[-1]
                height = max(m[3] - m[1], r[3] - r[1])
                if r[0] <= m[2] + gap_factor * height * 0.5:
                    m[0], m[1], m[2], m[3] = min(m[0], r[0]), min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3])
                    continue
            merged.append(list(r))
        page = key[0]
        boxes += [(page, a, b, c - a, d - b) for a, b, c, d in merged]
    return boxes


def summarize(entities: list[Entity]) -> dict[str, int]:
    summary = {"CNP": 0, "PERSON_NAME": 0, "ADDRESS": 0, "ID_SERIES": 0}
    for e in entities:
        summary[e.type] = summary.get(e.type, 0) + 1
    return summary
