"""Structured ID-card profile on top of the PII detector (Romanian CI / buletin, old and new layouts).

``extract_profile`` turns detector entities (+ OCR lines for a few unlabeled fields and the MRZ fallback) into a
normalized dictionary of field values. The result is an ``IDProfileRecord`` kept ONLY in the vault — callers outside
the server only ever see the field KEYS that were found.
"""
from __future__ import annotations

import re
from datetime import date

from server.pii_detector import ID_CARD_CONTEXT, build_lines, cnp_is_valid, strip_diacritics
from server.vault import DocumentRecord, Entity, IDProfileRecord

PROFILE_KEYS = (
    "last_name", "first_name", "full_name", "cnp", "id_series", "id_number", "id_series_number", "address_full",
    "county", "city", "street", "street_no", "building", "staircase", "floor", "apartment", "sector",
    "birth_date", "birth_place", "issue_date", "expiry_date", "issuing_authority", "nationality", "sex",
)
DERIVED_CONFIDENCE = 0.6

MRZ1_RE = re.compile(r"^[I1LT|]D[R][O0]U([A-Z<«]+)$")
MRZ2_RE = re.compile(r"^([A-Z]{2})(\d{6,7})")
SERIES_RE = re.compile(r"\b([A-Z]{2})\W{0,3}(?:N[RO]\.?\W{0,3})?(\d{6,7})\b")
DATE_PARTS = re.compile(r"(\d{1,2})[./\-\s](\d{1,2})[./\-\s](\d{2,4})")
NATIONALITY_LABEL = re.compile(r"cet[aă][tț]en[il1]e|nationalit|nationality", re.I)
SEX_LINE = re.compile(r"\bsex[eu]?\b[^A-Za-z]{0,12}\b([MF])\b", re.I)

# --- address components ("Str. Lalelelor nr. 10, bl. E2, sc. B, et. 3, ap. 24, Mun. Iasi, jud. Iasi")
_STOP = r"(?=\s*(?:,|;|\bnr\b|\bbl\b|\bsc\b|\bet\b|\bap\b|\bjud|\bmun\b|\bsector|\bsec\b|$))"
ADDR_PATTERNS = {
    "street": re.compile(r"\b(?:str(?:ada)?\.?|(bd\.?|b-dul|bulevardul|calea|aleea|sos\.?|[sș]oseaua|splaiul|"
                         r"intrarea|pia[tț]a))\s*([^,;]+?)" + _STOP, re.I),
    "street_no": re.compile(r"\bnr\.?\s*([0-9]+[A-Za-z]?(?:\s?[-/]\s?[0-9A-Za-z]+)?)", re.I),
    "building": re.compile(r"\bbl\.?\s*([A-Za-z0-9][A-Za-z0-9\-]*)", re.I),
    "staircase": re.compile(r"\bsc\.?\s*([A-Za-z0-9]+)", re.I),
    "floor": re.compile(r"\bet\.?\s*([0-9A-Za-z]+)", re.I),
    "apartment": re.compile(r"\bap\.?\s*([0-9A-Za-z]+)", re.I),
    "county": re.compile(r"\bjud(?:e[tț]ul)?\.?\s*([^,;]+?)(?=\s*(?:,|;|\bmun\b|\bsector|$))", re.I),
    "sector": re.compile(r"\bsec(?:tor(?:ul)?)?\.?\s*([1-6])\b", re.I),
    "city": re.compile(r"\b(?:mun(?:icipiul)?|ora[sș](?:ul)?|or|com(?:una)?|sat(?:ul)?|loc(?:alitatea)?)\.?\s+"
                       r"([^,;]+?)(?=\s*(?:,|;|\bjud|\bsector|\bsec\b|$))", re.I),
}


def display_case(value: str) -> str:
    """'POPESCU-IONESCU MARIA' -> 'Popescu-Ionescu Maria' (mixed-case input is kept as read)."""
    v = re.sub(r"\s+", " ", value).strip(" ,;:")
    return v.title() if v.isupper() else v


def norm_date(value: str) -> str | None:
    m = DATE_PARTS.search(value)
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    if len(m.group(3)) == 2:
        y += 2000 if y <= date.today().year % 100 + 10 else 1900
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{d:02d}.{mo:02d}.{y}"


def cnp_facts(cnp: str) -> dict[str, str]:
    """Sex and birth date encoded in a CNP."""
    s, yy, mm, dd = int(cnp[0]), int(cnp[1:3]), int(cnp[3:5]), int(cnp[5:7])
    century = {1: 1900, 2: 1900, 3: 1800, 4: 1800, 5: 2000, 6: 2000}.get(s)
    if century is None:                          # 7/8/9: residents / foreigners — century not encoded
        century = 2000 if yy <= date.today().year % 100 else 1900
    out = {"sex": "M" if s % 2 else "F"}
    if 1 <= mm <= 12 and 1 <= dd <= 31:
        out["birth_date"] = f"{dd:02d}.{mm:02d}.{century + yy}"
    return out


def parse_address(address: str) -> dict[str, str]:
    """Split a Romanian address into components. Missing components are simply absent."""
    text = re.sub(r"\s+", " ", address).strip()
    out: dict[str, str] = {}
    for key, rx in ADDR_PATTERNS.items():
        m = rx.search(text)
        if not m:
            continue
        val = (m.group(m.lastindex) if m.lastindex else m.group(0)).strip(" .,;")
        if key == "street" and m.group(1):       # keep the artery type when it is not a plain "strada"
            val = f"{m.group(1).strip()} {val}"
        if val:
            out[key] = display_case(val) if key in ("street", "city", "county") else val
    if "sector" in out:                          # Bucharest prints "Sector 3" instead of a county
        out.setdefault("county", "București")
        out.setdefault("city", "București")
    return out


def _mrz_names(entities: list[Entity]) -> tuple[str, str] | None:
    for e in entities:
        if e.type != "MRZ":
            continue
        m = MRZ1_RE.match(e.value.upper().replace("«", "<").replace(" ", ""))
        if not m:
            continue
        parts = [p for p in re.split(r"<{2,}", m.group(1)) if p]
        if len(parts) >= 2:
            sur, giv = parts[0].replace("<", " "), parts[1].replace("<", " ")
            if sur.isalpha() or " " in sur:
                return display_case(sur), display_case(giv)
    return None


def _mrz_series(entities: list[Entity]) -> tuple[str, str] | None:
    for e in entities:
        if e.type == "MRZ":
            m = MRZ2_RE.match(e.value.upper().replace("$", "S").replace(" ", ""))
            if m and not MRZ1_RE.match(e.value.upper()):
                return m.group(1), m.group(2)
    return None


def _line_after(lines, label_re) -> str | None:
    for i, line in enumerate(lines):
        m = label_re.search(line.norm)
        if not m:
            continue
        rest = line.text[m.end():].strip(" /:.-")
        rest = re.sub(r"^[A-Za-z/ ]*(?:nationalit\w*|nationality)\W*", "", rest, flags=re.I).strip(" /:.-")
        if rest:
            return rest
        if i + 1 < len(lines):
            return lines[i + 1].text
    return None


def extract_profile(doc: DocumentRecord, entities: list[Entity]) -> IDProfileRecord:
    lines = build_lines(doc.words)
    doc_text = " ".join(l.norm for l in lines)
    fields: dict[str, str] = {}
    conf: dict[str, float] = {}

    def put(key: str, value: str | None, c: float) -> None:
        if value and key not in fields:
            fields[key] = value.strip()
            conf[key] = round(float(c), 2)

    # names: labelled values first, then the MRZ, then (non-ID documents) the first person name as full name
    names = [e for e in entities if e.type == "PERSON_NAME"]
    sur = next((e for e in names if e.source.endswith("surname")), None)
    giv = next((e for e in names if e.source.endswith("given")), None)
    if sur:
        put("last_name", display_case(sur.value), sur.confidence)
    if giv:
        put("first_name", display_case(giv.value), giv.confidence)
    mrz_names = _mrz_names(entities)
    if mrz_names:
        put("last_name", mrz_names[0], 0.8)
        put("first_name", mrz_names[1], 0.8)
    if "last_name" not in fields and names:
        put("full_name", display_case(names[0].value), names[0].confidence)
    if "last_name" in fields and "first_name" in fields:
        put("full_name", f"{fields['last_name']} {fields['first_name']}",
            min(conf["last_name"], conf["first_name"]))

    # CNP (checksum-valid) and what it encodes
    cnp = next((re.sub(r"\D", "", e.value) for e in entities if e.type == "CNP"
                and cnp_is_valid(re.sub(r"\D", "", e.value))), None)
    if cnp:
        put("cnp", cnp, 0.99)

    # series + number
    series = None
    for e in sorted((e for e in entities if e.type == "ID_SERIES"), key=lambda e: e.source != "context"):
        m = SERIES_RE.search(e.value.upper())
        if m:
            series, sc = (m.group(1), m.group(2)), e.confidence
            break
    if series is None:
        series, sc = _mrz_series(entities), 0.8
    if series:
        put("id_series", series[0], sc)
        put("id_number", series[1], sc)
        put("id_series_number", f"{series[0]} {series[1]}", sc)

    # address (all ADDRESS entities in reading order) + components
    addr = [e for e in entities if e.type == "ADDRESS"]
    if addr:
        addr.sort(key=lambda e: min(e.word_indices) if e.word_indices else 0)
        full = re.sub(r"\s+", " ", ", ".join(dict.fromkeys(e.value.strip(" ,") for e in addr)))
        full = re.sub(r"\s+,", ",", full)
        # a value wrapped over two card lines joins without punctuation: "ap. 24 Mun. Iasi" -> "ap. 24, Mun. Iasi"
        full = re.sub(r"(?<=[^,\s])\s+(?=(?:Mun|Municipiul|Jud|Jude[tț]ul|Ora[sș]|Com|Comuna|Sat|Sector)\b)",
                      ", ", full)
        ac = min(e.confidence for e in addr)
        put("address_full", full, ac)
        for k, v in parse_address(full).items():
            put(k, v, ac)

    # issuer, dates, birth place
    issuer = next((e for e in entities if e.type == "ISSUER"), None)
    if issuer:
        put("issuing_authority", re.sub(r"\s+", " ", issuer.value).strip(" ,.;"), issuer.confidence)
    for e in entities:
        if e.type == "ID_DATE":
            dates = [d for d in (norm_date(x) for x in re.split(r"\s*[-–]\s*(?=\d{1,2}[./])", e.value)) if d]
            if len(dates) >= 2:
                put("issue_date", dates[0], e.confidence)
                put("expiry_date", dates[-1], e.confidence)
            elif dates:
                d, m, y = (int(x) for x in dates[0].split("."))
                put("expiry_date" if date(y, m, min(d, 28)) > date.today() else "issue_date", dates[0], 0.7)
        elif e.type == "BIRTH_DATE":
            put("birth_date", norm_date(e.value), e.confidence)
        elif e.type == "BIRTH_PLACE":
            put("birth_place", display_case(e.value), e.confidence)

    nat = _line_after(lines, NATIONALITY_LABEL)
    if nat:
        n = strip_diacritics(nat).upper()
        put("nationality", "Română" if ("ROU" in n or "ROMAN" in n) else display_case(nat.split("/")[0]), 0.8)
    sm = SEX_LINE.search(doc_text)
    if sm:
        put("sex", sm.group(1).upper(), 0.85)
    if cnp:
        for k, v in cnp_facts(cnp).items():
            put(k, v, DERIVED_CONFIDENCE)

    is_id_card = bool(ID_CARD_CONTEXT.search(doc_text)) or any(
        e.type == "MRZ" and MRZ1_RE.match(e.value.upper().replace(" ", "")) for e in entities)
    is_valid_id = bool(cnp) and "id_number" in fields
    fields = {k: fields[k] for k in PROFILE_KEYS if k in fields}
    return IDProfileRecord(doc.doc_id, fields, {k: conf[k] for k in fields}, is_id_card, is_valid_id)
