"""Layout-agnostic PII detection on word geometry (any language / template / column layout).

The Romanian rule set in ``pii_detector`` reads OCR *lines*. Real documents also come as bilingual ID cards
("Прізвище/ Surname" with the value on the next line(s)), two-column invoices ("Seller:" | "Client:" blocks) and
tables, where Tesseract's lines mix columns. Here words are regrouped into visual rows, rows are split into column
cells at wide gaps, and every cell is matched against a multilingual label lexicon (plus labels the user taught by
drawing manual blur boxes, see ``rule_learner``). The value of a label is taken right of it in the same cell, or
from the cells below it in the same column, until the next label / section header / a large vertical gap.

Besides word-level entities it returns *regions*: rectangles spanning a label's value block, so lines OCR could not
read (Cyrillic on a ron+eng pass, stamps, faint print) between the label and its value are redacted too.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from server.vault import Entity, Word

Box = tuple[int, int, int, int, int]          # (page, x, y, w, h)


def norm(text: str) -> str:
    """Lower-case, no diacritics (Latin + Cyrillic kept), apostrophes unified."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.replace("’", "'").replace("`", "'").lower()


# ----------------------------------------------------------------------------- label lexicon
# kind -> how the value is read: max rows below, what a value must look like, entity type
@dataclass(frozen=True)
class Kind:
    etype: str
    rows: int                 # max value rows below the label
    value: str                # name | text | date | id | iban | phone
    region: bool = False      # redact the whole block between label and last value row


KINDS = {
    "name": Kind("PERSON_NAME", 2, "name", True),
    "party": Kind("PARTY", 5, "text", True),
    "birth_date": Kind("BIRTH_DATE", 2, "date"),
    "id_date": Kind("ID_DATE", 2, "date"),
    "id": Kind("ID_SERIES", 2, "id"),
    "iban": Kind("IBAN", 1, "iban"),
    "address": Kind("ADDRESS", 3, "text", True),
    "birth_place": Kind("BIRTH_PLACE", 2, "name", True),
    "phone": Kind("PHONE", 1, "phone"),
}

_NO = r"(?:no\.?|mo\.|n[o°º]\.?|nr\.?|number|num[aă]r(?:ul)?|#|code|cod)"   # OCR: "No." -> "Mo."
LABELS: list[tuple[str, re.Pattern, bool]] = [   # (kind, regex, needs ':' or end-of-cell)
    ("name", re.compile(r"\b(?:surname|last\s*name|family\s*name|given\s*names?|first\s*names?|forenames?|"
                        r"middle\s*names?|patronymic|father'?s\s+name|maiden\s+name|nachname|vorname|cognome|"
                        r"apellidos?|holder'?s?\s+name|name\s+of\s+(?:the\s+)?holder|account\s+holder|"
                        r"nom\s+de\s+famille|pr[eé]noms?|"
                        r"full\s+name|name\s+and\s+surname)\b"), False),
    ("name", re.compile(r"(?<![\w-])(?:name|nom|nome|nombre|imya|im'ya)\b"), True),
    ("party", re.compile(r"\b(?:seller|buyer|client|customer|bill(?:ed)?\s+to|ship(?:ped)?\s+to|sold\s+to|"
                         r"issued\s+to|invoice\s+to|recipient|payer|payee|vendor|supplier|beneficiary|consignee|"
                         r"employee|employer|patient|tenant|landlord|lessee|lessor|contractor|guarantor|"
                         r"furnizor(?:ul)?|cump[aă]r[aă]tor(?:ul)?|v[aâ]nz[aă]tor(?:ul)?|beneficiar(?:ul)?|"
                         r"pacient(?:ul)?|chiria[sș](?:ul)?|locatar(?:ul)?|destinatar(?:ul)?|"
                         r"expediteur|destinataire|acheteur|vendeur|kunde|verk[aä]ufer|k[aä]ufer|empf[aä]nger)\b"),
     True),
    ("birth_date", re.compile(r"\b(?:date\s+of\s+birth|birth\s*date|d\.?\s?o\.?\s?b\b\.?|born(?:\s+on)?|"
                              r"n[eé]e?\s+le|date\s+de\s+naissance|geburtsdatum|data\s+di\s+nascita|"
                              r"fecha\s+de\s+nacimiento|data\s+(?:de\s+)?na[sș]ter(?:ii|e))\b"), False),
    ("id_date", re.compile(r"\b(?:date\s+of\s+(?:issue|expiry|expiration)|expiry\s+date|expiration\s+date|"
                           r"valid\s+(?:until|thru|through|to)|date\s+d.?expiration|date\s+de\s+d[eé]livrance|"
                           r"g[uü]ltig\s+bis|issue\s+date|data\s+(?:emiterii|eliber[aă]rii|expir[aă]rii))\b"), False),
    ("id", re.compile(r"\b(?:(?:document|passport|record|personal|identity|id|card|licen[cs]e|permit|"
                      r"social\s+security|national\s+id|insurance|policy|customer|client|member|patient|"
                      r"employee|registration|reference)\s*" + _NO + r"|"
                      r"tax\s*(?:id|no\.?|number|code)|tin|ssn|nif|nie|cod\s+client|steuer(?:nummer|-?id)|"
                      r"personal\s+code|codice\s+fiscale|pesel)\b"), False),   # (company CUI / VAT are not PII)
    ("iban", re.compile(r"\b(?:iban|bic|swift|account\s*(?:no\.?|number)|cont\s+(?:bancar|iban))\b"),
     False),
    ("address", re.compile(r"\b(?:address|adresse|anschrift|indirizzo|direcci[oó]n|residence|domicile|"
                           r"place\s+of\s+residence|home\s+address)\b"), False),
    ("birth_place", re.compile(r"\b(?:place\s+of\s+birth|lieu\s+de\s+naissance|geburtsort|luogo\s+di\s+nascita|"
                               r"lugar\s+de\s+nacimiento)\b"), False),
    ("phone", re.compile(r"\b(?:phone|tel(?:ephone)?|mobile|cell|fax)\b\.?"), True),
]
# a label line on an ID card ("Дата народження/ Date of birth") or a section that ends a value block
SECTION_STOP = re.compile(r"^(?:items?|summary|description|total|subtotal|no\.?|qty|quantity|unit|price|amount|"
                          r"notes?|terms|payment|bank\s+details?|signature|semn[aă]tura|services?|products?)\b")
# field labels that carry no PII themselves but end the value block above them
STOP_LABELS = re.compile(r"\b(?:sex|gender|nationality|citizenship|cet[aă][tț]enie|height|eyes?|issuing\s+authority|"
                         r"authority|signature|semn[aă]tura|photo|issued\s+by|emis[aă]?\s+de|eliberat[aă]?\s+de|"
                         r"d[eé]livr[eé]e?|valabil\w*|validit\w*)\b|na[l1i]{1,2}ona[l1i]{1,2}[l1i]?ty")
ID_DOC_CONTEXT = re.compile(r"passport|passeport|reisepass|pasaport|identity\s+card|id\s*card|carte\s+d.?identit|"
                            r"personalausweis|residence\s+permit|permis\s+de\s+[sș]edere|titre\s+de\s+s[eé]jour|"
                            r"driving\s+licen[cs]e|permis\s+de\s+(?:conducere|conduire)|carte\s+de\s+identitate|"
                            r"buletin|^p<[a-z]{3}|^id[a-z]{3}<")

DATE_ANY = re.compile(r"(?<!\d)(?:\d{1,2}[\s./\-]\d{1,2}[\s./\-](?:19|20)\d{2}|(?:19|20)\d{2}[./\-]\d{1,2}[./\-]\d{1,2}|"
                      r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2})(?!\d)",
                      re.I)
# words printed on identity documents that are never personal (document type / country / status)
DOC_WORDS = set("""romania roumanie romana romanian carte identitate identite identity card passport passeport pasaport
ukraine ukraina permis sedere temporara temporar permanent permanenta refugiat refugie refugee residence permit titre
sejour driving licence license conducere conduire republica republic european union uniunea europeana specimen
document national nationale buletin""".split())
KIND_PRIORITY = {"party": 0, "name": 1}            # every other (specific) kind: 2
KIND_PRIORITY.update({k: 2 for k in ("birth_date", "id_date", "id", "iban", "address", "birth_place", "phone")})
# "Adresa:", "Telefon mobil:" (a colon next to a blurred value is often read as ';')
FIELD_LABEL = re.compile(r"^[^\W\d_][^\W\d_.\- ]{1,24}(?:\s+[^\W\d_]{1,15}){0,2}\s*[:;]")
COMPANY = re.compile(r"\b(?:s\.?r\.?l|s\.?a|sc|llc|ltd|gmbh|inc|corp|plc|s\.?a\.?s|sarl|bv|nv|ag|srl)\b\.?", re.I)
IBAN_ANY = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")
LETTERS = re.compile(r"[^\W\d_]{2,}")


def _name_like(text: str) -> bool:
    """A name right of its label: capitalized letter tokens ('Popescu Andrei', 'POPESCU-IONESCU'), no OCR junk
    ('PF Pact fo |' after a bilingual label is noise)."""
    toks = text.replace(",", " ").split()
    return bool(toks) and len(toks) <= 5 and all(
        re.fullmatch(r"[^\W\d_][^\W\d_'\-.]+", t) and t[0].isupper() for t in toks) and any(len(t) >= 3 for t in toks)


def iban_checksum_ok(value: str) -> bool:
    v = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", v):
        return False
    digits = "".join(str(int(c, 36)) for c in v[4:] + v[:4])
    return int(digits) % 97 == 1


# ----------------------------------------------------------------------------- layout
@dataclass
class Cell:
    page: int
    idx: list[int]                     # word indices, left to right
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)
    row: int = 0

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)


def build_cells(words: list[Word]) -> tuple[list[list[Cell]], float]:
    """Visual rows (top to bottom) of column cells (left to right). Returns (rows, median word height)."""
    if not words:
        return [], 0.0
    # punctuation-only OCR noise ('—', '|', '`') would bridge column gaps and glue two fields into one cell
    keep = [i for i, w in enumerate(words) if re.search(r"[^\W_]", w.text)]
    if not keep:
        return [], 0.0
    hs = sorted(words[i].h for i in keep)
    med_h = float(hs[len(hs) // 2])
    order = sorted(keep, key=lambda i: (words[i].page, words[i].y + words[i].h / 2))
    rows: list[list[int]] = []
    for i in order:
        w = words[i]
        cy = w.y + w.h / 2
        if rows:
            last = rows[-1]
            ref = words[last[0]]
            rcy = sum(words[j].y + words[j].h / 2 for j in last) / len(last)
            if ref.page == w.page and abs(cy - rcy) <= 0.55 * max(med_h, min(w.h, 2 * med_h)):
                last.append(i)
                continue
        rows.append([i])
    out: list[list[Cell]] = []
    for r, row in enumerate(rows):
        row.sort(key=lambda i: words[i].x)
        rh = max(med_h, sorted(words[i].h for i in row)[len(row) // 2])
        cells: list[list[int]] = [[row[0]]]
        for a, b in zip(row, row[1:]):
            gap = words[b].x - (words[a].x + words[a].w)
            if gap > 1.6 * rh:
                cells.append([b])
            else:
                cells[-1].append(b)
        row_cells = []
        for c in cells:
            parts, spans, pos = [], [], 0
            for i in c:
                t = words[i].text
                spans.append((pos, pos + len(t)))
                parts.append(t)
                pos += len(t) + 1
            row_cells.append(Cell(words[c[0]].page, c, min(words[i].x for i in c), min(words[i].y for i in c),
                                  max(words[i].x + words[i].w for i in c), max(words[i].y + words[i].h for i in c),
                                  " ".join(parts), spans, r))
        out.append(row_cells)
    return out, med_h


def _words_after(cell: Cell, char_end: int) -> list[int]:
    return [cell.idx[k] for k, (s, _e) in enumerate(cell.spans) if s >= char_end]


# ----------------------------------------------------------------------------- detector
@dataclass
class LabelHit:
    kind: str
    cell: Cell
    end: int                           # char offset in the cell where the label ends
    source: str = "universal"


class UniversalDetector:
    def __init__(self, learned="shared") -> None:
        # "shared": the global learned_rules.json store, looked up at detection time (the path may change, and
        # rules learned in the web UI apply at once); None: no learned rules; or an explicit RuleStore
        self._learned = learned

    @property
    def learned(self):
        if self._learned == "shared":
            from server.rule_learner import RuleStore
            return RuleStore.shared()
        return self._learned

    # labels ------------------------------------------------------------
    def _labels(self, cell: Cell, id_context: bool) -> list[LabelHit]:
        t = norm(cell.text)
        # OCR often ends a label word with "o" for "e" ("Surnamo", "Namo", "Dato of birth"): try both spellings
        variants = {t, re.sub(r"(?<=[a-z]{2})o\b", "e", t)}
        hits: list[LabelHit] = []
        for kind, rx, strict in LABELS:
            for m in (m for v in variants for m in rx.finditer(v)):
                if kind == "id_date" and not id_context:
                    continue                                  # an invoice's "Date of issue" is not personal
                rest = t[m.end():]
                if strict and not (re.match(r"\s*[:;]", rest) or not LETTERS.search(rest) and not re.search(r"\d", rest)):
                    continue                                  # generic word ("Product name") — needs "Name:" form
                end = m.end() + len(rest) - len(rest.lstrip(" :;.-/|"))
                hits.append(LabelHit(kind, cell, end))
        if self.learned is not None:
            for rule, rx in self.learned.patterns():
                m = rx.search(t)
                if m:
                    rest = t[m.end():]
                    end = m.end() + len(rest) - len(rest.lstrip(" :;.-/|"))
                    hits.append(LabelHit(rule.kind, cell, end, f"learned:{rule.id}"))
        # several labels in one cell: the most specific kind wins ("IBAN client:" is an IBAN, not a party block),
        # then the right-most ("Прізвище/ Surname" — the value follows the last part)
        hits.sort(key=lambda h: (KIND_PRIORITY.get(h.kind, 0), h.end), reverse=True)
        return hits[:1] if hits else []

    def is_label_cell(self, cell: Cell, id_context: bool = True) -> bool:
        t = norm(cell.text)
        return bool(self._labels(cell, id_context)) or bool(SECTION_STOP.match(t.strip(" :"))) or \
            bool(STOP_LABELS.search(t))

    # values ------------------------------------------------------------
    @staticmethod
    def _accept(value_kind: str, text: str) -> bool:
        if value_kind in ("name", "text"):
            return bool(LETTERS.search(text))
        if value_kind == "date":       # a full date, or the readable part of one ("2025" of "13 12 2025")
            return bool(DATE_ANY.search(text) or re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))
        if value_kind == "id":
            compact = re.sub(r"[\s\-./]", "", text)
            return len(re.findall(r"\d", compact)) >= 4 or bool(re.search(r"[A-Z0-9]{5,}", compact) and
                                                                  re.search(r"\d", compact))
        if value_kind == "iban":       # IBAN shape, or a bank account number (mostly digits)
            compact = re.sub(r"[\s\-]", "", text.upper())
            return bool(re.search(r"[A-Z]{2}\d{2}[A-Z0-9]{8,}", compact) or re.search(r"\d{8,}", compact))
        if value_kind == "phone":
            return len(re.findall(r"\d", text)) >= 6
        return False

    @staticmethod
    def _value_words(value_kind: str, words: list[Word], idx: list[int]) -> list[int]:
        """Restrict a value run to the part that looks like the value (dates / ids inside longer text)."""
        if value_kind != "date":
            return idx
        text, spans, pos = [], [], 0
        for i in idx:
            spans.append((pos, pos + len(words[i].text)))
            text.append(words[i].text)
            pos += len(words[i].text) + 1
        m = DATE_ANY.search(" ".join(text)) or re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", " ".join(text))
        if not m:
            return []
        return [i for i, (s, e) in zip(idx, spans) if s < m.end() and e > m.start()]

    @staticmethod
    def _id_context_text(doc_text: str) -> bool:
        return bool(ID_DOC_CONTEXT.search(doc_text)) or sum(
            1 for kind, rx, _ in LABELS if kind in ("name", "birth_date", "id", "id_date") and rx.search(doc_text)) >= 3

    def id_context(self, words: list[Word]) -> bool:
        """Identity document (passport / ID card / permit, any language) by keywords or its label set."""
        t = " ".join(norm(w.text) for w in words)
        return self._id_context_text(t) or self._id_context_text(re.sub(r"(?<=[a-z]{2})o\b", "e", t))

    def detect(self, words: list[Word]) -> tuple[list[Entity], list[Box]]:
        rows, med_h = build_cells(words)
        if not rows:
            return [], []
        id_context = self.id_context(words)
        entities: list[Entity] = []
        regions: list[Box] = []
        label_cells: set[int] = set()
        hits_all: list[LabelHit] = []
        for row in rows:
            for cell in row:
                hs = self._labels(cell, id_context)
                if hs:
                    hits_all += hs
                    label_cells.add(id(cell))
        for hit in hits_all:
            kind = KINDS[hit.kind]
            cell = hit.cell
            right = _words_after(cell, hit.end)
            right_text = " ".join(words[i].text for i in right)
            value_rows: list[list[int]] = []
            if right and self._accept(kind.value, right_text) and (kind.value != "name" or _name_like(right_text)):
                value_rows.append(right)
            if hit.kind == "party" and value_rows and COMPANY.search(right_text):
                value_rows = []                               # a company ("SC ENERGIE VERDE SRL") is not PII
            elif not value_rows:                              # a party block only when the label stands alone
                value_rows += self._below(rows, cell, kind, words, med_h, label_cells, id_context)
            value_rows = [v for v in (self._value_words(kind.value, words, r) for r in value_rows) if v]
            if not value_rows:
                continue
            for r in value_rows:
                entities.append(Entity(kind.etype, " ".join(words[i].text for i in r), r, 0.8, hit.source))
            if kind.value in ("date", "id"):                 # partly read value: cover its whole row band
                # left edge = the label column (labels above/below are left-aligned; OCR may miss the label start)
                left = min([cell.x0] + [h.cell.x0 for h in hits_all if h.cell.page == cell.page
                                        and abs(h.cell.x0 - cell.x0) <= 6 * med_h])
                for r in value_rows:
                    if r is right or set(r) <= set(right):
                        continue
                    ys = [words[i].y for i in r] + [words[i].y + words[i].h for i in r]
                    x0 = min([left] + [words[i].x for i in r])
                    x1 = max([cell.x1] + [words[i].x + words[i].w for i in r])
                    regions.append((cell.page, x0, min(ys), x1 - x0, max(ys) - min(ys)))
            if kind.region:
                below_idx = [i for r in value_rows for i in r if i not in right]
                if below_idx:
                    x0 = min(min(words[i].x for i in below_idx), cell.x0)
                    x1 = max(words[i].x + words[i].w for i in below_idx)
                    y0 = cell.y1 + 1
                    y1 = max(words[i].y + words[i].h for i in below_idx)
                    if y1 > y0:
                        regions.append((cell.page, x0, y0, x1 - x0, y1 - y0))
        if id_context:                                       # every date on an identity document is personal
            covered = {i for e in entities for i in e.word_indices}
            for row in rows:
                for cell in row:
                    if id(cell) in label_cells and not _words_after(cell, 0):
                        continue
                    for m in DATE_ANY.finditer(cell.text):
                        idx = [cell.idx[k] for k, (s, e) in enumerate(cell.spans) if s < m.end() and e > m.start()]
                        if idx and not set(idx) <= covered:
                            entities.append(Entity("ID_DATE", m.group(0), idx, 0.75, "universal"))
        for row in rows:                                     # IBANs of any country (checksum-valid)
            for cell in row:
                for m in IBAN_ANY.finditer(cell.text.upper()):
                    if iban_checksum_ok(m.group(0)):
                        idx = [cell.idx[k] for k, (s, e) in enumerate(cell.spans) if s < m.end() and e > m.start()]
                        entities.append(Entity("IBAN", m.group(0).replace(" ", ""), idx, 0.95, "universal"))
        entities += self._address_blocks(rows, words, med_h)
        return entities, regions

    def _below(self, rows, cell: Cell, kind: Kind, words, med_h: float, label_cells: set[int],
               id_context: bool) -> list[list[int]]:
        """Value rows under a label, inside its column."""
        out: list[list[int]] = []
        row_cells = rows[cell.row]
        nxt = [c for c in row_cells if c.x0 > cell.x1]
        # values are usually left-aligned with the (bilingual) label, whose first part OCR may have missed
        col_x0 = cell.x0 - 5 * med_h
        col_x1 = min(c.x0 for c in nxt) - med_h * 0.5 if nxt else 10 ** 9
        prev_bottom = cell.y1
        for r in range(cell.row + 1, len(rows)):
            if len(out) >= kind.rows:
                break
            # word level: a cell may start with OCR junk left of the column ("i} MARIANA")
            hit = None
            for c in rows[r]:
                if c.page != cell.page:
                    continue
                sub = [i for i in c.idx if col_x0 <= words[i].x < col_x1 and words[i].x + words[i].w > cell.x0 - med_h
                       and re.search(r"[^\W_]", words[i].text)]          # punctuation-only OCR noise is no value
                if sub:
                    hit = (c, sub)
                    break
            if hit is None:
                if rows[r] and rows[r][0].page == cell.page and min(c.y0 for c in rows[r]) - prev_bottom > 2.2 * med_h:
                    break
                continue
            c, sub = hit
            text = " ".join(words[i].text for i in sub)
            y0, y1 = min(words[i].y for i in sub), max(words[i].y + words[i].h for i in sub)
            # (an unread line — e.g. Cyrillic on a ron+eng pass — may sit between label and value)
            if y0 - prev_bottom > 2.2 * max(med_h, cell.h, min(y1 - y0, 3 * med_h)):
                break
            if (id(c) in label_cells or SECTION_STOP.match(norm(text).strip(" :")) or STOP_LABELS.search(norm(text))
                    or FIELD_LABEL.match(text)):
                break
            if not self._accept(kind.value, text):
                if kind.value in ("date", "id"):
                    prev_bottom = y1
                    continue                                  # e.g. a garbled Cyrillic line before the value
                break
            out.append(sub)
            prev_bottom = y1
        return out

    def portrait_column_names(self, words: list[Word], portraits: list[Box]) -> list[Entity]:
        """Identity documents whose labels OCR cannot read (low-res photos): name-like lines in the text column
        right of the holder's photo are names (document-type words such as PERMIS DE SEDERE are skipped)."""
        if not portraits:
            return []
        page, px, py, pw, ph = max(portraits, key=lambda b: b[3] * b[4])
        ids = [i for i, w in enumerate(words) if w.page == page]      # cell index k -> global word index ids[k]
        rows, med_h = build_cells([words[i] for i in ids])
        out = []
        for row in rows:
            for c in row:
                if not (px + pw * 0.8 <= c.x0 <= px + pw + 12 * med_h and py <= c.y0 <= py + ph):
                    continue
                toks = [t for t in re.split(r"[\s,]+", c.text) if t]
                if not toks or not _name_like(" ".join(toks)):
                    continue
                if all(norm(t) in DOC_WORDS for t in toks) or self.is_label_cell(c):
                    continue
                out.append(Entity("PERSON_NAME", c.text, [ids[i] for i in c.idx], 0.7, "universal:portrait-column"))
                break                                     # the first cell of the row next to the photo
        return out

    @staticmethod
    def _address_blocks(rows, words, med_h: float) -> list[Entity]:
        """Unlabelled postal blocks: a 'City, ST 12345' / '12345 City' line plus the street line above it."""
        city_zip = re.compile(r"[^\W\d_]{2,}.*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|(?:^|,\s*)\d{4,6}\s+[A-ZÀ-Ž][^\W\d_]{2,}|"
                              r"\b[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b")
        street = re.compile(r"^\d{1,5}[A-Za-z]?\s+[A-Z][^\W\d_]+.*|.*\b(?:street|st\.|avenue|ave\.?|road|rd\.|lane|"
                            r"drive|blvd|boulevard|suite|apt|rue|stra[sß]e|str\.|strada|calea|vista|track|way|court)\b",
                            re.I)
        out: list[Entity] = []
        for r, row in enumerate(rows):
            for c in row:
                if not city_zip.search(c.text) or len(c.text) > 80 or ":" in c.text:
                    continue
                out.append(Entity("ADDRESS", c.text, list(c.idx), 0.7, "universal:block"))
                if r > 0:
                    above = [a for a in rows[r - 1] if a.x0 < c.x1 and a.x1 > c.x0 and c.y0 - a.y1 < 1.5 * med_h]
                    if above and street.match(above[0].text):
                        out.append(Entity("ADDRESS", above[0].text, list(above[0].idx), 0.7, "universal:block"))
        return out
