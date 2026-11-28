"""Hybrid PII detector for Romanian documents.

Deterministic layer (regex + checksums + label context) runs always; spaCy NER and a
LOCAL small language model (Ollama) are optional add-ons. Every entity is mapped back
to OCR word indices so it can be redacted.
"""
from __future__ import annotations

import json
import os
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

# --- OCR-tolerant label words -------------------------------------------------------------------------------
# OCR misreads labels in predictable ways: m -> rn/nn ("Prenurne"), ri -> n ("Sena"), i/l -> 1/|, o -> 0,
# dropped letters ("sera"). fz() turns a plain label word into a regex that accepts those variants. It only
# changes what we MATCH; the OCR text and its char-offset -> word mapping are never rewritten.
_FZ_CHARS = {"m": "(?:m|rn|nn|n)", "i": "[il1|!]?", "l": "[l1i|I]", "o": "[o0]", "a": "[aă@]", "e": "[eéc]",
             "s": "[sșş5]", "t": "[tțţ]", "u": "[uv]", "n": "(?:n|ri)"}


def fz(word: str) -> str:
    """Regex for `word` that tolerates common OCR errors (the digraph 'ri' is handled first)."""
    out, i = [], 0
    w = word.lower()
    while i < len(w):
        if w.startswith("ri", i):
            out.append("(?:ri|rl|r1|n|r)")
            i += 2
            continue
        c = w[i]
        out.append(_FZ_CHARS.get(c, re.escape(c)) if c.isalpha() else re.escape(c))
        i += 1
    return "".join(out)


FZ_NUME, FZ_PRENUME = fz("nume"), fz("prenume")
CNP_LABEL = r"\bC\.?\s?N\.?\s?P\b\.?"            # CNP, C.N.P., C N P

NAME_LABELS = [
    FZ_NUME + r"\s*(?:si|și|/)?\s*" + FZ_PRENUME + r"(?:le)?", fz("numele") + r"\s+si\s+" + fz("prenumele"),
    FZ_NUME + r"(?:le)?", FZ_PRENUME + r"(?:le)?",
    r"subsemnat(?:ul|a)", r"titular(?:ul)?", r"client(?:ul)?", r"beneficiar(?:ul)?", r"cumparator(?:ul)?",
    r"vanzator(?:ul)?", r"locatar(?:ul)?", r"proprietar(?:ul)?", r"solicitant(?:ul)?", r"angajat(?:ul)?",
    r"domnul", r"doamna", r"dl\.?", r"dna\.?", r"last\s*name", r"first\s*name", r"nom", r"prenom",
    r"semnatura", r"reprezentant(?:ul)?", r"intocmit", r"student(?:ul|a)?",
]
ADDRESS_LABELS = [fz("domiciliu") + r"(?:l)?", r"domiciliat(?:a)?\s+(?:in|în)", fz("adresa") + r"(?:\s+de\s+\w+)?",
                  r"adresse", r"address", r"sediul", r"resedinta", r"locul\s+consumului", r"loc\s+consum"]
ADDRESS_START = r"(?:str\.|strada|bd\.|b-dul|bulevardul|calea|aleea|sos\.|soseaua|piata|splaiul|intrarea)"
ADDRESS_CONT = re.compile(r"\b(nr|bl|sc|et|ap|jud|judet|mun|municipiul|oras|or|sat|com|comuna|sector|sec|cod)\b\.?", re.I)
# localities: county / town markers and the largest cities (prefixes, so OCR'd "Bucur" still counts)
LOCALITY_CUE = re.compile(r"^(?:mun|jud|judet\w*|sec|sector|com|comuna|oras|or|sat|municipiul)\.?\d*$|"
                          r"^(?:bucur|cluj|iasi|timis|const|craiov|brasov|galat|ploies|orade|brail|arad|pites|sibiu|"
                          r"bacau|targu|baia|buzau|botos|satu|ramnic|suceav|piatr|drobet|focsan|tulce|resit|slatin|"
                          r"calara|giurg|deva|hunedo|bistri|alba|zalau|sfant|vaslu|slobo|alexan|tarnov|medias)", re.I)
STOP_WORDS = re.compile(r"\b(cnp|domiciliat|domiciliu|nascut|nascuta|seria|sera|sena|legitimat|posesor|cu|in|în|"
                        r"data|tel|telefon\w*|e-?mails?|iban|adres\w*|identificat|act|furnizor\w*|total)\b", re.I)
NAME_TOKEN = re.compile(r"^[A-ZĂÂÎȘŞȚŢ][A-Za-zĂÂÎȘŞȚŢăâîșşțţ\-]{1,}$")
LABEL_WORD = re.compile(r"^(nume|prenume|nom|prenom|last|first|name|si|și|/)$", re.I)
LABEL_ONLY_REST = re.compile(r"[\s/:.\-]*(?:(?:nume|prenume|nom|prenom|last|first|name|si|adresse|address|"
                             r"adresa|domiciliu)[\s/:.\-]*)*", re.I)
ADDR_EXCLUDE = re.compile(r"\b(CNP|seria|emis[aă]?|valabil\w*|telefon|tel|e-?mail|IBAN)\b", re.I)

# --- Romanian ID card (CI / buletin) fields -----------------------------------------------------------------
ID_CARD_CONTEXT = re.compile(r"CARTE\s*DE\s*" + fz("identitate") + r"|IDENTITY\s+CARD|CARTE\s+D.?IDENTITE|"
                             + fz("buletin") + r"|[I1l]D[R][O0]U", re.I)
ISSUER_LABEL = re.compile(r"(?:" + fz("emis") + r"[aă]?|eliberat[aă]?)\s+de|d[eé]livr[eé]e?\s+par|[il1|]ssued\s+by|"
                          r"autoritate(?:a)?\s+emitent[aă]", re.I)
# words that belong to the multilingual issuer label itself, never to the value
ISSUER_LABEL_WORDS = re.compile(r"^[\s/:.\-]*(?:emis[aă]?|eliberat[aă]?|de|d[eé]livr[eé]e?|par|[il1|]ssued|by|"
                                r"autoritate(?:a)?|emitent[aă]?)[\s/:.\-]*$", re.I)
ISSUER_BARE = re.compile(r"\bD\.?E\.?P\.?A\.?B\.?D\b\.*|"           # Bucharest records office, usually alone
                         r"\b(?:S\.?P\.?C\.?L\.?E\.?P|S\.?P\.?C\.?E\.?P)\.?\s+[A-Z][\w.\-]*(?:\s+[A-Z0-9][\w.\-]*){0,3}|"
                         r"\bPoli[tț]ia\s+(?:mun\.?|municipiului|ora[sș]ului|jud\.?)?\s*[A-Z][\w\-]*(?:\s+[A-Z0-9][\w\-]*){0,2}", re.I)
VALIDITY_LABEL = re.compile(fz("valabil") + r"\w*|validit[eé]\w*|validity|data\s+(?:eliber[aă]rii|emiterii|expir[aă]rii)|"
                            r"date\s+of\s+(?:issue|expiry)|expir[aă]rii", re.I)
# dates printed with spaces on the electronic ID (2021+): "03 07 1983"
DATE_SPACED_RE = re.compile(r"(?<![\d.])(0?[1-9]|[12]\d|3[01])\s+(0?[1-9]|1[0-2])\s+((?:19|20)\d{2})(?!\d)")
# electronic ID document number: 2 letters + 7 digits ("SP1236343"); old CI: 2 letters + 6 digits
ID_DOC_NO = re.compile(r"(?<![A-Z0-9])([A-Z]{2})\s?(\d{7})(?!\d)")
DOC_NO_LABEL = re.compile(r"nr\.?\s*" + fz("document") + "|" + fz("document") + r"\s*n[o0]|num[aă]r\s+document", re.I)
SIGNATURE_LABEL = re.compile(fz("semnatura"), re.I)
# words of the signature label itself ("Semnatura titularului / Holder's signature"), OCR-tolerant
SIGNATURE_LABEL_WORDS = re.compile(r"^[\W_]*(?:semn[aă]tura|titularului|h\w{1,3}der\W?s|holder\W?s|signature|/)[\W_]*$", re.I)
# field-label words used to find where the label column starts on an ID card (portrait sits left of it)
ID_LABEL_ANCHOR = re.compile(r"^(?:" + "|".join(fz(w) for w in ("nume", "prenume", "semnatura", "domiciliu",
                                                                "cetatenie", "seria", "emisa", "valabilitate"))
                             + r"|sex|cnp|c\.n\.p\.?|nr\.?|data|loc)$", re.I)
BIRTH_PLACE_LABEL = re.compile(r"loc(?:ul)?\s+(?:de\s+)?" + fz("nast") + r"er(?:e|ii)|lieu\s+de\s+naissance|"
                               r"place\s+of\s+birth", re.I)
BIRTH_DATE_LABEL = re.compile(r"data\s+" + fz("nasterii") + r"|date\s+de\s+naissance|date\s+of\s+birth", re.I)
# MRZ line 2 of a Romanian ID even when OCR loses the '<' fillers: series+number, check digit, ROU, birth date
MRZ_LINE2 = re.compile(r"[A-Z$5]{2}\d{6,7}\W?\d?[R][O0]U\d{6,8}[MF<]?\d{5,}", re.I)
MRZ_LINE1 = re.compile(r"^[I1l]D[R][O0]U[A-Z<«]{3,}", re.I)
DATE_RE = re.compile(r"(?<!\d)\d{1,2}[./\-]\d{1,2}[./\-](?:\d{4}|\d{2})(?!\d)")
DATE_RANGE_RE = re.compile(r"(?<!\d)\d{1,2}[./]\d{1,2}[./]\d{2,4}\s*[-–]\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}(?!\d)")

# --- Unlabeled full names (no "Nume:" prefix) --------------------------------------------------------------
# Common Romanian given names (diacritics stripped, upper case) and frequent surnames. A run of capitalised
# tokens counts as a name only if it contains a known given name AND every other token is a known name or has
# a typical surname ending — this keeps "Str. Florilor", "SC Energie Verde SRL" etc. out.
RO_GIVEN_NAMES = set("""
ADRIAN ADRIANA ALEXANDRA ALEXANDRU ALEX ALIN ALINA ANA ANCA ANDREEA ANDREI ANGELA ANTON ANTONIA ANTONIU AURA AUREL
AURELIA AURELIAN BIANCA BOGDAN BOGDANA CAMELIA CARMEN CAROL CATALIN CATALINA CEZAR CLAUDIA CLAUDIU CONSTANTIN
CONSTANTA CORINA CORNEL CORNELIA COSMIN COSMINA CRISTIAN CRISTINA DAN DANA DANIEL DANIELA DARIUS DAVID DELIA DENISA
DIANA DOINA DORIN DORINA DRAGOS DUMITRU ECATERINA EDUARD ELENA ELISABETA EMANUEL EMIL EMILIA EUGEN EUGENIA FELICIA
FLAVIA FLORENTIN FLORENTINA FLORIAN FLORIN FLORINA GABRIEL GABRIELA GEORGE GEORGETA GEORGIANA GHEORGHE GRIGORE HORIA
IACOB IOANA ION IONEL IONELA IONUT IOSIF IRINA ISABELA IULIA IULIAN IULIANA LARISA LAURA LAURENTIU LAVINIA LIDIA
LILIANA LUCIA LUCIAN LUCIANA LUMINITA MADALINA MARCEL MARCELA MARIA MARIAN MARIANA MARIUS MATEI MELANIA MIHAELA MIHAI
MIHAIL MIHNEA MIRCEA MIRELA MIRUNA MONICA NARCISA NATALIA NECULAI NICOLAE NICOLETA NICHIFOR NICU NICUSOR OANA OCTAVIAN
OTILIA OVIDIU PAUL PAULA PETRE PETRU PETRONELA RADU RALUCA RAMONA REBECA RODICA ROXANA RUXANDRA SABINA SANDA SANDU
SEBASTIAN SERBAN SERGIU SILVIA SILVIU SIMONA SORIN SORINA STEFAN STEFANIA SVETLANA TEODOR TEODORA TIBERIU TRAIAN
TUDOR TUDORA VALENTIN VALENTINA VALERIA VALERIU VASILE VERONICA VICTOR VICTORIA VIOLETA VIORICA VIOREL VIRGIL
IOAN VLAD ZOE ZOITA CRISTI DECEBAL GELU HORATIU LIVIU NELU OLIMPIA PAVEL REMUS SEVER TOMA VALERICA ANDREIA DARIA
""".split())
RO_SURNAMES = set("""
POPA POP RADU DUMITRU STAN STOICA GHEORGHE MATEI CIOBANU RUSU DINU MUNTEANU LAZAR MARIN TUDOR FLOREA BALAN SAVA
TOMA NISTOR DRAGOMIR NEAGU ENACHE BARBU MOLDOVAN COSTEA SERBAN CRISTEA MOCANU MIHAI ROTARU UNGUREANU CONSTANTIN
VASILE NICOLAE ANDREI ILIE COMAN MANOLE OPREA PAVEL TANASE BUCUR DOBRE ZAHARIA ANGHEL NEACSU SANDU VLAD PETRE
CHIRILA BRATU CIOCAN DAN DOGARU GAVRILA LUNGU MAZILU NEGRU OANCEA PREDA SIMION SURDU TOADER URSU VOICU ZAMFIR
""".split())
SURNAME_SUFFIX = re.compile(r"(?:ESCU|EANU|ANU|ASU|IUC|CIUC|ACHE|OVICI|EVICI|ICI|ARIU|OIU|EA|OAIE|UTA|ITA|AN)$")
NAME_RUN_BREAK = re.compile(r"^(?:STR|STRADA|BD|B-DUL|BULEVARDUL|CALEA|ALEEA|SOS|SOSEAUA|PIATA|SPLAIUL|INTRAREA|MUN|"
                            r"MUNICIPIUL|JUD|JUDETUL|COM|COMUNA|SAT|ORAS|SC|SRL|SA|BANCA|SPCLEP|SPCEP|POLITIA|SCOALA|"
                            r"LICEUL|COLEGIUL|UNIVERSITATEA|SPITALUL|BISERICA|PARCUL|CARTIERUL|SECTOR|SF|SFANTUL)$")

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+(?:\.\s[A-Za-z0-9._%+\-]+)?@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # OCR often reads '@' as 'Q', '©' or '(a)'; require a common TLD to stay precise
    "EMAIL_OCR": re.compile(r"[A-Za-z0-9._%+\-]{2,}(?:Q|©|®|\(a\))[A-Za-z0-9\-]+\.(?:com|ro|net|org|eu|info)\b"),
    "IBAN": re.compile(r"\bRO\s?\d{2}\s?[A-Z]{4}(?:\s?[A-Z0-9]{4}){4}\b"),
    "PHONE": re.compile(r"(?<![\d])(?:\+40|0040|0)[\s.\-]?(?:7\d{2}|[23]\d{1,2})(?:[\s.\-]?\d{2,3}){2,3}(?![\d])"),
    "CNP": re.compile(r"(?<!\d)[1-9]\d{12}(?!\d)"),
    # "Seria SV nr 123456", OCR-tolerant: "Sena SV nr", "sera SS, na 099993"
    "ID_SERIES_LABELED": re.compile(r"\b" + fz("seria") + r"\s*[:.]?\s*([A-Z]{2})\s*[,;.]?\s*(?:N[RA]\.?|NUMAR(?:UL)?|"
                                    r"N[O0]\.?)?\s*[:.]?\s*(\d{6,7})\b", re.I),
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


def _one_edit(a: str, b: str) -> bool:
    """True if a and b differ by at most one substitution / insertion / deletion (OCR slips in names)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:k] + b[k + 1:] == a for k in range(len(b)))


def _strip_junk(text: str) -> str:
    """Drop OCR junk in front of a label: '| ', '>>: @) ', 'N ', ', 4 ' (non-letters and stray single letters)."""
    return re.sub(r"^(?:[\W\d_]+|\b[A-Za-z]\b)+", "", text).lstrip()


def _fix_capital_i(tok: str) -> str:
    """OCR often reads a leading capital I as l ("lonescu" -> "Ionescu")."""
    return "I" + tok[1:] if re.match(r"^l[a-zăâîșşțţ]{2,}$", tok) else tok


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
        # label punctuation as OCR reads it: "Adresa:", "Adresa;", "Nume ," (a blurred colon often comes back as ;)
        self._name_label_re = re.compile(r"^\s*(?:" + "|".join(NAME_LABELS) + r")\b\s*[:.;,\-]?\s*", re.I)
        self._addr_label_re = re.compile(r"(?:^|\b)(?:" + "|".join(ADDRESS_LABELS) + r")\b\s*[:.;,\-]?\s*", re.I)
        self._addr_start_re = re.compile(r"\b" + ADDRESS_START, re.I)

        # layout-agnostic multilingual labels + rules learned from manual blur boxes (shared learned_rules.json)
        from server.universal_detector import UniversalDetector
        self.universal = UniversalDetector()           # learned rules: the shared store, resolved per detection

    # ---------------------------------------------------------------- public
    def detect(self, words: list[Word], use_local_slm: bool = False,
               slm_base_url: str | None = None, slm_model: str | None = None) -> list[Entity]:
        return self.detect_with_regions(words, use_local_slm, slm_base_url, slm_model)[0]

    def detect_with_regions(self, words: list[Word], use_local_slm: bool = False, slm_base_url: str | None = None,
                            slm_model: str | None = None) -> tuple[list[Entity], list[tuple[int, int, int, int, int]]]:
        """Entities + extra regions (value blocks of labels, incl. lines OCR could not read) to redact."""
        lines = build_lines(words)
        uni_entities, regions = self.universal.detect(words)
        entities: list[Entity] = list(uni_entities)
        entities += self._regex_entities(lines, words)
        entities += self._name_entities(lines, words)
        entities += self._address_entities(lines, words)
        mrz = self._mrz_entities(lines)
        entities += mrz
        entities += self._mrz_derived_entities(mrz, lines, words)
        entities += self._id_card_entities(lines, words)
        entities += self._unlabeled_name_entities(lines, words)
        entities += self._locality_entities(lines, words, entities)
        if self._nlp is not None:
            entities += self._spacy_entities(lines)
        if use_local_slm and slm_base_url and slm_model:
            entities += self._slm_entities(lines, words, slm_base_url, slm_model)
        entities = self._propagate_names(entities, lines, words)
        return self._dedupe(entities), regions

    def _is_field_label(self, norm: str) -> bool:
        """True if a line is another form / ID-card field label rather than a value.

        Used before taking "the value on the next line": after redaction the value is gone, and the following
        label (e.g. "Loc nastere/Lieu de naissance") must not be mistaken for it."""
        s = _strip_junk(norm)
        return bool(self._name_label_re.match(s) or self._addr_label_re.match(s) or VALIDITY_LABEL.match(s)
                    or ISSUER_LABEL.search(s) or BIRTH_PLACE_LABEL.match(s) or BIRTH_DATE_LABEL.match(s)
                    or DOC_NO_LABEL.match(s) or SIGNATURE_LABEL.match(s)
                    or re.match(r"^(?:cet[aă][tț]enie|sex|data\b|cnp\b)", s, re.I)
                    or re.match(r"^[A-Za-zĂÂÎȘŞȚŢăâîșşțţ]+(?:\s+[a-zăâîșşțţ]+)?\s*:", s)   # "Adresa:", "Telefon mobil:"
                    or norm.count("<") >= 2 or re.search(r"[A-Za-z]\s*/\s*[A-Za-z]", norm))

    # ---------------------------------------------------------------- regex
    def _regex_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        out: list[Entity] = []
        doc_norm = " ".join(l.norm for l in lines).upper()
        id_context = any(k in doc_norm for k in ("CARTE DE IDENTITATE", "CARTE IDENTITATE", "BULETIN", "SERIA",
                                                  "IDENTITY CARD", " C.I.", "ACT DE IDENTITATE"))
        for li, line in enumerate(lines):
            t = line.norm
            prev = lines[li - 1].norm if li else ""
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
                elif (id_context or re.search(CNP_LABEL, t + " " + prev, re.I)) and t.count("<") < 2 \
                        and not MRZ_LINE2.search(t.replace(" ", "")):
                    # (MRZ lines contain 13-digit runs too; they are redacted as MRZ)
                    # wrong checksum (OCR slip, or a specimen card): still redact, but never use it for linking
                    out.append(Entity("CNP", m.group(0), _words_in_span(line, *m.span()), 0.6, "context"))
            for m in PATTERNS["PHONE"].finditer(t):
                digits = re.sub(r"\D", "", m.group(0))
                if 9 <= len(digits) <= 13 and not (len(digits) == 13 and cnp_is_valid(digits)):
                    out.append(Entity("PHONE", m.group(0), _words_in_span(line, *m.span()), 0.85))
            for m in PATTERNS["ID_SERIES_LABELED"].finditer(t):
                out.append(Entity("ID_SERIES", (m.group(1) + m.group(2)).upper(),
                                  _words_in_span(line, m.start(1), m.end(2)), 0.97, "context"))
            if id_context:
                doc_label = bool(DOC_NO_LABEL.search(t + " " + prev))
                for m in PATTERNS["ID_SERIES_BARE"].finditer(t):
                    if m.group(1) in RO_ID_SERIES or doc_label:
                        out.append(Entity("ID_SERIES", m.group(1) + m.group(2),
                                          _words_in_span(line, *m.span()), 0.8))
                for m in ID_DOC_NO.finditer(t.upper()):         # electronic ID: 2 letters + 7 digits
                    out.append(Entity("ID_SERIES", m.group(1) + m.group(2),
                                      _words_in_span(line, *m.span()), 0.85 if doc_label else 0.75))
            for m in PATTERNS["BIRTH_DATE"].finditer(t):
                out.append(Entity("BIRTH_DATE", m.group(1), _words_in_span(line, *m.span(1)), 0.9, "context"))
            # Lenient CNP: 13 noisy chars right after a CNP label (OCR confusions O->0, l->1 ...)
            for m in re.finditer(CNP_LABEL + r"\s*[:.]?\s*(\S{12,15})", t, re.I):
                raw = m.group(1).strip(".,;:")
                fixed = raw.translate(OCR_DIGIT_FIX)
                fixed = re.sub(r"\D", "", fixed)
                if len(fixed) == 13 and not any(e.type == "CNP" and e.value == fixed for e in out):
                    conf = 0.95 if cnp_is_valid(fixed) else 0.6
                    out.append(Entity("CNP", fixed, _words_in_span(line, m.start(1), m.start(1) + len(m.group(1))),
                                      conf, "context"))
        return out

    # ---------------------------------------------------------------- names
    def _take_name_tokens(self, line: Line, start_char: int, words: list[Word], value_line: bool = False) -> list[int]:
        """Name tokens right after `start_char`. value_line=True: the whole line is a value under a label
        (ID-card layout) — leading OCR junk ('>>:', '@)', '|') is skipped and tokens need 3+ letters, so blur
        texture read as 'Co' after redaction is not taken for a name."""
        picked: list[int] = []
        for k, (s, e) in enumerate(line.spans):
            if s < start_char:
                continue
            tok = words[line.word_idx[k]].text.strip(",.;:")
            if value_line and not picked and not re.search(r"[A-Za-zĂÂÎȘŞȚŢăâîșşțţ]{3,}", tok):
                continue                                   # junk before the value
            if value_line and not re.fullmatch(r"[A-Za-zĂÂÎȘŞȚŢăâîșşțţ\-]{3,}", tok):
                break
            if STOP_WORDS.fullmatch(strip_diacritics(tok)) or (picked and words[line.word_idx[k]].text.startswith("(")):
                break
            if LABEL_WORD.match(strip_diacritics(tok)) and not picked:
                continue
            # OCR reads a capital I as l ("lonescu", "lulia"): accept it as a name token right after a label too
            if NAME_TOKEN.match(tok) or re.match(r"^l[a-zăâîșşțţ]{2,}$", tok):
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
            for m in re.finditer(r"(?:^|(?<=[\s/]))(?:" + "|".join(NAME_LABELS) + r")\b\s*[:.;,\-]?\s*",
                                 line.norm, re.I):
                label = m.group(0).strip().lower()
                if label.startswith("semnatura") and not line.norm[m.end():].strip():
                    continue
                # bilingual ID-card label ("Nume/Nom/Last name"): the value is never to its right — what OCR puts
                # there is noise from the background guilloche ("PF Pact") — so only look below it
                bilingual = "/" in line.norm[max(0, m.start() - 1):m.end() + 1]
                idx = [] if bilingual else self._take_name_tokens(line, m.end(), words)
                # label alone on its line (ID-card layout) -> value is on the next line. OCR junk such as '| '
                # may precede the label, and the rest of the line holds the translation ('/Surname', '/Given names').
                # only look at the next line when the label is effectively alone ("Subsemnata [blurred] la data"
                # has a value slot on its own line — after redaction the next line is NOT its value)
                rest = line.norm[m.end():]
                alone = not _strip_junk(rest) or bool(LABEL_ONLY_REST.fullmatch(rest))
                label_starts_line = (not _strip_junk(line.norm[:m.start()]) and alone) or bilingual
                if not idx and label_starts_line and li + 1 < len(lines):
                    nxt = lines[li + 1]
                    if not self._is_field_label(nxt.norm):      # e.g. "Loc nastere/..." once the name is redacted
                        idx = self._take_name_tokens(nxt, 0, words, value_line=True)
                        # a second given name may wrap onto its own line ("CORINA" / "IOANA")
                        if idx and li + 2 < len(lines) and not self._is_field_label(lines[li + 2].norm):
                            more = self._take_name_tokens(lines[li + 2], 0, words, value_line=True)
                            rest = [i for i in lines[li + 2].word_idx if i not in more
                                    and re.search(r"[A-Za-z]{2,}", words[i].text)]
                            if more and not rest and all(words[i].text.strip(",.;:").isupper() for i in more):
                                idx = idx + more
                if not idx and bilingual:                    # OCR ordered the lines differently: go by position
                    idx = self._value_below(line, m, lines, words)
                if idx:
                    value = " ".join(_fix_capital_i(words[i].text.strip(",.;:")) for i in idx)
                    is_given = re.match(r"^(?:" + FZ_PRENUME + r"(?:le)?|prenom|first\s*name)\b", label, re.I)
                    is_sur = re.match(r"^(?:" + FZ_NUME + r"(?:le)?|nom|last\s*name)\b", label, re.I)
                    has_both = is_sur and re.search(FZ_PRENUME, label[2:], re.I)      # "Nume si prenume"
                    kind = "given" if is_given else "full" if has_both else "surname" if is_sur else "full"
                    out.append(Entity("PERSON_NAME", value, idx, 0.9, f"context:{kind}"))
        return out

    def _value_below(self, line: Line, m: re.Match, lines: list[Line], words: list[Word]) -> list[int]:
        """Name tokens located geometrically under a label: starting within ~3 label heights below it and
        horizontally overlapping the label (tolerates skew and lines that OCR orders differently)."""
        lab = _words_in_span(line, m.start(), m.end())
        if not lab:
            return []
        lx0 = min(words[i].x for i in lab)
        lh = max(words[i].h for i in lab)
        ly1 = max(words[i].y + words[i].h for i in lab)
        cands = [ln for ln in lines if ln is not line and not self._is_field_label(ln.norm)
                 and ly1 - 0.3 * lh <= min(words[i].y for i in ln.word_idx) <= ly1 + 3 * lh]
        for ln in sorted(cands, key=lambda ln: min(words[i].y for i in ln.word_idx)):
            near = [i for i in ln.word_idx if words[i].x + words[i].w >= lx0 - 2 * lh]
            if not near:
                continue
            start = ln.spans[ln.word_idx.index(near[0])][0]
            idx = self._take_name_tokens(ln, start, words, value_line=True)
            if idx:
                return idx
        return []

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
                nxt_norm = lines[li + 1].norm if li + 1 < len(lines) else ""
                looks_like_address = bool(ADDRESS_CONT.search(nxt_norm) or self._addr_start_re.search(nxt_norm)
                                          or re.search(r"\d", nxt_norm))
                if nxt_norm and looks_like_address and not ADDR_EXCLUDE.search(nxt_norm) \
                        and not self._name_label_re.match(nxt_norm):
                    idx = list(lines[li + 1].word_idx)
                    if li + 2 < len(lines) and ADDRESS_CONT.search(lines[li + 2].norm) \
                            and not self._name_label_re.match(lines[li + 2].norm) \
                            and not ADDR_EXCLUDE.search(lines[li + 2].norm):
                        idx += lines[li + 2].word_idx
                    value = " ".join(words[i].text for i in idx)
                    # a real address has a word of 3+ letters and some length (not 'Co' OCR'd off a blurred box)
                    if len(value) >= 6 and re.search(r"[A-Za-zĂÂÎȘŞȚŢăâîșşțţ]{3,}", value):
                        out.append(Entity("ADDRESS", value, idx, 0.85, "context"))
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
                # what follows the label must be real content — not blur texture read as '[' or 'Co'
                after = rest[: end - start] if end > start else value
                if len(value) >= 6 and (re.search(r"[A-Za-zĂÂÎȘŞȚŢăâîșşțţ]{3,}|\d", after)
                                        or len(idx) > len(_words_in_span(line, start, end))):
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
            elif len(compact) >= 20 and (MRZ_LINE1.search(compact) or MRZ_LINE2.search(compact)):
                out.append(Entity("MRZ", compact, list(line.word_idx), 0.85, "pattern"))   # fillers lost by OCR
        return out

    def _mrz_derived_entities(self, mrz: list[Entity], lines: list[Line], words: list[Word]) -> list[Entity]:
        """The MRZ spells out surname, given names and document number in a fixed format — use them to find
        the same values printed (and possibly badly OCR'd) elsewhere on the card."""
        names: set[str] = set()
        numbers: set[str] = set()
        for e in mrz:
            v = e.value.upper().replace("«", "<")
            m1 = MRZ_LINE1.search(v)
            if m1:
                body = re.sub(r"^[I1L]D[R][O0]U", "", v)
                names |= {t for t in re.split(r"<+", body) if len(t) >= 3 and t.isalpha()}
            # document number = the start of MRZ line 2 (anchored: the birth date / CNP digits come later)
            m2 = re.match(r"[^A-Z0-9$]*[A-Z0-9$]{2}(\d{6,7})(?=\W?\d?[R][O0]U)", v)
            if m2:
                numbers.add(m2.group(1))
        if not names and not numbers:
            return []
        mrz_words = {i for e in mrz for i in e.word_indices}
        out: list[Entity] = []
        for line in lines:
            for i in line.word_idx:
                if i in mrz_words:
                    continue
                raw = words[i].text.strip(",.;:|()[]{}")
                tok = strip_diacritics(_fix_capital_i(raw)).upper()
                if len(tok) >= 3 and any(tok == n or (len(n) >= 5 and _one_edit(tok, n)) for n in names):
                    out.append(Entity("PERSON_NAME", raw, [i], 0.85, "mrz"))
                digits = re.sub(r"\D", "", raw)
                if len(digits) >= 6 and any(n in digits for n in numbers):
                    out.append(Entity("ID_SERIES", raw, [i], 0.85, "mrz"))
        return out

    # ---------------------------------------------------------------- Romanian ID card fields
    def _id_card_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        """Issuing authority, validity / issue dates, place of birth and next-line birth dates.

        Issuer and validity are only treated as PII on identity documents (an invoice's "Data emiterii" is not)."""
        out: list[Entity] = []
        doc_text = " ".join(l.norm for l in lines)
        is_id_card = bool(ID_CARD_CONTEXT.search(doc_text))

        def value_after(li: int, start: int, max_next: int = 1) -> list[int]:
            """Words after `start` on line li; if the label is alone on its line, the next line(s)."""
            idx = _words_in_span(lines[li], start, len(lines[li].norm) + 1)
            if idx:
                return idx
            for k in range(1, max_next + 1):
                if li + k >= len(lines):
                    break
                nxt = lines[li + k].norm
                # the next line must be a value, not another field (after redaction the value itself is gone,
                # and the following label must not be mistaken for it)
                if self._is_field_label(nxt):
                    break
                return list(lines[li + k].word_idx)
            return []

        def text_of(idx: list[int]) -> str:
            return " ".join(words[i].text for i in idx)

        for li, line in enumerate(lines):
            t = line.norm
            if is_id_card:
                # issuing authority: value after the (last) "Emisa de / Delivree par / Issued by" label
                labels = list(ISSUER_LABEL.finditer(t))
                if labels:
                    start = labels[-1].end()
                    rest = t[start:]
                    start += len(rest) - len(rest.lstrip(" /:.-"))
                    stop = VALIDITY_LABEL.search(t, start)
                    idx = _words_in_span(line, start, stop.start() if stop else len(t) + 1) or value_after(li, len(t) + 1)
                    # drop leftover label words ("par/Issued by" when OCR split the label differently)
                    idx = [i for i in idx if not ISSUER_LABEL_WORDS.match(strip_diacritics(words[i].text))
                           and not all(ISSUER_LABEL_WORDS.match(p) for p in re.split(r"/", strip_diacritics(words[i].text)) if p)]
                    if idx and re.search(r"[A-Za-z]{3,}", text_of(idx)):
                        out.append(Entity("ISSUER", text_of(idx), idx, 0.9, "context"))
                for m in ISSUER_BARE.finditer(t):
                    out.append(Entity("ISSUER", m.group(0), _words_in_span(line, *m.span()), 0.85))
                # validity / issue / expiry dates
                vm = VALIDITY_LABEL.search(t)
                if vm:
                    dates = list(DATE_RANGE_RE.finditer(t, vm.end())) or list(DATE_RE.finditer(t, vm.end()))
                    if dates:
                        for d in dates:
                            out.append(Entity("ID_DATE", d.group(0), _words_in_span(line, *d.span()), 0.9, "context"))
                    elif li + 1 < len(lines):
                        nxt = lines[li + 1]
                        for d in (list(DATE_RANGE_RE.finditer(nxt.norm)) or list(DATE_RE.finditer(nxt.norm))):
                            out.append(Entity("ID_DATE", d.group(0), _words_in_span(nxt, *d.span()), 0.85, "context"))
                for d in DATE_RANGE_RE.finditer(t):                 # unlabeled "20.05.2023-20.05.2033"
                    out.append(Entity("ID_DATE", d.group(0), _words_in_span(line, *d.span()), 0.85))
                # every date on an ID card is personal (birth, issue, expiry). The electronic ID prints them with
                # spaces ("03 07 1983") under a multi-column label line; pick the type of the nearest label above.
                for d in list(DATE_SPACED_RE.finditer(t)) + list(DATE_RE.finditer(t)):
                    idx = _words_in_span(line, *d.span())
                    if not idx:
                        continue
                    etype = self._date_type_from_labels(lines[li - 1] if li else None, words, words[idx[0]].x)
                    out.append(Entity(etype, d.group(0), idx, 0.85, "context"))
                # card access number (CAN, 6 digits) printed right of "Semnatura titularului / Holder's signature"
                sm = SIGNATURE_LABEL.search(t)
                if sm:
                    after = [i for i in _words_in_span(line, sm.start(), len(t) + 1)
                             if not SIGNATURE_LABEL_WORDS.match(strip_diacritics(words[i].text))]
                    if sum(ch.isdigit() for i in after for ch in words[i].text) >= 3:
                        out.append(Entity("CAN", text_of(after), after, 0.8, "context"))
            # place of birth (any document)
            places = list(BIRTH_PLACE_LABEL.finditer(t))
            if places:
                start = places[-1].end()                # "Loc nastere/Lieu de naissance/Place of birth <value>"
                rest = t[start:]
                start += len(rest) - len(rest.lstrip(" /:.-"))
                idx = value_after(li, start)
                if idx and re.search(r"[A-Za-z]{3,}", text_of(idx)):
                    out.append(Entity("BIRTH_PLACE", text_of(idx), idx, 0.85, "context"))
            # birth date whose value sits on the next line (new CI layout)
            bm = BIRTH_DATE_LABEL.search(t)
            if bm and not DATE_RE.search(t, bm.end()) and li + 1 < len(lines):
                nxt = lines[li + 1]
                d = DATE_RE.search(nxt.norm)
                if d:
                    out.append(Entity("BIRTH_DATE", d.group(0), _words_in_span(nxt, *d.span()), 0.85, "context"))
        return out

    def _locality_entities(self, lines: list[Line], words: list[Word], found: list[Entity]) -> list[Entity]:
        """On identity documents, a run starting at a locality marker ("Mun.", "Jud.", "Sec.4") or a large
        city ("Bucuresti", even OCR'd as "Bucur") is a domicile / birth-place fragment — wherever it sits.
        Only adds what no other rule already covered."""
        if not ID_CARD_CONTEXT.search(" ".join(l.norm for l in lines)):
            return []
        covered = {i for e in found for i in e.word_indices}
        label_stop = re.compile(r"^(?:of|birth|lieu|naissance|place|sex|sexe|adresse|address|issued|by|par|"
                                r"nationality|nationalite)$", re.I)
        out: list[Entity] = []
        for line in lines:
            if self._is_field_label(line.norm) or line.text.count("<") >= 2:
                continue
            toks = [strip_diacritics(words[i].text).strip(",.;:|()[]{}") for i in line.word_idx]
            start = next((k for k, t in enumerate(toks) if t and LOCALITY_CUE.match(t)), None)
            if start is None:
                continue
            if start > 0 and re.fullmatch(r"[A-Za-z]{2,4}\.?", toks[start - 1]):   # garbled "Mun." before a city
                start -= 1
            idx = []
            for k in range(start, len(toks)):
                if label_stop.match(toks[k]) or re.search(r"[A-Za-z]/[A-Za-z]", toks[k]):
                    break
                idx.append(line.word_idx[k])
            idx = [i for i in idx if i not in covered]
            if idx and re.search(r"[A-Za-z]{3,}", " ".join(words[i].text for i in idx)):
                out.append(Entity("ADDRESS", " ".join(words[i].text for i in idx), idx, 0.7, "locality"))
        return out

    @staticmethod
    def _date_type_from_labels(label_line: Line | None, words: list[Word], x: int) -> str:
        """BIRTH_DATE if the closest date label above (by x position) is a birth-date label, else ID_DATE."""
        if label_line is None:
            return "ID_DATE"
        best, best_dx = "ID_DATE", None
        for rx, etype in ((BIRTH_DATE_LABEL, "BIRTH_DATE"), (VALIDITY_LABEL, "ID_DATE")):
            for m in rx.finditer(label_line.norm):
                idx = _words_in_span(label_line, *m.span())
                if idx:
                    dx = abs(words[idx[0]].x - x)
                    if best_dx is None or dx < best_dx:
                        best, best_dx = etype, dx
        return best

    # ---------------------------------------------------------------- unlabeled names
    def _unlabeled_name_entities(self, lines: list[Line], words: list[Word]) -> list[Entity]:
        """Full names with no label in front ("Ionescu Maria Elena", "NICHIFOR ANTONIU")."""
        out: list[Entity] = []

        def norm_tok(raw: str) -> str:
            return strip_diacritics(_fix_capital_i(raw.strip(",.;:()"))).upper()

        def name_like(raw: str) -> bool:
            tok = raw.strip(",.;:()")
            return bool(NAME_TOKEN.match(tok) or re.match(r"^l[a-zăâîșşțţ]{2,}$", tok))

        for line in lines:
            idx = line.word_idx
            k = 0
            while k < len(idx):
                # collect a run of name-like tokens; a trailing comma/period ends the run after that token
                run: list[int] = []
                prev = norm_tok(words[idx[k - 1]].text) if k > 0 else ""
                j = k
                while j < len(idx) and name_like(words[idx[j]].text) and len(run) < 6:
                    run.append(idx[j])
                    j += 1
                    if words[idx[j - 1]].text.rstrip().endswith((",", ".", ";", ":")):
                        break
                k = j + 1 if not run else j
                if len(run) < 2 or NAME_RUN_BREAK.match(prev.rstrip(".")):
                    continue
                best = self._best_name_window([norm_tok(words[i].text) for i in run])
                if best:
                    a, b = best
                    sel = run[a:b]
                    out.append(Entity("PERSON_NAME", " ".join(_fix_capital_i(words[i].text.strip(",.;:")) for i in sel),
                                      sel, 0.75, "lexicon"))
        return out

    @staticmethod
    def _best_name_window(toks: list[str]) -> tuple[int, int] | None:
        """Longest window of 2-4 tokens that looks like a person name, or None."""
        def known(t: str) -> bool:
            return t in RO_GIVEN_NAMES or t in RO_SURNAMES or bool(SURNAME_SUFFIX.search(t)) and len(t) >= 4

        for size in (4, 3, 2):
            for a in range(0, len(toks) - size + 1):
                win = toks[a:a + size]
                if any(NAME_RUN_BREAK.match(t) for t in win):
                    continue
                parts = [p for t in win for p in t.split("-")]
                if not any(p in RO_GIVEN_NAMES for p in parts):
                    continue
                if all(known(p) for p in parts):
                    return a, a + size
        return None

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
            prompt = ("You find personal data in ANY document (ID card, passport, invoice, contract, medical note, "
                      "diploma, form; any language or layout; OCR text may contain errors). Return ONLY JSON: "
                      '{"entities": [{"type": "PERSON_NAME|ADDRESS|ID_NUMBER|DATE_OF_BIRTH|PHONE|EMAIL|IBAN", '
                      '"text": "<exact substring of the document>"}]}. Do not include company names, prices or '
                      "document titles.\n\n" + text)
            resp = client.chat.completions.create(model=model, temperature=0,
                                                  messages=[{"role": "user", "content": prompt}])
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content[content.find("{"): content.rfind("}") + 1])
        except Exception:
            return []
        type_map = {"PERSON_NAME": "PERSON_NAME", "ADDRESS": "ADDRESS", "ID_NUMBER": "ID_SERIES",
                    "DATE_OF_BIRTH": "BIRTH_DATE", "PHONE": "PHONE", "EMAIL": "EMAIL", "IBAN": "IBAN"}
        items = [(type_map.get(str(e.get("type", "")).upper()), e.get("text")) for e in data.get("entities", [])
                 if isinstance(e, dict)]
        items += [("PERSON_NAME", v) for v in data.get("names", []) or []]        # older answer format
        items += [("ADDRESS", v) for v in data.get("addresses", []) or []]
        out = []
        for etype, value in items:
            if not etype or not isinstance(value, str) or len(value.strip()) < 3:
                continue
            value = value.strip()
            for line in lines:
                pos = line.text.find(value)
                if pos < 0:                       # tolerate OCR / model whitespace differences
                    m = re.search(r"\s*".join(map(re.escape, value.split())), line.text)
                    pos, value_len = (m.start(), m.end() - m.start()) if m else (-1, 0)
                else:
                    value_len = len(value)
                if pos >= 0:
                    idx = _words_in_span(line, pos, pos + value_len)
                    if idx:
                        out.append(Entity(etype, value, idx, 0.75, "slm"))
        return out

    @classmethod
    def slm_available(cls, base_url: str, timeout: float = 1.0) -> bool:
        """A local OpenAI-compatible server (Ollama / LM Studio) answers on localhost. Never probes remote hosts."""
        if not base_url or not cls._is_local(base_url):
            return False
        import urllib.request
        root = base_url.rstrip("/")
        for url in (root + "/models", root.rsplit("/v1", 1)[0] + "/api/tags"):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:      # noqa: S310 (localhost only)
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

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


# ----------------------------------------------------------------------------
# Faces (OpenCV YuNet) — layout-independent portrait detection
# ----------------------------------------------------------------------------
def _yunet(img, threshold: float) -> list[tuple[float, float, float, float, float]]:
    """(x, y, w, h, score) of faces in a BGR image, using OpenCV's YuNet model (models/ folder)."""
    import cv2
    from server import config
    if not os.path.isfile(config.FACE_MODEL_PATH):
        return []
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    h, w = img.shape[:2]
    long_side = max(h, w)
    s = 1280 / long_side if long_side > 1280 else (640 / long_side if long_side < 640 else 1.0)  # small scans: upscale
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    det = cv2.FaceDetectorYN.create(config.FACE_MODEL_PATH, "", (small.shape[1], small.shape[0]), threshold, 0.3, 5000)
    _, faces = det.detect(small)
    if faces is None:
        return []
    return [(f[0] / s, f[1] / s, f[2] / s, f[3] / s, float(f[14])) for f in faces]


def _rotate_bound(img, angle: float):
    """Rotate without cropping; returns (image, inverse affine matrix back to the original frame)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rotated = cv2.warpAffine(img, m, (nw, nh), borderValue=(255, 255, 255))
    return rotated, cv2.invertAffineTransform(m), np


def _ghost_faces(gray, face: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """The faint, smaller 'ghost' copy of the portrait printed on many ID cards: template-match the detected
    face (contrast-normalised) at smaller scales elsewhere on the card. Accepted only with a clear winner."""
    import cv2
    import numpy as np
    x, y, w, h = face
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    tpl = g[max(0, y):y + h, max(0, x):x + w]
    if tpl.size == 0 or min(tpl.shape) < 12:
        return []
    H, W = g.shape
    best: tuple[float, float, tuple[int, int], tuple[int, int]] | None = None
    for s in np.linspace(0.3, 0.9, 13):
        t = cv2.resize(tpl, (max(8, int(w * s)), max(8, int(h * s))))
        if t.shape[0] >= H or t.shape[1] >= W:
            continue
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        r[max(0, y - t.shape[0]):y + h, max(0, x - t.shape[1]):x + w] = -1     # not the portrait itself
        _, top, _, loc = cv2.minMaxLoc(r)
        th, tw = t.shape
        r2 = r.copy()                                                            # runner-up elsewhere
        r2[max(0, loc[1] - th):loc[1] + th, max(0, loc[0] - tw):loc[0] + tw] = -1
        _, second, _, _ = cv2.minMaxLoc(r2)
        if best is None or top > best[0]:
            best = (top, second, loc, (tw, th))
    if best and best[0] >= 0.42 and best[0] - best[1] >= 0.06:
        (gx, gy), (gw, gh) = best[2], best[3]
        return [(gx, gy, gw, gh)]
    return []


def _face_boxes(image_path: str, page_no: int) -> list[tuple[int, int, int, int, int]]:
    """Faces anywhere on the page (any layout), padded to cover hair, ears and chin, plus a ghost photo.

    Tries the upright page first, then rotated copies (sideways / tilted photos of cards)."""
    try:
        import cv2
        from server import config
        img = cv2.imread(image_path)
        if img is None:
            return []
        H, W = img.shape[:2]
        found = [(x, y, w, h) for x, y, w, h, _ in _yunet(img, config.FACE_SCORE_THRESHOLD)]
        if not found:
            for angle in (90, -90, 180, 20, -20, 35, -35):
                rot, inv, np = _rotate_bound(img, angle)
                hits = _yunet(rot, config.FACE_SCORE_THRESHOLD)
                for x, y, w, h, _ in hits:
                    corners = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]]).T
                    pts = inv @ corners
                    x0, y0, x1, y1 = pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()
                    found.append((x0, y0, x1 - x0, y1 - y0))
                if found:
                    break
        found = [(int(x), int(y), int(w), int(h)) for x, y, w, h in found if w > 8 and h > 8]
        if found:
            biggest = max(found, key=lambda f: f[2] * f[3])
            found += _ghost_faces(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), biggest)
    except Exception:
        return []
    r = config.FACE_PADDING_RATIO
    out = []
    for x, y, w, h in found:                       # sides r, more room above (hair) than below (chin)
        x0, y0 = max(0, int(x - r * w)), max(0, int(y - 1.6 * r * h))
        x1, y1 = min(W, int(x + w + r * w)), min(H, int(y + h + 1.2 * r * h))
        out.append((page_no, x0, y0, x1 - x0, y1 - y0))
    return out


def portrait_boxes(words: list[Word], pages: list) -> list[tuple[int, int, int, int, int]]:
    """Boxes over faces / the holder's photo, on any document.

    1. Face detection (YuNet) on every page, whatever the layout, including rotated photos and ghost images.
    2. Only on identity documents where no face was found: the layout rule — the text-free block left of the
       field labels, between the card title and the MRZ.
    3. On identity documents: the handwritten signature area under the signature label."""
    lines = build_lines(words)
    is_id = bool(lines) and bool(ID_CARD_CONTEXT.search(" ".join(l.norm for l in lines)))
    from server.universal_detector import UniversalDetector
    foreign_id = not is_id and UniversalDetector(learned=None).id_context(words)
    boxes: list[tuple[int, int, int, int, int]] = []
    for page in pages:
        faces = _face_boxes(page.base_image_path, page.number)
        if len(faces) > 1:                          # ghost photos sit on the background, never over printed text
            main = max(faces, key=lambda f: f[3] * f[4])
            texty = [w for w in words if w.page == page.number and re.search(r"[^\W\d_]{3,}", w.text)]

            def over_text(f) -> bool:
                _, x, y, fw, fh = f
                hits = [w for w in texty if w.x < x + fw and w.x + w.w > x and w.y < y + fh and w.y + w.h > y]
                return len(hits) >= 2
            faces = [f for f in faces if f is main or not over_text(f)]
        boxes += faces
        plines = [l for l in lines if words[l.word_idx[0]].page == page.number]
        if not plines or not (is_id or foreign_id):
            continue
        sig = _signature_boxes(plines, words, page)
        if not sig:                                 # unlabelled cursive signature (foreign IDs, passports)
            ink = ink_signature_box(page, words, faces)
            sig = [ink] if ink else []
        boxes += sig
        if not is_id:
            continue
        if faces:
            continue

        def bottom(l: Line) -> int:
            return max(words[i].y + words[i].h for i in l.word_idx)

        def top(l: Line) -> int:
            return min(words[i].y for i in l.word_idx)

        title = next((l for l in plines if ID_CARD_CONTEXT.search(l.norm)), None)
        # measure the title from real words: OCR glues tall '|' junk (card borders, flag stripes) into the line
        title_words = [i for i in (title.word_idx if title else []) if re.search(r"[A-Za-z]{2,}", words[i].text)]
        band_top = max(words[i].y + words[i].h for i in title_words) if title_words else \
            (bottom(title) if title else min(top(l) for l in plines))
        mrz = [l for l in plines if l.text.count("<") + l.text.count("«") >= 2]
        band_bottom = min(top(l) for l in mrz) if mrz else page.height
        band = [words[i] for l in plines for i in l.word_idx if band_top < words[i].y < band_bottom]
        # where the label column starts: field-label words if OCR found any (OCR also "reads" junk off the
        # photo itself, which would hide the gap), otherwise the left-most word in the band
        # only the first real word of a line counts (that is where a label sits; "nr." inside an address doesn't)
        anchors = []
        for l in plines:
            first = next((words[i] for i in l.word_idx if re.search(r"[A-Za-z]{2,}", words[i].text)), None)
            if first and band_top < first.y < band_bottom and \
                    ID_LABEL_ANCHOR.match(strip_diacritics(first.text).split("/")[0].strip(",.;:|()[]{}") or "-"):
                anchors.append(first)
        if band:
            left = min(w.x for w in (anchors or band))
            margin = int(page.width * 0.02)
            gap = max(4, int(page.width * 0.004))           # labels sit close to the photo on the electronic ID
            if left > page.width * 0.2 and band_bottom - band_top > page.height * 0.15:
                # start right under the card title: on the electronic ID the hair reaches up to the header row.
                # The redactor pads every box (capped at 1.5% of the long side): leave that room before the labels.
                pad = int(0.015 * max(page.width, page.height))
                boxes.append((page.number, margin, band_top + gap, max(1, left - gap - margin - pad),
                              band_bottom - band_top - gap - margin))
    return boxes


def ink_signature_box(page, words: list[Word], avoid: list[tuple[int, int, int, int, int]]
                      ) -> tuple[int, int, int, int, int] | None:
    """An unlabelled handwritten signature on an identity document: the darkest compact ink cluster in the lower
    part of the page that no OCR word explains (cursive is not read as text). At most one per page."""
    import cv2
    import numpy as np
    img = cv2.imread(page.base_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    H, W = img.shape
    pw = [w for w in words if w.page == page.number]
    text_h = int(np.median([w.h for w in pw])) if pw else max(12, H // 40)
    ink = (img < min(110, int(np.percentile(img, 8)))).astype(np.uint8) * 255   # dark ink only, not guilloche
    for w in pw:                                          # printed text is explained by OCR
        ink[max(0, w.y - 3):w.y + w.h + 3, max(0, w.x - 3):w.x + w.w + 3] = 0
    for _p, x, y, bw, bh in avoid:                        # faces / already redacted areas
        ink[max(0, y):y + bh, max(0, x):x + bw] = 0
    ink[: int(H * 0.6)] = 0
    m = int(min(H, W) * 0.03)                             # card borders / page edges
    ink[:, :m] = ink[:, W - m:] = ink[H - m:, :] = 0
    blob = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, text_h), max(3, text_h // 2))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    text_bottom = max((w.y + w.h for w in pw), default=0)
    best, best_key = None, None
    for k in range(1, n):
        x, y, bw, bh, _a = stats[k]
        if not (1.5 * text_h <= bw <= 0.35 * W and 0.7 * text_h <= bh <= 5 * text_h):
            continue
        mass = int((ink[y:y + bh, x:x + bw] > 0).sum())
        density = mass / float(bw * bh)
        if not 0.02 <= density <= 0.35:
            continue
        # a signature sits under the printed fields; unread print between fields must not win
        key = (y + bh / 2 >= text_bottom - text_h * 0.5, mass)
        if best_key is None or key > best_key:
            best, best_key = (int(x), int(y), int(bw), int(bh)), key
    if best is None:
        return None
    x, y, bw, bh = best                                   # faint stroke ends fall outside the dark core
    px, py = max(int(0.25 * bw), text_h // 2), max(int(0.35 * bh), text_h // 2)
    x0, y0 = max(0, x - px), max(0, y - py)
    return page.number, x0, y0, min(W, x + bw + px) - x0, min(H, y + bh + py) - y0


def _signature_boxes(plines: list[Line], words: list[Word], page) -> list[tuple[int, int, int, int, int]]:
    """The handwritten signature under "Semnatura titularului / Holder's signature" (OCR cannot read it)."""
    out = []
    for line in plines:
        m = SIGNATURE_LABEL.search(line.norm)
        if not m:
            continue
        label = [i for i in _words_in_span(line, m.start(), len(line.norm) + 1)
                 if SIGNATURE_LABEL_WORDS.match(strip_diacritics(words[i].text))]
        if not label:
            continue
        x0 = min(words[i].x for i in label)
        x1 = max(words[i].x + words[i].w for i in label)
        y1 = max(words[i].y + words[i].h for i in label)
        h = max(words[i].h for i in label)
        bx, by = max(0, x0 - h), y1
        bw, bh = min(page.width, x1 + h) - bx, min(page.height, y1 + int(5.5 * h)) - by
        if bw > 0 and bh > 0:
            out.append((page.number, bx, by, bw, bh))
    return out


def unread_value_boxes(words: list[Word], pages: list) -> list[tuple[int, int, int, int, int]]:
    """A name / signature label that ends its line with nothing OCR-readable after it ("Semnatura client:"
    with the handwriting or a faint name OCR could not read) -> blur the space right of it on that row.
    Bilingual ID-card labels ("Nume/Nom/Last name") are skipped: their value sits below, not to the right."""
    lines = build_lines(words)
    label_re = re.compile(r"(?:^|(?<=[\s/]))(?:" + "|".join(NAME_LABELS) + r")\b(?:\s+client(?:ului)?)?\s*[:.;,\-]?\s*$",
                          re.I)
    widths = {p.number: p.width for p in pages}
    out = []
    for line in lines:
        m = label_re.search(line.norm)
        if not m or "/" in line.norm:
            continue
        lab = _words_in_span(line, m.start(), m.end())
        if not lab:
            continue
        pg = words[lab[0]].page
        x1 = max(words[i].x + words[i].w for i in lab)
        y0 = min(words[i].y for i in lab)
        h = max(words[i].h for i in lab)
        bx = x1 + max(2, h // 3)
        bw = min(widths.get(pg, bx + 12 * h) - bx, 12 * h)
        if bw > h:
            out.append((pg, bx, max(0, y0 - h // 4), bw, int(h * 1.5)))
    return out


def extend_into_unread(boxes: list[tuple[int, int, int, int, int]], words: list[Word], page_widths: dict[int, int],
                       max_factor: float = 3.0) -> list[tuple[int, int, int, int, int]]:
    """Grow each text box to the right through space where OCR read nothing, up to the next word it did read
    on the same row (max `max_factor` x box height). OCR often drops the tail of a value (stamps, holograms,
    guilloche over 'Sec. 2'); a box that ends where the OCR'd words end would leave that tail readable."""
    out = []
    for (pg, x, y, w, h) in boxes:
        if h <= 0 or w > 8 * h * 10:                # skip non-text boxes (portraits etc. are handled whole)
            out.append((pg, x, y, w, h))
            continue
        right = x + w
        limit = min(page_widths.get(pg, right), right + int(max_factor * h))
        for wd in words:
            if wd.page != pg or wd.x < right:
                continue
            overlap = min(y + h, wd.y + wd.h) - max(y, wd.y)
            if overlap >= 0.4 * min(h, wd.h):
                limit = min(limit, wd.x - max(2, h // 6))
        out.append((pg, x, y, max(w, limit - x), h))
    return out


def summarize(entities: list[Entity]) -> dict[str, int]:
    summary = {"CNP": 0, "PERSON_NAME": 0, "ADDRESS": 0, "ID_SERIES": 0}
    for e in entities:
        summary[e.type] = summary.get(e.type, 0) + 1
    return summary
