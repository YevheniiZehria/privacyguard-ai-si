"""Privacy-preserving pipeline used by the MCP tools (and directly by the evaluation suite).

Every public method returns ONLY opaque ids + aggregate metadata. Raw text stays in the vault.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path

from server import config
from server.entity_linker import EntityLinker, entity_hashes
from server.ocr_engine import ocr_file, ocr_image
from server.image_preprocessing import load_bgr
from server.pii_detector import (PIIDetector, build_lines, entity_boxes, extend_into_unread, portrait_boxes,
                                 strip_diacritics, summarize, unread_value_boxes)
from server.form_filler import EXTRA_KEYS, fill_document
from server.id_extractor import PROFILE_KEYS, extract_profile
from server.redactor import Redactor, effective_padding, strength_name
from server.vault import DetectionRecord, DocumentRecord, FilledRecord, Page, RedactionRecord, Vault

FILLABLE_KEYS = frozenset(PROFILE_KEYS) | frozenset(EXTRA_KEYS)
MAX_EXTRA_LEN = 200

MIN_LEAK_LEN = 5
MAX_PADDING_PX = 50


def _norm(value: str) -> str:
    """Normalization used by the value-level leak check (letters/digits only, no diacritics, upper case)."""
    return re.sub(r"[^A-Z0-9]", "", strip_diacritics(value).upper())


def _fp(salt: bytes, norm: str) -> str:
    return hashlib.sha256(salt + norm.encode("utf-8")).hexdigest()


def _resolve(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = config.PROJECT_ROOT / p
    return str(p)


class PrivacyPipeline:
    def __init__(self, vault: Vault | None = None, output_dir: Path | None = None) -> None:
        self.vault = vault or Vault()
        self.linker = EntityLinker()
        self.detector = PIIDetector()
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)

    # 1 ----------------------------------------------------------------------
    def ocr_document(self, file_path: str) -> dict:
        path = _resolve(file_path)
        doc_id = self.vault.new_id("doc")
        res = ocr_file(path, doc_id, work_dir=self.output_dir / "work")
        self.vault.put_document(DocumentRecord(doc_id, path, res.pages, res.words, res.avg_confidence))
        first = res.pages[0]
        return {"doc_id": doc_id, "word_count": len(res.words), "avg_confidence": res.avg_confidence,
                "image_dimensions": [first.width, first.height], "page_count": len(res.pages),
                "status": "indexed"}

    # 2 ----------------------------------------------------------------------
    def detect_pii(self, doc_id: str, use_local_slm: bool = False) -> dict:
        doc = self.vault.get_document(doc_id)
        entities = self.detector.detect(doc.words, use_local_slm=use_local_slm,
                                        slm_base_url=config.LOCAL_SLM_BASE_URL, slm_model=config.LOCAL_SLM_MODEL)
        boxes = entity_boxes(entities, doc.words)
        # cover value tails OCR could not read (only grows into space with no OCR'd words)
        boxes = extend_into_unread(boxes, doc.words, {p.number: p.width for p in doc.pages})
        boxes += unread_value_boxes(doc.words, doc.pages)         # "Semnatura client: <unreadable>"
        portraits = portrait_boxes(doc.words, doc.pages)          # faces / ID photo / signature (no text)
        boxes += portraits
        hashes = entity_hashes(entities, self.vault.hmac_hex)
        det = DetectionRecord(self.vault.new_id("det"), doc_id, entities, boxes, hashes)
        self.vault.put_detection(det)
        self.linker.update_document(doc_id, hashes)
        summary = summarize(entities)
        if portraits:
            summary["PORTRAIT"] = len(portraits)
        return {"detection_id": det.detection_id, "doc_id": doc_id, "summary": summary,
                "total_boxes": len(boxes), "linkable_identifiers": sum(len(v) for v in hashes.values())}

    # 3 ----------------------------------------------------------------------
    def link_documents(self, doc_ids: list[str] | None = None) -> dict:
        return self.linker.link(doc_ids)

    # 4 ----------------------------------------------------------------------
    def redact_image(self, doc_id: str, detection_id: str, padding_px: int = 4,
                     blur_strength: str | int = "high", mode: str | None = None) -> dict:
        doc = self.vault.get_document(doc_id)
        det = self.vault.get_detection(detection_id)
        if det.doc_id != doc_id:
            raise ValueError("detection_id does not belong to doc_id")
        if not 0 <= padding_px <= MAX_PADDING_PX:
            raise ValueError(f"padding_px must be between 0 and {MAX_PADDING_PX}")
        redactor = Redactor(mode=mode, blur_strength=blur_strength, padding_px=padding_px)
        paths, drawn = redactor.redact_pages(doc.pages, det.boxes, doc_id, self.output_dir)
        rec = RedactionRecord(self.vault.new_id("res"), doc_id, detection_id, paths, drawn, padding_px,
                              blur_level=redactor.level, mode=redactor.mode, padding_scale=redactor.padding_scale)
        self.vault.put_result(rec)
        return {"result_id": rec.result_id, "output_path": paths[0], "boxes_drawn": drawn,
                "page_count": len(paths), "blur_strength": strength_name(redactor.level),
                "blur_level": redactor.level, "mode": redactor.mode}

    # 5 ----------------------------------------------------------------------
    def verify_redaction(self, result_id: str) -> dict:
        rec = self.vault.get_result(result_id)
        det = self.vault.get_detection(rec.detection_id)
        uncovered = leaks = 0
        rec.residual_types = []
        for page_no, path in enumerate(rec.output_paths, start=1):
            # The redacted image is already deskewed -> same coordinate frame as the redaction boxes.
            img = load_bgr(path)
            long_side = max(img.shape[:2])
            _, words = ocr_image(img, page_no=page_no, deskew=False)
            boxes = []
            for (pg, x, y, w, h) in det.boxes:          # same margin the redactor used, +3 px tolerance
                if pg == page_no:
                    pad = effective_padding(h, rec.padding_px, long_side, rec.padding_scale) + 3
                    boxes.append((x - pad, y - pad, x + w + pad, y + h + pad))

            def on_box(wd) -> bool:   # OCR "text" hallucinated on the edge of a black box is an artifact
                return any(wd.x < bx1 and wd.x + wd.w > bx0 and wd.y < by1 and wd.y + wd.h > by0
                           for bx0, by0, bx1, by1 in boxes)

            clean = [wd for wd in words if not on_box(wd)]
            # (a) generic re-detection on the visible (non-redacted) text -> boxes / padding too small
            new_entities = self.detector.detect(clean)
            uncovered += len(new_entities)
            rec.residual_types += [f"{e.type}:{e.source}:{e.value}" for e in new_entities]
            # (b) value-level leak check: do any original PII values still appear in the re-OCR text?
            text = _norm(" ".join(l.text for l in build_lines(words)))
            if det.entities:
                covered = {strip_diacritics(e.value).upper() for e in new_entities}
                for e in det.entities:
                    v = _norm(e.value)
                    if len(v) >= MIN_LEAK_LEN and v in text and strip_diacritics(e.value).upper() not in covered:
                        leaks += 1
                        rec.residual_types.append(f"LEAK:{e.type}:{e.value}")
            elif det.fingerprints:
                leaks += self._fingerprint_leaks(det.fingerprints, text, new_entities)
        residual = uncovered + leaks
        verified = residual == 0
        out = {"result_id": result_id, "verified": verified, "residual_pii_count": residual,
               "status": "PASS" if verified else "FAIL", "leak_count": leaks, "uncovered_count": uncovered,
               "suggested_blur_strength": None, "suggested_padding_px": None}
        if not verified:   # escalation ladder for the agent / UI (value-free guidance)
            out["suggested_blur_strength"] = "maximum"
            out["suggested_padding_px"] = min(MAX_PADDING_PX, rec.padding_px + (12 if uncovered else 6))
        return out

    @staticmethod
    def _fingerprint_leaks(fps: dict, text: str, new_entities) -> int:
        salt = bytes.fromhex(fps["salt"])
        covered = {_fp(salt, _norm(e.value)) for e in new_entities}
        windows: dict[int, set[str]] = {}         # hash every text window once per distinct value length
        leaks = 0
        for length, digest in fps.get("items", []):
            if digest in covered or length > len(text):
                continue
            if length not in windows:
                windows[length] = {_fp(salt, text[i:i + length]) for i in range(len(text) - length + 1)}
            leaks += digest in windows[length]
        return leaks

    # 6 ---------------------------------------------------------------------- re-apply without OCR
    def export_plan(self, detection_id: str) -> dict:
        """Everything needed to redact + verify the document again WITHOUT OCR: box coordinates and salted
        SHA-256 fingerprints of the PII values. Contains no raw text."""
        det = self.vault.get_detection(detection_id)
        if det.entities:
            salt = secrets.token_bytes(16)
            norms = sorted({n for n in (_norm(e.value) for e in det.entities) if len(n) >= MIN_LEAK_LEN})
            fps = {"salt": salt.hex(), "items": [[len(n), _fp(salt, n)] for n in norms]}
        else:
            fps = det.fingerprints or {"salt": secrets.token_hex(16), "items": []}
        return {"version": 1, "boxes": [[int(v) for v in b] for b in det.boxes], "fingerprints": fps}

    def import_plan(self, doc_id: str, page_paths: list, plan: dict, extra_boxes=()) -> str:
        """Restore a document (deskewed page images) + detection from export_plan() under the ORIGINAL doc_id,
        so redact_image writes the same output file names. Returns the new detection_id."""
        if not page_paths:
            raise ValueError("no page images to re-apply the redaction on")
        pages = []
        for n, path in enumerate(page_paths, start=1):
            h, w = load_bgr(str(path)).shape[:2]
            pages.append(Page(n, str(path), w, h))
        self.vault.put_document(DocumentRecord(doc_id, str(page_paths[0]), pages, [], 0.0))
        boxes = [tuple(int(v) for v in b) for b in [*plan.get("boxes", []), *extra_boxes]]
        det = DetectionRecord(self.vault.new_id("det"), doc_id, [], boxes, {},
                              fingerprints=plan.get("fingerprints") or None)
        self.vault.put_detection(det)
        return det.detection_id

    # 7 ---------------------------------------------------------------------- ID card -> form auto-fill
    def extract_id_profile(self, doc_id: str) -> dict:
        """Structure the ID card's fields inside the vault. Returns only WHICH fields were found."""
        doc = self.vault.get_document(doc_id)
        det = next((d for d in self.vault.detections_for(doc_id)), None)
        entities = det.entities if det and det.entities else self.detector.detect(doc.words)
        rec = extract_profile(doc, entities)
        self.vault.put_profile(rec)
        conf = list(rec.confidence.values())
        return {"doc_id": doc_id, "is_id_card": rec.is_id_card, "is_valid_id": rec.is_valid_id,
                "detected_fields": list(rec.fields), "field_count": len(rec.fields),
                "mean_confidence": round(sum(conf) / len(conf), 2) if conf else 0.0}

    def fill_form_from_id(self, id_doc_id: str, form_doc_id: str, extra_fields: dict[str, str] | None = None) -> dict:
        """Write the ID profile onto the form; the filled pages become a new document (OCR'd, so detect_pii ->
        redact_image -> verify_redaction work on it directly). Returns ids, paths and field KEYS only."""
        if id_doc_id == form_doc_id:
            raise ValueError("id_doc_id and form_doc_id must be different documents")
        profile = self.vault.get_profile(id_doc_id)
        form = self.vault.get_document(form_doc_id)
        extras: dict[str, str] = {}
        for k, v in (extra_fields or {}).items():
            if k not in FILLABLE_KEYS:
                raise ValueError(f"unknown extra field '{k}'. Allowed: {sorted(FILLABLE_KEYS)}")
            if v is not None and len(str(v)) > MAX_EXTRA_LEN:
                raise ValueError(f"extra field '{k}' is longer than {MAX_EXTRA_LEN} characters")
            if v:
                extras[k] = str(v)
        filled_id = self.vault.new_id("fill")
        out = fill_document(form, profile.fields, extras, self.output_dir, f"filled_{filled_id}")
        pages, words = [], []
        for n, path in enumerate(out.pages, start=1):
            img = load_bgr(path)
            _, page_words = ocr_image(img, page_no=n, deskew=False)
            h, w = img.shape[:2]
            pages.append(Page(n, path, w, h))
            words += page_words
        conf = [w.conf for w in words]
        self.vault.put_document(DocumentRecord(filled_id, out.pdf_path or out.pages[0], pages, words,
                                               round(sum(conf) / len(conf), 2) if conf else 0.0))
        self.vault.put_filled(FilledRecord(filled_id, form_doc_id, id_doc_id, out.pages, out.pdf_path,
                                           out.fields_filled, out.unfilled_fields))
        return {"filled_doc_id": filled_id, "output_path": out.pages[0], "pdf_path": out.pdf_path,
                "fields_filled": out.fields_filled, "unfilled_fields": out.unfilled_fields,
                "page_count": len(out.pages)}
