"""Privacy-preserving pipeline used by the MCP tools (and directly by the evaluation suite).

Every public method returns ONLY opaque ids + aggregate metadata. Raw text stays in the vault.
"""
from __future__ import annotations

import re
from pathlib import Path

from server import config
from server.entity_linker import EntityLinker, entity_hashes
from server.ocr_engine import ocr_file, ocr_image
from server.image_preprocessing import load_bgr
from server.pii_detector import PIIDetector, build_lines, entity_boxes, strip_diacritics, summarize
from server.redactor import redact_pages
from server.vault import DetectionRecord, DocumentRecord, RedactionRecord, Vault


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
        hashes = entity_hashes(entities, self.vault.hmac_hex)
        det = DetectionRecord(self.vault.new_id("det"), doc_id, entities, boxes, hashes)
        self.vault.put_detection(det)
        self.linker.update_document(doc_id, hashes)
        return {"detection_id": det.detection_id, "doc_id": doc_id, "summary": summarize(entities),
                "total_boxes": len(boxes), "linkable_identifiers": sum(len(v) for v in hashes.values())}

    # 3 ----------------------------------------------------------------------
    def link_documents(self, doc_ids: list[str] | None = None) -> dict:
        return self.linker.link(doc_ids)

    # 4 ----------------------------------------------------------------------
    def redact_image(self, doc_id: str, detection_id: str, padding_px: int = 4) -> dict:
        doc = self.vault.get_document(doc_id)
        det = self.vault.get_detection(detection_id)
        if det.doc_id != doc_id:
            raise ValueError("detection_id does not belong to doc_id")
        if not 0 <= padding_px <= 50:
            raise ValueError("padding_px must be between 0 and 50")
        paths, drawn = redact_pages(doc.pages, det.boxes, doc_id, self.output_dir, padding_px)
        rec = RedactionRecord(self.vault.new_id("res"), doc_id, detection_id, paths, drawn, padding_px)
        self.vault.put_result(rec)
        return {"result_id": rec.result_id, "output_path": paths[0], "boxes_drawn": drawn,
                "page_count": len(paths)}

    # 5 ----------------------------------------------------------------------
    def verify_redaction(self, result_id: str) -> dict:
        rec = self.vault.get_result(result_id)
        det = self.vault.get_detection(rec.detection_id)
        residual = 0
        pad = rec.padding_px + 3
        for page_no, path in enumerate(rec.output_paths, start=1):
            # The redacted image is already deskewed -> same coordinate frame as the redaction boxes.
            _, words = ocr_image(load_bgr(path), page_no=page_no, deskew=False)
            boxes = [(x - pad, y - pad, x + w + pad, y + h + pad) for (pg, x, y, w, h) in det.boxes if pg == page_no]

            def on_box(wd) -> bool:   # OCR "text" hallucinated on the edge of a black box is an artifact
                return any(wd.x < bx1 and wd.x + wd.w > bx0 and wd.y < by1 and wd.y + wd.h > by0
                           for bx0, by0, bx1, by1 in boxes)

            clean = [wd for wd in words if not on_box(wd)]
            # (a) generic re-detection on the visible (non-redacted) text
            new_entities = self.detector.detect(clean)
            residual += len(new_entities)
            # (b) value-level leak check: do any original PII values still appear in the re-OCR text?
            text = re.sub(r"[^A-Z0-9]", "", strip_diacritics(" ".join(l.text for l in build_lines(words))).upper())
            covered = {strip_diacritics(e.value).upper() for e in new_entities}
            rec.residual_types = [f"{e.type}:{e.source}:{e.value}" for e in new_entities]
            for e in det.entities:
                v = re.sub(r"[^A-Z0-9]", "", strip_diacritics(e.value).upper())
                if len(v) >= 5 and v in text and strip_diacritics(e.value).upper() not in covered:
                    residual += 1
                    rec.residual_types.append(f"LEAK:{e.type}:{e.value}")
        verified = residual == 0
        return {"result_id": result_id, "verified": verified, "residual_pii_count": residual,
                "status": "PASS" if verified else "FAIL"}
