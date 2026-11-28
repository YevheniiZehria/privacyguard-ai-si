"""Pipeline tests on a rendered synthetic document (requires Tesseract)."""
import json
import re

import pytest
from PIL import Image, ImageDraw, ImageFont

from server.pii_detector import cnp_control_digit
from server.pipeline import PrivacyPipeline


@pytest.fixture
def doc(tmp_path):
    c = "185050822507"
    c += cnp_control_digit(c)
    img = Image.new("RGB", (1600, 420), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("arial.ttf", 34)
    d.text((40, 40), f"Subsemnatul Manole Carol Victor, CNP {c},", fill="black", font=f)
    d.text((40, 120), "domiciliat in Str. Teilor nr. 4, bl. A2, ap. 7, Iasi", fill="black", font=f)
    d.text((40, 200), "telefon 0745 111 222, e-mail carol.manole@example.com", fill="black", font=f)
    d.text((40, 280), "Semnatura: Manole", fill="black", font=f)
    p = tmp_path / "doc.png"
    img.rotate(2, expand=True, fillcolor="white").save(p)
    return str(p), c


def test_full_pipeline_outputs_are_pii_free(doc, tmp_path):
    path, cnp = doc
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    red = pipe.redact_image(ocr["doc_id"], det["detection_id"])
    ver = pipe.verify_redaction(red["result_id"])
    assert det["summary"]["CNP"] == 1 and det["summary"]["PERSON_NAME"] >= 2 and det["summary"]["ADDRESS"] == 1
    assert ver["status"] == "PASS"
    blob = json.dumps([ocr, det, red, ver, pipe.link_documents()])
    assert cnp not in blob and "Manole" not in blob and "example.com" not in blob
    assert not re.search(r"\d{13}", blob)


def test_verification_fails_on_unredacted_image(doc, tmp_path):
    path, _ = doc
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    pipe.vault.get_detection(det["detection_id"]).boxes = []        # simulate a redaction that drew nothing
    red = pipe.redact_image(ocr["doc_id"], det["detection_id"])
    ver = pipe.verify_redaction(red["result_id"])
    assert ver["status"] == "FAIL" and ver["residual_pii_count"] >= 4


def test_blur_strength_maximum_passes_and_invalid_rejected(doc, tmp_path):
    path, _ = doc
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    red = pipe.redact_image(ocr["doc_id"], det["detection_id"], padding_px=8, blur_strength="maximum")
    assert red["blur_level"] == 10 and red["blur_strength"] == "maximum" and red["mode"] == "blur"
    ver = pipe.verify_redaction(red["result_id"])
    assert ver["status"] == "PASS" and ver["leak_count"] == 0 and ver["suggested_blur_strength"] is None
    with pytest.raises(ValueError):
        pipe.redact_image(ocr["doc_id"], det["detection_id"], blur_strength="ultra")
    with pytest.raises(ValueError):
        pipe.redact_image(ocr["doc_id"], det["detection_id"], blur_strength=11)


def test_plan_round_trip_reapplies_without_ocr(doc, tmp_path):
    path, cnp = doc
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    plan = pipe.export_plan(det["detection_id"])
    blob = json.dumps(plan)
    assert cnp not in blob and "MANOLE" not in blob.upper() and "example" not in blob
    pages = [p.base_image_path for p in pipe.vault.get_document(ocr["doc_id"]).pages]
    pipe.vault.clear()                                              # raw text is gone
    det_id = pipe.import_plan(ocr["doc_id"], pages, json.loads(blob))
    red = pipe.redact_image(ocr["doc_id"], det_id, blur_strength=4)
    assert red["output_path"].endswith(f"redacted_{ocr['doc_id']}.png")
    assert pipe.verify_redaction(red["result_id"])["status"] == "PASS"

    # nothing redacted -> the restored detection still FAILs verification
    det_id = pipe.import_plan(ocr["doc_id"], pages, {"boxes": [], "fingerprints": plan["fingerprints"]})
    ver = pipe.verify_redaction(pipe.redact_image(ocr["doc_id"], det_id)["result_id"])
    assert ver["status"] == "FAIL"
    # the salted fingerprints find known values inside arbitrary re-OCR text (no raw values needed)
    from server.pipeline import _norm
    text = _norm(f"xx CNP {cnp} yy e-mail carol.manole@example.com zz")
    assert PrivacyPipeline._fingerprint_leaks(plan["fingerprints"], text, []) >= 2
    assert PrivacyPipeline._fingerprint_leaks(plan["fingerprints"], _norm("nothing to see here"), []) == 0


def test_fail_returns_value_free_escalation_hint(doc, tmp_path):
    path, cnp = doc
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    pipe.vault.get_detection(det["detection_id"]).boxes = []
    ver = pipe.verify_redaction(pipe.redact_image(ocr["doc_id"], det["detection_id"], padding_px=4)["result_id"])
    assert ver["status"] == "FAIL" and ver["uncovered_count"] >= 1
    assert ver["suggested_blur_strength"] == "maximum" and ver["suggested_padding_px"] == 16
    blob = json.dumps(ver)
    assert cnp not in blob and "Manole" not in blob and not re.search(r"\d{13}", blob)
