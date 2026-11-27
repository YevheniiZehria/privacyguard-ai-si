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
