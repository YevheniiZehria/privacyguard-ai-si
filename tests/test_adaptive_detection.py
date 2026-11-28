"""Adaptive detection (requires Tesseract): perspective dewarp, universal multilingual / column layouts,
learning from manual blur boxes, Cyrillic second pass and the low-coverage safety net."""
import json
import random
import re

import cv2
import numpy as np
import pytest
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from evaluation.generate_synthetic_docs import id_card, make_people, render
from server import config
from server.image_preprocessing import find_document_quad, load_bgr, preprocess
from server.ocr_engine import needs_foreign_pass, ocr_image
from server.pii_detector import PIIDetector
from server.pipeline import PrivacyPipeline
from server.rule_learner import LearnedRule, RuleStore, learn_from_box, validate

FONT = "arial.ttf"


def _bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _readable(path: str) -> str:
    _, words = ocr_image(load_bgr(path), deskew=False)
    return re.sub(r"[^A-Z0-9А-ЯІЇЄҐ]", "", " ".join(w.text for w in words).upper())


def _run(pipe: PrivacyPipeline, path: str) -> tuple[dict, dict, str]:
    ocr = pipe.ocr_document(path)
    det = pipe.detect_pii(ocr["doc_id"])
    red = pipe.redact_image(ocr["doc_id"], det["detection_id"])
    ver = pipe.verify_redaction(red["result_id"])
    return det, ver, red["output_path"]


@pytest.fixture
def rules_path(tmp_path, monkeypatch):
    path = tmp_path / "learned_rules.json"
    monkeypatch.setenv("LEARNED_RULES_PATH", str(path))
    return path


@pytest.fixture(scope="module")
def person():
    rng = random.Random(7)
    fake = Faker("ro_RO")
    fake.seed_instance(7)
    return make_people(rng, fake)[0]


# ----------------------------------------------------------------------------- 1. perspective dewarp
def _photo_of_card(person) -> np.ndarray:
    lines = id_card(person)
    img, _ = render(lines, "arial", size=32, width=1500, line_gap=1.75, mrz_from=len(lines) - 2, border=True)
    card = _bgr(img)
    h, w = card.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[380, 260], [1880, 140], [1990, 1360], [240, 1210]])      # trapezoid, phone at an angle
    bg = np.full((1700, 2250, 3), (40, 55, 60), np.uint8)
    return cv2.warpPerspective(card, cv2.getPerspectiveTransform(src, dst), (2250, 1700), dst=bg,
                               borderMode=cv2.BORDER_TRANSPARENT)


def test_perspective_photo_is_dewarped_and_fully_redacted(person, tmp_path, monkeypatch):
    photo = _photo_of_card(person)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), photo)
    assert find_document_quad(photo) is not None
    pre = preprocess(photo)
    assert pre.dewarped and pre.base_bgr.shape[0] < photo.shape[0]          # flattened to the card itself

    monkeypatch.setattr(config, "DEWARP_ENABLED", False)
    _, raw_words = ocr_image(photo)
    monkeypatch.setattr(config, "DEWARP_ENABLED", True)
    _, flat_words = ocr_image(photo)
    mean = lambda ws: float(np.mean([w.conf for w in ws]))                  # noqa: E731
    assert mean(flat_words) >= mean(raw_words)

    det, ver, out = _run(PrivacyPipeline(output_dir=tmp_path / "out"), str(path))
    assert ver["status"] == "PASS"
    text = _readable(out)
    for value in (person.cnp, person.surname.upper(), person.number, "LALELELOR"):
        assert value not in text, value


def test_scans_are_not_dewarped():
    for f in ("synthetic_data/01_id_card_P1.png", "synthetic_data/03_invoice_P1.jpg", "synthetic_data/02_form_P1.png"):
        assert find_document_quad(load_bgr(f)) is None


# ----------------------------------------------------------------------------- 2. learning from manual boxes
def _account_doc(name: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    f = ImageFont.truetype(FONT, 34)
    img = Image.new("RGB", (1500, 320), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "EXTRAS DE CONT BANCAR", fill="black", font=f)
    d.text((40, 130), "Titular cont:", fill="black", font=f)
    x = d.textbbox((40, 130), "Titular cont:", font=f)[2] + 20
    d.text((x, 130), name, fill="black", font=f)
    x0, y0, x1, y1 = d.textbbox((x, 130), name, font=f)
    d.text((40, 220), "Sold disponibil: 1.250,00 lei", fill="black", font=f)
    return img, (x0 - 4, y0 - 4, x1 - x0 + 8, y1 - y0 + 8)


def test_learning_round_trip_stores_no_value_and_redacts_next_document(rules_path, tmp_path):
    img, box = _account_doc("Kvarnstrom Elsbeth")             # names outside the Romanian lexicon
    bgr = _bgr(img)
    _, words = ocr_image(bgr, deskew=False)
    assert not PIIDetector(use_spacy=False).detect(words)       # nothing is known about "Titular cont" yet

    masked = bgr.copy()
    x, y, w, h = box
    masked[y:y + h, x:x + w] = 255
    _, label_words = ocr_image(masked, deskew=False)
    rule = learn_from_box(label_words, box, "Kvarnstrom Elsbeth", *img.size)
    assert rule and rule.label == "titular cont" and rule.relation == "right" and rule.shape == "ALPHA_WORDS"
    RuleStore.shared().add(rule)
    stored = rules_path.read_text(encoding="utf-8")
    assert "titular cont" in stored and "Kvarnstrom" not in stored and "Elsbeth" not in stored

    img2, _ = _account_doc("Thorvaldsen Ingrid")
    p2 = tmp_path / "second.png"
    img2.save(p2)
    det, ver, out = _run(PrivacyPipeline(output_dir=tmp_path / "out"), str(p2))
    assert det["summary"]["PERSON_NAME"] >= 1 and ver["status"] == "PASS"
    assert "THORVALDSEN" not in _readable(out) and "INGRID" not in _readable(out)


@pytest.mark.parametrize("label", ["cnp 1960523331234", "client a@b.ro", "iban ro49aaaa1b31007593840000",
                                   "popescu", "tel 0745 111 222", "x", "a de"])
def test_validator_rejects_values(label):
    rule = LearnedRule("r_abcdefgh", label, "right", "ALPHA_WORDS", 0.1, 0.1, 0.1, 0.1)
    assert not validate(rule)


def test_validator_accepts_labels_and_drops_tampered_rules(rules_path):
    assert validate(LearnedRule("r_abcdefgh", "beneficiar plata", "below", "ALPHA_WORDS", 0.1, 0.2, 0.3, 0.05))
    rules_path.write_text(json.dumps({"rules": [
        {"id": "r_abcdefgh", "label": "cod client", "relation": "right", "shape": "DIGITS_5_8", "x_rel": 0.1,
         "y_rel": 0.1, "w_rel": 0.1, "h_rel": 0.1},
        {"id": "r_hijklmno", "label": "1960523331234", "relation": "right", "shape": "DIGITS_9_13", "x_rel": 0.1,
         "y_rel": 0.1, "w_rel": 0.1, "h_rel": 0.1}]}), encoding="utf-8")
    assert [r.label for r in RuleStore.shared().rules()] == ["cod client"]


def test_learned_rules_api(rules_path, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from ui import api as ui_api
    monkeypatch.setattr(ui_api, "STORE", ui_api.Store(tmp_path / "ws"))
    with TestClient(ui_api.app) as c:
        assert c.post("/api/login", json={"username": ui_api.UI_USERNAME,
                                          "passphrase": ui_api.UI_PASSPHRASE}).status_code == 200
        fid = c.post("/api/folders", json={"name": "Learn"}).json()["folder_id"]
        img, box = _account_doc("Kvarnstrom Elsbeth")
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        c.post(f"/api/folders/{fid}/upload", files=[("files", ("extras.png", buf.getvalue(), "image/png"))])
        import time
        for _ in range(300):
            docs = c.get(f"/api/folders/{fid}/documents").json()["documents"]
            if docs and docs[0]["status"] not in ("QUEUED", "SCANNING"):
                break
            time.sleep(0.3)
        did = docs[0]["doc_id"]
        x, y, w, h = box
        r = c.post(f"/api/folders/{fid}/documents/{did}/manual-redact", json={"page": 1, "x": x, "y": y, "w": w, "h": h})
        assert r.status_code == 200
        learned = r.json()["learned_rule"]
        assert learned and learned["label"] == "titular cont" and learned["relation"] == "right"
        assert "Kvarnstrom" not in r.text and "Elsbeth" not in r.text
        rules = c.get("/api/learned-rules").json()["rules"]
        assert [x["label"] for x in rules] == ["titular cont"]
        assert c.delete("/api/learned-rules/r_zzzzzzzz").status_code == 404
        assert c.delete(f"/api/learned-rules/{rules[0]['id']}").status_code == 200
        assert c.get("/api/learned-rules").json()["rules"] == []
        assert c.delete("/api/learned-rules").json() == {"removed": 0}
        assert "available" in c.get("/api/slm-status").json()


# ----------------------------------------------------------------------------- 3. universal layouts
def test_english_two_column_invoice_and_contract(tmp_path):
    f = ImageFont.truetype(FONT, 30)
    fb = ImageFont.truetype("arialbd.ttf", 30)
    img = Image.new("RGB", (1700, 1100), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), "Invoice no: 40378170", fill="black", font=fb)
    d.text((80, 110), "Date of issue: 10/15/2012", fill="black", font=f)
    col = [("Seller:", "Client:"), ("Patel, Thompson and Montgomery", "Jackson, Odonnell and Jackson"),
           ("356 Kyle Vista", "267 John Track Suite 841"), ("New James, MA 46228", "Jenniferville, PA 98601")]
    for i, (a, b) in enumerate(col):
        d.text((80, 220 + 45 * i), a, fill="black", font=fb if i == 0 else f)
        d.text((900, 220 + 45 * i), b, fill="black", font=fb if i == 0 else f)
    d.text((80, 420), "Tax Id: 958-74-3511", fill="black", font=f)
    d.text((900, 420), "Tax Id: 998-87-7723", fill="black", font=f)
    d.text((80, 465), "IBAN: GB82WEST12345698765432", fill="black", font=f)
    d.text((80, 580), "ITEMS", fill="black", font=fb)
    d.text((80, 630), "Description         Qty     Net price     Gross", fill="black", font=f)
    d.text((80, 680), "Corkscrew Gift Box     1        7,50          8,25", fill="black", font=f)
    d.text((80, 800), "Account holder: Reinholt Vasquez", fill="black", font=f)
    d.text((80, 850), "Date of birth: 24.08.1991        Customer No.: 77310452", fill="black", font=f)
    path = tmp_path / "invoice_en.png"
    img.save(path)
    det, ver, out = _run(PrivacyPipeline(output_dir=tmp_path / "out"), str(path))
    text = _readable(out)
    for secret in ("THOMPSON", "ODONNELL", "KYLEVISTA", "JOHNTRACK", "46228", "98601", "958743511", "998877723",
                   "GB82WEST", "REINHOLT", "VASQUEZ", "24081991", "77310452"):
        assert secret not in text, secret
    for kept in ("INVOICENO40378170", "ITEMS", "CORKSCREW"):          # non-personal content stays readable
        assert kept in text, kept
    assert ver["status"] == "PASS"


# ----------------------------------------------------------------------------- 4. foreign ID (Cyrillic)
def test_bilingual_passport_card_is_covered(tmp_path):
    f = ImageFont.truetype(FONT, 30)
    fs = ImageFont.truetype(FONT, 18)
    img = Image.new("RGB", (1300, 820), (236, 242, 250))
    d = ImageDraw.Draw(img)
    d.text((420, 30), "УКРАЇНА  UKRAINE   PASSPORT", fill=(30, 30, 40), font=f)
    rows = [("Прізвище/ Surname", ["ТКАЧЕНКО", "TKACHENKO"]), ("Ім'я/ Name", ["МАР'ЯНА", "MARIANA"]),
            ("Дата народження/ Date of birth", ["24 08 1991"]), ("Документ №/ Document No.", ["FA 123456"])]
    y = 110
    for label, values in rows:
        d.text((420, y), label, fill=(60, 60, 70), font=fs)
        y += 30
        for v in values:
            d.text((420, y), v, fill=(20, 20, 30), font=f)
            y += 44
        y += 14
    path = tmp_path / "passport.png"
    img.save(path)
    _, words = ocr_image(load_bgr(str(path)))
    assert needs_foreign_pass([w for w in words if w.text.isupper()]) or any(
        re.search("[А-ЯІЇЄ]", w.text) for w in words)                       # Cyrillic was read (second pass)
    det, ver, out = _run(PrivacyPipeline(output_dir=tmp_path / "out"), str(path))
    text = _readable(out)
    for secret in ("TKACHENKO", "MARIANA", "24081991", "123456", "ТКАЧЕНКО"):
        assert secret not in text, secret


# ----------------------------------------------------------------------------- 5. safety net
def test_id_document_with_nothing_detected_needs_review(tmp_path, monkeypatch):
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    monkeypatch.setattr(pipe.detector, "detect_with_regions", lambda *a, **k: ([], []))
    det, ver, _ = _run(pipe, "synthetic_data/01_id_card_P1.png")
    assert ver["status"] == "FAIL" and ver["low_coverage"] is True
    blob = json.dumps(ver)
    assert not re.search(r"\d{13}", blob)
