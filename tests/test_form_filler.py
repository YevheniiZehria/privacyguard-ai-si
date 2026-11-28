"""Auto-fill a blank form from an ID card (requires Tesseract): extraction, flat + AcroForm filling, privacy."""
import json
import random
import re

import pymupdf
import pytest
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from agent.cli_agent import privacy_guard
from evaluation.generate_synthetic_docs import id_card, make_people, render
from server.form_filler import _covers_diacritics, match_widget, resolve_font
from server.id_extractor import cnp_facts, parse_address
from server.image_preprocessing import load_bgr
from server.ocr_engine import ocr_image
from server.pipeline import PrivacyPipeline

FONT = "arial.ttf"


@pytest.fixture(scope="module")
def person():
    rng = random.Random(7)
    fake = Faker("ro_RO")
    fake.seed_instance(7)
    return make_people(rng, fake)[0]     # Dinu Nichifor Antoniu, Str. Lalelelor nr. 10, bl. E2, ap. 24, Timis


def _id_image(person, tmp_path, drop_name_values=False):
    lines = id_card(person)
    if drop_name_values:                 # printed names unreadable -> only the MRZ carries them
        lines = [[("", None)] if seg and seg[0][1] == "PERSON_NAME" else seg for seg in lines]
    img, _ = render(lines, "arial", size=32, width=1500, line_gap=1.75, mrz_from=len(lines) - 2, border=True)
    path = tmp_path / ("id_mrz.png" if drop_name_values else "id.png")
    img.save(path)
    return str(path)


def _form(rows, path, width=1700):
    """rows: [[("t", label) | ("line", px) | ("dots", n)], ...] — a blank Romanian form."""
    font = ImageFont.truetype(FONT, 30)
    img = Image.new("RGB", (width, 120 + 90 * len(rows)), "white")
    d = ImageDraw.Draw(img)
    y = 60
    for row in rows:
        x = 60
        for kind, val in row:
            if kind == "t":
                d.text((x, y), val, fill=(20, 20, 20), font=font)
                x = d.textbbox((x, y), val, font=font)[2] + 12
            elif kind == "line":
                d.line((x, y + 34, x + val, y + 34), fill=(20, 20, 20), width=2)
                x += val + 14
            else:
                d.text((x, y), "." * val, fill=(20, 20, 20), font=font)
                x = d.textbbox((x, y), "." * val, font=font)[2] + 12
        y += 90
    img.save(path)
    return str(path)


FORM_SIMPLE = [[("t", "CERERE")],
               [("t", "Subsemnatul"), ("line", 700)],
               [("t", "CNP"), ("line", 420), ("t", "domiciliat in"), ("line", 640)],
               [("t", "posesor al CI seria"), ("line", 120), ("t", "nr."), ("line", 260),
                ("t", "eliberat de"), ("dots", 30)],
               [("t", "Telefon:")]]
FORM_MULTI = [[("t", "Nume"), ("line", 520), ("t", "Prenume"), ("line", 600)],
              [("t", "Str."), ("line", 420), ("t", "Nr."), ("line", 110), ("t", "Bl."), ("line", 110),
               ("t", "Sc."), ("line", 90), ("t", "Ap."), ("line", 110)],
              [("t", "Județul"), ("line", 500), ("t", "Data nașterii"), ("line", 300)],
              [("t", "Telefon"), ("line", 500)]]


@pytest.fixture(scope="module")
def setup(person, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fill")
    pipe = PrivacyPipeline(output_dir=tmp / "out")
    id_doc = pipe.ocr_document(_id_image(person, tmp))["doc_id"]
    prof = pipe.extract_id_profile(id_doc)
    return pipe, tmp, id_doc, prof


def _ocr_in(path, box):
    """OCR text (letters/digits only, upper case) inside a drawn-text box of the filled page."""
    img = load_bgr(path)
    x0, y0, x1, y1 = box
    crop = img[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
    crop = __import__("cv2").copyMakeBorder(crop, 30, 30, 30, 30, 0, value=(255, 255, 255))
    _, words = ocr_image(crop, deskew=False)
    return re.sub(r"[^A-Z0-9]", "", " ".join(w.text for w in words).upper())


# ----------------------------------------------------------------------------- extraction
def test_id_profile_fields(setup, person):
    pipe, _, id_doc, prof = setup
    assert prof["is_id_card"] and prof["is_valid_id"]
    rec = pipe.vault.get_profile(id_doc).fields
    assert rec["last_name"] == "Dinu" and rec["first_name"] == "Nichifor Antoniu"
    assert rec["full_name"] == "Dinu Nichifor Antoniu"
    assert rec["cnp"] == person.cnp
    assert (rec["id_series"], rec["id_number"]) == (person.series, person.number)
    assert rec["street"] == "Lalelelor" and rec["street_no"] == "10" and rec["building"] == "E2"
    assert rec["apartment"] == "24" and rec["county"] == "Timis" and rec["city"] == "Timis"
    assert rec["birth_date"] == person.birth and rec["sex"] == "M"
    assert rec["issue_date"] == "20.05.2023" and rec["expiry_date"] == "20.05.2033"
    assert rec["issuing_authority"] == "SPCLEP Timis" and rec["nationality"] == "Română"
    assert "ap. 24, Mun. Timis" in rec["address_full"]


def test_mrz_fallback_recovers_names(person, tmp_path):
    pipe = PrivacyPipeline(output_dir=tmp_path / "out")
    doc = pipe.ocr_document(_id_image(person, tmp_path, drop_name_values=True))["doc_id"]
    res = pipe.extract_id_profile(doc)
    rec = pipe.vault.get_profile(doc).fields
    assert {"last_name", "first_name", "full_name"} <= set(res["detected_fields"])
    assert rec["last_name"] == "Dinu" and rec["first_name"] == "Nichifor Antoniu"


@pytest.mark.parametrize("addr,expected", [
    ("Str. Lalelelor nr. 10, bl. E2, sc. B, et. 3, ap. 24, Mun. Iasi, jud. Iasi",
     {"street": "Lalelelor", "street_no": "10", "building": "E2", "staircase": "B", "floor": "3",
      "apartment": "24", "city": "Iasi", "county": "Iasi"}),
    ("Bd. Unirii nr. 5A, Sector 3", {"street": "Bd. Unirii", "street_no": "5A", "sector": "3",
                                     "county": "București", "city": "București"}),
    ("Sat Vadu, Com. Moara, jud. Suceava", {"city": "Vadu", "county": "Suceava"}),
])
def test_parse_address(addr, expected):
    got = parse_address(addr)
    assert {k: got.get(k) for k in expected} == expected


def test_cnp_facts():
    assert cnp_facts("1820313355978") == {"sex": "M", "birth_date": "13.03.1982"}
    assert cnp_facts("6040229123456")["birth_date"] == "29.02.2004"


# ----------------------------------------------------------------------------- flat forms
def test_flat_form_fill_places_values_in_slots(setup, person):
    pipe, tmp, id_doc, _ = setup
    form_doc = pipe.ocr_document(_form(FORM_SIMPLE, tmp / "form_simple.png"))["doc_id"]
    res = pipe.fill_form_from_id(id_doc, form_doc)
    assert {"full_name", "cnp", "address_full", "id_series", "id_number", "issuing_authority"} <= set(res["fields_filled"])
    assert res["unfilled_fields"] == ["phone"]                     # "Telefon:" exists, no value given
    rec = pipe.vault.get_filled(res["filled_doc_id"])
    out =pipe.vault.get_document(res["filled_doc_id"])
    text = re.sub(r"[^A-Z0-9]", "", " ".join(w.text for w in out.words).upper())
    assert person.cnp in text and "DINU" in text and "LALELELOR" in text and person.number in text
    assert rec.output_paths == [res["output_path"]]


def test_flat_form_values_do_not_overlap_labels(setup, person, tmp_path):
    pipe, tmp, id_doc, _ = setup
    from server.form_filler import fill_document
    form_doc = pipe.ocr_document(_form(FORM_SIMPLE, tmp / "form_simple2.png"))["doc_id"]
    fields = pipe.vault.get_profile(id_doc).fields
    out = fill_document(pipe.vault.get_document(form_doc), fields, {}, tmp_path, "filled_chk")
    for p in out.placements:
        for (_, lx0, ly0, lx1, ly1) in out.label_boxes:
            x0, y0, x1, y1 = p.box
            assert not (x0 < lx1 and x1 > lx0 and y0 < ly1 and y1 > ly0), (p.key, p.box, (lx0, ly0, lx1, ly1))
    by_key = {p.key: p.box for p in out.placements}
    assert person.cnp in _ocr_in(out.pages[0], by_key["cnp"])         # the CNP sits in the CNP slot
    assert "DINU" in _ocr_in(out.pages[0], by_key["full_name"])


def test_multi_field_form_and_extras(setup):
    pipe, tmp, id_doc, _ = setup
    form_doc = pipe.ocr_document(_form(FORM_MULTI, tmp / "form_multi.png"))["doc_id"]
    res = pipe.fill_form_from_id(id_doc, form_doc)
    assert {"last_name", "first_name", "street", "street_no", "building", "apartment", "county",
            "birth_date"} <= set(res["fields_filled"])
    assert "phone" in res["unfilled_fields"] and "staircase" in res["unfilled_fields"]   # no value -> blank
    res2 = pipe.fill_form_from_id(id_doc, form_doc, {"phone": "0745 111 222"})
    assert "phone" in res2["fields_filled"]
    text = " ".join(w.text for w in pipe.vault.get_document(res2["filled_doc_id"]).words)
    assert "0745" in text and "Lalelelor" in text
    with pytest.raises(ValueError):
        pipe.fill_form_from_id(id_doc, form_doc, {"password": "x"})


def test_font_has_romanian_diacritics():
    font = ImageFont.truetype(resolve_font(), 40)
    assert _covers_diacritics(font)
    assert font.getmask("Ștefănescu Țăndărei").getbbox()


# ----------------------------------------------------------------------------- AcroForm
def test_acroform_widgets_are_filled(setup, tmp_path):
    pipe, _, id_doc, _ = setup
    pdf = tmp_path / "acro.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for i, (name, label) in enumerate([("nume", "Nume"), ("prenume", "Prenume"), ("cnp", "CNP"),
                                       ("adresa_domiciliu", "Adresa"), ("seria_ci", "Seria CI"),
                                       ("telefon", "Telefon")]):
        page.insert_text((50, 80 + i * 40), label, fontsize=11)
        w = pymupdf.Widget()
        w.field_name, w.field_type = name, pymupdf.PDF_WIDGET_TYPE_TEXT
        w.rect = pymupdf.Rect(150, 66 + i * 40, 560, 86 + i * 40)
        page.add_widget(w)
    doc.save(pdf)
    form_doc = pipe.ocr_document(str(pdf))["doc_id"]
    prof = pipe.vault.get_profile(id_doc)
    prof.fields["last_name"] = "Ștefănescu"                         # a value Helvetica cannot show
    try:
        res = pipe.fill_form_from_id(id_doc, form_doc)
    finally:
        prof.fields["last_name"] = "Dinu"
    assert res["pdf_path"] and {"last_name", "first_name", "cnp", "address_full", "id_series"} <= set(res["fields_filled"])
    assert res["unfilled_fields"] == ["phone"]
    with pymupdf.open(res["pdf_path"]) as out:
        values = {w.field_name: w.field_value for w in out[0].widgets()}
        text = out[0].get_text()
    assert values["cnp"] == pipe.vault.get_profile(id_doc).fields["cnp"] and values["prenume"] == "Nichifor Antoniu"
    assert "Ștefănescu" in text                                     # stamped with the TTF font


@pytest.mark.parametrize("name,key", [("Nume", "last_name"), ("prenumele", "first_name"), ("C.N.P.", "cnp"),
                                      ("Adresa domiciliu", "address_full"), ("seria_ci", "id_series"),
                                      ("Data nașterii", "birth_date"), ("txtJudet", "county"), ("xyz", None)])
def test_match_widget(name, key):
    assert match_widget(name) == key


# ----------------------------------------------------------------------------- privacy + pipeline
def test_tool_outputs_are_pii_free_and_filled_doc_is_redactable(setup, person):
    pipe, tmp, id_doc, prof = setup
    form_doc = pipe.ocr_document(_form(FORM_SIMPLE, tmp / "form_priv.png"))["doc_id"]
    res = pipe.fill_form_from_id(id_doc, form_doc)
    blob = json.dumps([prof, res])
    for secret in (person.cnp, person.surname, person.surname.upper(), "Lalelelor", person.number, "SPCLEP"):
        assert secret not in blob
    assert privacy_guard(prof) == [] and privacy_guard(res) == []
    assert privacy_guard({**res, "fields_filled": ["cnp", person.surname]})    # a value in a key list is blocked
    det = pipe.detect_pii(res["filled_doc_id"])
    assert det["summary"]["CNP"] >= 1
    red = pipe.redact_image(res["filled_doc_id"], det["detection_id"])
    assert pipe.verify_redaction(red["result_id"])["status"] == "PASS"


def test_fill_requires_profile_and_distinct_docs(setup):
    pipe, _, id_doc, _ = setup
    with pytest.raises(ValueError):
        pipe.fill_form_from_id(id_doc, id_doc)
    with pytest.raises(KeyError):
        pipe.fill_form_from_id("doc_missing", id_doc)
