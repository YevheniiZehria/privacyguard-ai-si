"""Web terminal API tests: login, folders on disk, automatic redaction (requires Tesseract)."""
import io
import json
import re
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from server.pii_detector import cnp_control_digit
from ui import api as ui_api

CNP = "296101233123" + cnp_control_digit("296101233123")


@pytest.fixture
def anon(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_api, "STORE", ui_api.Store(tmp_path / "ws"))
    with TestClient(ui_api.app) as c:
        yield c


@pytest.fixture
def client(anon):
    r = anon.post("/api/login", json={"username": ui_api.UI_USERNAME, "passphrase": ui_api.UI_PASSPHRASE})
    assert r.status_code == 200
    return anon


def _doc_image() -> Image.Image:
    img = Image.new("RGB", (1500, 260), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("arial.ttf", 34)
    d.text((40, 40), f"Subsemnata Ionescu Maria Elena, CNP {CNP}", fill="black", font=f)
    d.text((40, 130), "domiciliata in Str. Florilor nr. 3, Iasi", fill="black", font=f)
    return img


def _png() -> bytes:
    buf = io.BytesIO()
    _doc_image().save(buf, format="PNG")
    return buf.getvalue()


def _pdf_two_pages() -> bytes:
    buf = io.BytesIO()
    a, b = _doc_image(), _doc_image()
    a.save(buf, "PDF", save_all=True, append_images=[b], resolution=300)   # renders back at 1500 px wide
    return buf.getvalue()


def _wait_idle(c, folder_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        docs = c.get(f"/api/folders/{folder_id}/documents").json()["documents"]
        if docs and not any(d["status"] in ("QUEUED", "SCANNING") for d in docs):
            return docs
        time.sleep(0.4)
    raise AssertionError("documents were not processed in time")


# ----------------------------------------------------------------------------- auth
def test_login_logout_and_protection(anon):
    assert anon.get("/api/session").json() == {"authenticated": False, "username": None}
    assert anon.get("/api/folders").status_code == 401
    assert anon.post("/api/folders", json={"name": "x"}).status_code == 401
    bad = anon.post("/api/login", json={"username": ui_api.UI_USERNAME, "passphrase": "wrong"})
    assert bad.status_code == 401 and "pg_session" not in bad.cookies

    r = anon.post("/api/login", json={"username": ui_api.UI_USERNAME, "passphrase": ui_api.UI_PASSPHRASE})
    assert r.status_code == 200 and r.json()["token"]
    assert "httponly" in r.headers["set-cookie"].lower()
    assert anon.get("/api/session").json()["authenticated"] is True
    assert anon.get("/api/folders").status_code == 200

    token = r.json()["token"]                      # bearer works too (cookie-less clients)
    fresh = TestClient(ui_api.app)
    assert fresh.get("/api/folders", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    assert anon.post("/api/logout").status_code == 200
    assert fresh.get("/api/folders", headers={"Authorization": f"Bearer {token}"}).status_code == 401


# ----------------------------------------------------------------------------- folders
def test_folder_crud_persists_on_disk(client):
    f = client.post("/api/folders", json={"name": "  Client_Set_01 "}).json()
    assert re.fullmatch(r"f_[0-9a-f]{8}", f["folder_id"]) and f["name"] == "Client_Set_01"
    meta_path = ui_api.STORE.root / f["folder_id"] / "metadata.json"
    assert json.loads(meta_path.read_text())["name"] == "Client_Set_01"

    listed = client.get("/api/folders").json()["folders"]
    assert [x["folder_id"] for x in listed] == [f["folder_id"]] and listed[0]["document_count"] == 0

    # a new Store over the same directory sees the folder (survives a restart)
    assert ui_api.Store(ui_api.STORE.root).list()[0]["name"] == "Client_Set_01"

    assert client.post("/api/folders", json={"name": "   "}).status_code == 422
    assert client.delete(f"/api/folders/{f['folder_id']}").status_code == 200
    assert not meta_path.parent.exists()
    assert client.get("/api/folders").json()["folders"] == []


def test_upload_redacts_automatically(client):
    f = client.post("/api/folders", json={"name": "Contracts"}).json()
    fid = f["folder_id"]
    r = client.post(f"/api/folders/{fid}/upload", files=[("files", ("scan one.png", _png(), "image/png")),
                                                         ("files", ("scan one.png", _png(), "image/png"))])
    assert r.status_code == 200
    added = r.json()["added"]
    assert len(added) == 2 and all(d["status"] == "QUEUED" for d in added)
    assert "stored" not in added[0] and "pipeline_doc_id" not in added[0]       # no disk paths / internal ids

    docs = _wait_idle(client, fid)
    assert all(d["status"] == "BLURRED" and d["boxes_drawn"] >= 1 and d["residual_pii_count"] == 0 for d in docs)

    did = docs[0]["doc_id"]
    for kind in ("original", "blurred"):
        img = client.get(f"/api/folders/{fid}/documents/{did}/image/{kind}")
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    dl = client.get(f"/api/folders/{fid}/documents/{did}/download")
    assert dl.headers["content-type"] == "image/png" and "scan_one_blurred.png" in dl.headers["content-disposition"]

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/folders/{fid}/download-all").content))
    assert sorted(z.namelist()) == ["scan_one_blurred.png", "scan_one_blurred_2.png"]   # duplicate names deduped

    # raw PII never lands in metadata or API responses
    blob = (ui_api.STORE.root / fid / "metadata.json").read_text() + json.dumps(client.get("/api/folders").json()) \
        + json.dumps(client.get(f"/api/folders/{fid}/documents").json())
    assert CNP not in blob and "Ionescu" not in blob and "Florilor" not in blob

    hub = client.get("/api/folders").json()["folders"][0]
    assert hub["document_count"] == 2 and hub["blurred_count"] == 2

    # manual blur box: changes the redacted page, is stored, survives a rescan, and can be undone
    before = client.get(f"/api/folders/{fid}/documents/{did}/image/blurred").content
    m = client.post(f"/api/folders/{fid}/documents/{did}/manual-redact", json={"page": 1, "x": 900, "y": 150, "w": 300, "h": 80})
    assert m.status_code == 200 and m.json()["manual_boxes"] == [[1, 900, 150, 300, 80]]
    assert client.get(f"/api/folders/{fid}/documents/{did}/image/blurred").content != before
    assert client.post(f"/api/folders/{fid}/documents/{did}/manual-redact",
                       json={"page": 9, "x": 0, "y": 0, "w": 10, "h": 10}).status_code == 400

    rescan = client.post(f"/api/folders/{fid}/documents/{did}/rescan")
    assert rescan.status_code == 200 and rescan.json()["status"] == "QUEUED"
    again = next(d for d in _wait_idle(client, fid) if d["doc_id"] == did)
    assert again["status"] == "BLURRED" and again["boxes_drawn"] >= 1 and again["manual_boxes"]
    undo = client.delete(f"/api/folders/{fid}/documents/{did}/manual-redact")
    assert undo.status_code == 200 and undo.json()["status"] == "QUEUED"
    again = next(d for d in _wait_idle(client, fid) if d["doc_id"] == did)
    assert again["manual_boxes"] == [] and again["status"] == "BLURRED"

    assert client.delete(f"/api/folders/{fid}/documents/{did}").status_code == 200
    assert len(client.get(f"/api/folders/{fid}/documents").json()["documents"]) == 1


def test_multipage_pdf_downloads_as_pdf(client):
    fid = client.post("/api/folders", json={"name": "PDFs"}).json()["folder_id"]
    client.post(f"/api/folders/{fid}/upload", files=[("files", ("two pages.pdf", _pdf_two_pages(), "application/pdf"))])
    doc = _wait_idle(client, fid)[0]
    assert doc["status"] == "BLURRED" and doc["page_count"] == 2
    assert client.get(f"/api/folders/{fid}/documents/{doc['doc_id']}/image/blurred?page=2").status_code == 200
    assert client.get(f"/api/folders/{fid}/documents/{doc['doc_id']}/image/blurred?page=3").status_code == 404
    dl = client.get(f"/api/folders/{fid}/documents/{doc['doc_id']}/download")
    assert dl.headers["content-type"] == "application/pdf" and dl.content.startswith(b"%PDF")


def test_rejects_bad_input(client):
    fid = client.post("/api/folders", json={"name": "Inbox"}).json()["folder_id"]
    assert client.post(f"/api/folders/{fid}/upload",
                       files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))]).status_code == 400
    assert not list((ui_api.STORE.root / fid / "original").iterdir())         # nothing written
    assert client.get("/api/folders/../../etc/documents").status_code == 404
    assert client.get("/api/folders/f_00000000/documents").status_code == 404
    assert client.get(f"/api/folders/{fid}/documents/d_0000000000/download").status_code == 404
    assert client.get(f"/api/folders/{fid}/documents/nope/image/original").status_code == 404
    assert client.get(f"/api/folders/{fid}/download-all").status_code == 404   # nothing redacted yet


def test_safe_upload_names():
    assert ui_api._safe_name("../../etc/passwd") == "passwd"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", ui_api._safe_name("Buletin Ion Popescu (1).png"))


def test_index_is_public(anon):
    page = anon.get("/").text
    assert "PrivacyGuard" in page and "AUTHENTICATE_SESSION" in page


# ----------------------------------------------------------------------------- blur calibration
def test_calibration_persists_and_is_inherited(client):
    fid = client.post("/api/folders", json={"name": "Calib"}).json()["folder_id"]
    got = client.get(f"/api/folders/{fid}/calibration").json()
    assert got["calibration"] == {"blur_level": 7, "padding_px": 4, "mode": "blur", "deep_scan": False} and got["presets"]["maximum"] == 10

    r = client.put(f"/api/folders/{fid}/calibration", json={"blur_level": 9, "padding_px": 12, "mode": "blur"})
    assert r.status_code == 200
    meta = json.loads((ui_api.STORE.root / fid / "metadata.json").read_text())
    assert meta["calibration"] == {"blur_level": 9, "padding_px": 12, "mode": "blur", "deep_scan": False}
    for bad in ({"blur_level": 11}, {"blur_level": 0}, {"padding_px": 30}, {"mode": "pixel"}):
        assert client.put(f"/api/folders/{fid}/calibration", json=bad).status_code == 422

    client.post(f"/api/folders/{fid}/upload", files=[("files", ("a.png", _png(), "image/png"))])
    doc = _wait_idle(client, fid)[0]
    assert doc["calibration"] == {"blur_level": 9, "padding_px": 12, "mode": "blur", "deep_scan": False}   # inherited by new uploads
    assert doc["status"] == "BLURRED"


def test_reapply_without_ocr_updates_preview_and_status(client, monkeypatch):
    fid = client.post("/api/folders", json={"name": "Reapply"}).json()["folder_id"]
    client.post(f"/api/folders/{fid}/upload", files=[("files", ("a.png", _png(), "image/png")),
                                                     ("files", ("b.png", _png(), "image/png"))])
    docs = _wait_idle(client, fid)
    did = docs[0]["doc_id"]
    meta = json.loads((ui_api.STORE.root / fid / "metadata.json").read_text())
    pdoc = meta["documents"][0]["pipeline_doc_id"]
    plan = ui_api.STORE.root / fid / "out" / "work" / f"{pdoc}_plan.json"
    assert plan.is_file() and CNP not in plan.read_text() and "Ionescu" not in plan.read_text()
    before = client.get(f"/api/folders/{fid}/documents/{did}/image/blurred").content

    from server import pipeline as pl
    def no_ocr(*_a, **_k):
        raise AssertionError("re-apply must not run OCR again")
    monkeypatch.setattr(pl.PrivacyPipeline, "ocr_document", no_ocr)

    r = client.post(f"/api/folders/{fid}/documents/{did}/reapply",
                    json={"calibration": {"blur_level": 10, "padding_px": 16, "mode": "blur"}})
    assert r.status_code == 200 and r.json()["status"] == "QUEUED"
    doc = next(d for d in _wait_idle(client, fid) if d["doc_id"] == did)
    assert doc["status"] == "BLURRED" and doc["residual_pii_count"] == 0
    assert doc["calibration"] == {"blur_level": 10, "padding_px": 16, "mode": "blur", "deep_scan": False}
    assert client.get(f"/api/folders/{fid}/documents/{did}/image/blurred").content != before

    # solid black mode on the whole folder, saved as the new default
    r = client.post(f"/api/folders/{fid}/reapply",
                    json={"calibration": {"blur_level": 5, "padding_px": 6, "mode": "black"}, "save_as_default": True})
    assert r.status_code == 200 and r.json()["queued"] == 2
    docs = _wait_idle(client, fid)
    assert all(d["status"] == "BLURRED" and d["calibration"]["mode"] == "black" for d in docs)
    from server.image_preprocessing import load_bgr
    red = load_bgr(str(ui_api.STORE.root / fid / "out" / f"redacted_{pdoc}.png"))
    assert (red.max(axis=2) == 0).mean() > 0.05                     # solid black boxes are drawn
    assert client.get(f"/api/folders/{fid}/calibration").json()["calibration"]["mode"] == "black"

    assert client.post(f"/api/folders/{fid}/documents/{did}/reapply",
                       json={"calibration": {"blur_level": 11}}).status_code == 422
    blob = (ui_api.STORE.root / fid / "metadata.json").read_text() + r.text \
        + json.dumps(client.get(f"/api/folders/{fid}/documents").json())
    assert CNP not in blob and "Ionescu" not in blob and "Florilor" not in blob


def test_reapply_rejects_unfinished_documents(client):
    fid = client.post("/api/folders", json={"name": "Busy"}).json()["folder_id"]
    with ui_api.STORE.lock:                          # a document that has not been redacted yet
        meta = ui_api.STORE.load(fid)
        meta["documents"].append({"doc_id": "d_0123456789", "name": "x.png", "stored": "x.png", "size": 1,
                                  "created": 0, "status": "ERROR", "pipeline_doc_id": None, "page_count": None,
                                  "boxes_drawn": None, "residual_pii_count": None, "processed": None, "error": "x"})
        ui_api.STORE.save(meta)
    assert client.post(f"/api/folders/{fid}/documents/d_0123456789/reapply", json={}).status_code == 409
    assert client.post(f"/api/folders/{fid}/reapply", json={}).json()["queued"] == 0


# ----------------------------------------------------------------------------- auto-fill from ID card
def _id_card_png() -> tuple[bytes, object]:
    import random
    from faker import Faker
    from evaluation.generate_synthetic_docs import id_card, make_people, render
    rng = random.Random(7)
    fake = Faker("ro_RO")
    fake.seed_instance(7)
    p = make_people(rng, fake)[0]
    lines = id_card(p)
    img, _ = render(lines, "arial", size=32, width=1500, line_gap=1.75, mrz_from=len(lines) - 2, border=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), p


def _blank_form_png() -> bytes:
    f = ImageFont.truetype("arial.ttf", 30)
    img = Image.new("RGB", (1700, 400), "white")
    d = ImageDraw.Draw(img)
    for y, label, length in ((60, "Subsemnatul", 700), (150, "CNP", 420), (240, "Telefon", 400)):
        d.text((60, y), label, fill=(20, 20, 20), font=f)
        x = d.textbbox((60, y), label, font=f)[2] + 12
        d.line((x, y + 34, x + length, y + 34), fill=(20, 20, 20), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_autofill_from_id_card(client):
    fid = client.post("/api/folders", json={"name": "Autofill"}).json()["folder_id"]
    id_png, person = _id_card_png()
    r = client.post(f"/api/folders/{fid}/upload?source=1", files=[("files", ("buletin.png", id_png, "image/png")),
                                                                 ("files", ("cerere.png", _blank_form_png(), "image/png"))])
    added = r.json()["added"]
    assert [d["status"] for d in added] == ["SOURCE", "SOURCE"]          # stored, not queued for redaction
    time.sleep(0.5)
    docs = client.get(f"/api/folders/{fid}/documents").json()["documents"]
    assert all(d["status"] == "SOURCE" for d in docs)
    id_doc, form_doc = added[0]["doc_id"], added[1]["doc_id"]

    assert client.post(f"/api/folders/{fid}/autofill", json={"id_doc_id": id_doc, "form_doc_id": id_doc}).status_code == 422
    assert client.post(f"/api/folders/{fid}/autofill", json={"id_doc_id": id_doc, "form_doc_id": form_doc,
                                                             "extras": {"cnp": "x"}}).status_code == 422
    r = client.post(f"/api/folders/{fid}/autofill", json={"id_doc_id": id_doc, "form_doc_id": form_doc,
                                                          "extras": {"phone": "0745 999 888"}})
    assert r.status_code == 200 and r.json()["status"] == "QUEUED" and r.json()["role"] == "filled"
    filled = next(d for d in _wait_idle(client, fid) if d["doc_id"] == r.json()["doc_id"])
    assert filled["status"] == "FILLED", filled.get("error")
    assert {"full_name", "cnp", "phone"} <= set(filled["filled_fields"])

    # metadata / API: field keys only — no names, CNP, series, nor the typed phone number
    meta_text = (ui_api.STORE.root / fid / "metadata.json").read_text(encoding="utf-8")
    blob = meta_text + json.dumps(client.get(f"/api/folders/{fid}/documents").json())
    for secret in (person.cnp, person.surname, person.surname.upper(), "Lalelelor", person.number, "0745 999 888"):
        assert secret not in blob

    did = filled["doc_id"]
    for kind in ("original", "filled"):
        assert client.get(f"/api/folders/{fid}/documents/{did}/image/{kind}").status_code == 200
    dl = client.get(f"/api/folders/{fid}/documents/{did}/download")
    assert dl.status_code == 200 and dl.headers["content-type"] == "image/png"
    assert "cerere_completat.png" in dl.headers["content-disposition"]
    assert client.get(f"/api/folders/{fid}/download-all").status_code == 404   # FILLED is never in the blurred zip

    copy = client.post(f"/api/folders/{fid}/documents/{did}/anonymize-copy")
    assert copy.status_code == 200 and copy.json()["status"] == "QUEUED"
    anon = next(d for d in _wait_idle(client, fid) if d["doc_id"] == copy.json()["doc_id"])
    assert anon["status"] == "BLURRED" and anon["boxes_drawn"] >= 2
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/folders/{fid}/download-all").content))
    assert z.namelist() == ["cerere_completat_anonim_blurred.png"]

    # deleting the filled form keeps the blank form's upload; a source can still be redacted on demand
    assert client.delete(f"/api/folders/{fid}/documents/{did}").status_code == 200
    assert (ui_api.STORE.root / fid / "original").joinpath(added[1]["name"]).exists() is False   # names are prefixed
    assert len(list((ui_api.STORE.root / fid / "original").glob(f"{form_doc}_*"))) == 1
    red = client.post(f"/api/folders/{fid}/documents/{form_doc}/redact")
    assert red.status_code == 200 and red.json()["status"] == "QUEUED"
    assert next(d for d in _wait_idle(client, fid) if d["doc_id"] == form_doc)["status"] == "BLURRED"
