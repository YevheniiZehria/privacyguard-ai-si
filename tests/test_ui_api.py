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

    rescan = client.post(f"/api/folders/{fid}/documents/{did}/rescan")
    assert rescan.status_code == 200 and rescan.json()["status"] == "QUEUED"
    again = next(d for d in _wait_idle(client, fid) if d["doc_id"] == did)
    assert again["status"] == "BLURRED" and again["boxes_drawn"] >= 1

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
