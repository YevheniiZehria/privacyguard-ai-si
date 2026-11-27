"""Web dashboard API tests (direct mode; requires Tesseract)."""
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


@pytest.fixture
def client():
    with TestClient(ui_api.app) as c:
        c.post("/api/reset")
        yield c
        c.post("/api/reset")


def _doc_png(cnp: str) -> bytes:
    img = Image.new("RGB", (1500, 260), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("arial.ttf", 34)
    d.text((40, 40), f"Subsemnata Ionescu Maria Elena, CNP {cnp}", fill="black", font=f)
    d.text((40, 130), "domiciliata in Str. Florilor nr. 3, Iasi", fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wait_for_job(c, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get("/api/state").json()
        if not st["job"]["running"]:
            return st
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_upload_process_download_reset(client):
    cnp = "296101233123" + cnp_control_digit("296101233123")
    r = client.post("/api/upload", files=[("files", ("scan one.png", _doc_png(cnp), "image/png")),
                                          ("files", ("scan two.png", _doc_png(cnp), "image/png"))])
    assert r.status_code == 200
    added = r.json()["added"]
    assert len(added) == 2 and "path" not in added[0]          # absolute paths never reach the page

    assert client.post("/api/process", json={"mode": "direct", "padding_px": 6}).status_code == 200
    st = _wait_for_job(client)
    files = st["files"]
    assert all(f["status"] == "VERIFIED PASS" for f in files)
    assert all(f["summary"]["CNP"] == 1 and f["has_after"] for f in files)
    assert st["links"]["clusters"] and len(st["links"]["clusters"][0]["doc_ids"]) == 2

    # every event that crossed the boundary is free of raw PII
    blob = json.dumps(st["events"]) + json.dumps(st["files"]) + json.dumps(st["links"])
    assert cnp not in blob and "Ionescu" not in blob and "Florilor" not in blob
    tool_events = [e for e in st["events"] if e["type"] == "tool"]
    assert len(tool_events) == 9 and all(e["pii_pattern_hits"] == 0 for e in tool_events)

    fid = files[0]["file_id"]
    assert client.get(f"/api/image/{fid}/before").headers["content-type"] == "image/png"
    assert client.get(f"/api/image/{fid}/after").status_code == 200
    assert "attachment" in client.get(f"/api/download/{fid}").headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(client.get("/api/download-all").content))
    names = z.namelist()
    assert "manifest.json" in names and sum(n.endswith(".png") for n in names) == 2
    assert cnp not in z.read("manifest.json").decode()

    uploaded = [ui_api.STATE.files[f["file_id"]].path for f in files]
    assert client.post("/api/reset").status_code == 200
    assert not any(__import__("os").path.exists(p) for p in uploaded)   # uploads are deleted on reset


def test_rejects_bad_uploads_and_unknown_ids(client):
    r = client.post("/api/upload", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
    assert client.get("/api/image/nope/after").status_code == 404
    assert client.post("/api/process", json={"mode": "direct"}).status_code == 400      # empty queue
    assert client.post("/api/process", json={"mode": "direct", "padding_px": 99}).status_code == 422


def test_safe_upload_names():
    assert ui_api._safe_name("../../etc/passwd") == "passwd"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", ui_api._safe_name("Buletin Ion Popescu (1).png"))


def test_index_and_config(client):
    assert "PrivacyGuard" in client.get("/").text
    cfg = client.get("/api/config").json()
    assert cfg["providers"]["direct"]["available"] is True
    assert set(cfg["pii_types"]) >= {"CNP", "MRZ"}
