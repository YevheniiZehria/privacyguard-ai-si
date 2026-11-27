"""PrivacyGuard web terminal — FastAPI backend.

Run:  python -m ui.api            (opens http://127.0.0.1:8000)
      python -m ui.api --port 8010 --no-browser

Flow: login -> folder hub -> folder dashboard. Every document dropped into a folder is queued and run through
the unchanged server pipeline (OCR -> PII detection -> irreversible redaction -> verification) by one background
worker. Folders persist on disk:

    workspace_folders/<folder_id>/metadata.json   folder name, timestamps, per-document status and counts
                                  original/        uploaded files (deleted with the folder)
                                  out/             pipeline output: work/ (deskewed pages) + redacted_*.png

metadata.json holds ids, statuses and counts only — never OCR text or PII values (those stay in the in-memory
vault and are cleared after each document). The server binds to 127.0.0.1 and every /api route except login and
session requires a session token (HttpOnly cookie, or `Authorization: Bearer`).
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
import webbrowser
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import config  # noqa: F401  (loads .env)

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".pdf"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PADDING_PX = 4

UI_USERNAME = os.getenv("UI_USERNAME", "").strip() or "admin"
UI_PASSPHRASE = os.getenv("UI_PASSPHRASE", "").strip() or "privacyguard"
SESSION_COOKIE = "pg_session"
SESSION_TTL_S = 12 * 3600

FOLDER_RE = re.compile(r"^f_[0-9a-f]{8}$")
DOC_RE = re.compile(r"^d_[0-9a-f]{10}$")
ACTIVE = ("QUEUED", "SCANNING")
FINISHED = ("BLURRED", "REVIEW")      # REVIEW = redacted, but the verification pass still found PII-like text


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:80]
    return stem or "document"


# ----------------------------------------------------------------------------- sessions
class Sessions:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def create(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._tokens = {t: v for t, v in self._tokens.items() if v[1] > now}
            self._tokens[token] = (username, now + SESSION_TTL_S)
        return token

    def user(self, token: str | None) -> str | None:
        if not token:
            return None
        with self._lock:
            entry = self._tokens.get(token)
            if not entry or entry[1] < time.time():
                self._tokens.pop(token, None)
                return None
            return entry[0]

    def drop(self, token: str | None) -> None:
        with self._lock:
            self._tokens.pop(token or "", None)


SESSIONS = Sessions()


def _token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(SESSION_COOKIE)


def require_session(request: Request) -> str:
    user = SESSIONS.user(_token(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


# ----------------------------------------------------------------------------- disk store
class Store:
    """Folders on disk. All mutations of metadata.json go through one lock and atomic replace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    # paths -------------------------------------------------------------
    def folder_dir(self, folder_id: str) -> Path:
        if not FOLDER_RE.match(folder_id):
            raise HTTPException(404, "Unknown folder")
        d = self.root / folder_id
        if not (d / "metadata.json").is_file():
            raise HTTPException(404, "Unknown folder")
        return d

    def original_path(self, folder_id: str, doc: dict) -> Path:
        return self.root / folder_id / "original" / doc["stored"]

    def page_paths(self, folder_id: str, doc: dict, kind: Literal["original", "blurred"]) -> list[Path]:
        pdoc, n = doc.get("pipeline_doc_id"), doc.get("page_count") or 0
        if not pdoc:
            return []
        out = self.root / folder_id / "out"
        if kind == "original":                      # deskewed page = the exact frame the redaction was drawn on
            paths = [out / "work" / f"{pdoc}_p{i}.png" for i in range(1, n + 1)]
        else:
            paths = [out / f"redacted_{pdoc}{'' if i == 1 else f'_p{i}'}.png" for i in range(1, n + 1)]
        return [p for p in paths if p.is_file()]

    # metadata ----------------------------------------------------------
    def load(self, folder_id: str) -> dict:
        d = self.folder_dir(folder_id)
        return json.loads((d / "metadata.json").read_text(encoding="utf-8"))

    def save(self, meta: dict) -> None:
        d = self.root / meta["folder_id"]
        tmp = d / "metadata.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, d / "metadata.json")

    def list(self) -> list[dict]:
        out = []
        with self.lock:
            for d in sorted(self.root.iterdir()):
                if d.is_dir() and FOLDER_RE.match(d.name) and (d / "metadata.json").is_file():
                    out.append(folder_summary(self.load(d.name)))
        return sorted(out, key=lambda f: f["created"], reverse=True)

    def create(self, name: str) -> dict:
        folder_id = f"f_{uuid.uuid4().hex[:8]}"
        d = self.root / folder_id
        (d / "original").mkdir(parents=True)
        (d / "out").mkdir()
        meta = {"folder_id": folder_id, "name": name, "created": time.time(), "documents": []}
        with self.lock:
            self.save(meta)
        return meta

    def delete(self, folder_id: str) -> None:
        with self.lock:
            d = self.folder_dir(folder_id)
            shutil.rmtree(d, ignore_errors=True)

    def update_doc(self, folder_id: str, doc_id: str, **fields) -> dict | None:
        """Returns the updated document, or None if the folder/document disappeared meanwhile."""
        with self.lock:
            try:
                meta = self.load(folder_id)
            except HTTPException:
                return None
            doc = next((x for x in meta["documents"] if x["doc_id"] == doc_id), None)
            if doc is None:
                return None
            doc.update(fields)
            self.save(meta)
            return doc


def folder_summary(meta: dict) -> dict:
    docs = meta["documents"]
    return {"folder_id": meta["folder_id"], "name": meta["name"], "created": meta["created"],
            "document_count": len(docs),
            "blurred_count": sum(1 for d in docs if d["status"] in FINISHED),
            "processing_count": sum(1 for d in docs if d["status"] in ACTIVE)}


def public_doc(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in ("stored", "pipeline_doc_id")}


STORE = Store(Path(os.getenv("UI_WORKSPACE", "").strip() or ROOT / "workspace_folders"))


# ----------------------------------------------------------------------------- worker
JOBS: queue.Queue[tuple[str, str]] = queue.Queue()
_worker_started = threading.Event()


def process_document(folder_id: str, doc_id: str, pipe) -> None:
    store = STORE
    try:
        meta = store.load(folder_id)
    except HTTPException:
        return
    doc = next((x for x in meta["documents"] if x["doc_id"] == doc_id), None)
    if not doc or doc["status"] != "QUEUED":
        return
    store.update_doc(folder_id, doc_id, status="SCANNING", error=None)
    pipe.output_dir = store.root / folder_id / "out"
    try:
        ocr = pipe.ocr_document(str(store.original_path(folder_id, doc)))
        det = pipe.detect_pii(ocr["doc_id"])
        red = pipe.redact_image(ocr["doc_id"], det["detection_id"], PADDING_PX)
        ver = pipe.verify_redaction(red["result_id"])
        store.update_doc(folder_id, doc_id, status="BLURRED" if ver["verified"] else "REVIEW",
                         pipeline_doc_id=ocr["doc_id"], page_count=ocr["page_count"],
                         boxes_drawn=red["boxes_drawn"], residual_pii_count=ver["residual_pii_count"],
                         processed=time.time())
    except Exception as exc:                       # pipeline errors never contain document text
        store.update_doc(folder_id, doc_id, status="ERROR", error=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        pipe.vault.clear()                         # raw OCR text / PII values do not outlive the job
        if not (store.root / folder_id / "metadata.json").exists():
            shutil.rmtree(store.root / folder_id, ignore_errors=True)   # folder deleted mid-scan


def worker_loop() -> None:
    from server.pipeline import PrivacyPipeline
    pipe = PrivacyPipeline()
    while True:
        folder_id, doc_id = JOBS.get()
        try:
            process_document(folder_id, doc_id, pipe)
        finally:
            JOBS.task_done()


def start_worker() -> None:
    if _worker_started.is_set():
        return
    _worker_started.set()
    threading.Thread(target=worker_loop, name="redaction-worker", daemon=True).start()


def requeue_interrupted() -> None:
    """Documents caught mid-scan by a restart start over (the vault is in-memory)."""
    for f in STORE.list():
        if not f["processing_count"]:
            continue
        for doc in STORE.load(f["folder_id"])["documents"]:
            if doc["status"] in ACTIVE:
                STORE.update_doc(f["folder_id"], doc["doc_id"], status="QUEUED")
                JOBS.put((f["folder_id"], doc["doc_id"]))


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_worker()
    requeue_interrupted()
    yield


app = FastAPI(title="PrivacyGuard AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    passphrase: str = Field(..., max_length=256)


@app.post("/api/login")
def login(req: LoginRequest) -> JSONResponse:
    ok_user = hmac.compare_digest(req.username.strip().encode(), UI_USERNAME.encode())
    ok_pass = hmac.compare_digest(req.passphrase.encode(), UI_PASSPHRASE.encode())
    if not (ok_user and ok_pass):
        time.sleep(0.4)                            # slow down guessing
        raise HTTPException(401, "Access denied")
    token = SESSIONS.create(UI_USERNAME)
    resp = JSONResponse({"token": token, "username": UI_USERNAME, "expires_in": SESSION_TTL_S})
    # Lax, not Strict: Strict cookies are withheld from every request of a page that was opened from another
    # site/app, which logs the user out on each such visit. Lax still blocks them on cross-site POST/DELETE.
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL_S, httponly=True, samesite="lax", path="/")
    return resp


@app.post("/api/logout")
def logout(request: Request) -> JSONResponse:
    SESSIONS.drop(_token(request))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/session")
def session(request: Request) -> dict:
    user = SESSIONS.user(_token(request))
    return {"authenticated": bool(user), "username": user}


api = APIRouter(prefix="/api", dependencies=[Depends(require_session)])


class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


@api.get("/folders")
def list_folders() -> dict:
    return {"folders": STORE.list()}


@api.post("/folders")
def create_folder(req: FolderCreate) -> dict:
    name = re.sub(r"\s+", " ", req.name).strip()
    if not name:
        raise HTTPException(422, "Folder name is empty")
    return folder_summary(STORE.create(name))


@api.delete("/folders/{folder_id}")
def delete_folder(folder_id: str) -> dict:
    STORE.delete(folder_id)
    return {"ok": True}


@api.get("/folders/{folder_id}/documents")
def list_documents(folder_id: str) -> dict:
    with STORE.lock:
        meta = STORE.load(folder_id)
    return {"folder": folder_summary(meta), "documents": [public_doc(d) for d in meta["documents"]]}


@api.post("/folders/{folder_id}/upload")
async def upload(folder_id: str, files: list[UploadFile] = File(...)) -> dict:
    d = STORE.folder_dir(folder_id)
    payloads = []
    for up in files:                               # validate everything before writing anything
        ext = Path(up.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG")
        data = await up.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{up.filename}: larger than 25 MB")
        if not data:
            raise HTTPException(400, f"{up.filename}: empty file")
        payloads.append((Path(up.filename or f"document{ext}").name, data))
    added = []
    with STORE.lock:
        meta = STORE.load(folder_id)
        for name, data in payloads:
            doc_id = f"d_{uuid.uuid4().hex[:10]}"
            stored = f"{doc_id}_{_safe_name(name)}"
            (d / "original").mkdir(exist_ok=True)
            (d / "original" / stored).write_bytes(data)
            doc = {"doc_id": doc_id, "name": name, "stored": stored, "size": len(data), "created": time.time(),
                   "status": "QUEUED", "pipeline_doc_id": None, "page_count": None, "boxes_drawn": None,
                   "residual_pii_count": None, "processed": None, "error": None}
            meta["documents"].append(doc)
            added.append(doc)
        STORE.save(meta)
    for doc in added:
        JOBS.put((folder_id, doc["doc_id"]))
    return {"added": [public_doc(x) for x in added]}


def _doc(folder_id: str, doc_id: str) -> dict:
    if not DOC_RE.match(doc_id):
        raise HTTPException(404, "Unknown document")
    with STORE.lock:
        meta = STORE.load(folder_id)
    doc = next((x for x in meta["documents"] if x["doc_id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Unknown document")
    return doc


@api.delete("/folders/{folder_id}/documents/{doc_id}")
def delete_document(folder_id: str, doc_id: str) -> dict:
    with STORE.lock:
        doc = _doc(folder_id, doc_id)
        meta = STORE.load(folder_id)
        meta["documents"] = [x for x in meta["documents"] if x["doc_id"] != doc_id]
        STORE.save(meta)
        paths = [STORE.original_path(folder_id, doc), *STORE.page_paths(folder_id, doc, "original"),
                 *STORE.page_paths(folder_id, doc, "blurred")]
        if doc.get("pipeline_doc_id"):
            paths += list((STORE.root / folder_id / "out" / "work").glob(f"{doc['pipeline_doc_id']}_*"))
        for p in paths:
            p.unlink(missing_ok=True)
    return {"ok": True}


@api.get("/folders/{folder_id}/documents/{doc_id}/image/{kind}")
def document_image(folder_id: str, doc_id: str, kind: Literal["original", "blurred"],
                   page: int = Query(1, ge=1)) -> FileResponse:
    doc = _doc(folder_id, doc_id)
    pages = STORE.page_paths(folder_id, doc, kind)
    if not pages and kind == "original":           # not scanned yet: show the upload itself if it is an image
        src = STORE.original_path(folder_id, doc)
        if src.suffix.lower() in {".png", ".jpg", ".jpeg"} and src.is_file():
            pages = [src]
    if page > len(pages):
        raise HTTPException(404, "Image not available")
    return FileResponse(pages[page - 1], headers={"Cache-Control": "no-store"})


def _export(folder_id: str, doc: dict) -> tuple[str, bytes]:
    """(file name, bytes): one page -> PNG, several pages -> a single PDF."""
    pages = STORE.page_paths(folder_id, doc, "blurred")
    if doc["status"] not in FINISHED or not pages:
        raise HTTPException(404, "Document not redacted yet")
    stem = _safe_name(Path(doc["name"]).stem)
    if len(pages) == 1:
        return f"{stem}_blurred.png", pages[0].read_bytes()
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in pages]
    buf = io.BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=300)
    return f"{stem}_blurred.pdf", buf.getvalue()


@api.get("/folders/{folder_id}/documents/{doc_id}/download")
def download_document(folder_id: str, doc_id: str) -> Response:
    name, data = _export(folder_id, _doc(folder_id, doc_id))
    media = "application/pdf" if name.endswith(".pdf") else "image/png"
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"',
                                                     "Cache-Control": "no-store"})


@api.get("/folders/{folder_id}/download-all")
def download_all(folder_id: str) -> Response:
    with STORE.lock:
        meta = STORE.load(folder_id)
    buf = io.BytesIO()
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in meta["documents"]:
            if doc["status"] not in FINISHED:
                continue
            name, data = _export(folder_id, doc)
            base, ext = os.path.splitext(name)
            i = 2
            while name in used:
                name, i = f"{base}_{i}{ext}", i + 1
            used.add(name)
            zf.writestr(name, data)
    if not used:
        raise HTTPException(404, "No redacted documents in this folder yet")
    fname = f"{_safe_name(meta['name'])}_blurred.zip"
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"', "Cache-Control": "no-store"})


app.include_router(api)


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="PrivacyGuard web terminal")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"PrivacyGuard terminal on {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
