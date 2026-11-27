"""PrivacyGuard web dashboard — FastAPI backend.

Run:  python -m ui.api            (opens http://127.0.0.1:8000)
      python -m ui.api --port 8010 --no-browser

Two execution modes:
  * direct  — calls server/pipeline.py in-process (fast, deterministic, no LLM)
  * ollama / gemini — runs the real autonomous agent (agent/cli_agent.py), which spawns the MCP server
    over stdio; every tool call is streamed to the browser together with the exact text the LLM received.

The server binds to 127.0.0.1 only. Uploaded documents live in ui/uploads/ (git-ignored) and are deleted
on reset. Images are served only from known paths inside ui/uploads/, synthetic_data/ and output/.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import config

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = Path(__file__).resolve().parent / "uploads"
SYNTHETIC = ROOT / "synthetic_data"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".pdf"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PII_TYPES = ["CNP", "PERSON_NAME", "ADDRESS", "ID_SERIES", "IBAN", "PHONE", "EMAIL", "BIRTH_DATE", "MRZ"]
INBOX_ID = "f_inbox"
Policy = Literal["full", "pii"]


# ----------------------------------------------------------------------------- state
@dataclass
class Folder:
    """A document set. policy=full blurs every page entirely; policy=pii runs the targeted redaction pipeline.
    auto=True processes new documents as soon as they are added to the folder."""
    folder_id: str
    name: str
    policy: Policy = "full"
    auto: bool = True
    links: dict | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return asdict(self)


@dataclass
class FileEntry:
    file_id: str
    name: str
    path: str
    source: Literal["upload", "demo"]
    folder_id: str = INBOX_ID
    status: str = "QUEUED"          # QUEUED | OCR DONE | PII DETECTED | REDACTED | VERIFIED PASS | VERIFIED FAIL | ERROR
    doc_id: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    avg_confidence: float | None = None
    image_dimensions: list[int] | None = None
    summary: dict[str, int] = field(default_factory=dict)
    total_boxes: int | None = None
    boxes_drawn: int | None = None
    result_id: str | None = None
    residual_pii_count: int | None = None
    error: str | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")                       # never expose absolute local paths to the page
        d["has_before"] = self.before_path() is not None
        d["has_after"] = self.after_path() is not None
        return d

    def before_path(self) -> Path | None:
        if self.doc_id:
            p = config.OUTPUT_DIR / "work" / f"{self.doc_id}_p1.png"     # deskewed page = redaction frame
            if p.exists():
                return p
        p = Path(self.path)
        return p if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.exists() else None

    def after_path(self) -> Path | None:
        if self.doc_id:
            p = config.OUTPUT_DIR / f"redacted_{self.doc_id}.png"
            if p.exists():
                return p
        return None


class Hub:
    """Thread-safe event bus: worker threads publish, SSE subscribers consume."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.history: deque[dict] = deque(maxlen=3000)
        self.lock = threading.Lock()
        self.seq = 0

    def publish(self, event: dict) -> None:
        with self.lock:
            self.seq += 1
            event = {"seq": self.seq, "ts": time.time(), **event}
            self.history.append(event)
        if self.loop is None:
            return
        for q in list(self.subscribers):
            self.loop.call_soon_threadsafe(q.put_nowait, event)


class AppState:
    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.folders: dict[str, Folder] = {}
        self.links: dict | None = None
        self.job: dict = {"running": False, "mode": None, "folder_id": None, "started": None, "finished": None,
                          "stats": None}
        self.settings: dict = {"mode": "direct", "padding_px": 4, "use_local_slm": False, "tts": False}
        self.pending: list[str] = []          # folders waiting for an automatic run
        self.lock = threading.RLock()
        self.hub = Hub()
        self.reset_folders()

    def reset_folders(self) -> None:
        self.folders = {INBOX_ID: Folder(INBOX_ID, "INBOX", policy="pii", auto=False, created=0.0)}
        self.pending.clear()

    def by_doc(self, doc_id: str) -> FileEntry | None:
        return next((f for f in self.files.values() if f.doc_id == doc_id), None)

    def scoped(self, folder_id: str | None) -> list[FileEntry]:
        return [f for f in self.files.values() if folder_id is None or f.folder_id == folder_id]

    def emit_file(self, entry: FileEntry) -> None:
        self.hub.publish({"type": "file", "file": entry.public()})

    def emit_folders(self) -> None:
        self.hub.publish({"type": "folders", "folders": [f.public() for f in self.folders.values()]})

    def log(self, message: str, level: str = "info") -> None:
        self.hub.publish({"type": "log", "level": level, "message": message})


STATE = AppState()


# ----------------------------------------------------------------------------- privacy helpers
def scan_for_pii(text: str) -> list[str]:
    """Same patterns the agent's client-side privacy guard uses."""
    from agent.cli_agent import LEAK_PATTERNS
    return [p.pattern for p in LEAK_PATTERNS if p.search(text)]


def boundary_event(step: int, tool: str, args: dict, result: dict, llm_content: str | None, mode: str) -> dict:
    content = llm_content if llm_content is not None else json.dumps(result, ensure_ascii=False)
    hits = scan_for_pii(json.dumps(args, ensure_ascii=False) + content)
    return {"type": "tool", "mode": mode, "step": step, "tool": tool, "args": args, "result": result,
            "llm_content": content, "sent_to_llm": mode != "direct", "pii_pattern_hits": len(hits)}


def apply_tool_result(tool: str, args: dict, result: dict, alias_to_file: dict[str, str]) -> None:
    """Update file entries from an (id-only) tool result — shared by both execution modes."""
    if "error" in result:
        entry = None
        if tool == "ocr_document":
            entry = STATE.files.get(alias_to_file.get(str(args.get("file_path")), ""))
        elif args.get("doc_id"):
            entry = STATE.by_doc(args["doc_id"])
        if entry:
            entry.status, entry.error = "ERROR", result["error"][:300]
            STATE.emit_file(entry)
        return
    entry: FileEntry | None = None
    if tool == "ocr_document":
        entry = STATE.files.get(alias_to_file.get(str(args.get("file_path")), ""))
        if entry:
            entry.doc_id = result["doc_id"]
            entry.page_count = result.get("page_count")
            entry.word_count = result.get("word_count")
            entry.avg_confidence = result.get("avg_confidence")
            entry.image_dimensions = result.get("image_dimensions")
            entry.status = "OCR DONE"
    elif tool == "detect_pii":
        entry = STATE.by_doc(result.get("doc_id", ""))
        if entry:
            entry.summary = {k: int(v) for k, v in (result.get("summary") or {}).items()}
            entry.total_boxes = result.get("total_boxes")
            entry.status = "PII DETECTED"
    elif tool == "redact_image":
        entry = STATE.by_doc(args.get("doc_id", ""))
        if entry:
            entry.result_id = result["result_id"]
            entry.boxes_drawn = result.get("boxes_drawn")
            entry.status = "REDACTED"
    elif tool == "verify_redaction":
        entry = next((f for f in STATE.files.values() if f.result_id == result.get("result_id")), None)
        if entry:
            entry.residual_pii_count = result.get("residual_pii_count")
            entry.status = "VERIFIED PASS" if result.get("verified") else "VERIFIED FAIL"
    elif tool == "link_documents":
        STATE.links = result
        folder = STATE.folders.get(STATE.job.get("folder_id") or "")
        if folder:
            folder.links = result
        else:                               # whole-queue run: give every folder the clusters of its own docs
            for fo in STATE.folders.values():
                fo.links = _links_for(result, {f.doc_id for f in STATE.scoped(fo.folder_id) if f.doc_id})
        STATE.hub.publish({"type": "links", "links": result, "folder_id": STATE.job.get("folder_id")})
        STATE.emit_folders()
    if entry:
        STATE.emit_file(entry)


def _links_for(links: dict, doc_ids: set[str]) -> dict | None:
    """Restrict a link_documents result to one folder's documents."""
    if not doc_ids:
        return None
    clusters = [c for c in links.get("clusters", []) if set(c["doc_ids"]) & doc_ids]
    return {**links, "clusters": clusters,
            "unlinked_doc_ids": [d for d in links.get("unlinked_doc_ids", []) if d in doc_ids]}


def speak(stats: dict) -> None:
    """TTS in a separate process (SAPI5 must own its thread; keeps the server responsive)."""
    try:
        subprocess.Popen([sys.executable, "-m", "agent.tts_notifier", json.dumps(stats)], cwd=str(ROOT),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def job_stats(files: list[FileEntry]) -> dict:
    clusters = (STATE.links or {}).get("clusters", [])
    return {
        "documents_processed": sum(1 for f in files if f.doc_id),
        "boxes_redacted": sum(f.boxes_drawn or 0 for f in files),
        "verified_pass": sum(1 for f in files if f.status == "VERIFIED PASS"),
        "verified_fail": sum(1 for f in files if f.status == "VERIFIED FAIL"),
        "clusters": len(clusters),
        "linked_documents": sum(len(c["doc_ids"]) for c in clusters),
        "status": "completed",
    }


# ----------------------------------------------------------------------------- workers
def blur_page(img):
    """Irreversible full-page blur: heavy pixelation (≈48 blocks on the long side) followed by a wide Gaussian."""
    import cv2
    h, w = img.shape[:2]
    f = max(w, h) / 48
    small = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, int(min(w, h) / 20) | 1)
    return cv2.GaussianBlur(out, (k, k), 0)


def run_full_blur(files: list[FileEntry]) -> None:
    """policy=full: blur every page of every document, then re-OCR the result to prove nothing is readable.

    Lives in the UI layer and only reuses server utilities, so the redaction pipeline itself is unchanged.
    Output files follow the pipeline's naming, so before/after images and downloads work the same way."""
    from server.image_preprocessing import load_bgr, save_png
    from server.ocr_engine import ocr_image
    from server.pdf_reader import render_pdf_pages
    from server.pii_detector import PIIDetector

    detector = PIIDetector()
    work = config.OUTPUT_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    step = 0
    for i, entry in enumerate(files):
        alias = f"FILE_{i + 1}"
        step += 1
        outs: list[Path] = []
        try:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            src = Path(entry.path)
            sources = render_pdf_pages(str(src), str(work), doc_id) if src.suffix.lower() == ".pdf" else [str(src)]
            dims = None
            for n, s in enumerate(sources, start=1):
                img = load_bgr(s)
                dims = dims or [int(img.shape[1]), int(img.shape[0])]
                save_png(str(work / f"{doc_id}_p{n}.png"), img)                    # "before" frame
                out = config.OUTPUT_DIR / f"redacted_{doc_id}{'' if n == 1 else f'_p{n}'}.png"
                save_png(str(out), blur_page(img))
                outs.append(out)
            result = {"doc_id": doc_id, "page_count": len(outs), "image_dimensions": dims, "status": "blurred"}
            entry.doc_id, entry.page_count, entry.image_dimensions = doc_id, len(outs), dims
            entry.boxes_drawn, entry.summary, entry.status = len(outs), {}, "REDACTED"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            entry.status, entry.error = "ERROR", result["error"][:300]
        STATE.emit_file(entry)
        STATE.hub.publish(boundary_event(step, "blur_document", {"file_path": alias}, result, None, "direct"))
        if "error" in result:
            continue

        step += 1
        try:
            residual = 0
            for n, out in enumerate(outs, start=1):
                _, words = ocr_image(load_bgr(str(out)), page_no=n, deskew=False)
                residual += len(detector.detect(words))
            result = {"doc_id": entry.doc_id, "verified": residual == 0, "residual_pii_count": residual,
                      "status": "PASS" if residual == 0 else "FAIL"}
            entry.residual_pii_count = residual
            entry.status = "VERIFIED PASS" if residual == 0 else "VERIFIED FAIL"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            entry.status, entry.error = "ERROR", result["error"][:300]
        STATE.emit_file(entry)
        STATE.hub.publish(boundary_event(step, "verify_redaction", {"doc_id": entry.doc_id}, result, None, "direct"))


def run_direct(files: list[FileEntry], padding_px: int, use_local_slm: bool, link: bool = True) -> None:
    from server.pipeline import PrivacyPipeline
    pipe = PrivacyPipeline()                       # fresh vault + link graph per run
    alias_to_file = {f"FILE_{i + 1}": f.file_id for i, f in enumerate(files)}
    step = 0

    def call(tool: str, args: dict, fn, *fargs) -> dict:
        nonlocal step
        step += 1
        try:
            result = fn(*fargs)
        except Exception as exc:                    # pipeline messages never contain document text
            result = {"error": f"{type(exc).__name__}: {exc}"}
        apply_tool_result(tool, args, result, alias_to_file)
        STATE.hub.publish(boundary_event(step, tool, args, result, None, "direct"))
        return result

    for alias, entry in zip(alias_to_file, files):
        ocr = call("ocr_document", {"file_path": alias}, pipe.ocr_document, entry.path)
        if "error" in ocr:
            continue
        det = call("detect_pii", {"doc_id": ocr["doc_id"], "use_local_slm": use_local_slm},
                   pipe.detect_pii, ocr["doc_id"], use_local_slm)
        if "error" in det:
            continue
        red = call("redact_image", {"doc_id": ocr["doc_id"], "detection_id": det["detection_id"],
                                    "padding_px": padding_px},
                   pipe.redact_image, ocr["doc_id"], det["detection_id"], padding_px)
        if "error" in red:
            continue
        call("verify_redaction", {"result_id": red["result_id"]}, pipe.verify_redaction, red["result_id"])
    if link:
        call("link_documents", {}, pipe.link_documents)


def run_agent_mode(files: list[FileEntry], provider: str, padding_px: int, use_local_slm: bool) -> dict | None:
    from agent.cli_agent import run_agent
    alias_to_file = {f"FILE_{i + 1}": f.file_id for i, f in enumerate(files)}
    final: dict = {}

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "tool":
            apply_tool_result(ev["tool"], ev["args"], ev["result"], alias_to_file)
            STATE.hub.publish(boundary_event(ev["step"], ev["tool"], ev["args"], ev["result"], ev["llm_content"],
                                             provider))
        elif t == "start":
            STATE.hub.publish({"type": "agent", "event": "start", "provider": ev["provider"], "model": ev["model"],
                               "tools": ev["tools"], "llm_user_message": ev["llm_user_message"],
                               "pii_pattern_hits": len(scan_for_pii(ev["llm_user_message"]))})
        elif t == "done":
            final.update(ev["stats"])
        else:
            STATE.hub.publish({"type": "agent", "event": t, **{k: v for k, v in ev.items() if k != "type"}})

    task = "Anonymize every input document, verify each redaction and link related documents."
    if padding_px != 4:
        task += f" Call redact_image with padding_px={padding_px}."
    if use_local_slm:
        task += " Call detect_pii with use_local_slm=true."
    # Own event loop in this worker thread (Windows Proactor loop -> subprocess support for the MCP stdio server)
    code = asyncio.run(run_agent(task, [f.path for f in files], provider, None,
                                 int(config_env("MAX_STEPS", "40")), tts=False, verbose=False, on_event=on_event))
    return final if code in (0, 1) else None


def config_env(name: str, default: str) -> str:
    import os
    return os.getenv(name) or default


def start_job(folder_id: str | None, only_new: bool = False) -> None:
    """Caller holds STATE.lock and has checked that no job is running."""
    s = STATE.settings
    STATE.job = {"running": True, "mode": s["mode"], "folder_id": folder_id, "only_new": only_new,
                 "started": time.time(), "finished": None, "stats": None}
    STATE.hub.publish({"type": "job", "state": "started", "mode": s["mode"], "folder_id": folder_id})
    threading.Thread(target=job_thread, args=(s["mode"], s["padding_px"], s["use_local_slm"], s["tts"], folder_id,
                                              only_new), daemon=True).start()


def maybe_autorun(folder_id: str) -> bool:
    """Process new documents of an auto folder now, or as soon as the running job ends."""
    with STATE.lock:
        folder = STATE.folders.get(folder_id)
        if not folder or not folder.auto:
            return False
        if STATE.job["running"]:
            if folder_id not in STATE.pending:
                STATE.pending.append(folder_id)
        else:
            start_job(folder_id, only_new=True)
    return True


def job_thread(mode: str, padding_px: int, use_local_slm: bool, tts: bool, folder_id: str | None = None,
               only_new: bool = False) -> None:
    with STATE.lock:
        files = [f for f in STATE.scoped(folder_id) if not only_new or f.status in ("QUEUED", "ERROR")]
        for f in files:
            f.status, f.doc_id, f.result_id, f.error = "QUEUED", None, None, None
            f.summary, f.boxes_drawn, f.total_boxes, f.residual_pii_count = {}, None, None, None
            STATE.emit_file(f)
        blur = [f for f in files if STATE.folders.get(f.folder_id, STATE.folders[INBOX_ID]).policy == "full"]
        pii = [f for f in files if f not in blur]
        if not only_new and pii:
            STATE.links = None
    if not only_new and pii:
        STATE.hub.publish({"type": "links", "links": None, "folder_id": folder_id})
    scope = STATE.folders[folder_id].name if folder_id in STATE.folders else "all folders"
    STATE.log(f"Job started — {scope}: {len(blur)} full-blur, {len(pii)} PII doc(s), mode={mode}, "
              f"padding_px={padding_px}{' (auto)' if only_new else ''}")
    t0 = time.time()
    try:
        if blur:
            run_full_blur(blur)
        if pii:
            if mode == "direct":
                # auto runs only see the new documents, so they would overwrite the folder's links with a subset
                run_direct(pii, padding_px, use_local_slm, link=not only_new)
            else:
                run_agent_mode(pii, mode, padding_px, use_local_slm)
    except Exception as exc:
        STATE.log(f"Job failed: {type(exc).__name__}: {exc}", "error")
    stats = job_stats(files)
    with STATE.lock:
        STATE.job.update(running=False, finished=time.time(), stats=stats)
    STATE.hub.publish({"type": "job", "state": "finished", "mode": mode, "folder_id": folder_id, "stats": stats,
                       "seconds": round(time.time() - t0, 1)})
    if tts:
        speak(stats)
    with STATE.lock:                                   # drain folders that received files while we were busy
        while STATE.pending and not STATE.job["running"]:
            nxt = STATE.pending.pop(0)
            if nxt in STATE.folders and any(f.status == "QUEUED" for f in STATE.scoped(nxt)):
                start_job(nxt, only_new=True)


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.hub.loop = asyncio.get_running_loop()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="PrivacyGuard AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config() -> dict:
    import os
    ollama_ok, ollama_models = False, []
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get("http://localhost:11434/api/tags")
            ollama_models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except Exception:
        pass
    return {
        "providers": {
            "direct": {"available": True, "label": "Direct Pipeline"},
            "ollama": {"available": ollama_ok and any(m.startswith("qwen2.5:7b") for m in ollama_models),
                       "label": "Ollama Local (qwen2.5:7b)", "detail": "running" if ollama_ok else "not reachable"},
            "gemini": {"available": bool(os.getenv("GEMINI_API_KEY") or (os.getenv("LLM_API_KEY") and
                                         "your-" not in os.getenv("LLM_API_KEY", ""))),
                       "label": "Gemini Cloud", "detail": "needs GEMINI_API_KEY or LLM_API_KEY in .env"},
        },
        "pii_types": PII_TYPES,
        "synthetic_available": (SYNTHETIC / "ground_truth.json").exists(),
    }


@app.get("/api/state")
def get_state() -> dict:
    with STATE.lock:
        return {"files": [f.public() for f in STATE.files.values()], "links": STATE.links, "job": STATE.job,
                "folders": [f.public() for f in STATE.folders.values()], "settings": STATE.settings,
                "events": list(STATE.hub.history)[-400:]}


# ----------------------------------------------------------------------------- folders
class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=48)
    policy: Policy = "full"
    auto: bool = True


class FolderPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=48)
    policy: Policy | None = None
    auto: bool | None = None


def _folder(folder_id: str) -> Folder:
    folder = STATE.folders.get(folder_id)
    if not folder:
        raise HTTPException(404, "Unknown folder")
    return folder


@app.post("/api/folders")
def create_folder(req: FolderCreate) -> dict:
    folder = Folder(f"f_{uuid.uuid4().hex[:8]}", req.name.strip(), req.policy, req.auto)
    with STATE.lock:
        STATE.folders[folder.folder_id] = folder
    STATE.emit_folders()
    STATE.log(f"Folder {folder.folder_id} created — policy={folder.policy}, auto={folder.auto}")
    return folder.public()


@app.patch("/api/folders/{folder_id}")
def update_folder(folder_id: str, req: FolderPatch) -> dict:
    with STATE.lock:
        folder = _folder(folder_id)
        if req.name is not None:
            folder.name = req.name.strip()
        if req.policy is not None:
            folder.policy = req.policy
        if req.auto is not None:
            folder.auto = req.auto
    STATE.emit_folders()
    if req.auto:                     # switching auto on picks up documents that are still waiting
        if any(f.status == "QUEUED" for f in STATE.scoped(folder_id)):
            maybe_autorun(folder_id)
    return folder.public()


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str) -> dict:
    if folder_id == INBOX_ID:
        raise HTTPException(400, "INBOX cannot be deleted")
    with STATE.lock:
        if STATE.job["running"]:
            raise HTTPException(409, "A job is running")
        _folder(folder_id)
        for f in STATE.scoped(folder_id):
            STATE.files.pop(f.file_id, None)
            if f.source == "upload":
                Path(f.path).unlink(missing_ok=True)
            STATE.hub.publish({"type": "file_removed", "file_id": f.file_id})
        STATE.folders.pop(folder_id)
        if folder_id in STATE.pending:
            STATE.pending.remove(folder_id)
    STATE.emit_folders()
    return {"ok": True}


class Settings(BaseModel):
    mode: Literal["direct", "ollama", "gemini"] = "direct"
    padding_px: int = Field(4, ge=0, le=12)
    use_local_slm: bool = False
    tts: bool = False


@app.put("/api/settings")
def put_settings(req: Settings) -> dict:
    with STATE.lock:
        STATE.settings = req.model_dump()
    return STATE.settings


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue()
    STATE.hub.subscribers.add(q)

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            STATE.hub.subscribers.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:80]
    return stem or "document"


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), folder_id: str = Form(INBOX_ID)) -> dict:
    folder = _folder(folder_id)
    if STATE.job["running"] and not folder.auto:
        raise HTTPException(409, "A job is running")
    added = []
    for up in files:
        ext = Path(up.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
        data = await up.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{up.filename}: larger than 25 MB")
        file_id = uuid.uuid4().hex[:10]
        dest = UPLOADS / f"{file_id}_{_safe_name(up.filename or 'document' + ext)}"
        dest.write_bytes(data)
        entry = FileEntry(file_id, Path(up.filename or dest.name).name, str(dest), "upload", folder_id)
        with STATE.lock:
            STATE.files[file_id] = entry
        STATE.emit_file(entry)
        added.append(entry.public())
    STATE.emit_folders()
    return {"added": added, "auto": bool(added) and maybe_autorun(folder_id)}


@app.post("/api/load-demo")
def load_demo(folder_id: str = Query(INBOX_ID)) -> dict:
    folder = _folder(folder_id)
    if STATE.job["running"] and not folder.auto:
        raise HTTPException(409, "A job is running")
    if not (SYNTHETIC / "ground_truth.json").exists():
        raise HTTPException(404, "Synthetic dataset missing — run python -m evaluation.generate_synthetic_docs")
    known = {f.path for f in STATE.scoped(folder_id)}
    added = []
    for p in sorted(SYNTHETIC.iterdir()):
        if p.suffix.lower() in ALLOWED_EXT and str(p) not in known:
            entry = FileEntry(uuid.uuid4().hex[:10], p.name, str(p), "demo", folder_id)
            with STATE.lock:
                STATE.files[entry.file_id] = entry
            STATE.emit_file(entry)
            added.append(entry.public())
    STATE.emit_folders()
    return {"added": added, "auto": bool(added) and maybe_autorun(folder_id)}


@app.delete("/api/files/{file_id}")
def remove_file(file_id: str) -> dict:
    if STATE.job["running"]:
        raise HTTPException(409, "A job is running")
    with STATE.lock:
        entry = STATE.files.pop(file_id, None)
    if not entry:
        raise HTTPException(404, "Unknown file")
    if entry.source == "upload":
        Path(entry.path).unlink(missing_ok=True)
    STATE.hub.publish({"type": "file_removed", "file_id": file_id})
    return {"ok": True}


@app.post("/api/reset")
def reset() -> dict:
    if STATE.job["running"]:
        raise HTTPException(409, "A job is running")
    with STATE.lock:
        for f in STATE.files.values():
            if f.source == "upload":
                Path(f.path).unlink(missing_ok=True)       # uploaded documents are not kept
        STATE.files.clear()
        STATE.links = None
        STATE.reset_folders()
        STATE.hub.history.clear()
    STATE.hub.publish({"type": "reset"})
    STATE.emit_folders()
    return {"ok": True}


class ProcessRequest(Settings):
    folder_id: str | None = None          # None = every document in every folder


@app.post("/api/process")
def process(req: ProcessRequest) -> dict:
    with STATE.lock:
        if STATE.job["running"]:
            raise HTTPException(409, "A job is already running")
        if req.folder_id is not None:
            _folder(req.folder_id)
        if not STATE.scoped(req.folder_id):
            raise HTTPException(400, "No documents in the queue")
        STATE.settings = req.model_dump(exclude={"folder_id"})
        start_job(req.folder_id)
    return {"started": True, "mode": req.mode, "folder_id": req.folder_id}


def _image_response(path: Path | None) -> FileResponse:
    if path is None or not path.exists():
        raise HTTPException(404, "Image not available yet")
    resolved = path.resolve()
    allowed = [config.OUTPUT_DIR.resolve(), UPLOADS.resolve(), SYNTHETIC.resolve()]
    if not any(resolved.is_relative_to(a) for a in allowed):
        raise HTTPException(403, "Path outside allowed folders")
    return FileResponse(resolved, headers={"Cache-Control": "no-store"})


@app.get("/api/image/{file_id}/{kind}")
def image(file_id: str, kind: Literal["before", "after"]) -> FileResponse:
    entry = STATE.files.get(file_id)
    if not entry:
        raise HTTPException(404, "Unknown file")
    return _image_response(entry.before_path() if kind == "before" else entry.after_path())


@app.get("/api/download/{file_id}")
def download(file_id: str) -> FileResponse:
    entry = STATE.files.get(file_id)
    if not entry or not entry.after_path():
        raise HTTPException(404, "No redacted image for this file")
    resp = _image_response(entry.after_path())
    resp.headers["Content-Disposition"] = f'attachment; filename="redacted_{entry.doc_id}.png"'
    return resp


@app.get("/api/download-all")
def download_all(folder_id: str | None = Query(None)) -> Response:
    files = STATE.scoped(folder_id)
    folder = _folder(folder_id) if folder_id else None
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            p = f.after_path()
            if p:
                zf.write(p, f"redacted_{f.doc_id}.png")
                count += 1
        manifest = {"folder": folder.public() if folder else None,
                    "documents": [{"doc_id": f.doc_id, "folder_id": f.folder_id, "status": f.status,
                                   "boxes_drawn": f.boxes_drawn, "summary": f.summary} for f in files if f.doc_id],
                    "links": folder.links if folder else STATE.links}
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))   # ids + counts only
    if not count:
        raise HTTPException(404, "No redacted documents yet")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="privacyguard_redacted.zip"'})


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="PrivacyGuard web dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"PrivacyGuard dashboard on {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
