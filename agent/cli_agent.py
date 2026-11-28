"""PrivacyGuard CLI agent: OpenAI-compatible tool-calling loop over the local MCP server.

Privacy guarantees enforced on THIS side as well:
  * File paths are replaced by aliases (FILE_1, FILE_2, ...) before the LLM sees them.
  * Every tool result passes a guard (allow-listed keys + CNP/e-mail/IBAN pattern scan)
    before it is appended to the conversation.

Examples:
  python -m agent.cli_agent --provider ollama synthetic_data/*.png
  python -m agent.cli_agent --provider gemini --dir synthetic_data "Anonymize all documents and link them"
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, InternalServerError,
                    RateLimitError)

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from agent.tts_notifier import announce

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "api_key": "ollama"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "local-model", "api_key": "lm-studio"},
}
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}

SYSTEM_PROMPT = """You are PrivacyGuard, an autonomous document-anonymization agent.
You control a LOCAL MCP server that holds every document in a secure vault. You never see document
text: you only receive opaque ids (doc_*, det_*, res_*), counts, scores and booleans.

For EVERY input file:
  1. ocr_document(file_path=<alias such as FILE_1>)
  2. detect_pii(doc_id)
  3. redact_image(doc_id, detection_id)  — omit padding_px / blur_strength so the safe defaults (4 px, "high")
     are used
  4. verify_redaction(result_id). If status is "FAIL", self-correct before moving on:
     - reason about WHY it failed from the counts only: leak_count > 0 means a known value is still readable
       (blur too weak -> raise blur_strength); uncovered_count > 0 means PII-like text sits outside the boxes
       (coverage too small -> raise padding_px more);
     - call redact_image(doc_id, detection_id, blur_strength="maximum", padding_px=<previous + 6..12>), or
       simply use suggested_blur_strength / suggested_padding_px from the verification result;
     - verify_redaction again on the NEW result_id. Retry at most 2 times per document; if it still fails,
       report it as FAIL (needs manual review).
After all files are processed call link_documents() once.

FORM FILLING (e.g. "Completează cererea cu datele din buletin" / "fill the form from the ID card"):
  1. ocr_document on every alias.
  2. extract_id_profile(doc_id) on each document: the one with is_id_card=true is the ID card, the other one
     is the blank form. You only see field NAMES (detected_fields), never their values.
  3. fill_form_from_id(id_doc_id, form_doc_id) — pass extra_fields (phone, email, date, purpose) ONLY with
     values the user wrote in the task; never invent them.
  4. Only if the user also asked to anonymize/blur: detect_pii -> redact_image -> verify_redaction on the
     returned filled_doc_id. Do not redact the inputs and do not call link_documents in a fill-only task.
  5. Report filled_doc_id, page_count and the filled / unfilled field NAMES.
If a tool returns an error, read it, fix the arguments or skip that file, and continue.
Finish with a short report: per file alias -> doc_id, boxes drawn, verification status; then the document
clusters with their shared entity TYPES and confidence. Never ask the user questions; act autonomously."""

ALLOWED_KEYS = {
    "doc_id", "word_count", "avg_confidence", "image_dimensions", "page_count", "status", "detection_id",
    "summary", "total_boxes", "linkable_identifiers", "CNP", "PERSON_NAME", "ADDRESS", "ID_SERIES", "IBAN",
    "PHONE", "EMAIL", "BIRTH_DATE", "MRZ", "total_documents", "clusters", "unlinked_doc_ids", "unknown_doc_ids",
    "cluster_id", "doc_ids", "shared_entity_types", "link_strength", "links", "doc_a", "doc_b",
    "shared_entity_count", "confidence", "result_id", "output_path", "boxes_drawn", "verified",
    "residual_pii_count", "error", "tool", "blur_strength", "blur_level", "mode", "leak_count",
    "uncovered_count", "suggested_blur_strength", "suggested_padding_px", "is_id_card", "is_valid_id",
    "detected_fields", "field_count", "mean_confidence", "filled_doc_id", "pdf_path", "fields_filled",
    "unfilled_fields", "low_coverage", "PARTY",
}
# field-name lists must contain field NAMES only — a value smuggled into such a list is blocked
FIELD_LIST_KEYS = {"detected_fields", "fields_filled", "unfilled_fields"}
FIELD_VOCAB = {
    "last_name", "first_name", "full_name", "cnp", "id_series", "id_number", "id_series_number", "address_full",
    "county", "city", "street", "street_no", "building", "staircase", "floor", "apartment", "sector",
    "birth_date", "birth_place", "issue_date", "expiry_date", "issuing_authority", "nationality", "sex",
    "phone", "email", "date", "purpose",
}
PATH_KEYS = ("error.", "output_path.", "pdf_path.")
FILL_TASK_WORDS = ("anonim", "anonym", "blur", "redact", "estomp")
MAX_ESCALATIONS = 2
LEAK_PATTERNS = [re.compile(r"(?<!\d)[1-9]\d{12}(?!\d)"), re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
                 re.compile(r"\bRO\d{2}[A-Z]{4}"), re.compile(r"\b[A-Z]{2}\s?\d{6}\b")]


# ----------------------------------------------------------------------------- privacy guard
def privacy_guard(obj: Any, path: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ALLOWED_KEYS:
                problems.append(f"unexpected key '{path}{k}'")
            if k in FIELD_LIST_KEYS and isinstance(v, list) and any(x not in FIELD_VOCAB for x in v):
                problems.append(f"non-field-name item in '{path}{k}'")
            problems += privacy_guard(v, f"{path}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            problems += privacy_guard(v, path)
    elif isinstance(obj, str) and not path.endswith(PATH_KEYS):
        if any(p.search(obj) for p in LEAK_PATTERNS):
            problems.append(f"possible PII in '{path.rstrip('.')}'")
    return problems


# ----------------------------------------------------------------------------- self-correction
def escalation_args(prev_args: dict, verify_payload: dict) -> dict:
    """Next redact_image arguments after a FAIL verification (ids + numbers only, never document content).

    Strength always escalates to "maximum"; padding grows by 12 px when text was left outside the boxes
    (uncovered_count > 0), otherwise by 6 px. The server's suggested_* values win when they are larger."""
    pad = int(prev_args.get("padding_px", 4) or 4)
    step = 12 if int(verify_payload.get("uncovered_count") or 0) > 0 else 6
    pad = max(pad + step, int(verify_payload.get("suggested_padding_px") or 0))
    return {"doc_id": prev_args.get("doc_id", ""), "detection_id": prev_args.get("detection_id", ""),
            "blur_strength": "maximum", "padding_px": min(50, pad)}


# ----------------------------------------------------------------------------- MCP -> OpenAI schema
def clean_schema(schema: Any) -> Any:
    """Make MCP JSON schemas acceptable to strict OpenAI-compatible providers (Gemini, Mistral)."""
    if isinstance(schema, dict):
        if "anyOf" in schema:
            non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
            if len(non_null) == 1:
                merged = {k: v for k, v in schema.items() if k != "anyOf"}
                merged.update(non_null[0])
                schema = merged
        return {k: clean_schema(v) for k, v in schema.items() if k not in ("title", "$schema", "default")}
    if isinstance(schema, list):
        return [clean_schema(s) for s in schema]
    return schema


def to_openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                               "parameters": clean_schema(t.input_schema)}}
            for t in mcp_tools]


def tool_payload(result) -> dict:
    if result.is_error:
        return {"error": " ".join(getattr(c, "text", "") for c in result.content).strip() or "tool error"}
    if result.structured_content:
        return dict(result.structured_content)
    text = result.content[0].text if result.content else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "non-JSON tool output suppressed by privacy guard"}


# ----------------------------------------------------------------------------- LLM with backoff
async def chat_with_retry(client: AsyncOpenAI, max_retries: int, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)) \
                or status in RETRYABLE_STATUS
            if not retryable or attempt == max_retries:
                raise
            delay = min(2 ** attempt, 32) + random.uniform(0, 1)
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after") \
                if getattr(exc, "response", None) is not None else None
            if retry_after and str(retry_after).replace(".", "").isdigit():
                delay = max(delay, float(retry_after))
            print(f"  ! LLM error ({type(exc).__name__}{f' {status}' if status else ''}); "
                  f"retry {attempt + 1}/{max_retries} in {delay:.1f}s", file=sys.stderr)
            await asyncio.sleep(delay)


# ----------------------------------------------------------------------------- agent
class Stats:
    def __init__(self) -> None:
        self.data = {"documents_processed": 0, "boxes_redacted": 0, "verified_pass": 0, "verified_fail": 0,
                     "clusters": 0, "linked_documents": 0, "tool_errors": 0, "status": "completed"}
        self._latest_boxes: dict[str, int] = {}
        self._verify: dict[str, bool] = {}        # doc_id -> latest verification outcome
        self._res_doc: dict[str, str] = {}        # result_id -> doc_id

    def record(self, name: str, args: dict, payload: dict) -> None:
        if "error" in payload:
            self.data["tool_errors"] += 1
            return
        if name == "ocr_document":
            self._latest_boxes.setdefault("doc:" + payload.get("doc_id", ""), 0)
            self.data["documents_processed"] = sum(1 for k in self._latest_boxes if k.startswith("doc:"))
        elif name == "redact_image":
            self._latest_boxes["red:" + args.get("doc_id", "")] = payload.get("boxes_drawn", 0)
            self._res_doc[payload.get("result_id", "")] = args.get("doc_id", "")
            self.data["boxes_redacted"] = sum(v for k, v in self._latest_boxes.items() if k.startswith("red:"))
        elif name == "verify_redaction":
            rid = payload.get("result_id", "")
            self._verify[self._res_doc.get(rid, rid)] = bool(payload.get("verified"))
            self.data["verified_pass"] = sum(self._verify.values())
            self.data["verified_fail"] = len(self._verify) - self.data["verified_pass"]
        elif name == "link_documents":
            self.data["clusters"] = len(payload.get("clusters", []))
            self.data["linked_documents"] = sum(len(c["doc_ids"]) for c in payload.get("clusters", []))


async def run_agent(task: str, files: list[str], provider: str, model: str | None, max_steps: int,
                    tts: bool, verbose: bool, llm: Any = None,
                    on_event: Callable[[dict], None] | None = None) -> int:
    """Run the agent. ``on_event`` (optional) receives id-only progress events, e.g. for the web UI."""

    def emit(event: dict) -> None:
        if on_event is not None:
            try:
                on_event(event)
            except Exception:   # a broken observer must never break the agent
                pass

    preset = PROVIDERS.get(provider, {})
    base_url = os.getenv("LLM_BASE_URL") if provider == "env" else preset["base_url"]
    api_key = (os.getenv("LLM_API_KEY") if provider == "env" or "api_key" not in preset else preset["api_key"])
    if provider in ("gemini", "mistral"):
        api_key = os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("LLM_API_KEY")
    model = model or (os.getenv("LLM_MODEL") if provider == "env" else preset["model"])
    if llm is None:
        if not base_url or not model or not api_key:
            print("LLM_BASE_URL / LLM_MODEL / LLM_API_KEY missing — see .env.example", file=sys.stderr)
            emit({"type": "error", "message": f"No API key / base URL / model configured for provider '{provider}'."})
            return 2
        # Local CPU models need a long first call while the model is loaded into memory
        is_local = any(h in base_url for h in ("localhost", "127.0.0.1"))
        timeout = float(os.getenv("LLM_TIMEOUT_S") or (300 if is_local else 60))
        llm = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))

    aliases = {f"FILE_{i + 1}": str(Path(f).resolve()) for i, f in enumerate(files)}
    stats = Stats()
    print(f"PrivacyGuard agent | provider={provider} model={model} | {len(files)} file(s) | MAX_STEPS={max_steps}")

    params = StdioServerParameters(command=sys.executable, args=["-m", "server.mcp_server"], cwd=str(ROOT),
                                   env={"PYTHONIOENCODING": "utf-8"})
    async with Client(params) as mcp_client:
        tools = to_openai_tools((await mcp_client.list_tools()).tools)
        user_msg = task + "\n\nInput files (aliases): " + (", ".join(aliases) if aliases else "none")
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        emit({"type": "start", "provider": provider, "model": model, "aliases": list(aliases),
              "tools": [t["function"]["name"] for t in tools], "llm_user_message": user_msg})
        # Workflow tracker (ids only): lets the agent catch skipped steps and duplicate work by small models
        ocr_cache: dict[str, dict] = {}          # alias -> ocr payload
        result_doc: dict[str, str] = {}          # result_id -> doc_id
        verified_docs: set[str] = set()
        last_redact: dict[str, dict] = {}        # doc_id -> latest redact_image args
        failed_verify: dict[str, dict] = {}      # doc_id -> latest FAIL verification payload
        escalations: dict[str, int] = {}         # doc_id -> redact_image calls after the first one
        linked_after_last_doc = False
        fill_docs: set[str] = set()              # docs of a completed form fill (exempt from redaction nudges)
        wants_redaction = any(w in task.lower() for w in FILL_TASK_WORDS)
        nudges = 0

        def pending_work() -> list[str]:
            todo = [f"{a}: not processed yet (call ocr_document)" for a in aliases if a not in ocr_cache]
            for p in ocr_cache.values():
                d = p["doc_id"]
                if d in verified_docs or (d in fill_docs and not wants_redaction):
                    continue
                fail = failed_verify.get(d)
                if fail is None:
                    todo.append(f"{d}: redaction not verified yet")
                elif escalations.get(d, 0) < MAX_ESCALATIONS and d in last_redact:
                    nxt = escalation_args(last_redact[d], fail)
                    todo.append(f"{d}: verification FAILED (leak_count={fail.get('leak_count', 0)}, "
                                f"uncovered_count={fail.get('uncovered_count', 0)}) — call redact_image("
                                f"doc_id='{nxt['doc_id']}', detection_id='{nxt['detection_id']}', "
                                f"blur_strength='maximum', padding_px={nxt['padding_px']}) then verify_redaction")
            if len(ocr_cache) >= 2 and not linked_after_last_doc and not fill_docs:
                todo.append("link_documents has not been called after the last document")
            return todo

        for step in range(1, max_steps + 1):
            try:
                resp = await chat_with_retry(llm, max_retries, model=model, messages=messages, tools=tools,
                                             tool_choice="auto", temperature=0)
            except Exception as exc:
                print(f"LLM provider failed permanently: {type(exc).__name__}: {exc}", file=sys.stderr)
                emit({"type": "error", "message": f"LLM provider failed: {type(exc).__name__}: {exc}"})
                stats.data["status"] = "failed"
                break
            msg = resp.choices[0].message
            assistant: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [{"id": tc.id, "type": "function",
                                            "function": {"name": tc.function.name,
                                                         "arguments": tc.function.arguments or "{}"}}
                                           for tc in msg.tool_calls]
            messages.append(assistant)

            if not msg.tool_calls:
                todo = pending_work()
                if todo and nudges < 3:
                    nudges += 1
                    print(f"  ! workflow incomplete, nudging the model ({nudges}/3): {len(todo)} item(s) pending",
                          file=sys.stderr)
                    messages.append({"role": "user", "content": "Do not ask questions; continue autonomously. "
                                     "Still pending:\n- " + "\n- ".join(todo)})
                    emit({"type": "nudge", "step": step, "pending": todo, "model_said": msg.content or ""})
                    continue
                print("\n=== Agent report ===\n" + (msg.content or "(empty)"))
                emit({"type": "report", "step": step, "content": msg.content or ""})
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    real_args = dict(args)
                    if name == "ocr_document":
                        alias = str(args.get("file_path", ""))
                        if alias not in aliases:
                            raise ValueError(f"Unknown file alias '{alias}'. Use one of {list(aliases)}")
                        real_args["file_path"] = aliases[alias]
                    if name == "ocr_document" and args["file_path"] in ocr_cache:
                        # never index the same file twice (a duplicate would falsely "link" to itself)
                        payload = dict(ocr_cache[args["file_path"]])
                        payload["status"] = "indexed"
                    else:
                        payload = tool_payload(await mcp_client.call_tool(name, real_args))
                    if "error" not in payload:
                        if name == "ocr_document":
                            if args["file_path"] not in ocr_cache:
                                linked_after_last_doc = False
                            ocr_cache[args["file_path"]] = payload
                        elif name == "redact_image":
                            d = args.get("doc_id", "")
                            result_doc[payload["result_id"]] = d
                            if d in last_redact:
                                escalations[d] = escalations.get(d, 0) + 1
                            last_redact[d] = dict(args)
                        elif name == "verify_redaction":
                            d = result_doc.get(payload.get("result_id", ""), "")
                            if payload.get("verified"):
                                verified_docs.add(d)
                                failed_verify.pop(d, None)
                            else:
                                failed_verify[d] = payload
                        elif name == "link_documents":
                            linked_after_last_doc = True
                        elif name == "fill_form_from_id":
                            fill_docs |= {args.get("id_doc_id", ""), args.get("form_doc_id", "")}
                except json.JSONDecodeError:
                    args, payload = {}, {"error": "Arguments were not valid JSON.", "tool": name}
                except Exception as exc:   # MCP / transport / validation errors -> back to the LLM
                    args = args if "args" in locals() else {}
                    payload = {"error": f"{type(exc).__name__}: {exc}", "tool": name}

                problems = privacy_guard(payload)
                if problems:
                    print(f"  ! privacy guard blocked {name} output: {problems}", file=sys.stderr)
                    payload = {"error": "Tool output blocked by client privacy guard.", "tool": name}
                stats.record(name, args, payload)
                shown = json.dumps(payload, ensure_ascii=False)
                print(f"[step {step:02d}] {name}({json.dumps(args)}) -> {shown if verbose else shown[:160]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": shown})
                # `llm_content` is byte-for-byte what the LLM receives for this tool call
                emit({"type": "tool", "step": step, "tool": name, "args": args, "result": payload,
                      "llm_content": shown, "privacy_guard": "blocked" if problems else "clean"})
        else:
            stats.data["status"] = "stopped at the step limit"
            print(f"\nMAX_STEPS ({max_steps}) reached — stopping.", file=sys.stderr)
            emit({"type": "error", "message": f"MAX_STEPS ({max_steps}) reached — stopping."})

    emit({"type": "done", "stats": dict(stats.data)})
    announce(stats.data, enabled=tts)
    return 0 if stats.data["status"] == "completed" else 1


def collect_files(paths: list[str], directory: str | None) -> list[str]:
    out: list[str] = []
    for p in paths:
        out += [g for g in glob.glob(p)] or ([p] if Path(p).suffix else [])
    if directory:
        out += sorted(str(f) for f in Path(directory).iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return list(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="PrivacyGuard autonomous anonymization agent")
    ap.add_argument("items", nargs="*", help="Files/globs to process; a non-path argument is used as the task text")
    ap.add_argument("--dir", help="Process every image/PDF in this folder")
    ap.add_argument("--provider", default="env", choices=["env", *PROVIDERS],
                    help="'env' uses LLM_BASE_URL/LLM_MODEL/LLM_API_KEY from .env")
    ap.add_argument("--model", help="Override the model name")
    ap.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "25")))
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    task_words = [i for i in a.items if not Path(i).suffix and not glob.glob(i)]
    task = " ".join(task_words) or "Anonymize every input document, verify each redaction and link related documents."
    files = collect_files([i for i in a.items if i not in task_words], a.dir)
    if not files:
        ap.error("no input files")
    t0 = time.time()
    code = asyncio.run(run_agent(task, files, a.provider, a.model, a.max_steps, not a.no_tts, a.verbose))
    print(f"Done in {time.time() - t0:.1f}s")
    sys.exit(code)


if __name__ == "__main__":
    main()
