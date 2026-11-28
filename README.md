# PrivacyGuard AI — Proiect #03: Anonimizarea actelor

An autonomous **CLI agent** connected to a local **MCP server** that:

1. OCRs scanned Romanian documents (ID cards, application forms, invoices, PDFs) with OpenCV + Tesseract,
2. detects PII (CNP with official checksum, names, addresses, ID series, IBAN, phone, e-mail, birth date, MRZ),
3. redacts it with an irreversible blur (noise + pixelation + Gaussian; `REDACTION_MODE=black` for solid boxes) and **verifies** the redaction by re-running OCR,
4. **links documents that belong to the same person** via salted HMAC-SHA256 hashes,
5. announces the result with **offline text-to-speech**,

while the LLM (cloud or local) **never receives raw document text or PII** — only opaque ids, counts and scores.

> All documents in this repository are **synthetic and fictional** (generated with Faker, marked
> "SPECIMEN - DATE FICTIVE"). Real documents placed in the project root are git-ignored.

---

## Architecture

```
 ┌──────────────────────────┐   OpenAI-compatible API    ┌──────────────────────────┐
 │ LLM (Gemini / Mistral /   │◄──────────────────────────►│ agent/cli_agent.py        │
 │ Ollama / LM Studio)       │  sees ONLY: FILE_n aliases, │ • tool-calling loop        │
 └──────────────────────────┘  doc_/det_/res_ ids, counts  │ • MAX_STEPS, retry/backoff │
                                                           │ • file-path aliasing       │
                                                           │ • client privacy guard     │
                                                           │ • agent/tts_notifier.py    │
                                                           └────────────┬─────────────┘
                                                                        │ MCP over stdio
 ┌──────────────────────────────────────────────────────────────────────▼─────────────┐
 │ server/mcp_server.py  (MCP SDK v2 MCPServer — the v2 name of FastMCP)               │
 │  ocr_document → detect_pii → redact_image → verify_redaction      link_documents     │
 │ ─────────────────────────────────────────────────────────────────────────────────── │
 │ server/pipeline.py   orchestration, returns ids + metadata only                       │
 │ server/vault.py      in-memory vault: raw text, word boxes, entities, session secret  │
 │ server/ocr_engine.py + image_preprocessing.py + pdf_reader.py   OpenCV + Tesseract    │
 │ server/pii_detector.py   regex + checksums + label context (+ optional spaCy / SLM)   │
 │ server/entity_linker.py  HMAC-SHA256 bipartite graph (networkx) → clusters           │
 │ server/redactor.py       noise+pixelate+blur regions → output/redacted_<doc_id>.png  │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

### Privacy-by-design (zero PII leakage to the LLM)

| Layer | Mechanism |
|---|---|
| Vault | Raw OCR text, boxes and entity values live only in the MCP server process memory (`server/vault.py`). |
| Tool contracts | Every tool returns a strict Pydantic model of ids, counts, scores and booleans. |
| Hashing | Linking uses `HMAC-SHA256(session_secret, TYPE|normalized_value)`; the secret is random per server session (or `VAULT_SECRET`). |
| File names | The agent replaces file paths with `FILE_1…FILE_n` aliases before talking to the LLM. |
| Client guard | Every tool result is checked against an allow-list of keys and scanned for CNP / e-mail / IBAN / ID patterns before reaching the LLM. |
| Local SLM | `detect_pii(use_local_slm=True)` only sends text to an endpoint on `localhost`; any other host is refused. |
| Errors | Tool errors are converted to messages that never echo document content. |

### Cross-document entity linking (corelare automată)

`detect_pii` normalizes high-entropy identifiers (CNP digits, `SERIES+NUMBER`, IBAN, lower-cased e-mail,
last 9 phone digits, diacritic-free **sorted** name tokens; ID-card *Nume* + *Prenume* fields are combined)
and adds their salted hashes to a bipartite graph `doc ↔ entity_hash`. `link_documents` computes, for every
pair of documents, the shared entity **types** and a confidence `1 − Π(1 − w_type)`
(CNP 0.99, ID series/MRZ 0.97, IBAN 0.92, e-mail 0.88, phone 0.80, name 0.60), then returns connected
components as clusters. Unvalidated CNPs and propagated name fragments are redacted but not used for linking.

Example output (no raw text):

```json
{"cluster_id": "cluster_1", "doc_ids": ["doc_2532e961f2ee", "doc_4a5555c08fb0", "doc_fc3f1365726f"],
 "shared_entity_types": ["CNP", "ID_SERIES", "PERSON_NAME", "PHONE"], "link_strength": 0.6,
 "links": [{"doc_a": "doc_2532e961f2ee", "doc_b": "doc_fc3f1365726f",
            "shared_entity_types": ["CNP", "PERSON_NAME"], "shared_entity_count": 2, "confidence": 0.996}]}
```

---

## MCP tools

| Tool | Input | Returns (only) |
|---|---|---|
| `ocr_document` | `file_path` | `doc_id, word_count, avg_confidence, image_dimensions, page_count, status` |
| `detect_pii` | `doc_id, use_local_slm=false` | `detection_id, doc_id, summary{CNP, PERSON_NAME, ADDRESS, ID_SERIES, IBAN, PHONE, EMAIL, BIRTH_DATE, MRZ}, total_boxes, linkable_identifiers` |
| `link_documents` | `doc_ids?` | `total_documents, clusters[{cluster_id, doc_ids, shared_entity_types, link_strength, links[]}], unlinked_doc_ids, unknown_doc_ids` |
| `redact_image` | `doc_id, detection_id, padding_px=4, blur_strength="high"` (`low/medium/high/maximum` or `1..10`) | `result_id, output_path, boxes_drawn, page_count, blur_strength, blur_level, mode` |
| `verify_redaction` | `result_id` | `result_id, verified, residual_pii_count, status: PASS/FAIL, leak_count, uncovered_count, suggested_blur_strength, suggested_padding_px` |
| `extract_id_profile` | `doc_id` | `doc_id, is_id_card, is_valid_id, detected_fields[] (field KEYS), field_count, mean_confidence` |
| `fill_form_from_id` | `id_doc_id, form_doc_id, extra_fields?{phone,email,date,purpose}` | `filled_doc_id, output_path, pdf_path, fields_filled[], unfilled_fields[], page_count` |

**Self-correction:** when `verify_redaction` returns `FAIL`, the agent reads `leak_count` (a known value is still
readable → blur too weak) and `uncovered_count` (PII-like text outside the boxes → padding too small) and re-runs
`redact_image` with `blur_strength="maximum"` and +6…12 px padding (at most twice), then verifies again.

**OCR pipeline:** grayscale → deskew (projection-profile search ±10°) → 2× upscale for small scans →
non-local-means denoising → adaptive Gaussian threshold → Tesseract `ron+eng` (`image_to_data`, word boxes),
with a grayscale fallback when confidence is low. Redaction is drawn on the deskewed original-resolution
image so boxes align exactly. PDFs are rendered at 300 DPI per page.

**PII detection:** CNP regex + official checksum (key `279146358279`, date and county validation) plus a
lenient OCR-confusion pass after a `CNP` label; labelled `SERIA XX NR 123456` and bare series (in ID
context); IBAN with mod-97; Romanian phone numbers; e-mail (also when OCR reads `@` as `Q`); names after
labels (`Nume`, `Prenume`, `Subsemnatul/a`, `Client`, `Titular`, `Semnătura`, …, including bilingual ID-card
labels with the value on the next line); addresses after `Domiciliu`/`Adresa`/`Domiciliat în` or starting with
`Str./Bd./Calea…`, including continuation lines; MRZ lines. **Unlabeled full names** (`Ionescu Maria Elena`,
`NICHIFOR ANTONIU`) via a lexicon of common Romanian given names + surnames / surname endings, and OCR's `I`→`l`
slip (`lonescu`) is tolerated. Labels are matched **OCR-tolerantly** (`Sena`/`sera` for `Seria`, `C.N.P.`,
`Prenurne`, `Domiciliul`, …). On identity cards (old CI, electronic ID 2021+, any layout) also: issuing authority
(`SPCLEP X`, `D.E.P.A.B.D.`), validity / issue / expiry dates (also `03 07 1983`), 2+6 and 2+7 document numbers,
place of birth, locality fragments (`Mun.`, `Jud.`, `Sec.4`, large cities), card access number, the signature area,
and names / document number **read from the MRZ** and redacted wherever they appear on the card. Values are found
right of a label or under it (by line order and by position, tolerant to skew). Detected name tokens are
propagated to other occurrences (signatures). Optional: spaCy `ro_core_news_sm`, local Ollama SLM.

**Faces / portraits:** OpenCV **YuNet** (`models/face_detection_yunet_2023mar.onnx`, MIT licence, from
`opencv/opencv_zoo`) finds faces anywhere on any document, retrying rotated copies (90°/180°, ±20°, ±35°); the
box is padded by `FACE_PADDING_RATIO` (0.3) to cover hair, ears and chin. The faint *ghost* photo on ID cards is
found by template-matching the detected face. Only when no face is found does an ID-card layout rule apply.

**Blur calibration:** every box gets a margin of `max(REDACTION_MIN_PADDING_PX, REDACTION_PADDING_RATIO × box
height)` (capped at 1.5% of the image's long side, both × a strength-dependent multiplier) and is grown right
through space where OCR read nothing (value tails under stamps). Noise amplitude, pixelation cell and Gaussian
kernel adapt to the **text height, image resolution and local contrast**; `blur_strength` (1–10, default `high` = 7)
raises them further, and even level 1 keeps a safety floor (cell ≥ 0.35 × text height). A residual-legibility
guard re-blurs any region where stroke-scale detail survives. `REDACTION_MODE=black` draws solid boxes instead.

**Form auto-fill from an ID card:** `server/id_extractor.py` turns the detector's output into a structured profile
(names — with an MRZ fallback —, CNP and what it encodes, series/number, address split into street / nr / bl / sc /
et / ap / city / county / sector, birth date/place, issue/expiry dates, issuer, nationality, sex) kept only in the
vault. `server/form_filler.py` fills **AcroForm PDFs** (field names/tooltips matched fuzzily; values with ș/ț are
stamped with a TrueType font) and **flat forms** (scans, flat PDFs, images): Romanian labels (`Subsemnatul`, `CNP`,
`domiciliat în`, `Str.`, `Nr.`, `Bl.`, `posesor al CI seria … nr.`, `eliberat de`, `Data nașterii`, …) are found
OCR-tolerantly on a copy with rule lines removed, the blank after/below each label is located (underline, dotted
line, `____`, empty box, or space after `Label:`), and values are drawn with auto-fitted size, document ink colour
and wrapping onto the next blank line. Font: `FORM_FONT_PATH`, else DejaVu/Liberation, else Arial/Segoe UI (must
contain ăâîșț). The filled pages become a new document that can go straight through `detect_pii → redact_image`.

**Verification:** the redacted image is re-OCR'd; text hallucinated on or around redacted regions is ignored;
the detector runs on the remaining visible text and every original PII value is searched for in the new text.
Any hit → `FAIL`.

---

## Setup (Windows)

```powershell
cd "D:\SI\PrivacyGuard AI-si"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env
```

* **Tesseract 5** with `ron` and `eng` traineddata (e.g. UB-Mannheim build). Set `TESSERACT_CMD` in `.env`
  (default in the template: `D:\Tesseract\tesseract.exe`).
* **LLM provider** — set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`, or use `--provider`:
  * Gemini (cloud): `https://generativelanguage.googleapis.com/v1beta/openai/`, model `gemini-2.5-flash`
    (`GEMINI_API_KEY` or `LLM_API_KEY`).
  * Mistral (cloud): `https://api.mistral.ai/v1`.
  * Ollama (local): `ollama pull qwen2.5:7b`, base URL `http://localhost:11434/v1`.
  * LM Studio (local): `http://localhost:1234/v1`.
* Optional NER: `pip install spacy && python -m spacy download ro_core_news_sm`.

## Usage

```powershell
# 1. Generate the synthetic dataset (already committed; re-run to regenerate)
python -m evaluation.generate_synthetic_docs

# 2. Run the agent (cloud or local LLM)
python -m agent.cli_agent --provider gemini --dir synthetic_data
python -m agent.cli_agent --provider ollama synthetic_data\01_id_card_P1.png synthetic_data\02_form_P1.png
python -m agent.cli_agent --provider env --max-steps 40 --no-tts --dir synthetic_data "Anonymize and link"

# 3. MCP server standalone / MCP Inspector
python -m server.mcp_server
npx @modelcontextprotocol/inspector .venv\Scripts\python.exe -m server.mcp_server

# 4. Smoke test of all five tools over stdio (no LLM)
python -m tests.smoke_mcp_client

# 5. Tests and evaluation
python -m pytest -q
python -m evaluation.run_eval
python -m evaluation.generate_synthetic_docs --seed 2026 --out output/heldout_data
python -m evaluation.run_eval --data output/heldout_data --tag heldout
```

**Agent resilience:** `MAX_STEPS` cap; exponential backoff with jitter (and `Retry-After`) for HTTP 408/409/429/5xx,
timeouts and connection errors; malformed tool arguments, unknown aliases and MCP tool errors are returned to
the LLM as structured `{"error": ..., "tool": ...}` tool messages so it can recover. MCP JSON schemas are cleaned
(`anyOf[null]`, `title`, `default`) for strict providers such as Gemini.

**Workflow guardrail (for small local models):** the agent tracks the workflow by id only. Indexing the
same file alias twice returns the cached `doc_id` (no duplicate document that would falsely link to itself),
and if the model stops early — e.g. asks *"Would you like me to link the documents?"* — the agent lists the
pending steps (unprocessed files, unverified redactions, missing `link_documents`) and lets it continue (max 3 nudges).

**Tested with a real local model:** `qwen2.5:7b` via Ollama on CPU, all 12 synthetic documents: 12 OCR /
detect / redact / verify calls, 12/12 PASS, `link_documents` → 4 clusters covering all 12 documents
(one nudge was needed), ~7 minutes end to end.

**TTS:** at the end of every run `agent/tts_notifier.py` speaks e.g. *"Task completed. 12 documents processed,
92 PII regions redacted. Verification: 12 passed, 0 failed. 4 document link groups found, covering 12 documents."*
using `pyttsx3` (Windows SAPI5, offline, CPU) with a `piper` CLI fallback (`PIPER_EXE`, `PIPER_VOICE`).

---

## Web GUI

```powershell
python -m ui.api                 # opens http://127.0.0.1:8000
python -m ui.api --port 8010 --no-browser
```

A black & white terminal-style web app (FastAPI backend in `ui/api.py`, vanilla HTML/CSS/JS in `ui/static/`,
no build step, no CDN) on top of the same pipeline:

1. **Login** — username + access passphrase from `.env` (`UI_USERNAME`, `UI_PASSPHRASE`; local-dev defaults
   `admin` / `privacyguard` — change them). Sessions are random tokens kept in memory (HttpOnly, SameSite=Lax cookie, 12 h; a server restart logs you out).
2. **Folder hub** — create, open and delete document sets; each card shows creation time and `redacted/total`.
3. **Folder dashboard** — drag & drop PDF/PNG/JPG files (or click to browse). Every file is queued and run through
   `ocr_document → detect_pii → redact_image → verify_redaction` automatically by one background worker. Status goes
   `[ QUEUED ] → [ SCANNING... ] → [ BLURRED ]` (`[ REVIEW ]` if the verification pass still finds PII-like text,
   `[ ERROR ]` on failure). Per file: **Preview** (original / redacted / side-by-side, page stepper for PDFs,
   zoom / pan) with a **`[ + MANUAL BLUR BOX ]`** safety net — drag a rectangle over anything left readable and
   the same irreversible blur is applied at once (`POST …/documents/{doc}/manual-redact`; boxes survive a rescan,
   *Undo manual* reprocesses the document) — **Download** (PNG, or one PDF for multi-page documents),
   **Rescan**, and **Download all (.zip)** for the whole folder.
4. **Blur calibration** — a folder panel (strength 1–10, padding 0–24 px, `IRREVERSIBLE BLUR` / `SOLID BLACK`,
   *Save default*, *Apply & re-verify all*) and the same controls in the preview for one document. Re-apply re-runs
   redaction + verification **without OCR** from `out/work/<doc>_plan.json` (box coordinates + salted value
   fingerprints, no text); new uploads inherit the folder default.
5. **`[ AUTO-FILL FROM ID ]`** — pick (or upload as `[ SOURCE ]`, not redacted) an ID card and a blank form, add
   optional Telefon / E-mail / Data / Scop, and *Fill form*: the worker produces a `[ FILLED ]` document (preview
   *Blank* vs *Filled*, download PNG/PDF, **Blur copy** = an anonymized copy that is redacted automatically).
   Filled forms contain personal data and are never included in *Download all (.zip)*; `metadata.json` stores only
   which field keys were filled, and the optional typed values are kept in memory only.

Folders persist on disk in `workspace_folders/<folder_id>/` (git-ignored; override with `UI_WORKSPACE`):
`metadata.json` (names, timestamps, statuses, counts, numeric blur calibration and filled field *keys* only),
`original/` (uploads) and `out/` (pipeline output).
Deleting a folder removes all of it.

Privacy notes: the server binds to `127.0.0.1` only and every API route except login requires a session. OCR text
and PII values live only in the in-memory vault and are cleared after each document; nothing about them is written
to `metadata.json` or sent to the page. Uploaded originals stay on disk until you delete the document or folder.
The web app always uses the direct pipeline; the LLM agent remains available from the CLI.

## Evaluation

12 synthetic documents: 4 fictional people × (ID card, application form, utility invoice); 7 fonts,
rotations from −3.5° to +4°, Gaussian noise (σ 0/6/12), blur, JPEG compression for invoices. People 3 and 4
share a surname (hard negative for linking). Detection is scored by matching values against ground truth
(tolerating OCR character errors); redaction is scored by re-OCR'ing the redacted image and searching for every
**ground-truth** value (independent of the detector). Full reports: [evaluation/results.md](evaluation/results.md),
[evaluation/results_heldout.md](evaluation/results_heldout.md).

| Dataset | Detection P / R / F1 (micro) | GT values readable after redaction | verify PASS | Linking P / R (pairs) |
|---|---|---|---|---|
| Development set (seed 7) | 1.000 / 1.000 / 1.000 | 0 / 92 | 12 / 12 | 1.000 / 1.000 |
| Held-out values (seed 2026) | 1.000 / 0.989 / 0.995 | 2 / 92 | 11 / 12 | 1.000 / 1.000 |

**Read these numbers honestly:** the detector rules were tuned on the development set, and the held-out set
uses the *same three templates* with new names/values, so it measures robustness to new values, fonts and noise
— not to unseen document layouts. In the held-out run one OCR-garbled e-mail was missed; `verify_redaction`
correctly reported that document as `FAIL`, which is exactly the safety net the agent uses (re-redact with
larger padding / flag for review). Real-world scans with other layouts will score lower.

## Project structure

```
server/       MCP server, vault, OCR, PII detector, linker, redactor, pipeline, id_extractor, form_filler
agent/        CLI agent loop + offline TTS
ui/           web terminal: api.py (FastAPI, login + folders) and static/ (index.html, app.js, styles.css)
evaluation/   synthetic dataset generator, evaluation suite, results*.md
synthetic_data/  12 generated documents + ground_truth.json
tests/        unit tests, pipeline tests, agent-loop test with a scripted fake LLM, MCP smoke client
output/       redacted images, work files, eval reports (git-ignored)
```

## Notes

* **MCP SDK version:** the project uses `mcp` 2.x, where `FastMCP` was renamed to `MCPServer`
  (`from mcp.server.mcpserver import MCPServer`). The decorator API (`@mcp.tool()`) and stdio transport are the
  same; `mcp<2` users can swap the import for `from mcp.server.fastmcp import FastMCP`.
* The vault is in-memory: restarting the server clears documents, detections and the link graph, and a new
  random HMAC secret makes old hashes uncorrelatable (by design).
* `output_path` reveals only the project folder and the opaque `doc_id`.
