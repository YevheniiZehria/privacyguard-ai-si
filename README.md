# PrivacyGuard AI — Proiect #03: Anonimizarea actelor

An autonomous **CLI agent** connected to a local **MCP server** that:

1. OCRs scanned Romanian documents (ID cards, application forms, invoices, PDFs) with OpenCV + Tesseract,
2. detects PII (CNP with official checksum, names, addresses, ID series, IBAN, phone, e-mail, birth date, MRZ),
3. redacts it with solid black boxes and **verifies** the redaction by re-running OCR,
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
 │ server/redactor.py       filled (0,0,0) rectangles → output/redacted_<doc_id>.png     │
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
| `redact_image` | `doc_id, detection_id, padding_px=4` | `result_id, output_path, boxes_drawn, page_count` |
| `verify_redaction` | `result_id` | `result_id, verified, residual_pii_count, status: PASS/FAIL` |

**OCR pipeline:** grayscale → deskew (projection-profile search ±10°) → 2× upscale for small scans →
non-local-means denoising → adaptive Gaussian threshold → Tesseract `ron+eng` (`image_to_data`, word boxes),
with a grayscale fallback when confidence is low. Redaction is drawn on the deskewed original-resolution
image so boxes align exactly. PDFs are rendered at 300 DPI per page.

**PII detection:** CNP regex + official checksum (key `279146358279`, date and county validation) plus a
lenient OCR-confusion pass after a `CNP` label; labelled `SERIA XX NR 123456` and bare series (in ID
context); IBAN with mod-97; Romanian phone numbers; e-mail (also when OCR reads `@` as `Q`); names after
labels (`Nume`, `Prenume`, `Subsemnatul/a`, `Client`, `Titular`, `Semnătura`, …, including bilingual ID-card
labels with the value on the next line); addresses after `Domiciliu`/`Adresa`/`Domiciliat în` or starting with
`Str./Bd./Calea…`, including continuation lines; MRZ lines. Detected name tokens are propagated to other
occurrences (signatures). Optional: spaCy `ro_core_news_sm`, local Ollama SLM.

**Verification:** the redacted image is re-OCR'd; text hallucinated on the edges of black boxes is ignored;
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

**TTS:** at the end of every run `agent/tts_notifier.py` speaks e.g. *"Task completed. 12 documents processed,
92 PII regions redacted. Verification: 12 passed, 0 failed. 4 document link groups found, covering 12 documents."*
using `pyttsx3` (Windows SAPI5, offline, CPU) with a `piper` CLI fallback (`PIPER_EXE`, `PIPER_VOICE`).

---

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
server/       MCP server, vault, OCR, PII detector, linker, redactor, pipeline
agent/        CLI agent loop + offline TTS
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
