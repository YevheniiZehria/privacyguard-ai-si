"""Agent loop integration test with a scripted fake LLM (no network, no API key).

Covers: MCP tool wiring, file aliasing, retry on HTTP 503, tool errors returned to the
LLM as tool messages, privacy guard, MAX_STEPS, and final stats.
"""
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import InternalServerError

from agent import cli_agent
from agent.cli_agent import privacy_guard, run_agent

ROOT = Path(__file__).resolve().parent.parent


def _msg(content=None, calls=None):
    tcs = [SimpleNamespace(id=f"c{i}_{n}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls or [])]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tcs or None))])


class ScriptedLLM:
    """Plays the role of the LLM: drives the documented workflow by reading tool results."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.seen_messages = []

    async def create(self, model, messages, **_):
        self.calls += 1
        self.seen_messages = messages
        if self.calls == 1:   # transient provider failure -> must be retried
            req = httpx.Request("POST", "http://fake/v1/chat/completions")
            raise InternalServerError("unavailable", response=httpx.Response(503, request=req), body=None)
        last = messages[-1]
        if last["role"] == "user":
            return _msg(calls=[("ocr_document", {"file_path": "FILE_1"}), ("ocr_document", {"file_path": "NOPE"})])
        results = {}
        for m in messages:
            if m["role"] == "tool":
                results.setdefault("all", []).append(json.loads(m["content"]))
        latest = results["all"][-1]
        if "doc_id" in latest and "detection_id" not in latest and latest.get("status") == "indexed":
            return _msg(calls=[("detect_pii", {"doc_id": latest["doc_id"]})])
        if "detection_id" in latest:
            return _msg(calls=[("redact_image", {"doc_id": latest["doc_id"], "detection_id": latest["detection_id"]})])
        if "boxes_drawn" in latest:
            return _msg(calls=[("verify_redaction", {"result_id": latest["result_id"]})])
        if "verified" in latest:
            return _msg(calls=[("link_documents", {})])
        if "clusters" in latest:
            return _msg(content="Done.")
        # an error message (e.g. the bad alias) — continue with whatever is pending
        ocr_ok = [r for r in results["all"] if r.get("status") == "indexed"]
        return _msg(calls=[("detect_pii", {"doc_id": ocr_ok[-1]["doc_id"]})])


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from server.pii_detector import cnp_control_digit
    c = "296101233123"
    c += cnp_control_digit(c)
    img = Image.new("RGB", (1500, 300), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("arial.ttf", 36)
    d.text((40, 40), f"Subsemnata Ionescu Maria, CNP {c}", fill="black", font=f)
    d.text((40, 120), "domiciliata in Str. Florilor nr. 3, Iasi", fill="black", font=f)
    path = tmp_path / "sample.png"
    img.save(path)
    return str(path), c


def test_agent_loop_end_to_end(sample_image, monkeypatch):
    path, cnp = sample_image
    real_sleep = asyncio.sleep
    monkeypatch.setattr(cli_agent.asyncio, "sleep", lambda *_: real_sleep(0))
    llm = ScriptedLLM()
    code = asyncio.run(run_agent("Anonymize", [path], "env", "fake", max_steps=12, tts=False, verbose=True, llm=llm))
    assert code == 0
    transcript = json.dumps(llm.seen_messages)
    assert cnp not in transcript and "Ionescu" not in transcript and path not in transcript
    assert "Unknown file alias" in transcript          # tool error went back to the LLM
    assert any('"status": "PASS"' in m.get("content", "") for m in llm.seen_messages)


def test_max_steps_stops_loop(sample_image, monkeypatch):
    path, _ = sample_image

    class Looping:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **_):
            return _msg(calls=[("link_documents", {})])

    code = asyncio.run(run_agent("x", [path], "env", "fake", max_steps=3, tts=False, verbose=False, llm=Looping()))
    assert code == 1


def test_privacy_guard_blocks_raw_values():
    assert privacy_guard({"doc_id": "doc_1", "summary": {"CNP": 1}}) == []
    assert privacy_guard({"text": "Ion Popescu"})
    assert privacy_guard({"doc_id": "1960523331234"})


class LazyLLM:
    """Small-model behaviour: indexes FILE_1 twice, then stops and asks a question instead of linking."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.turn = 0
        self.nudged = False
        self.doc_ids = []

    async def create(self, messages, **_):
        self.turn += 1
        tool_msgs = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        self.doc_ids = [t["doc_id"] for t in tool_msgs if t.get("status") == "indexed"]
        if messages[-1]["role"] == "user" and "Still pending" in messages[-1]["content"]:
            self.nudged = True
            return _msg(calls=[("link_documents", {})])
        if self.turn == 1:
            return _msg(calls=[("ocr_document", {"file_path": "FILE_1"}), ("ocr_document", {"file_path": "FILE_1"})])
        if self.turn == 2:
            return _msg(calls=[("ocr_document", {"file_path": "FILE_2"})])
        if self.turn == 3:
            return _msg(content="Would you like me to link the documents?")
        return _msg(content="Done.")


def test_duplicate_ocr_is_cached_and_skipped_steps_are_nudged(sample_image, tmp_path):
    path, _ = sample_image
    from PIL import Image
    second = tmp_path / "second.png"
    Image.open(path).save(second)
    llm = LazyLLM()
    asyncio.run(run_agent("x", [path, str(second)], "env", "fake", max_steps=10, tts=False, verbose=False, llm=llm))
    assert llm.doc_ids[0] == llm.doc_ids[1]          # FILE_1 twice -> same doc_id, no duplicate in the vault
    assert len(set(llm.doc_ids)) == 2
    assert llm.nudged                                 # the agent pushed the model to finish the workflow


# ----------------------------------------------------------------------------- self-correction on FAIL
def test_escalation_args():
    prev = {"doc_id": "doc_1", "detection_id": "det_1"}
    weak = {"status": "FAIL", "leak_count": 1, "uncovered_count": 0}
    assert cli_agent.escalation_args(prev, weak) == {"doc_id": "doc_1", "detection_id": "det_1",
                                                     "blur_strength": "maximum", "padding_px": 10}
    short = {"status": "FAIL", "leak_count": 0, "uncovered_count": 2, "suggested_padding_px": 16}
    assert cli_agent.escalation_args({**prev, "padding_px": 10}, short)["padding_px"] == 22
    assert cli_agent.escalation_args({**prev, "padding_px": 45}, short)["padding_px"] == 50   # capped
    assert "maximum" in cli_agent.SYSTEM_PROMPT and "FAIL" in cli_agent.SYSTEM_PROMPT


class FakeMCP:
    """In-process stand-in for the MCP server: the first verification FAILs (weak blur), later ones PASS
    only when the redaction was escalated to blur_strength='maximum'."""

    calls: list = []

    def __init__(self, *_a, **_k):
        FakeMCP.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def list_tools(self):
        tool = lambda n: SimpleNamespace(name=n, description=n, input_schema={"type": "object", "properties": {}})
        return SimpleNamespace(tools=[tool(n) for n in ("ocr_document", "detect_pii", "redact_image",
                                                         "verify_redaction", "link_documents")])

    async def call_tool(self, name, args):
        FakeMCP.calls.append((name, dict(args)))
        n = len(FakeMCP.calls)
        if name == "ocr_document":
            out = {"doc_id": "doc_a", "word_count": 20, "avg_confidence": 91.0, "image_dimensions": [1500, 300],
                   "page_count": 1, "status": "indexed"}
        elif name == "detect_pii":
            out = {"detection_id": "det_a", "doc_id": "doc_a", "summary": {"CNP": 1}, "total_boxes": 2,
                   "linkable_identifiers": 1}
        elif name == "redact_image":
            strength = args.get("blur_strength", "high")
            out = {"result_id": f"res_{n}", "output_path": "out/redacted_doc_a.png", "boxes_drawn": 2,
                   "page_count": 1, "blur_strength": strength, "blur_level": 10 if strength == "maximum" else 7,
                   "mode": "blur"}
            FakeMCP.last_strength = strength
        elif name == "verify_redaction":
            ok = FakeMCP.last_strength == "maximum"
            out = {"result_id": args["result_id"], "verified": ok, "residual_pii_count": 0 if ok else 1,
                   "status": "PASS" if ok else "FAIL", "leak_count": 0 if ok else 1, "uncovered_count": 0,
                   "suggested_blur_strength": None if ok else "maximum", "suggested_padding_px": None if ok else 10}
        else:
            out = {"total_documents": 1, "clusters": [], "unlinked_doc_ids": ["doc_a"], "unknown_doc_ids": []}
        return SimpleNamespace(is_error=False, structured_content=out, content=[])


class StubbornLLM:
    """Runs the workflow, then tries to finish right after a FAIL; follows the agent's nudge literally."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.nudges = []

    async def create(self, messages, **_):
        last = messages[-1]
        if last["role"] == "user" and "Still pending" in last["content"]:
            self.nudges.append(last["content"])
            m = re.search(r"redact_image\(doc_id='(\w+)', detection_id='(\w+)', blur_strength='(\w+)', "
                          r"padding_px=(\d+)\)", last["content"])
            if m:
                return _msg(calls=[("redact_image", {"doc_id": m[1], "detection_id": m[2], "blur_strength": m[3],
                                                     "padding_px": int(m[4])})])
            return _msg(content="Done.")
        if last["role"] == "user":
            return _msg(calls=[("ocr_document", {"file_path": "FILE_1"})])
        res = json.loads(last["content"])
        if res.get("status") == "indexed":
            return _msg(calls=[("detect_pii", {"doc_id": res["doc_id"]})])
        if "detection_id" in res:
            return _msg(calls=[("redact_image", {"doc_id": "doc_a", "detection_id": res["detection_id"]})])
        if "boxes_drawn" in res:
            return _msg(calls=[("verify_redaction", {"result_id": res["result_id"]})])
        return _msg(content="All done.")                 # ignores the FAIL -> must be nudged


def test_agent_escalates_after_verify_fail(sample_image, monkeypatch):
    path, _ = sample_image
    monkeypatch.setattr(cli_agent, "Client", FakeMCP)
    events = []
    llm = StubbornLLM()
    code = asyncio.run(run_agent("x", [path], "env", "fake", max_steps=12, tts=False, verbose=False, llm=llm,
                                 on_event=events.append))
    assert code == 0
    assert llm.nudges and "verification FAILED" in llm.nudges[0] and "blur_strength='maximum'" in llm.nudges[0]
    redacts = [a for n, a in FakeMCP.calls if n == "redact_image"]
    assert len(redacts) == 2 and redacts[1]["blur_strength"] == "maximum" and redacts[1]["padding_px"] >= 10
    verifies = [e["result"] for e in events if e.get("type") == "tool" and e["tool"] == "verify_redaction"]
    assert [v["status"] for v in verifies] == ["FAIL", "PASS"]
    assert all(e.get("privacy_guard") == "clean" for e in events if e.get("type") == "tool")   # new keys allowed
    stats = next(e["stats"] for e in events if e["type"] == "done")
    assert stats["verified_pass"] == 1 and stats["verified_fail"] == 0      # per document, latest outcome


# ----------------------------------------------------------------------------- form filling
class FillMCP(FakeMCP):
    """FILE_1 = blank form, FILE_2 = ID card; only the ID card reports is_id_card."""

    async def list_tools(self):
        tool = lambda n: SimpleNamespace(name=n, description=n, input_schema={"type": "object", "properties": {}})
        return SimpleNamespace(tools=[tool(n) for n in ("ocr_document", "extract_id_profile", "fill_form_from_id",
                                                         "detect_pii", "redact_image", "verify_redaction",
                                                         "link_documents")])

    async def call_tool(self, name, args):
        FakeMCP.calls.append((name, dict(args)))
        if name == "ocr_document":
            d = "doc_form" if args["file_path"].endswith("form.png") else "doc_id"
            out = {"doc_id": d, "word_count": 30, "avg_confidence": 90.0, "image_dimensions": [1500, 900],
                   "page_count": 1, "status": "indexed"}
        elif name == "extract_id_profile":
            is_id = args["doc_id"] == "doc_id"
            fields = ["full_name", "cnp", "id_series", "id_number", "address_full"] if is_id else []
            out = {"doc_id": args["doc_id"], "is_id_card": is_id, "is_valid_id": is_id, "detected_fields": fields,
                   "field_count": len(fields), "mean_confidence": 0.9 if is_id else 0.0}
        elif name == "fill_form_from_id":
            out = {"filled_doc_id": "fill_1", "output_path": "out/filled_fill_1.png", "pdf_path": None,
                   "fields_filled": ["full_name", "cnp", "address_full"], "unfilled_fields": ["phone"],
                   "page_count": 1}
        else:
            return await super().call_tool(name, args)
        return SimpleNamespace(is_error=False, structured_content=out, content=[])


class FillLLM:
    """Follows the documented fill workflow, then reports (no redaction, no linking)."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.nudged = False
        self.profiles = {}

    async def create(self, messages, **_):
        last = messages[-1]
        if last["role"] == "user" and "Still pending" in last["content"]:
            self.nudged = True
            return _msg(content="Done.")
        if last["role"] == "user":
            return _msg(calls=[("ocr_document", {"file_path": "FILE_1"}), ("ocr_document", {"file_path": "FILE_2"})])
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        docs = [r["doc_id"] for r in results if r.get("status") == "indexed"]
        prof = {r["doc_id"]: r for r in results if "is_id_card" in r}
        if not prof:
            return _msg(calls=[("extract_id_profile", {"doc_id": d}) for d in docs])
        if not any("filled_doc_id" in r for r in results):
            id_doc = next(d for d, r in prof.items() if r["is_id_card"])
            form = next(d for d in docs if d != id_doc)
            return _msg(calls=[("fill_form_from_id", {"id_doc_id": id_doc, "form_doc_id": form})])
        return _msg(content="Filled: full_name, cnp, address_full; unfilled: phone.")


def test_agent_fills_form_without_redaction_nudges(tmp_path, monkeypatch):
    from PIL import Image
    files = []
    for n in ("form.png", "id.png"):
        Image.new("RGB", (50, 50), "white").save(tmp_path / n)
        files.append(str(tmp_path / n))
    monkeypatch.setattr(cli_agent, "Client", FillMCP)
    events = []
    llm = FillLLM()
    code = asyncio.run(run_agent("Completează cererea cu datele din buletin", files, "env", "fake", max_steps=8,
                                 tts=False, verbose=False, llm=llm, on_event=events.append))
    assert code == 0 and not llm.nudged                 # fill-only task: no redaction / link_documents nudges
    fill = [a for n, a in FakeMCP.calls if n == "fill_form_from_id"]
    assert fill == [{"id_doc_id": "doc_id", "form_doc_id": "doc_form"}]
    assert not any(n in ("redact_image", "link_documents") for n, _ in FakeMCP.calls)
    assert all(e.get("privacy_guard") == "clean" for e in events if e.get("type") == "tool")
    assert "extract_id_profile" in cli_agent.SYSTEM_PROMPT and "fill_form_from_id" in cli_agent.SYSTEM_PROMPT
