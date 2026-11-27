"""Agent loop integration test with a scripted fake LLM (no network, no API key).

Covers: MCP tool wiring, file aliasing, retry on HTTP 503, tool errors returned to the
LLM as tool messages, privacy guard, MAX_STEPS, and final stats.
"""
import asyncio
import json
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
