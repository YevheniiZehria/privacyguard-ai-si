"""End-to-end smoke test: spawn the MCP server over stdio and exercise all 5 tools.

Usage:  python -m tests.smoke_mcp_client [image ...]
Without arguments it uses the first 3 synthetic documents (run evaluation/generate_synthetic_docs.py first).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def payload(result) -> dict:
    if result.is_error:
        return {"error": " ".join(getattr(c, "text", "") for c in result.content)}
    if result.structured_content:
        return result.structured_content
    return json.loads(result.content[0].text)


async def main(files: list[str]) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "server.mcp_server"], cwd=str(ROOT))
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", [t.name for t in tools.tools])
        doc_ids = []
        for f in files:
            ocr = payload(await client.call_tool("ocr_document", {"file_path": f}))
            print("ocr_document     ", ocr)
            det = payload(await client.call_tool("detect_pii", {"doc_id": ocr["doc_id"]}))
            print("detect_pii       ", det)
            red = payload(await client.call_tool("redact_image", {"doc_id": ocr["doc_id"],
                                                                  "detection_id": det["detection_id"]}))
            print("redact_image     ", red)
            ver = payload(await client.call_tool("verify_redaction", {"result_id": red["result_id"]}))
            print("verify_redaction ", ver)
            doc_ids.append(ocr["doc_id"])
        print("link_documents   ", json.dumps(payload(await client.call_tool("link_documents", {})), indent=1))
        print("error handling   ", payload(await client.call_tool("detect_pii", {"doc_id": "doc_missing"})))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        gt = ROOT / "synthetic_data" / "ground_truth.json"
        args = [d["file"] for d in json.loads(gt.read_text(encoding="utf-8"))["documents"][:3]]
    asyncio.run(main(args))
