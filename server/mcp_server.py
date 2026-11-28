"""PrivacyGuard MCP server (stdio).

Built with the official Python MCP SDK v2 ``MCPServer`` (the v2 name of ``FastMCP``).
Tools exchange ONLY opaque ids, counts, scores and booleans — never raw document text.

Run:        python -m server.mcp_server
Inspector:  npx @modelcontextprotocol/inspector .venv\\Scripts\\python.exe -m server.mcp_server
"""
from __future__ import annotations

import logging
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from server.pipeline import PrivacyPipeline

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("privacyguard")

mcp = MCPServer(
    "privacyguard",
    instructions=(
        "Local privacy vault for document anonymization. Workflow per document: "
        "ocr_document -> detect_pii -> redact_image -> verify_redaction. If verify_redaction returns FAIL, "
        "call redact_image again with an escalated blur_strength (e.g. high -> maximum) and a larger padding_px "
        "(leak_count > 0: blur too weak; uncovered_count > 0: boxes/padding too small; suggested_* fields give the "
        "next step), then verify again. To fill a blank form from an ID card: ocr_document both files, call "
        "extract_id_profile on each (the one with is_id_card=true is the ID), then fill_form_from_id(id_doc_id, "
        "form_doc_id); the returned filled_doc_id can go through detect_pii -> redact_image like any document. "
        "Call link_documents after "
        "detect_pii on several documents to find which ones belong to the same person. "
        "You only ever receive opaque ids and counts; raw PII never leaves this server."
    ),
)
pipeline = PrivacyPipeline()


# ----------------------------------------------------------------------------- schemas
class OcrResult(BaseModel):
    doc_id: str = Field(description="Opaque document id")
    word_count: int
    avg_confidence: float = Field(description="Mean Tesseract word confidence (0-100)")
    image_dimensions: list[int] = Field(description="[width, height] of page 1 in pixels")
    page_count: int
    status: Literal["indexed"]


class PIISummary(BaseModel):
    CNP: int = 0
    PERSON_NAME: int = 0
    ADDRESS: int = 0
    ID_SERIES: int = 0
    IBAN: int = 0
    PHONE: int = 0
    EMAIL: int = 0
    BIRTH_DATE: int = 0
    MRZ: int = 0


class DetectionResult(BaseModel):
    detection_id: str
    doc_id: str
    summary: PIISummary = Field(description="Number of detected entities per PII type")
    total_boxes: int = Field(description="Number of merged bounding boxes to redact")
    linkable_identifiers: int = Field(description="Salted hashes added to the cross-document link graph")


class DocLink(BaseModel):
    doc_a: str
    doc_b: str
    shared_entity_types: list[str]
    shared_entity_count: int
    confidence: float


class DocCluster(BaseModel):
    cluster_id: str
    doc_ids: list[str]
    shared_entity_types: list[str]
    link_strength: float = Field(description="Weakest pairwise link confidence inside the cluster")
    links: list[DocLink]


class LinkResult(BaseModel):
    total_documents: int
    clusters: list[DocCluster]
    unlinked_doc_ids: list[str]
    unknown_doc_ids: list[str]


class RedactionResult(BaseModel):
    result_id: str
    output_path: str
    boxes_drawn: int
    page_count: int
    blur_strength: str = Field(description="Preset name (low/medium/high/maximum) or the numeric level")
    blur_level: int = Field(description="Blur level 1..10 actually applied")
    mode: Literal["blur", "black"]


class VerificationResult(BaseModel):
    result_id: str
    verified: bool
    residual_pii_count: int
    status: Literal["PASS", "FAIL"]
    leak_count: int = Field(description="Known PII values still readable (blur too weak -> raise blur_strength)")
    uncovered_count: int = Field(description="PII-like text outside the boxes (coverage short -> raise padding_px)")
    suggested_blur_strength: Literal["low", "medium", "high", "maximum"] | None = None
    suggested_padding_px: int | None = None


class IDProfileResult(BaseModel):
    doc_id: str
    is_id_card: bool = Field(description="The document looks like a Romanian ID card (CI / buletin)")
    is_valid_id: bool = Field(description="A checksum-valid CNP and a series/number were found")
    detected_fields: list[str] = Field(description="Profile field KEYS found (values stay in the vault)")
    field_count: int
    mean_confidence: float


class FillResult(BaseModel):
    filled_doc_id: str = Field(description="New document id of the filled form (usable with detect_pii)")
    output_path: str
    pdf_path: str | None
    fields_filled: list[str] = Field(description="Form field KEYS that were written")
    unfilled_fields: list[str] = Field(description="Form field KEYS found on the form but without a value")
    page_count: int


def _safe(op: str, fn, *args, **kwargs):
    """Run a pipeline step; convert failures into ToolErrors with non-sensitive messages."""
    try:
        return fn(*args, **kwargs)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        log.warning("%s failed: %s", op, msg)
        raise ToolError(f"{op} failed: {msg}") from None
    except Exception as exc:  # never echo internals that could contain document text
        log.exception("%s crashed", op)
        raise ToolError(f"{op} failed with {type(exc).__name__}; see server log.") from None


# ----------------------------------------------------------------------------- tools
@mcp.tool()
def ocr_document(
    file_path: Annotated[str, Field(description="Path to a PNG/JPG/TIFF/PDF document (absolute or project-relative)")],
) -> OcrResult:
    """Preprocess (deskew, denoise, threshold) and OCR a document; store text in the local vault. Returns a doc_id."""
    return OcrResult(**_safe("ocr_document", pipeline.ocr_document, file_path))


@mcp.tool()
def detect_pii(
    doc_id: Annotated[str, Field(description="doc_id returned by ocr_document")],
    use_local_slm: Annotated[bool, Field(description="Also use a LOCAL Ollama model for names/addresses")] = False,
) -> DetectionResult:
    """Detect PII (CNP with checksum, names, addresses, ID series, IBAN, phone, email) and update the link graph."""
    return DetectionResult(**_safe("detect_pii", pipeline.detect_pii, doc_id, use_local_slm))


@mcp.tool()
def link_documents(
    doc_ids: Annotated[list[str] | None, Field(description="Subset of doc_ids to link; omit for all")] = None,
) -> LinkResult:
    """Group documents that belong to the same person via matching salted identifier hashes."""
    return LinkResult(**_safe("link_documents", pipeline.link_documents, doc_ids))


@mcp.tool()
def redact_image(
    doc_id: Annotated[str, Field(description="doc_id returned by ocr_document")],
    detection_id: Annotated[str, Field(description="detection_id returned by detect_pii")],
    padding_px: Annotated[int, Field(ge=0, le=50, description="Extra pixels around each box")] = 4,
    blur_strength: Annotated[Literal["low", "medium", "high", "maximum"] | Annotated[int, Field(ge=1, le=10)],
                             Field(description="Blur intensity: low|medium|high|maximum or 1..10. "
                                               "Escalate to 'maximum' after a FAIL verification")] = "high",
) -> RedactionResult:
    """Irreversibly blur every detected PII region (or black-box it with REDACTION_MODE=black) and save output/redacted_<doc_id>.png.
    Blur cell size, kernel and noise adapt to text height, image resolution and contrast; blur_strength raises them further."""
    return RedactionResult(**_safe("redact_image", pipeline.redact_image, doc_id, detection_id, padding_px,
                                   blur_strength))


@mcp.tool()
def verify_redaction(
    result_id: Annotated[str, Field(description="result_id returned by redact_image")],
) -> VerificationResult:
    """Re-run OCR + PII detection on the redacted image and confirm nothing readable remains.
    On FAIL, re-run redact_image with suggested_blur_strength / suggested_padding_px and verify again."""
    return VerificationResult(**_safe("verify_redaction", pipeline.verify_redaction, result_id))


@mcp.tool()
def extract_id_profile(
    doc_id: Annotated[str, Field(description="doc_id of an ID card returned by ocr_document")],
) -> IDProfileResult:
    """Structure the ID card's fields (names, CNP, series/number, address parts, dates, issuer) inside the vault.
    Returns only which field keys were found — never the values."""
    return IDProfileResult(**_safe("extract_id_profile", pipeline.extract_id_profile, doc_id))


@mcp.tool()
def fill_form_from_id(
    id_doc_id: Annotated[str, Field(description="doc_id of the ID card (after extract_id_profile)")],
    form_doc_id: Annotated[str, Field(description="doc_id of the blank form returned by ocr_document")],
    extra_fields: Annotated[dict[str, str] | None, Field(
        description="Optional values the user gave that are not on the ID: phone, email, date, purpose")] = None,
) -> FillResult:
    """Fill the blank form (PDF form fields or flat scan/image) with the ID card's data; writes
    output/filled_<id>.png (+ .pdf) and registers it as a new document. Returns field keys only."""
    return FillResult(**_safe("fill_form_from_id", pipeline.fill_form_from_id, id_doc_id, form_doc_id,
                              extra_fields))


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
