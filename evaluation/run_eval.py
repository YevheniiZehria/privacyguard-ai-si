"""Automated evaluation on the synthetic dataset (runs the pipeline directly, no LLM).

Metrics
  * Entity detection: Precision / Recall / F1 per PII type (value matching after normalization).
  * Redaction: residual PII values still readable after re-OCR of the redacted image (leak check
    against GROUND TRUTH, independent of the detector) + verify_redaction PASS rate.
  * Linking: pairwise Precision / Recall / F1 of "same person" document pairs vs ground truth.

Usage: python -m evaluation.run_eval   (generates the dataset first if missing)
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from tabulate import tabulate

from server.image_preprocessing import load_bgr
from server.ocr_engine import ocr_image
from server.pii_detector import build_lines, strip_diacritics
from server.pipeline import PrivacyPipeline

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "synthetic_data" / "ground_truth.json"
OUT_DIR = ROOT / "output" / "eval"
TYPES = ["CNP", "PERSON_NAME", "ADDRESS", "ID_SERIES", "IBAN", "PHONE", "EMAIL", "BIRTH_DATE", "MRZ"]


def norm(etype: str, value: str) -> str:
    v = re.sub(r"[^A-Z0-9]", "", strip_diacritics(value).upper())
    if etype == "ID_SERIES":
        v = v.replace("NR", "")
    if etype == "PHONE":
        v = v[-9:]
    return v


def matches(etype: str, gt_value: str, det_value: str) -> bool:
    g, d = norm(etype, gt_value), norm(etype, det_value)
    if not g or not d:
        return False
    if g == d:
        return True
    # tolerate OCR character errors (e.g. IDROU -> TDROU, 56@ -> 56e@): the region is still found and redacted
    if SequenceMatcher(None, g, d).ratio() >= 0.85:
        return True
    if etype in ("ADDRESS", "MRZ", "PERSON_NAME"):
        # multi-line / partial spans: most of the GT characters must appear (in order) inside the detection
        sm = SequenceMatcher(None, g, d, autojunk=False)
        covered = sum(b.size for b in sm.get_matching_blocks())
        return covered >= 0.85 * len(g)
    return False


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(GT_PATH.parent), help="Folder with ground_truth.json")
    ap.add_argument("--tag", default="", help="Report suffix, e.g. heldout")
    a = ap.parse_args()
    gt_path = Path(a.data).resolve() / "ground_truth.json"
    if not gt_path.exists():
        from evaluation import generate_synthetic_docs
        generate_synthetic_docs.main(["--out", a.data])
    gt = json.loads(gt_path.read_text(encoding="utf-8"))["documents"]
    out_dir = OUT_DIR / (a.tag or "main")
    pipe = PrivacyPipeline(output_dir=out_dir)

    tp_det = defaultdict(int)      # detected entities that match some GT occurrence
    n_det = defaultdict(int)
    tp_gt = defaultdict(int)       # GT occurrences found
    n_gt = defaultdict(int)
    leaks_total, gt_total = 0, 0
    per_doc = []
    doc_person = {}
    t0 = time.time()

    for doc in gt:
        t = time.time()
        ocr = pipe.ocr_document(str(ROOT / doc["file"]))
        det = pipe.detect_pii(ocr["doc_id"])
        red = pipe.redact_image(ocr["doc_id"], det["detection_id"])
        ver = pipe.verify_redaction(red["result_id"])
        doc_person[ocr["doc_id"]] = doc["person"]

        detected = pipe.vault.get_detection(det["detection_id"]).entities   # local-only access for scoring
        gts = doc["entities"]
        for g in gts:
            n_gt[g["type"]] += 1
            if any(e.type == g["type"] and matches(g["type"], g["value"], e.value) for e in detected):
                tp_gt[g["type"]] += 1
        for e in detected:
            n_det[e.type] += 1
            if any(g["type"] == e.type and matches(e.type, g["value"], e.value) for g in gts):
                tp_det[e.type] += 1

        # Leak check against ground truth on the redacted output
        _, words = ocr_image(load_bgr(red["output_path"]))
        text = re.sub(r"[^A-Z0-9]", "", strip_diacritics(" ".join(l.text for l in build_lines(words))).upper())
        leaks = [g for g in gts if len(norm(g["type"], g["value"])) >= 5 and norm(g["type"], g["value"]) in text]
        leaks_total += len(leaks)
        gt_total += len(gts)
        per_doc.append([Path(doc["file"]).name, doc["kind"], doc["font"], doc["rotation_deg"], ocr["avg_confidence"],
                        len(gts), sum(det["summary"].values()), red["boxes_drawn"], len(leaks), ver["status"],
                        f"{time.time() - t:.1f}s"])

    # Linking
    link = pipe.link_documents()
    pred_pairs = set()
    for c in link["clusters"]:
        pred_pairs |= {tuple(sorted(p)) for p in itertools.combinations(c["doc_ids"], 2)}
    ids = list(doc_person)
    gt_pairs = {tuple(sorted((a, b))) for a, b in itertools.combinations(ids, 2) if doc_person[a] == doc_person[b]}
    lp = len(pred_pairs & gt_pairs) / len(pred_pairs) if pred_pairs else 0.0
    lr = len(pred_pairs & gt_pairs) / len(gt_pairs) if gt_pairs else 0.0

    # ------------------------------------------------------------------ report
    rows = []
    for t in TYPES:
        if not n_gt[t] and not n_det[t]:
            continue
        p = tp_det[t] / n_det[t] if n_det[t] else 0.0
        r = tp_gt[t] / n_gt[t] if n_gt[t] else 0.0
        rows.append([t, n_gt[t], n_det[t], f"{p:.3f}", f"{r:.3f}", f"{f1(p, r):.3f}"])
    # micro totals only over the types the ground truth labels; extra ID-card fields (issuer, validity, place of
    # birth, portrait) are redacted too but have no ground truth, so they are listed separately and not scored
    scored_det = sum(n_det[t] for t in TYPES)
    P = sum(tp_det[t] for t in TYPES) / max(scored_det, 1)
    R = sum(tp_gt.values()) / max(sum(n_gt.values()), 1)
    rows.append(["**ALL (micro)**", sum(n_gt.values()), scored_det, f"{P:.3f}", f"{R:.3f}", f"{f1(P, R):.3f}"])
    extra = {t: n for t, n in sorted(n_det.items()) if t not in TYPES and n}

    passes = sum(1 for d in per_doc if d[9] == "PASS")
    clusters_desc = [f"{c['cluster_id']}: {len(c['doc_ids'])} docs, persons "
                     f"{sorted({doc_person[d] for d in c['doc_ids']})}, types {c['shared_entity_types']}, "
                     f"strength {c['link_strength']}" for c in link["clusters"]]

    det_tbl = tabulate(rows, headers=["Type", "GT", "Detected", "Precision", "Recall", "F1"], tablefmt="github")
    doc_tbl = tabulate(per_doc, headers=["File", "Kind", "Font", "Rot (deg)", "OCR conf", "GT PII", "Detected",
                                         "Boxes", "GT leaks", "Verify", "Time"], tablefmt="github")
    link_tbl = tabulate([[len(gt_pairs), len(pred_pairs), f"{lp:.3f}", f"{lr:.3f}", f"{f1(lp, lr):.3f}",
                          len(link["clusters"])]],
                        headers=["GT pairs", "Pred pairs", "Precision", "Recall", "F1", "Clusters"], tablefmt="github")
    report = "\n".join([
        "# PrivacyGuard evaluation report" + (f" ({a.tag})" if a.tag else ""), "",
        f"Dataset: {len(gt)} synthetic documents ({len(set(doc_person.values()))} fictional people). "
        f"Total runtime {time.time() - t0:.0f}s.", "",
        "## PII detection", det_tbl, "",
        *([f"Also redacted (not in ground truth, not scored): "
           + ", ".join(f"{k} {v}" for k, v in extra.items()), ""] if extra else []),
        "## Redaction", f"* Ground-truth PII values still readable after redaction: **{leaks_total} / {gt_total}** "
        f"(redaction recall {1 - leaks_total / max(gt_total, 1):.3f})",
        f"* verify_redaction PASS: **{passes} / {len(per_doc)}**", "",
        doc_tbl, "",
        "## Cross-document linking", link_tbl, "", *[f"* {c}" for c in clusters_desc], "",
    ])
    suffix = f"_{a.tag}" if a.tag else ""
    (out_dir / "eval_report.md").write_text(report, encoding="utf-8")
    (ROOT / "evaluation" / f"results{suffix}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
