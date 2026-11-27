"""Cross-document entity linking via salted HMAC-SHA256 hashes.

Each document is connected (bipartite graph) to the hashes of its high-entropy
identifiers. Documents sharing a hash are linked; connected components form
clusters (one person / case). Only types, counts and scores leave this module.
"""
from __future__ import annotations

import itertools
import re
from typing import Callable

import networkx as nx

from server.pii_detector import strip_diacritics
from server.vault import Entity

# Probability that a shared value of this type means "same person".
TYPE_WEIGHTS = {"CNP": 0.99, "ID_SERIES": 0.97, "MRZ": 0.97, "IBAN": 0.92, "EMAIL": 0.88,
                "PHONE": 0.8, "PERSON_NAME": 0.6}
MIN_LINK_CONFIDENCE = 0.5


def normalize(etype: str, value: str) -> str | None:
    v = strip_diacritics(value).upper()
    if etype == "CNP":
        d = re.sub(r"\D", "", v)
        return d if len(d) == 13 else None
    if etype in ("ID_SERIES", "IBAN"):
        return re.sub(r"[^A-Z0-9]", "", v) or None
    if etype == "EMAIL":
        return v.lower().strip()
    if etype == "PHONE":
        d = re.sub(r"\D", "", v)
        return d[-9:] if len(d) >= 9 else None
    if etype == "PERSON_NAME":
        tokens = sorted(t for t in re.split(r"[\s\-]+", re.sub(r"[^A-Z\s\-]", "", v)) if len(t) >= 2)
        return " ".join(tokens) if len(tokens) >= 2 else None
    return None


def entity_hashes(entities: list[Entity], hmac_fn: Callable[[str], str]) -> dict[str, set[str]]:
    """Compute salted hashes per entity type. Surname + given-name fields are combined into a full name."""
    values: dict[str, set[str]] = {}
    surnames = [e.value for e in entities if e.type == "PERSON_NAME" and e.source == "context:surname"]
    given = [e.value for e in entities if e.type == "PERSON_NAME" and e.source == "context:given"]
    composed = [f"{s} {g}" for s, g in itertools.product(surnames, given)]
    for e in entities:
        if e.type not in TYPE_WEIGHTS or e.source == "propagated":
            continue
        if e.type == "CNP" and e.confidence < 0.9:      # unvalidated CNPs are redacted but not used for linking
            continue
        n = normalize(e.type, e.value)
        if n:
            values.setdefault(e.type, set()).add(n)
    for full in composed:
        n = normalize("PERSON_NAME", full)
        if n:
            values.setdefault("PERSON_NAME", set()).add(n)
    return {t: {hmac_fn(f"{t}|{v}") for v in vs} for t, vs in values.items()}


class EntityLinker:
    def __init__(self) -> None:
        self.graph = nx.Graph()

    def update_document(self, doc_id: str, hashes: dict[str, set[str]]) -> None:
        node = ("doc", doc_id)
        if self.graph.has_node(node):
            self.graph.remove_edges_from(list(self.graph.edges(node)))
        self.graph.add_node(node)
        for etype, hs in hashes.items():
            for h in hs:
                hnode = ("hash", etype, h)
                self.graph.add_node(hnode)
                self.graph.add_edge(node, hnode)
        # drop orphan hash nodes
        self.graph.remove_nodes_from([n for n in list(self.graph.nodes)
                                      if n[0] == "hash" and self.graph.degree(n) == 0])

    def doc_ids(self) -> list[str]:
        return [n[1] for n in self.graph.nodes if n[0] == "doc"]

    def link(self, doc_ids: list[str] | None = None) -> dict:
        all_docs = self.doc_ids()
        selected = [d for d in (doc_ids or all_docs) if d in all_docs]
        unknown = [d for d in (doc_ids or []) if d not in all_docs]

        pair_links = []
        doc_graph = nx.Graph()
        doc_graph.add_nodes_from(selected)
        for a, b in itertools.combinations(selected, 2):
            ha = set(self.graph.neighbors(("doc", a)))
            hb = set(self.graph.neighbors(("doc", b)))
            shared = ha & hb
            if not shared:
                continue
            types = sorted({n[1] for n in shared})
            miss = 1.0
            for t in types:
                miss *= 1.0 - TYPE_WEIGHTS.get(t, 0.5)
            conf = round(1.0 - miss, 4)
            if conf < MIN_LINK_CONFIDENCE:
                continue
            pair_links.append({"doc_a": a, "doc_b": b, "shared_entity_types": types,
                               "shared_entity_count": len(shared), "confidence": conf})
            doc_graph.add_edge(a, b, confidence=conf)

        clusters, unlinked = [], []
        for i, comp in enumerate(sorted(nx.connected_components(doc_graph), key=lambda c: sorted(c))):
            members = sorted(comp)
            if len(members) < 2:
                unlinked.extend(members)
                continue
            links = [l for l in pair_links if l["doc_a"] in comp and l["doc_b"] in comp]
            clusters.append({
                "cluster_id": f"cluster_{len(clusters) + 1}",
                "doc_ids": members,
                "shared_entity_types": sorted({t for l in links for t in l["shared_entity_types"]}),
                "link_strength": round(min(l["confidence"] for l in links), 4),
                "links": links,
            })
        return {"total_documents": len(selected), "clusters": clusters,
                "unlinked_doc_ids": sorted(unlinked), "unknown_doc_ids": unknown}
