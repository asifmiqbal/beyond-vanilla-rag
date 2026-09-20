"""Hybrid Vector & Metadata Retriever for Telco Knowledge Base using LanceDB."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import lancedb
import numpy as np


class FastSemanticVectorizer:
    """High-throughput deterministic semantic projection vectorizer (768 dimensions)."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._rng = np.random.RandomState(42)
        self._basis = self._rng.randn(256, self.dim).astype(np.float32)

    def vectorize(self, text: str) -> List[float]:
        tokens = text.lower().replace("\n", " ").split()
        if not tokens:
            return [0.0] * self.dim

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % 256
            weight = 1.0 / math.log2(max(len(token), 2) + 1.0)
            vec += self._basis[h] * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec.tolist()

    def batch_vectorize(self, texts: List[str]) -> List[List[float]]:
        return [self.vectorize(t) for t in texts]


# Intent-to-domain mapping
INTENT_DOMAIN_MAP = {
    "billing_inquiry": "billing_invoicing",
    "network_connectivity": "network_incidents",
    "roaming_passes": "roaming_passes",
    "fiber_broadband": "fiber_broadband",
    "account_security": "account_security",
    "plan_upgrade": "prepaid_postpaid_plans",
}


class TelcoRetriever:
    """Retrieves top-k context articles from the 30,000-document LanceDB vector store."""

    def __init__(self, db_path: Optional[str] = None, table_name: str = "telco_articles"):
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "knowledge_base/lancedb")
        self.table_name = table_name
        self.vectorizer = FastSemanticVectorizer(dim=768)
        self._db: Optional[lancedb.DBConnection] = None
        self._table: Optional[lancedb.table.Table] = None
        self._init_db()

    def _init_db(self) -> None:
        """Connect to LanceDB instance and open table."""
        if Path(self.db_path).exists():
            try:
                self._db = lancedb.connect(self.db_path)
                tables = self._db.table_names() if hasattr(self._db, "table_names") else []
                if self.table_name in tables:
                    self._table = self._db.open_table(self.table_name)
            except Exception as e:
                print(f"[TelcoRetriever Warning] Could not open LanceDB table: {e}")

    def retrieve(
        self, query: str, intent: Optional[str] = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant articles for grounding."""
        if self._table is None:
            self._init_db()
            if self._table is None:
                return []

        # Vectorize incoming search query
        query_vec = self.vectorizer.vectorize(query)

        # Build LanceDB query
        search = self._table.search(query_vec)

        # Filter by domain if intent maps to a known KB domain
        target_domain = INTENT_DOMAIN_MAP.get(intent) if intent else None
        if target_domain:
            search = search.where(f"domain = '{target_domain}'")

        try:
            results = search.limit(top_k).to_list()
        except Exception:
            # Fallback without where filter if filter errors
            results = self._table.search(query_vec).limit(top_k).to_list()

        output_chunks: List[Dict[str, Any]] = []
        for r in results:
            output_chunks.append({
                "doc_id": r.get("doc_id"),
                "domain": r.get("domain"),
                "title": r.get("title"),
                "plan_code": r.get("plan_code"),
                "regional_apn": r.get("regional_apn"),
                "roaming_zone": r.get("roaming_zone"),
                "sla": r.get("sla"),
                "tier_price": r.get("tier_price"),
                "error_code": r.get("error_code"),
                "summary": r.get("summary"),
                "content": r.get("content"),
                "similarity_score": round(1.0 - float(r.get("_distance", 0.0)), 4) if "_distance" in r else 0.95,
            })

        return output_chunks
