from __future__ import annotations
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticMemory:
    """Tiny in-process vector store for news/reflections."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._texts: List[str] = []
        self._emb = np.zeros((0, 384), dtype=np.float32)

    def add(self, texts: List[str]):
        if not texts:
            return
        vecs = self.model.encode(texts, normalize_embeddings=True)
        self._texts.extend(texts)
        self._emb = np.vstack([self._emb, vecs]) if self._emb.size else vecs

    # our original method
    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self._texts:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q, self._emb)[0]
        idx = sims.argsort()[::-1][:k]
        return [{"text": self._texts[i], "score": float(sims[i])} for i in idx]

    # alias that matches Medium article signature/fields
    def search_memory(self, query: str, k: int = 3) -> List[Dict]:
        """
        Returns items with 'text' and 'distance' like the article uses.
        We define distance = 1 - cosine_sim (normalized), just for display.
        """
        hits = self.search(query, k=k)
        out = []
        for h in hits:
            score = float(h["score"])
            out.append({"text": h["text"], "distance": float(max(0.0, 1.0 - score))})
        return out
