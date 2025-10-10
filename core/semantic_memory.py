# core/semantic_memory.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

# Numpy + cosine for a lightweight CPU-only path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence-transformers, but never hard-crash if the stack isn't healthy
try:
    from sentence_transformers import SentenceTransformer
except Exception as _e:
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = os.getenv("SEMMEM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class SemanticMemory:
    """
    Tiny in-process vector store for news/reflections with a *safe* CPU init.

    - Forces device="cpu" to avoid the PyTorch 'meta tensor' crash on some Windows/GPU setups.
    - If sentence-transformers cannot load or SEMMEM_DISABLE=1 is set, runs in 'disabled' mode:
      it will store raw texts and return most-recent hits instead of vector search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # allow turning this off from .env quickly
        self.disabled = os.getenv("SEMMEM_DISABLE", "0") == "1"

        self._texts: List[str] = []
        self._emb: Optional[np.ndarray] = None  # NxD numpy array (when enabled)
        self._model: Optional[SentenceTransformer] = None

        if self.disabled:
            print("[SemanticMemory] Disabled via SEMMEM_DISABLE=1")
            return

        if SentenceTransformer is None:
            print("[SemanticMemory] sentence-transformers unavailable; running disabled.")
            self.disabled = True
            return

        # Force CPU to avoid GPU meta-tensor issues
        try:
            # SentenceTransformer supports device='cpu' in recent versions
            self._model = SentenceTransformer(model_name, device="cpu")  # type: ignore[arg-type]
            print(f"[SemanticMemory] Loaded model on CPU: {model_name}")
        except Exception as e:
            print(f"[SemanticMemory] Model init failed ({e}); running disabled.")
            self.disabled = True
            self._model = None

    # --------------------------- Public API ---------------------------

    def add(self, texts: List[str]) -> None:
        """
        Add a batch of texts. If enabled, we compute embeddings on CPU and
        append to the numpy matrix; otherwise, we only store the raw texts.
        """
        if not texts:
            return

        # normalize input
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return

        self._texts.extend(texts)

        if self.disabled or self._model is None:
            # store only raw texts
            return

        try:
            # returns numpy array (D dims); normalize=True gives cosine-ready vectors
            vecs = self._model.encode(texts, normalize_embeddings=True)
            if self._emb is None:
                self._emb = np.asarray(vecs, dtype=np.float32)
            else:
                self._emb = np.vstack([self._emb, np.asarray(vecs, dtype=np.float32)])
        except Exception as e:
            print(f"[SemanticMemory] encode failed ({e}); switching to disabled mode.")
            self.disabled = True
            self._emb = None  # keep texts; search() will return recency

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Cosine-similarity search (if enabled). In disabled mode, returns most-recent items.
        """
        if not self._texts:
            return []

        if self.disabled or self._model is None or self._emb is None:
            # recency fallback
            tail = self._texts[-k:]
            tail = list(reversed(tail))
            return [{"text": t, "score": 0.0} for t in tail]

        try:
            q = self._model.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(q, self._emb)[0]  # shape: [N]
            idx = sims.argsort()[::-1][:k]
            return [{"text": self._texts[i], "score": float(sims[i])} for i in idx]
        except Exception as e:
            print(f"[SemanticMemory] search failed ({e}); returning recency.")
            tail = self._texts[-k:]
            tail = list(reversed(tail))
            return [{"text": t, "score": 0.0} for t in tail]

    def search_memory(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Adapter to the signature you were using elsewhere.
        Returns items with 'text' and 'distance' (distance = 1 - cosine_sim for display).
        """
        hits = self.search(query, k=k)
        out: List[Dict[str, Any]] = []
        for h in hits:
            score = float(h.get("score", 0.0))
            out.append({"text": h["text"], "distance": float(max(0.0, 1.0 - score))})
        return out
