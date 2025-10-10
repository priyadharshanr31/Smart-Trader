# agents/long_term_agent.py
from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a LONG-TERM macro strategist. You analyze WEEKLY bars plus a short list of "
    "recent news/reflection snippets to judge regime. "
    "Return a STRICT JSON object ONLY (no prose) with keys: "
    '{"vote":"BUY|SELL|HOLD","confidence":0..1,"rationale":"one or two short lines"}.'
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence in [0,1] using weekly context and recent news.\n\n"
    "Weekly snapshot (tail=10):\n{table}\n\n"
    "Recent News (semantic top-k):\n{news}\n"
    "Return ONLY JSON as specified."
)

REQ_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

class LongTermAgent(BaseAgent):
    MIN_ROWS = 200
    TAIL_N = 10

    def __init__(self, name, llm, config, semantic_memory):
        super().__init__(name, llm, config)
        self.semantic_memory = semantic_memory  # may be None

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('long_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "(no long-term data)"
        if len(df) < self.MIN_ROWS:
            return "HOLD", 0.5, f"(long-term rows < {self.MIN_ROWS})"

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            return "HOLD", 0.5, f"(missing cols: {missing})"

        tail = df.tail(self.TAIL_N)[REQ_COLS]
        if tail.isna().any().any():
            return "HOLD", 0.5, "(NaNs in long-term tail window)"

        ticker = df['ticker'].iloc[-1]
        table = tail.to_string(index=False)

        # Semantic hits (robust)
        if not self.semantic_memory:
            news = "(semantic memory disabled)"
        else:
            try:
                hits = self.semantic_memory.search_memory(f"{ticker} market sentiment", k=3)
                if hits:
                    news = "\n".join([f"- {h['text']} (distance: {float(h.get('distance', 0)):.2f})" for h in hits])
                else:
                    news = "(no recent articles)"
            except Exception:
                news = "(no recent articles)"

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table, "news": news},
        )

        # sanitize confidence
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        return decision, conf, raw
