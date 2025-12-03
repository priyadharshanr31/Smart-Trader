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
    # Minimum number of rows for the long‑term agent.  Crypto pairs and
    # newly added tickers often have fewer than 200 weekly bars.  Relaxing
    # this threshold to 100 ensures the long‑term agent participates in
    # decision making for such assets rather than immediately returning
    # HOLD due to insufficient history.
    MIN_ROWS = 100
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

        # ---------------------------------------------------------------
        # Fallback heuristic for long‑term: If the LLM returns HOLD with
        # confidence ≤0.5, fall back on a simple weekly trend analysis.
        # Use 10‑week and 30‑week moving averages to gauge macro regime
        # and determine BUY/SELL signals.  If the short MA is above the
        # long MA with positive slope, we BUY; if below with negative
        # slope, we SELL; otherwise HOLD.  Confidence scales with the
        # relative separation of the MAs.  This ensures that even when
        # the generative model is offline, the agent can still make
        # informed macro decisions.
        if decision.upper() == "HOLD" and conf <= 0.5:
            try:
                close = df['close']
                ma10_series = close.rolling(window=10).mean()
                ma30_series = close.rolling(window=30).mean()
                ma10_last = float(ma10_series.iloc[-1])
                ma30_last = float(ma30_series.iloc[-1])
                # compute slopes over last 10 periods (approx two months)
                ma10_prev = float(ma10_series.iloc[-10]) if len(ma10_series) >= 10 else ma10_last
                ma30_prev = float(ma30_series.iloc[-10]) if len(ma30_series) >= 10 else ma30_last
                slope10 = ma10_last - ma10_prev
                slope30 = ma30_last - ma30_prev
                buy_signal = False
                sell_signal = False
                if (ma10_last > ma30_last) and (slope10 > 0 or slope30 > 0):
                    buy_signal = True
                if (ma10_last < ma30_last) and (slope10 < 0 or slope30 < 0):
                    sell_signal = True
                if buy_signal and not sell_signal:
                    decision = "BUY"
                    distance = abs(ma10_last - ma30_last) / (ma30_last + 1e-9)
                    conf = min(1.0, 0.5 + distance * 2.0)
                elif sell_signal and not buy_signal:
                    decision = "SELL"
                    distance = abs(ma10_last - ma30_last) / (ma30_last + 1e-9)
                    conf = min(1.0, 0.5 + distance * 2.0)
                else:
                    decision = "HOLD"
                    conf = 0.4
                raw = f"{raw} [fallback_heuristic_longterm]"
            except Exception:
                pass

        return decision, conf, raw
