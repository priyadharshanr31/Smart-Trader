from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a MID-TERM (swing) trend analyst. You consider DAILY bars with "
    "5/20-day moving averages, RSI, MACD (and its signal), and Bollinger Bands. "
    "Return a STRICT JSON object ONLY (no prose) with keys: "
    '{"vote":"BUY|SELL|HOLD","confidence":0..1,"rationale":"one or two short lines"}.'
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence in [0,1] using the daily context below.\n\n"
    "Daily data (last 20 rows):\n{table}\n\n"
    "5DMA (tail=5):\n{ma5}\n\n"
    "20DMA (tail=5):\n{ma20}\n"
    "Return ONLY JSON as specified."
)

REQ_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

class MidTermAgent(BaseAgent):
    MIN_ROWS = 120
    TAIL_N = 20

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('mid_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "(no mid-term data)"
        if len(df) < self.MIN_ROWS:
            return "HOLD", 0.5, f"(mid-term rows < {self.MIN_ROWS})"

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            return "HOLD", 0.5, f"(missing cols: {missing})"

        tail = df.tail(self.TAIL_N)[REQ_COLS]
        if tail.isna().any().any():
            return "HOLD", 0.5, "(NaNs in mid-term tail window)"

        ticker = df['ticker'].iloc[-1]
        table = tail.to_string(index=False)
        ma5 = df['close'].rolling(window=5).mean().tail(5).to_string(index=False)
        ma20 = df['close'].rolling(window=20).mean().tail(5).to_string(index=False)

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table, "ma5": ma5, "ma20": ma20},
        )

        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        return decision, conf, raw
