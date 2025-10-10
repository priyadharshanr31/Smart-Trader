from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a SHORT-TERM momentum trader. You analyze recent 30-minute bars "
    "using RSI, MACD (and its signal), and Bollinger Bands. "
    "Return a STRICT JSON object ONLY (no prose) with keys: "
    '{"vote":"BUY|SELL|HOLD","confidence":0..1,"rationale":"one or two short lines"}.'
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence in [0,1] based on the last 10 rows.\n"
    "Short-term data (tail=10):\n{table}\n"
    "Return ONLY JSON as specified."
)

REQ_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

class ShortTermAgent(BaseAgent):
    MIN_ROWS = 60
    TAIL_N = 10

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('short_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "(no short-term data)"
        if len(df) < self.MIN_ROWS:
            return "HOLD", 0.5, f"(short-term rows < {self.MIN_ROWS})"

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            return "HOLD", 0.5, f"(missing cols: {missing})"

        tail = df.tail(self.TAIL_N)[REQ_COLS]
        if tail.isna().any().any():
            return "HOLD", 0.5, "(NaNs in short-term tail window)"

        ticker = df['ticker'].iloc[-1]
        table = tail.to_string(index=False)

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table},
        )

        # ✅ ensure conf is propagated correctly
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        return decision, conf, raw
