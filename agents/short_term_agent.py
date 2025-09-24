from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a SHORT-TERM momentum trader. You analyze intraday bars (≈30m) "
    "with RSI/MACD/Bollinger and produce a directional vote."
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence 0..1 based on the last 10 rows.\n\n"
    "Short-term data (tail=10):\n{table}\n"
)

class ShortTermAgent(BaseAgent):
    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('short_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "No short-term data"
        ticker = df['ticker'].iloc[-1]
        table = df[['close','rsi','macd','macd_signal','upper_band','lower_band']].tail(10).to_string()

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table},
        )
        return decision, conf, raw
