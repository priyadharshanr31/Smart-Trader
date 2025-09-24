from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a MID-TERM (swing) trend analyst. You consider daily bars, "
    "5/20-day moving averages, RSI, MACD, and produce a vote."
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence 0..1 using the daily context below.\n\n"
    "Daily data (last 20 rows):\n{table}\n\n"
    "5DMA (tail=5):\n{ma5}\n\n"
    "20DMA (tail=5):\n{ma20}\n"
)

class MidTermAgent(BaseAgent):
    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('mid_term')
        if df is None or df.empty or len(df) < 20:
            return "HOLD", 0.5, "Insufficient mid-term data"
        ticker = df['ticker'].iloc[-1]
        table = df[['close','rsi','macd','macd_signal','upper_band','lower_band']].tail(20).to_string()
        ma5 = df['close'].rolling(window=5).mean().tail(5).to_string()
        ma20 = df['close'].rolling(window=20).mean().tail(5).to_string()

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table, "ma5": ma5, "ma20": ma20},
        )
        return decision, conf, raw
