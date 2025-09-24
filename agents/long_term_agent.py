from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a LONG-TERM macro strategist. You consider weekly bars plus "
    "recent news (semantic hits) to judge regime and produce a vote."
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence 0..1 using weekly context and recent news.\n\n"
    "Weekly snapshot (tail=10):\n{table}\n\n"
    "Recent News (semantic top-k):\n{news}\n"
)

class LongTermAgent(BaseAgent):
    def __init__(self, name, llm, config, semantic_memory):
        super().__init__(name, llm, config)
        self.semantic_memory = semantic_memory

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('long_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "No long-term data"
        ticker = df['ticker'].iloc[-1]
        table = df[['close','rsi','macd','macd_signal','upper_band','lower_band']].tail(10).to_string()

        # Use the article-style API: search_memory returns {text, distance}
        try:
            hits = self.semantic_memory.search_memory(f"{ticker} market sentiment", k=3)
            if hits:
                news = "\n".join([f"- {h['text']} (distance: {h['distance']:.2f})" for h in hits])
            else:
                news = "(no recent articles)"
        except Exception:
            news = "(no recent articles)"

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table, "news": news},
        )
        return decision, conf, raw
