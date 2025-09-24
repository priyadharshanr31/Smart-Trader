# agents/suggestions_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

class SuggestionsAgent:
    """
    A standalone idea-generation agent that uses Finnhub's news-sentiment metrics
    + recent headlines and asks an LLM to decide: BUY / HOLD / AVOID.
    """

    def __init__(self, llm, finnhub_client, min_conf: float = 0.60):
        """
        llm: your LCTraderLLM instance (must implement vote_structured(system_msg, user_template, variables))
        finnhub_client: FinnhubClient instance
        min_conf: minimum LLM confidence to recommend BUY
        """
        self.llm = llm
        self.fh = finnhub_client
        self.min_conf = float(min_conf)

        # System instruction specialized for suggestions (NOT using the 3-agent prompts)
        self.system_msg = (
            "You are an equity idea-generation analyst. Use Finnhub news-sentiment metrics and recent headlines "
            "to judge whether the stock is attractive to BUY now, or HOLD (watchlist), or AVOID.\n"
            "Rules:\n"
            "- Prefer BUY when companyNewsScore is above sector average proxies and bullishPercent > bearishPercent,\n"
            "  with positive or improving buzz.\n"
            "- Prefer AVOID when bearishPercent dominates or sentiment is clearly negative.\n"
            "- If signals conflict or are weak, choose HOLD.\n"
            "Output STRICTLY as: VOTE: BUY|HOLD|SELL, CONFIDENCE: 0.00-1.00\n"
            "(Use SELL to indicate AVOID.)"
        )

        # Template for the user content
        self.user_template = (
            "TICKER: {symbol}\n\n"
            "Finnhub News Sentiment (most recent):\n"
            "- companyNewsScore: {companyNewsScore}\n"
            "- buzz.articlesInLastWeek: {articlesInLastWeek}\n"
            "- buzz.buzz: {buzz}\n"
            "- buzz.weeklyAverage: {weeklyAverage}\n"
            "- sentiment.bullishPercent: {bullishPercent}\n"
            "- sentiment.bearishPercent: {bearishPercent}\n\n"
            "Recent headlines (max 8):\n"
            "{headlines_block}\n"
        )

    def _format_headlines(self, headlines: List[Dict[str, Any]], max_items: int = 8) -> str:
        lines = []
        for it in headlines[:max_items]:
            h = (it.get("headline") or "").strip()
            s = (it.get("summary") or "").strip()
            if h or s:
                if s and len(s) > 180:
                    s = s[:177] + "..."
                lines.append(f"- {h}" + (f" — {s}" if s else ""))
        return "\n".join(lines) if lines else "- (no recent headlines available)"

    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Returns a dict:
        {
          'symbol': 'AAPL',
          'recommendation': 'BUY'|'HOLD'|'AVOID',
          'confidence': 0.0-1.0,
          'companyNewsScore': float|None,
          'bullishPercent': float|None,
          'bearishPercent': float|None,
          'buzz': float|None,
          'articlesInLastWeek': int|None,
          'weeklyAverage': float|None,
          'headlines': [ ... ],
          'raw': '<LLM raw text>'
        }
        """
        sent = self.fh.news_sentiment(symbol) or {}
        headlines = self.fh.company_news_struct(symbol, days=14, max_items=12) or []

        # Extract metrics with safe defaults
        buzz = (sent.get("buzz") or {})
        sentiment = (sent.get("sentiment") or {})
        variables = {
            "symbol": symbol.upper(),
            "companyNewsScore": float(sent.get("companyNewsScore") or 0.0),
            "articlesInLastWeek": int(buzz.get("articlesInLastWeek") or 0),
            "buzz": float(buzz.get("buzz") or 0.0),
            "weeklyAverage": float(buzz.get("weeklyAverage") or 0.0),
            "bullishPercent": float(sentiment.get("bullishPercent") or 0.0),
            "bearishPercent": float(sentiment.get("bearishPercent") or 0.0),
            "headlines_block": self._format_headlines(headlines, max_items=8),
        }

        # Ask LLM for a structured recommendation (we map SELL -> AVOID)
        decision, confidence, raw = self.llm.vote_structured(
            system_msg=self.system_msg,
            user_template=self.user_template,
            variables=variables,
        )
        rec = "AVOID" if decision == "SELL" else decision  # SELL == AVOID for non-held ideas

        return {
            "symbol": variables["symbol"],
            "recommendation": rec,
            "confidence": float(confidence or 0.0),
            "companyNewsScore": variables["companyNewsScore"],
            "bullishPercent": variables["bullishPercent"],
            "bearishPercent": variables["bearishPercent"],
            "buzz": variables["buzz"],
            "articlesInLastWeek": variables["articlesInLastWeek"],
            "weeklyAverage": variables["weeklyAverage"],
            "headlines": headlines[:8],
            "raw": raw,
        }
