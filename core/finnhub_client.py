# core/finnhub_client.py
from __future__ import annotations
import requests
import datetime as dt
from typing import List, Dict, Any, Optional

FINNHUB_BASE = "https://finnhub.io/api/v1"

class FinnhubClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    # --------- News Sentiment (NEW) ---------
    def news_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        GET /news-sentiment?symbol=XYZ
        Returns something like:
        {
          "buzz": {"articlesInLastWeek": 153, "buzz": 0.24, "weeklyAverage": 275.57},
          "companyNewsScore": 0.53,
          "sentiment": {"bearishPercent": 0.28, "bullishPercent": 0.28},
          "symbol": "AAPL"
        }
        """
        try:
            url = f"{FINNHUB_BASE}/news-sentiment"
            params = {"symbol": symbol.upper(), "token": self.api_key}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    # --------- Simple string lists (used by Macro agent memory) ---------
    def company_news(self, symbol: str, days: int = 30) -> List[str]:
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        url = f"{FINNHUB_BASE}/company-news"
        params = {"symbol": symbol.upper(), "from": start.isoformat(), "to": end.isoformat(), "token": self.api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        out = []
        for it in items[:50]:
            headline = (it.get("headline") or "").strip()
            summary = (it.get("summary") or "").strip()
            if headline or summary:
                out.append((headline + (": " if headline and summary else "") + summary).strip())
        return out

    def crypto_news(self, max_items: int = 50) -> List[str]:
        url = f"{FINNHUB_BASE}/news"
        for category in ("crypto", "general"):
            try:
                params = {"category": category, "token": self.api_key}
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                items = r.json() if isinstance(r.json(), list) else []
                out = []
                for it in items[:max_items]:
                    headline = (it.get("headline") or "").strip()
                    summary = (it.get("summary") or "").strip()
                    if headline or summary:
                        out.append((headline + (": " if headline and summary else "") + summary).strip())
                if out:
                    return out
            except Exception:
                continue
        return []

    # --------- Structured news (for UI tables) ---------
    def company_news_struct(self, symbol: str, days: int = 7, max_items: int = 50) -> List[Dict]:
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        url = f"{FINNHUB_BASE}/company-news"
        params = {"symbol": symbol.upper(), "from": start.isoformat(), "to": end.isoformat(), "token": self.api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        out: List[Dict] = []
        for it in items[:max_items]:
            out.append({
                "symbol": symbol.upper(),
                "headline": it.get("headline"),
                "summary": it.get("summary"),
                "source": it.get("source"),
                "url": it.get("url"),
                "datetime": it.get("datetime"),  # epoch seconds
            })
        return out

    def general_news_struct(self, max_items: int = 50) -> List[Dict]:
        url = f"{FINNHUB_BASE}/news"
        params = {"category": "general", "token": self.api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        out: List[Dict] = []
        for it in items[:max_items]:
            out.append({
                "headline": it.get("headline"),
                "summary": it.get("summary"),
                "source": it.get("source"),
                "url": it.get("url"),
                "datetime": it.get("datetime"),
            })
        return out

    def crypto_news_struct(self, max_items: int = 50) -> List[Dict]:
        for category in ("crypto", "general"):
            try:
                url = f"{FINNHUB_BASE}/news"
                params = {"category": category, "token": self.api_key}
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                items = r.json() if isinstance(r.json(), list) else []
                out: List[Dict] = []
                for it in items[:max_items]:
                    out.append({
                        "headline": it.get("headline"),
                        "summary": it.get("summary"),
                        "source": it.get("source"),
                        "url": it.get("url"),
                        "datetime": it.get("datetime"),
                        "category": category,
                    })
                if out:
                    return out
            except Exception:
                continue
        return []
