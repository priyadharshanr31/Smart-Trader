# run_scheduler.py
from __future__ import annotations
import os, threading, time
from datetime import datetime, timedelta
from typing import Dict, Set
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone
from autonomous_runner import run_once
from core.trader import AlpacaTrader
from config import settings
from core.finnhub_client import FinnhubClient
from dotenv import load_dotenv
load_dotenv()

ny = timezone("America/New_York")
sched = BlockingScheduler(timezone=ny)

WATCHLIST_STOCKS = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,NVDA").split(",") if s.strip()]
WATCHLIST_CRYPTO = [s.strip() for s in os.getenv("WATCHLIST_CRYPTO", "BTC/USD,ETH/USD").split(",") if s.strip()]

# ------------- 30m bar-close loops -------------
@sched.scheduled_job("cron", day_of_week="mon-fri", hour="10-16", minute="2,32")
def stocks_halfhour():
    for s in WATCHLIST_STOCKS:
        print(run_once(s, is_crypto=False, trigger="bar_close_30m"))

@sched.scheduled_job("cron", minute="2,32")
def crypto_halfhour():
    for c in WATCHLIST_CRYPTO:
        print(run_once(c, is_crypto=True, trigger="bar_close_30m"))

# ------------- Optional realtime pollers (price/news) -------------
ENABLE_PRICE_POLLER = os.getenv("ENABLE_PRICE_POLLER", "1") == "1"
ENABLE_NEWS_POLLER  = os.getenv("ENABLE_NEWS_POLLER", "1") == "1"

def _owned_symbols(trader: AlpacaTrader) -> Set[str]:
    return {p["symbol"] for p in trader.list_positions()}

def price_poller():
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    last_price: Dict[str, float] = {}
    last_run: Dict[str, float] = {}
    while True:
        try:
            owned = _owned_symbols(trader)
            for sym in owned:
                p = trader.last_price(sym) or 0.0
                if p <= 0:
                    continue
                prev = last_price.get(sym, p)
                last_price[sym] = p
                # simple trigger: >1% move since last snapshot and at least 2 minutes since last decision
                if prev > 0 and abs(p - prev) / prev >= 0.01:
                    if (time.time() - last_run.get(sym, 0)) > 120:
                        print(run_once(sym, is_crypto=False, trigger="price_event"))
                        last_run[sym] = time.time()
        except Exception:
            pass
        time.sleep(20)

def news_poller():
    fh = FinnhubClient(settings.finnhub_key)
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    last_seen_ts: Dict[str, int] = {}
    while True:
        try:
            for sym in _owned_symbols(trader):
                items = fh.company_news_struct(sym, days=3, max_items=5)
                if not items:
                    continue
                latest = int(items[0].get("datetime") or 0)
                if latest > last_seen_ts.get(sym, 0):
                    print(run_once(sym, is_crypto=False, trigger="news_event", news_boost=True))
                    last_seen_ts[sym] = latest
        except Exception:
            pass
        time.sleep(120)

if __name__ == "__main__":
    if ENABLE_PRICE_POLLER:
        threading.Thread(target=price_poller, daemon=True).start()
    if ENABLE_NEWS_POLLER:
        threading.Thread(target=news_poller, daemon=True).start()
    sched.start()
