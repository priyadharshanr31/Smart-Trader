# run_scheduler.py
from __future__ import annotations
import os, threading, time
from typing import Dict, List
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone
from autonomous_runner import run_once
from core.trader import AlpacaTrader, _to_broker_symbol
from core.positions import read_ledger, write_ledger
from config import settings
from core.finnhub_client import FinnhubClient
from dotenv import load_dotenv, find_dotenv
# Load .env from project root to pick up API keys (e.g., GEMINI_API_KEY).  Using
# find_dotenv ensures we locate the .env file even if run_scheduler is invoked
# from outside its containing directory.
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv(override=True)

ny = timezone("America/New_York")
sched = BlockingScheduler(timezone=ny)

WATCHLIST_STOCKS = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,NVDA,ORCL,AMD,PLTR,INTC").split(",") if s.strip()]
WATCHLIST_CRYPTO = [s.strip() for s in os.getenv("WATCHLIST_CRYPTO", "BTC/USD,ETH/USD,SOL/USD").split(",") if s.strip()]

# ---------- helpers ----------
def _to_display_symbol(sym: str, asset_class: str) -> str:
    s = (sym or "").upper().replace(" ", "")
    if "crypto" in (asset_class or "").lower():
        for q in ("USDT", "USDC", "USD", "EUR", "BTC", "ETH"):
            if s.endswith(q) and len(s) > len(q):
                base = s[: -len(q)]
                return f"{base}/{q}"
    return s

def _owned_positions(trader: AlpacaTrader) -> List[Dict]:
    return trader.list_positions()

def reconcile_ledger_with_broker(trader: AlpacaTrader):
    """
    Drop ledger entries that no longer exist at broker (prevents stale SELL_NO_POSITION).
    """
    broker = {p["symbol"]: float(p["qty"]) for p in trader.list_positions()}  # broker symbols (e.g., BTCUSD)
    ledger = read_ledger()  # keys like 'BTC/USD'
    changed = False
    for sym in list(ledger.keys()):
        if broker.get(_to_broker_symbol(sym), 0.0) <= 0.0:
            ledger.pop(sym, None)
            changed = True
    if changed:
        write_ledger(ledger)

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

def price_poller():
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    last_price: Dict[str, float] = {}
    last_run: Dict[str, float] = {}
    while True:
        try:
            for pos in _owned_positions(trader):
                sym_broker = pos.get("symbol", "")
                ac = (pos.get("asset_class") or "").lower()
                display = _to_display_symbol(sym_broker, ac)
                is_crypto = "crypto" in ac

                p = trader.last_price(display) or 0.0
                if p <= 0:
                    continue
                prev = last_price.get(display, p)
                last_price[display] = p

                # simple trigger: >1% move and ≥2 min since last run
                if prev > 0 and abs(p - prev) / prev >= 0.01:
                    if (time.time() - last_run.get(display, 0)) > 120:
                        print(run_once(display, is_crypto=is_crypto, trigger="price_event"))
                        last_run[display] = time.time()
        except Exception:
            pass
        time.sleep(20)

def news_poller():
    fh = FinnhubClient(settings.finnhub_key)
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    last_seen_ts: Dict[str, int] = {}
    while True:
        try:
            for pos in _owned_positions(trader):
                ac = (pos.get("asset_class") or "").lower()
                if "crypto" in ac:
                    continue  # skip crypto for company news
                sym = pos.get("symbol", "")
                latest_items = fh.company_news_struct(sym, days=3, max_items=5)
                if not latest_items:
                    continue
                latest = int(latest_items[0].get("datetime") or 0)
                if latest > last_seen_ts.get(sym, 0):
                    print(run_once(sym, is_crypto=False, trigger="news_event", news_boost=True))
                    last_seen_ts[sym] = latest
        except Exception:
            pass
        time.sleep(120)

if __name__ == "__main__":
    # Initialize DB tables (creates if missing)
    from core.db import init_db
    init_db()

    # Reconcile local ledger once on startup
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    reconcile_ledger_with_broker(trader)

    if ENABLE_PRICE_POLLER:
        threading.Thread(target=price_poller, daemon=True).start()
    if ENABLE_NEWS_POLLER:
        threading.Thread(target=news_poller, daemon=True).start()
    sched.start()
