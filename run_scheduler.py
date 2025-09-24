# run_scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone
from autonomous_runner import run_once
import os

ny = timezone("America/New_York")
sched = BlockingScheduler(timezone=ny)

WATCHLIST_STOCKS = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,NVDA").split(",") if s.strip()]
WATCHLIST_CRYPTO = [s.strip() for s in os.getenv("WATCHLIST_CRYPTO", "BTC/USD,ETH/USD").split(",") if s.strip()]

# Stocks: US regular hours, 30m bar close + ~2 min lag
@sched.scheduled_job("cron", day_of_week="mon-fri", hour="10-16", minute="2,32")
def stocks_halfhour():
    for s in WATCHLIST_STOCKS:
        print(run_once(s, is_crypto=False))

# Crypto: 24/7, every 30 min at :02 and :32
@sched.scheduled_job("cron", minute="2,32")
def crypto_halfhour():
    for c in WATCHLIST_CRYPTO:
        print(run_once(c, is_crypto=True))

if __name__ == "__main__":
    sched.start()
