# config.py
from __future__ import annotations
import os

class Settings:
    # Cache / data
    data_dir = os.getenv("DATA_DIR", "data")

    # Price horizons (keep short at 30m to match your schedule)
    short_interval     = os.getenv("SHORT_INTERVAL", "30m")
    short_period       = os.getenv("SHORT_PERIOD", "60d")
    short_lookback     = int(os.getenv("SHORT_LOOKBACK", "300"))
    mid_daily_lookback = int(os.getenv("MID_DAILY_LOOKBACK", "400"))
    long_weekly_lookback = int(os.getenv("LONG_WEEKLY_LOOKBACK", "520"))

    # Agent/debate threshold (UI & runner)
    mean_confidence_to_act = float(os.getenv("MEAN_CONFIDENCE_TO_ACT", "0.60"))

    # APIs
    alpaca_key      = os.getenv("ALPACA_API_KEY_ID", "")
    alpaca_secret   = os.getenv("ALPACA_API_SECRET_KEY", "")
    alpaca_base_url = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
    finnhub_key     = os.getenv("FINNHUB_API_KEY", "")
    gemini_key      = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
