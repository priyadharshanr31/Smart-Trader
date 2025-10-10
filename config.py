# config.py
from __future__ import annotations
import os

class Settings:
    # Cache / data
    data_dir = os.getenv("DATA_DIR", "data")

    # Price horizons
    short_interval       = os.getenv("SHORT_INTERVAL", "30m")
    short_period         = os.getenv("SHORT_PERIOD", "60d")
    short_lookback       = int(os.getenv("SHORT_LOOKBACK", "300"))
    mid_daily_lookback   = int(os.getenv("MID_DAILY_LOOKBACK", "400"))
    long_weekly_lookback = int(os.getenv("LONG_WEEKLY_LOOKBACK", "520"))

    # =========================
    # Debate thresholds (tune here to reduce HOLDs)
    # =========================
    # Lower 'enter' threshold => easier to trigger BUY
    # Lower 'exit' threshold  => easier to trigger SELL
    # Feel free to tune via environment:
    #   MEAN_CONFIDENCE_TO_ACT, EXIT_CONFIDENCE_TO_ACT
    mean_confidence_to_act = float(os.getenv("MEAN_CONFIDENCE_TO_ACT", "0.30"))
    exit_confidence_to_act = float(os.getenv("EXIT_CONFIDENCE_TO_ACT", "0.30"))

    # APIs
    alpaca_key      = os.getenv("ALPACA_API_KEY_ID", "")
    alpaca_secret   = os.getenv("ALPACA_API_SECRET_KEY", "")
    alpaca_base_url = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
    finnhub_key     = os.getenv("FINNHUB_API_KEY", "")
    gemini_key      = os.getenv("GEMINI_API_KEY", "")
    gemini_model    = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

    # ---- Policy caps & throttles ----
    # cash floor: keep ≥40% of equity as cash
    CASH_FLOOR_PCT = float(os.getenv("CASH_FLOOR_PCT", "0.40"))
    # total exposure per symbol ≤5% of equity
    PER_SYMBOL_EXPOSURE_CAP_PCT = float(os.getenv("PER_SYMBOL_EXPOSURE_CAP_PCT", "0.05"))
    # per-trade horizon caps (percent of equity)
    HORIZON_TRADE_CAP_PCT = {
        "short": float(os.getenv("CAP_SHORT_PCT", "0.02")),
        "mid":   float(os.getenv("CAP_MID_PCT",   "0.03")),
        "long":  float(os.getenv("CAP_LONG_PCT",  "0.05")),
    }
    # shares throttles
    MAX_SHARES_PER_BUY     = int(os.getenv("MAX_SHARES_PER_BUY", "5"))
    MAX_SHARES_PER_SYMBOL  = int(os.getenv("MAX_SHARES_PER_SYMBOL", "20"))
    DAILY_BUY_LIMIT_PER_SYMBOL = int(os.getenv("DAILY_BUY_LIMIT_PER_SYMBOL", "2"))
    REBUY_COOLDOWN_MINUTES = int(os.getenv("REBUY_COOLDOWN_MINUTES", "60"))

settings = Settings()
