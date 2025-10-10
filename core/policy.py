# core/policy.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import List
from config import settings

# ----- $ caps -----

def cash_floor_remaining(cash: float, equity: float) -> float:
    """Dollars available above the cash floor (≥40% equity kept as cash)."""
    reserve = settings.CASH_FLOOR_PCT * float(equity or 0.0)
    return max(0.0, float(cash or 0.0) - reserve)

def horizon_trade_cap(horizon: str, equity: float) -> float:
    pct = float(settings.HORIZON_TRADE_CAP_PCT.get(horizon, 0.02))
    return max(0.0, pct * float(equity or 0.0))

def per_symbol_cap_remaining(symbol_mv: float, equity: float) -> float:
    cap = settings.PER_SYMBOL_EXPOSURE_CAP_PCT * float(equity or 0.0)
    return max(0.0, cap - float(symbol_mv or 0.0))

def compute_allowed_notional(horizon: str, cash: float, equity: float, symbol_mv: float) -> float:
    """Dollar notional allowed for a BUY according to caps."""
    return min(
        cash_floor_remaining(cash, equity),
        horizon_trade_cap(horizon, equity),
        per_symbol_cap_remaining(symbol_mv, equity),
    )

# ----- share caps & throttles -----

def clamp_qty_by_share_caps(desired_qty: float, current_qty: float) -> float:
    """Clamp desired order qty by max-per-buy and max total shares per symbol."""
    desired_qty = min(float(desired_qty), float(settings.MAX_SHARES_PER_BUY))
    remaining = max(0.0, float(settings.MAX_SHARES_PER_SYMBOL) - float(current_qty or 0.0))
    return max(0.0, min(desired_qty, remaining))

def too_soon_since_last_buy(symbol: str, runs_for_symbol: List[dict]) -> bool:
    """Cooldown to avoid back-to-back buys."""
    now = datetime.now(timezone.utc)
    for r in runs_for_symbol:
        if r.get("action") != "BUY":
            continue
        try:
            last = datetime.fromisoformat(r["when"].replace("Z","+00:00"))
        except Exception:
            continue
        if (now - last).total_seconds() / 60.0 < float(settings.REBUY_COOLDOWN_MINUTES):
            return True
        break
    return False

def hit_daily_buy_limit(symbol: str, runs_for_symbol: List[dict]) -> bool:
    """No more than N buy executions per symbol per UTC day."""
    today = datetime.now(timezone.utc).date()
    buys = 0
    for r in runs_for_symbol:
        if r.get("action") != "BUY":
            continue
        try:
            day = datetime.fromisoformat(r["when"].replace("Z","+00:00")).date()
        except Exception:
            continue
        if day == today:
            buys += 1
            if buys >= int(settings.DAILY_BUY_LIMIT_PER_SYMBOL):
                return True
    return False
