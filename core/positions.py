# core/positions.py
from __future__ import annotations
import json, os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone

STATE_DIR   = "state"
LEDGER_PATH = os.path.join(STATE_DIR, "positions.json")

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_ledger() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.exists(LEDGER_PATH):
        return {}
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def write_ledger(data: Dict[str, Any]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    return read_ledger().get(symbol.upper())

def upsert_position(symbol: str, info: Dict[str, Any]) -> None:
    data = read_ledger()
    data[symbol.upper()] = info
    write_ledger(data)

def remove_position(symbol: str) -> None:
    data = read_ledger()
    data.pop(symbol.upper(), None)
    write_ledger(data)

# ---- timeboxing ----
HORIZON_TIMEBOX = {
    "short": timedelta(days=1),   # within a day
    "mid":   timedelta(days=7),   # ~1 week
    "long":  timedelta(days=60),  # ~3 months
}

def set_timebox_on_entry(symbol: str, horizon: str, qty: float, entry_price: float, notional: float):
    """Create a new ledger row with timebox for first entry of a symbol."""
    ledger = read_ledger()
    dur = HORIZON_TIMEBOX.get(horizon, timedelta(days=7))
    timebox_until = (datetime.now(timezone.utc) + dur).replace(microsecond=0).isoformat().replace("+00:00","Z")
    ledger[symbol.upper()] = {
        "symbol": symbol.upper(),
        "horizon": horizon,
        "qty": float(qty),
        "entry_price": float(entry_price),
        "notional": float(notional),
        "entered_at": _now_iso(),
        "timebox_until": timebox_until,
    }
    write_ledger(ledger)
    return timebox_until

def merge_entry(ledger: dict, symbol: str, horizon: str, add_qty: float, add_price: float, add_notional: float, reset_timebox: bool=False):
    """Add to an existing position. By default, KEEP the earliest timebox."""
    sym = symbol.upper()
    row = ledger.get(sym)
    if row:
        row["qty"] = float(row.get("qty", 0.0) + add_qty)
        row["notional"] = float(row.get("notional", 0.0) + add_notional)
        row["entry_price"] = float(row["notional"] / max(row["qty"], 1e-9))
        row["horizon"] = row.get("horizon") or horizon
        row["entered_at"] = row.get("entered_at") or _now_iso()
        if reset_timebox:
            row["timebox_until"] = row.get("timebox_until")  # keep existing; change if you want extension
        ledger[sym] = row
    else:
        ledger[sym] = {
            "symbol": sym,
            "horizon": horizon,
            "qty": float(add_qty),
            "entry_price": float(add_price),
            "notional": float(add_notional),
            "entered_at": _now_iso(),
            # timebox will be set by set_timebox_on_entry at first entry
        }
    return ledger
