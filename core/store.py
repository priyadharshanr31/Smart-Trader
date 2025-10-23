# core/store.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any

from core.db import SessionLocal
from core.models import Run


def _as_dt(ts_iso: str) -> datetime:
    """
    "2025-10-22T12:34:56Z" -> timezone-aware datetime
    Safe if ts_iso is None (falls back to now UTC).
    """
    if not ts_iso:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def save_run_dict(d: Dict[str, Any]) -> None:
    """
    Persist a 'run' dict (the same one you append to JSONL) into MySQL.
    Designed to be best-effort: callers can swallow exceptions so trading never blocks.
    """
    r = Run(
        ts_utc=_as_dt(d.get("when")),
        symbol=(d.get("symbol") or "").upper(),
        trigger=d.get("trigger") or "",
        action=d.get("action") or "",
        decision=d.get("decision") or {},
        reason=d.get("reason"),
        qty=d.get("qty"),
        entry_price=d.get("entry_price"),
        order_id=d.get("order_id"),
        account_cash=(d.get("account") or {}).get("cash"),
        account_equity=(d.get("account") or {}).get("equity"),
    )
    with SessionLocal() as s:
        s.add(r)
        s.commit()
