# core/models.py
from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Index, Text, JSON
)
from core.db import Base


class Run(Base):
    """
    Minimal audit row for each automation decision.
    We keep the JSON decision payload so you can query confidence,
    horizon scores, etc., later if needed.
    """
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # UTC timestamp of the run (timezone-aware)
    ts_utc = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    symbol = Column(String(32), nullable=False)     # e.g., "AAPL", "BTC/USD"
    trigger = Column(String(32), nullable=False)    # e.g., "bar_close_30m"
    action = Column(String(32), nullable=False)     # BUY / SELL / HOLD / SUGGEST_BUY / ...

    decision = Column(JSON, nullable=False)         # {"action":..., "target_horizon":..., "confidence":..., "scores":...}
    reason = Column(Text, nullable=True)

    qty = Column(Float, nullable=True)              # filled qty (for BUY)
    entry_price = Column(Float, nullable=True)      # avg fill price (for BUY)
    order_id = Column(String(64), nullable=True)

    account_cash = Column(Float, nullable=True)
    account_equity = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_runs_symbol_ts", "symbol", "ts_utc"),
        Index("ix_runs_order_id", "order_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover (debug convenience)
        return f"<Run {self.ts_utc} {self.symbol} {self.action}>"
