# core/trader.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional, List, Tuple, Union

from alpaca_trade_api.rest import REST  # consistent import

Number = Union[int, float]


class AlpacaTrader:
    """
    Thin wrapper around alpaca-trade-api used by the UI and automation.
    - Reads creds from args or environment (ALPACA_* or APCA_*).
    - Normalizes account/position objects to plain dicts.
    """

    def __init__(
        self,
        key_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Accept both prefixes so Streamlit (ALPACA_*) and the SDK (APCA_*) are happy.
        key_id = key_id or os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        secret_key = (
            secret_key
            or os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
        )
        base_url = (
            (base_url or os.getenv("ALPACA_PAPER_BASE_URL") or os.getenv("APCA_API_BASE_URL"))
            or "https://paper-api.alpaca.markets"
        ).strip()

        if not key_id or not secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY "
                "or APCA_API_KEY_ID/APCA_API_SECRET_KEY (in .env or environment)."
            )

        self.client = REST(key_id=key_id, secret_key=secret_key, base_url=base_url)

    # ---------------- Account ----------------
    def get_account(self) -> Dict[str, Any]:
        a = self.client.get_account()
        return {
            "cash": float(getattr(a, "cash", 0.0)),
            "equity": float(getattr(a, "equity", 0.0)),
            "buying_power": float(getattr(a, "buying_power", 0.0)),
            "portfolio_value": float(
                getattr(a, "portfolio_value", getattr(a, "equity", 0.0))
            ),
            "status": getattr(a, "status", ""),
            "multiplier": getattr(a, "multiplier", None),
        }

    # convenience alias used in one place
    def account_balances(self) -> Dict[str, Any]:
        return self.get_account()

    # ---------------- Quotes ----------------
    def last_price(self, symbol: str) -> Optional[float]:
        """
        Works for equities ('AAPL') and crypto ('BTC/USD').
        Tries equity first, then crypto endpoint.
        """
        try:
            t = self.client.get_latest_trade(symbol)
            px = getattr(t, "price", None)
            if px is not None:
                return float(px)
        except Exception:
            pass
        try:
            t = self.client.get_latest_crypto_trade(symbol)
            px = getattr(t, "price", None)
            if px is not None:
                return float(px)
        except Exception:
            pass
        return None

    # ---------------- Positions ----------------
    def list_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            for p in self.client.list_positions():
                out.append(
                    {
                        "symbol": getattr(p, "symbol", ""),
                        "asset_class": getattr(p, "asset_class", ""),  # 'us_equity' or 'crypto'
                        "qty": float(getattr(p, "qty", getattr(p, "qty_available", 0.0))),
                        "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0)),
                        "current_price": float(
                            getattr(p, "current_price", getattr(p, "asset_current_price", 0.0))
                        ),
                        "market_value": float(getattr(p, "market_value", 0.0)),
                        "cost_basis": float(getattr(p, "cost_basis", 0.0)),
                        "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0)),
                        "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0.0)),
                        "exchange": getattr(p, "exchange", ""),
                    }
                )
        except Exception:
            return []
        return out

    def position_mv(self, symbol: str) -> float:
        try:
            p = self.client.get_position(symbol)
            return float(getattr(p, "market_value", 0.0))
        except Exception:
            return 0.0

    def position_qty(self, symbol: str) -> float:
        try:
            p = self.client.get_position(symbol)
            return float(getattr(p, "qty", getattr(p, "qty_available", 0.0)))
        except Exception:
            return 0.0

    def positions_symbols_by_class(self) -> Tuple[List[str], List[str]]:
        """Return (equity_symbols, crypto_symbols)."""
        eq, cr = [], []
        for p in self.list_positions():
            ac = (p.get("asset_class") or "").lower()
            (cr if "crypto" in ac else eq).append(p["symbol"])
        return eq, cr

    # ---------------- Orders ----------------
    def market_buy(self, symbol: str, qty: Number, client_order_id: Optional[str] = None) -> str:
        o = self.client.submit_order(
            symbol=symbol,
            qty=str(qty),
            side="buy",
            type="market",
            time_in_force="day",
            client_order_id=client_order_id,
        )
        return o.id

    def market_sell(self, symbol: str, qty: Number, client_order_id: Optional[str] = None) -> str:
        o = self.client.submit_order(
            symbol=symbol,
            qty=str(qty),
            side="sell",
            type="market",
            time_in_force="day",
            client_order_id=client_order_id,
        )
        return o.id

    # Backward-compat aliases
    def market_buy_qty(self, symbol: str, qty: Number) -> str:
        return self.market_buy(symbol, qty)

    def market_sell_qty(self, symbol: str, qty: Number) -> str:
        return self.market_sell(symbol, qty)

    def close_position(self, symbol: str) -> Optional[str]:
        try:
            r = self.client.close_position(symbol)
            return getattr(r, "id", None)
        except Exception:
            return None
