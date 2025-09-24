# core/trader.py
from __future__ import annotations
from alpaca_trade_api import REST
from typing import Optional, List, Dict

class AlpacaTrader:
    def __init__(self, key_id: str, secret_key: str, base_url: str):
        self.client = REST(key_id=key_id, secret_key=secret_key, base_url=base_url)

    # ---------- Quotes ----------
    def last_price(self, symbol: str) -> Optional[float]:
        """
        Supports stocks (e.g., 'AAPL') and crypto (e.g., 'BTC/USD').
        Tries both equity and crypto latest trade endpoints.
        """
        try:
            t = self.client.get_latest_trade(symbol)
            return float(getattr(t, "price", None) or getattr(t, "p", None))
        except Exception:
            try:
                t = self.client.get_latest_crypto_trade(symbol)
                return float(getattr(t, "price", None) or getattr(t, "p", None))
            except Exception:
                return None

    # ---------- Account ----------
    def get_account(self) -> Dict:
        """
        Returns key account fields for dashboard metrics.
        """
        acct = self.client.get_account()
        return {
            "equity": float(getattr(acct, "equity", 0.0)),
            "cash": float(getattr(acct, "cash", 0.0)),
            "buying_power": float(getattr(acct, "buying_power", 0.0)),
            "portfolio_value": float(getattr(acct, "portfolio_value", getattr(acct, "equity", 0.0))),
            "multiplier": getattr(acct, "multiplier", None),
            "status": getattr(acct, "status", None),
        }

    # ---------- Positions ----------
    def list_positions(self) -> List[Dict]:
        """
        Returns a list of positions (both stocks & crypto) in a simple dict format.
        """
        out: List[Dict] = []
        try:
            positions = self.client.list_positions()
        except Exception:
            positions = []
        for p in positions:
            try:
                out.append({
                    "symbol": getattr(p, "symbol", ""),
                    "asset_class": getattr(p, "asset_class", ""),
                    "qty": float(getattr(p, "qty", getattr(p, "qty_available", 0.0))),
                    "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0)),
                    "current_price": float(getattr(p, "current_price", 0.0)),
                    "market_value": float(getattr(p, "market_value", 0.0)),
                    "cost_basis": float(getattr(p, "cost_basis", 0.0)),
                    "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0)),
                    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0.0)),
                    "exchange": getattr(p, "exchange", ""),
                })
            except Exception:
                continue
        return out

    def position_qty(self, symbol: str) -> float:
        """Return current long qty if any; 0.0 if flat."""
        try:
            pos = self.client.get_position(symbol)
            return float(getattr(pos, "qty", getattr(pos, "qty_available", 0.0)))
        except Exception:
            return 0.0

    # ---------- Orders ----------
    def market_buy(self, symbol: str, qty: float | int) -> str:
        o = self.client.submit_order(
            symbol=symbol, qty=str(qty), side='buy', type='market', time_in_force='gtc'
        )
        return o.id

    def market_sell(self, symbol: str, qty: float | int) -> str:
        o = self.client.submit_order(
            symbol=symbol, qty=str(qty), side='sell', type='market', time_in_force='gtc'
        )
        return o.id
