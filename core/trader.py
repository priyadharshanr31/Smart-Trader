# core/trader.py
from __future__ import annotations
import os, time
from typing import Any, Dict, Optional, List, Tuple, Union

from alpaca_trade_api.rest import REST  # consistent import

Number = Union[int, float]

# ---------- symbol helpers ----------
def _to_broker_symbol(sym: str) -> str:
    """
    UI/logic may use BTC/USD; Alpaca positions/orders use BTCUSD (no slash).
    Equities (AAPL) are passed through unchanged.
    """
    s = (sym or "").upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}{quote}"
    return s

def _to_crypto_pair(sym: str) -> str:
    """
    Convert broker format back to pair if needed, e.g., BTCUSD -> BTC/USD.
    Used mainly for quotes; best-effort.
    """
    s = (sym or "").upper().replace(" ", "")
    for q in ("USDT", "USDC", "USD", "EUR", "BTC", "ETH"):
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            return f"{base}/{q}"
    return s if "/" in s else s

def _is_crypto_symbol(sym: str) -> bool:
    """
    Heuristic: BTC/USD or BTCUSD/ETHUSDT/etc. (broker format that ends with a known quote)
    """
    s = (sym or "").upper().replace(" ", "")
    if "/" in s:
        return True
    for q in ("USDT", "USDC", "USD", "EUR", "BTC", "ETH"):
        if s.endswith(q) and len(s) > len(q):
            return True
    return False


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
                "or APCA_API_KEY_ID/APCA_API_SECRET_KEY."
            )

        self.client = REST(key_id=key_id, secret_key=secret_key, base_url=base_url)

    # ---------------- Account ----------------
    def get_account(self) -> Dict[str, Any]:
        a = self.client.get_account()
        return {
            "cash": float(getattr(a, "cash", 0.0)),
            "equity": float(getattr(a, "equity", 0.0)),
            "buying_power": float(getattr(a, "buying_power", 0.0)),
            "portfolio_value": float(getattr(a, "portfolio_value", getattr(a, "equity", 0.0))),
            "status": getattr(a, "status", ""),
            "multiplier": getattr(a, "multiplier", None),
        }

    # convenience alias
    def account_balances(self) -> Dict[str, Any]:
        return self.get_account()

    # ---------------- Quotes ----------------
    def last_price(self, symbol: str) -> Optional[float]:
        """
        Works for equities ('AAPL') and crypto ('BTC/USD' or 'BTCUSD').
        Try both equity and crypto endpoints with multiple symbol shapes.
        """
        candidates = [
            (symbol or "").upper(),
            _to_broker_symbol(symbol),
            _to_crypto_pair(symbol),
        ]
        # Equities first
        for sym_try in candidates:
            try:
                t = self.client.get_latest_trade(sym_try)
                px = getattr(t, "price", None)
                if px is not None:
                    return float(px)
            except Exception:
                pass
        # Crypto endpoint
        for sym_try in candidates:
            try:
                t = self.client.get_latest_crypto_trade(sym_try)
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
                        "symbol": getattr(p, "symbol", ""),   # broker symbol e.g., BTCUSD
                        "asset_class": getattr(p, "asset_class", ""),  # 'us_equity' or 'crypto'
                        "qty": float(getattr(p, "qty", getattr(p, "qty_available", 0.0)) or 0.0),
                        "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                        "current_price": float(getattr(p, "current_price", getattr(p, "asset_current_price", 0.0)) or 0.0),
                        "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
                        "cost_basis": float(getattr(p, "cost_basis", 0.0) or 0.0),
                        "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                        "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0.0) or 0.0),
                        "exchange": getattr(p, "exchange", ""),
                    }
                )
        except Exception:
            return []
        return out

    def _position_qty_via_list(self, symbol: str) -> float:
        """Fallback: scan list_positions() if get_position() fails."""
        bsym = _to_broker_symbol(symbol)
        for p in self.list_positions():
            if (p.get("symbol") or "").upper() == bsym:
                return float(p.get("qty") or 0.0)
        return 0.0

    def position_mv(self, symbol: str) -> float:
        bsym = _to_broker_symbol(symbol)
        try:
            p = self.client.get_position(bsym)
            return float(getattr(p, "market_value", 0.0) or 0.0)
        except Exception:
            # Fallback: estimate from list_positions
            for pos in self.list_positions():
                if (pos.get("symbol") or "").upper() == bsym:
                    return float(pos.get("market_value") or 0.0)
            return 0.0

    def position_qty(self, symbol: str) -> float:
        bsym = _to_broker_symbol(symbol)
        try:
            p = self.client.get_position(bsym)
            q = float(getattr(p, "qty", getattr(p, "qty_available", 0.0)) or 0.0)
        except Exception:
            q = self._position_qty_via_list(symbol)
        # Treat dust as zero
        return 0.0 if abs(q) < 1e-12 else q

    def positions_symbols_by_class(self) -> Tuple[List[str], List[str]]:
        """Return (equity_symbols, crypto_symbols) using broker symbols."""
        eq, cr = [], []
        for p in self.list_positions():
            ac = (p.get("asset_class") or "").lower()
            (cr if "crypto" in ac else eq).append(p["symbol"])
        return eq, cr

    # ---------------- Orders ----------------
    def _await_fills(
        self, order_id: str, orig_symbol: str, requested_qty: Number, timeout_s: float = 3.0
    ) -> Tuple[float, float]:
        """
        Try to get filled_qty and avg_price for a just-submitted market order.
        Falls back to (requested_qty, last_price) if not filled fast enough.
        """
        end = time.time() + max(0.2, timeout_s)
        filled_qty = 0.0
        avg_px = 0.0
        while time.time() < end:
            try:
                o = self.client.get_order(order_id)
                status = (getattr(o, "status", "") or "").lower()
                fq = getattr(o, "filled_qty", None)
                ap = getattr(o, "filled_avg_price", None)
                if fq: filled_qty = float(fq)
                if ap: avg_px = float(ap)
                if status in ("filled", "partially_filled", "canceled", "done_for_day"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        if filled_qty <= 0.0:
            filled_qty = float(requested_qty)
        if avg_px <= 0.0:
            lp = self.last_price(orig_symbol)
            if lp:
                avg_px = float(lp)
        return filled_qty, avg_px

    def _tif_for(self, symbol: str) -> str:
        """
        Alpaca constraint: crypto does NOT allow time_in_force='day'.
        Use 'gtc' for crypto, 'day' for equities.
        """
        return "gtc" if _is_crypto_symbol(symbol) else "day"

    def market_buy(
        self, symbol: str, qty: Number, client_order_id: Optional[str] = None
    ) -> Tuple[str, float, float]:
        bsym = _to_broker_symbol(symbol)
        tif = self._tif_for(symbol)
        o = self.client.submit_order(
            symbol=bsym,
            qty=str(qty),
            side="buy",
            type="market",
            time_in_force=tif,  # <-- crypto: gtc, equities: day
            client_order_id=client_order_id,
        )
        order_id = getattr(o, "id", None)
        fq, ap = self._await_fills(order_id, symbol, qty)
        return order_id, fq, ap

    def market_sell(
        self, symbol: str, qty: Number, client_order_id: Optional[str] = None
    ) -> Tuple[str, float, float]:
        bsym = _to_broker_symbol(symbol)
        tif = self._tif_for(symbol)
        o = self.client.submit_order(
            symbol=bsym,
            qty=str(qty),
            side="sell",
            type="market",
            time_in_force=tif,  # <-- crypto: gtc, equities: day
            client_order_id=client_order_id,
        )
        order_id = getattr(o, "id", None)
        fq, ap = self._await_fills(order_id, symbol, qty)
        return order_id, fq, ap

    # Backward-compat aliases (return (order_id, filled_qty, avg_px))
    def market_buy_qty(self, symbol: str, qty: Number) -> Tuple[str, float, float]:
        return self.market_buy(symbol, qty)

    def market_sell_qty(self, symbol: str, qty: Number) -> Tuple[str, float, float]:
        return self.market_sell(symbol, qty)

    def close_position(self, symbol: str) -> Optional[str]:
        """
        Preferred: broker 'close_position'.
        Fallback: if that fails (e.g., 404), fetch qty and submit a market SELL.
        """
        bsym = _to_broker_symbol(symbol)
        try:
            r = self.client.close_position(bsym)
            return getattr(r, "id", None)
        except Exception:
            # Fallback: try explicit market sell of whatever qty we find
            qty = self.position_qty(symbol)
            if qty > 0:
                try:
                    order_id, *_ = self.market_sell(symbol, qty)
                    return order_id
                except Exception:
                    return None
            return None
