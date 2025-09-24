# autonomous_runner.py
from __future__ import annotations
import os, json, time
from datetime import datetime, timezone
from typing import Dict, Any

from config import settings
from core.data_manager import DataManager
from core.finnhub_client import FinnhubClient
from core.semantic_memory import SemanticMemory
from core.llm import LCTraderLLM
from core.debate import Debate
from core.trader import AlpacaTrader
from core.policy import entry_policy, exit_policy

from agents.short_term_agent import ShortTermAgent
from agents.mid_term_agent import MidTermAgent
from agents.long_term_agent import LongTermAgent
from dotenv import load_dotenv
load_dotenv()

STATE_DIR  = "state"
STATE_PATH = os.path.join(STATE_DIR, "last_processed.json")
LOG_PATH   = os.path.join(STATE_DIR, "auto_runs.jsonl")   # NEW
os.makedirs(STATE_DIR, exist_ok=True)

WATCHLIST_STOCKS = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,NVDA").split(",") if s.strip()]
WATCHLIST_CRYPTO = [s.strip() for s in os.getenv("WATCHLIST_CRYPTO", "BTC/USD,ETH/USD").split(",") if s.strip()]

COOLDOWN_MINUTES     = int(os.getenv("COOLDOWN_MINUTES", "60"))
ALLOW_SHORTS_STOCKS  = os.getenv("ALLOW_SHORTS_STOCKS", "0") == "1"
NOTIONAL_STOCK       = float(os.getenv("NOTIONAL_STOCK", "1000"))
NOTIONAL_CRYPTO      = float(os.getenv("NOTIONAL_CRYPTO", "50"))

def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"last_bar_ts": {}, "last_trade_ts": {}}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def _append_log(entry: Dict[str, Any]) -> None:
    """Append one compact JSON line so UI can show recent activity."""
    try:
        entry["_ts"] = datetime.now(timezone.utc).isoformat()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _minutes_since(epoch: int) -> float:
    return (time.time() - epoch) / 60.0

def _latest_bar_time(df_30m) -> str | None:
    if df_30m is None or df_30m.empty:
        return None
    return str(df_30m["time"].iloc[-1])

def _qty_for_stock(broker: AlpacaTrader, symbol: str) -> int:
    px = broker.last_price(symbol) or 0.0
    if px <= 0:
        return 0
    return max(1, int(NOTIONAL_STOCK // px))

def _qty_for_crypto(broker: AlpacaTrader, symbol: str) -> float:
    px = broker.last_price(symbol) or 0.0
    if px <= 0:
        return 0.0
    qty = NOTIONAL_CRYPTO / px
    return float(f"{qty:.6f}")

def run_once(symbol: str, is_crypto: bool) -> Dict[str, Any]:
    if os.getenv("AUTO_PAUSED", "0") == "1":
        out = {"status": "paused", "symbol": symbol}
        _append_log({"kind": "crypto" if is_crypto else "stock", **out})
        return out

    state = _load_state()
    last_bar_ts = state.get("last_bar_ts", {})
    last_trade  = state.get("last_trade_ts", {})

    dm = DataManager()
    sm = SemanticMemory()
    fh = FinnhubClient(api_key=settings.finnhub_key) if settings.finnhub_key else None
    llm = LCTraderLLM(gemini_key=settings.gemini_key)
    debate = Debate(mean_conf_to_act=settings.mean_confidence_to_act)
    broker = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)

    snapshot = dm.layered_snapshot_crypto(symbol) if is_crypto else dm.layered_snapshot(symbol)
    df_30m   = snapshot["short_term"]
    df_daily = snapshot["mid_term"]

    bar_ts = _latest_bar_time(df_30m)
    key = f"{'C' if is_crypto else 'S'}::{symbol.upper()}"
    if not bar_ts:
        out = {"status": "skip", "symbol": symbol, "reason": "no_data"}
        _append_log({"kind": "crypto" if is_crypto else "stock", "bar_ts": None, **out})
        return out
    if last_bar_ts.get(key) == bar_ts:
        out = {"status": "skip", "symbol": symbol, "bar_ts": bar_ts, "reason": "already_processed"}
        _append_log({"kind": "crypto" if is_crypto else "stock", **out})
        return out

    lt = last_trade.get(key)
    if lt and _minutes_since(lt) < COOLDOWN_MINUTES:
        out = {"status": "cooldown", "symbol": symbol, "bar_ts": bar_ts, "wait_mins": round(COOLDOWN_MINUTES - _minutes_since(lt), 1)}
        _append_log({"kind": "crypto" if is_crypto else "stock", **out})
        return out

    try:
        if fh:
            arts = fh.crypto_news() if is_crypto else fh.company_news(symbol, days=45)
            sm.add(arts[:30])
    except Exception:
        pass

    short = ShortTermAgent("ShortTerm", llm, {})
    mid   = MidTermAgent("MidTerm", llm, {})
    long  = LongTermAgent("LongTerm", llm, {}, sm)

    votes = []
    for agent in (short, mid, long):
        d, c, raw = agent.vote(snapshot)
        votes.append({"agent": agent.name, "decision": d, "confidence": c})  # store slim version (no raw) in log

    final_decision, final_conf = debate.run(votes)
    display_symbol = df_30m["ticker"].iloc[-1] if not df_30m.empty else symbol
    held_qty = 0.0
    try:
        held_qty = broker.position_qty(display_symbol)
    except Exception:
        pass

    # EXIT path
    if held_qty > 0:
        should_exit, why_exit = exit_policy(final_decision, final_conf)
        if should_exit:
            oid = broker.market_sell(display_symbol, held_qty)
            state["last_bar_ts"][key] = bar_ts
            state["last_trade_ts"][key] = int(time.time())
            _save_state(state)
            out = {
                "status": "exit",
                "symbol": display_symbol,
                "bar_ts": bar_ts,
                "order_id": oid,
                "reason": why_exit,
                "final": {"decision": final_decision, "confidence": final_conf},
            }
            _append_log({
                "kind": "crypto" if is_crypto else "stock",
                **out,
                "votes": votes
            })
            return out

    # ENTRY path
    can_enter, why = entry_policy(final_decision, final_conf, df_30m, df_daily, votes)
    if not can_enter:
        state["last_bar_ts"][key] = bar_ts
        _save_state(state)
        out = {
            "status": "hold",
            "symbol": display_symbol,
            "bar_ts": bar_ts,
            "reason": why,
            "final": {"decision": final_decision, "confidence": final_conf},
        }
        _append_log({"kind": "crypto" if is_crypto else "stock", **out, "votes": votes})
        return out

    # We can enter
    if final_decision == "BUY":
        if held_qty > 0:
            out = {"status": "hold", "symbol": display_symbol, "bar_ts": bar_ts, "reason": "already_long",
                   "final": {"decision": final_decision, "confidence": final_conf}}
        else:
            qty = _qty_for_crypto(broker, display_symbol) if is_crypto else _qty_for_stock(broker, display_symbol)
            if qty and qty > 0:
                oid = broker.market_buy(display_symbol, qty)
                out = {"status": "entered_long", "symbol": display_symbol, "bar_ts": bar_ts, "order_id": oid, "qty": qty,
                       "final": {"decision": final_decision, "confidence": final_conf}}
            else:
                out = {"status": "hold", "symbol": display_symbol, "bar_ts": bar_ts, "reason": "qty_zero",
                       "final": {"decision": final_decision, "confidence": final_conf}}
    else:  # SELL signal
        if held_qty > 0:
            oid = broker.market_sell(display_symbol, held_qty)
            out = {"status": "closed_long", "symbol": display_symbol, "bar_ts": bar_ts, "order_id": oid, "qty": held_qty,
                   "final": {"decision": final_decision, "confidence": final_conf}}
        else:
            if not is_crypto and ALLOW_SHORTS_STOCKS:
                qty = _qty_for_stock(broker, display_symbol)
                if qty and qty > 0:
                    oid = broker.market_sell(display_symbol, qty)  # paper short
                    out = {"status": "opened_short", "symbol": display_symbol, "bar_ts": bar_ts, "order_id": oid, "qty": qty,
                           "final": {"decision": final_decision, "confidence": final_conf}}
                else:
                    out = {"status": "hold", "symbol": display_symbol, "bar_ts": bar_ts, "reason": "qty_zero",
                           "final": {"decision": final_decision, "confidence": final_conf}}
            else:
                out = {"status": "hold", "symbol": display_symbol, "bar_ts": bar_ts, "reason": "no_position_no_shorts",
                       "final": {"decision": final_decision, "confidence": final_conf}}

    state["last_bar_ts"][key] = bar_ts
    state["last_trade_ts"][key] = int(time.time())
    _save_state(state)

    _append_log({"kind": "crypto" if is_crypto else "stock", **out, "votes": votes})
    return out

if __name__ == "__main__":
    for s in WATCHLIST_STOCKS:
        print(run_once(s, is_crypto=False))
        time.sleep(1)
    for c in WATCHLIST_CRYPTO:
        print(run_once(c, is_crypto=True))
        time.sleep(1)
