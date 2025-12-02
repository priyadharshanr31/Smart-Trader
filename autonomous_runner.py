# autonomous_runner.py
from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import Dict, Any, List

from config import settings
from core.data_manager import DataManager
from core.debate import Debate, summarize_reason_2lines
from core.llm import LCTraderLLM
from core.policy import (
    compute_allowed_notional, clamp_qty_by_share_caps,
    too_soon_since_last_buy, hit_daily_buy_limit
)
from core.trader import AlpacaTrader
from core.positions import read_ledger, write_ledger, set_timebox_on_entry, merge_entry
from core.semantic_memory import SemanticMemory
from core.store import save_run_dict   # dual-write to MySQL
from agents.short_term_agent import ShortTermAgent
from agents.mid_term_agent import MidTermAgent
from agents.long_term_agent import LongTermAgent

STATE_DIR = "state"
RUN_LOG = os.path.join(STATE_DIR, "auto_runs.jsonl")
os.makedirs(STATE_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")


def _append_run(line: Dict[str, Any]) -> None:
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    try:
        save_run_dict(line)
    except Exception as e:
        print(f"[runs->mysql] save failed: {e}")


def _read_recent_runs(max_lines=500) -> List[dict]:
    if not os.path.exists(RUN_LOG):
        return []
    out: List[dict] = []
    with open(RUN_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out[-max_lines:]


# ---- timebox enforcement on every cycle ----
def enforce_timeboxes(trader: AlpacaTrader):
    ledger = read_ledger()
    now = datetime.now(timezone.utc)
    changed = False
    for sym, meta in list(ledger.items()):
        tb = meta.get("timebox_until")
        if not tb:
            continue
        try:
            until = datetime.fromisoformat(tb.replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if now >= until:
            try:
                order_id = trader.close_position(sym)
                _append_run({
                    "when": _now_iso(), "symbol": sym, "trigger": "timebox_expired",
                    "action": "FORCED_EXIT",
                    "reason": f"Timebox expired for {meta.get('horizon')}",
                    "order_id": order_id,
                })
            except Exception as e:
                _append_run({
                    "when": _now_iso(), "symbol": sym, "trigger": "timebox_expired",
                    "action": "FORCED_EXIT_FAILED",
                    "error": str(e),
                })
            ledger.pop(sym, None)
            changed = True
    if changed:
        write_ledger(ledger)


def run_once(
    symbol: str,
    is_crypto: bool = False,
    trigger: str = "bar_close_30m",
    news_boost: bool = False
) -> Dict[str, Any]:
    """
    One full decision cycle for a symbol.
    """
    sym = symbol.upper()

    print(f"\n[run_once] ===== RUN START for {sym} (is_crypto={is_crypto}, trigger={trigger}) =====")

    # toolbelt
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)

    # 1) enforce timeboxes first
    enforce_timeboxes(trader)

    # 2) data + agents
    dm = DataManager()

    # SAFE semantic memory init — never let this crash the job
    try:
        sm = SemanticMemory()
    except Exception as e:
        print(f"[run_once] SemanticMemory init failed: {e} — continuing without it.")
        sm = None

    # LLM with current Gemini key
    llm = LCTraderLLM(model=settings.gemini_model, api_key=settings.gemini_key)

    # DEBUG: which Gemini key is this run using?
    try:
        print(f"[run_once] Gemini key fingerprint: {llm.debug_key_fingerprint()}")
    except Exception as e:
        print(f"[run_once] Could not get Gemini key fingerprint: {e}")

    debate = Debate(
        enter_th=settings.mean_confidence_to_act,
        exit_th=settings.exit_confidence_to_act
    )

    snap = dm.layered_snapshot_crypto(sym) if is_crypto else dm.layered_snapshot(sym)

    short = ShortTermAgent("ShortTerm", llm, {})
    mid   = MidTermAgent("MidTerm", llm, {})
    long_ = LongTermAgent("LongTerm", llm, {}, sm)

    # --- Agent votes + debug logging ---
    votes: List[Dict[str, Any]] = []
    for agent in (short, mid, long_):
        d, c, raw = agent.vote(snap)
        v = {"agent": agent.name, "decision": d, "confidence": float(c), "raw": raw}
        votes.append(v)

    print(f"[run_once] Votes for {sym} (trigger={trigger}):")
    for v in votes:
        raw_preview = str(v["raw"])
        if len(raw_preview) > 300:
            raw_preview = raw_preview[:300] + "..."
        print(
            f"  - {v['agent']}: {v['decision']} (conf={v['confidence']:.2f})\n"
            f"    rationale: {raw_preview}"
        )

    decision = debate.horizon_decide(votes)
    reason = summarize_reason_2lines(votes, decision)

    # DEBUG: final ensemble decision
    try:
        print(
            f"[run_once] Final decision for {sym}: "
            f"{decision.get('action')} "
            f"(horizon={decision.get('target_horizon')}, "
            f"conf={float(decision.get('confidence', 0.0)):.2f})"
        )
    except Exception:
        print(f"[run_once] Final decision for {sym}: {decision}")

    # 3) account & exposure
    acct = trader.account_balances()
    cash = acct["cash"]; equity = acct["equity"]

    # Primary price source = Alpaca; fallback to the snapshot's latest close
    alpaca_px = trader.last_price(sym)
    snap_px = None
    try:
        st = snap.get("short_term")
        if st is not None and not st.empty and "close" in st.columns:
            snap_px = float(st["close"].iloc[-1])
    except Exception:
        snap_px = None
    last = float(alpaca_px or 0.0) or float(snap_px or 0.0)  # fallback if Alpaca returns None/0

    ledger = read_ledger()
    held_qty_ledger = float((ledger.get(sym, {}) or {}).get("qty", 0.0))
    symbol_mv = (last * held_qty_ledger) if last and held_qty_ledger else 0.0

    # 4) SELL: execute if either ledger or broker shows qty > 0
    if decision["action"] == "SELL":
        held_qty_broker = trader.position_qty(sym)
        if (held_qty_ledger > 0.0) or (held_qty_broker > 0.0):
            try:
                order_id = trader.close_position(sym)
                if sym in ledger:
                    ledger.pop(sym, None)
                    write_ledger(ledger)
                line = {
                    "when": _now_iso(), "symbol": sym, "trigger": trigger,
                    "decision": decision, "action": "SELL",
                    "reason": reason + f" (pre-sell qty: ledger={held_qty_ledger:.8f}, broker={held_qty_broker:.8f})",
                    "order_id": order_id,
                    "account": trader.account_balances(),
                }
                _append_run(line)
                return line
            except Exception as e:
                line = {
                    "when": _now_iso(), "symbol": sym, "trigger": trigger,
                    "decision": decision, "action": "SELL_FAILED", "error": str(e)
                }
                _append_run(line)
                return line
        else:
            line = {
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SELL_NO_POSITION",
                "reason": f"{reason} (no open position: ledger={held_qty_ledger:.8f}, broker={held_qty_broker:.8f})"
            }
            _append_run(line)
            return line

    # 5) BUY with caps/throttles
    if decision["action"] == "BUY":
        horizon = decision.get("target_horizon")

        # Guard 1: require a valid horizon
        if horizon not in ("short", "mid", "long"):
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: missing/invalid target_horizon)"
            })
            return {"action": "SUGGEST_BUY"}

        # Guard 2: need a working price
        if last <= 0:
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + f" (blocked: no price; alpaca_px={alpaca_px}, snap_px={snap_px})"
            })
            return {"action": "SUGGEST_BUY"}

        # throttles
        runs = list(reversed(_read_recent_runs(400)))
        runs_for_symbol = [r for r in runs if r.get("symbol") == sym]

        if hit_daily_buy_limit(sym, runs_for_symbol):
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: daily buy limit)"
            })
            return {"action": "SUGGEST_BUY"}

        if too_soon_since_last_buy(sym, runs_for_symbol):
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + f" (blocked: cooldown {settings.REBUY_COOLDOWN_MINUTES}m)"
            })
            return {"action": "SUGGEST_BUY"}

        # $ caps
        notional_allowed = compute_allowed_notional(horizon, cash, equity, symbol_mv)
        if notional_allowed <= 0:
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: cash/exposure caps)"
            })
            return {"action": "SUGGEST_BUY"}

        desired_qty = notional_allowed / last
        desired_qty = clamp_qty_by_share_caps(desired_qty, held_qty_ledger)
        if desired_qty <= 0:
            _append_run({
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: share cap reached)"
            })
            return {"action": "SUGGEST_BUY"}

        # execute buy (shares) — returns (order_id, filled_qty, avg_px)
        order_id, filled_qty, avg_px = trader.market_buy_qty(sym, desired_qty)

        # first entry vs add to existing
        if sym not in ledger:
            tb_until = set_timebox_on_entry(sym, horizon, filled_qty, avg_px, filled_qty * avg_px)
        else:
            merge_entry(ledger, sym, horizon, filled_qty, avg_px, filled_qty * avg_px, reset_timebox=False)
            write_ledger(ledger)
            tb_until = ledger.get(sym, {}).get("timebox_until")

        line = {
            "when": _now_iso(), "symbol": sym, "trigger": trigger,
            "decision": decision, "action": "BUY",
            "qty": filled_qty, "entry_price": avg_px,
            "order_id": order_id, "timebox_until": tb_until,
            "reason": reason, "account": trader.account_balances(),
        }
        _append_run(line)
        return line

    # 6) HOLD
    line = {
        "when": _now_iso(), "symbol": sym, "trigger": trigger,
        "decision": decision, "action": "HOLD", "reason": reason
    }
    _append_run(line)
    return line
