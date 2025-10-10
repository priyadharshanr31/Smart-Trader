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

def _read_recent_runs(max_lines=500) -> List[dict]:
    if not os.path.exists(RUN_LOG):
        return []
    out: List[dict] = []
    with open(RUN_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
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

def run_once(symbol: str,
             is_crypto: bool = False,
             trigger: str = "bar_close_30m",
             news_boost: bool = False) -> Dict[str, Any]:
    """
    One full decision cycle for a symbol.
    - Enforce timeboxes (forced exits)
    - Build layered data
    - Get agent votes
    - Debate → final decision with horizon
    - Policy: execute SELL immediately, BUY with caps/throttles or log SUGGEST_BUY
    """
    sym = symbol.upper()

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

    llm = LCTraderLLM(model=settings.gemini_model, api_key=settings.gemini_key)

    # >>> KEY PART: lower thresholds come from config to reduce HOLD frequency
    debate = Debate(
        enter_th=settings.mean_confidence_to_act,
        exit_th=settings.exit_confidence_to_act
    )

    snap = dm.layered_snapshot_crypto(sym) if is_crypto else dm.layered_snapshot(sym)

    short = ShortTermAgent("ShortTerm", llm, {})
    mid   = MidTermAgent("MidTerm", llm, {})
    long_ = LongTermAgent("LongTerm", llm, {}, sm)

    votes = []
    for agent in (short, mid, long_):
        d, c, raw = agent.vote(snap)
        votes.append({"agent": agent.name, "decision": d, "confidence": float(c), "raw": raw})

    decision = debate.horizon_decide(votes)
    reason = summarize_reason_2lines(votes, decision)

    # 3) account & exposure
    acct = trader.account_balances()
    cash = acct["cash"]; equity = acct["equity"]
    last = trader.last_price(sym) or 0.0
    ledger = read_ledger()
    held_qty = float(ledger.get(sym, {}).get("qty", 0.0))
    symbol_mv = (last * held_qty) if last and held_qty else 0.0

    # 4) SELL: always execute if held
    if decision["action"] == "SELL":
        if sym in ledger:
            try:
                order_id = trader.close_position(sym)
                ledger.pop(sym, None)
                write_ledger(ledger)
                line = {
                    "when": _now_iso(), "symbol": sym, "trigger": trigger,
                    "decision": decision, "action": "SELL",
                    "reason": reason, "order_id": order_id,
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
                "decision": decision, "action": "SELL_NO_POSITION", "reason": reason
            }
            _append_run(line)
            return line

    # 5) BUY with caps/throttles
    if decision["action"] == "BUY" and decision.get("target_horizon") in ("short","mid","long") and last > 0:
        horizon = decision["target_horizon"]

        # throttles
        runs = list(reversed(_read_recent_runs(400)))  # newest first after reverse below
        runs_for_symbol = [r for r in runs if r.get("symbol") == sym]

        if hit_daily_buy_limit(sym, runs_for_symbol):
            line = {
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: daily buy limit)"
            }
            _append_run(line); return line

        if too_soon_since_last_buy(sym, runs_for_symbol):
            line = {
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + f" (blocked: cooldown {settings.REBUY_COOLDOWN_MINUTES}m)"
            }
            _append_run(line); return line

        # $ caps
        notional_allowed = compute_allowed_notional(horizon, cash, equity, symbol_mv)
        if notional_allowed <= 0:
            line = {
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: cash/exposure caps)"
            }
            _append_run(line); return line

        desired_qty = notional_allowed / last
        desired_qty = clamp_qty_by_share_caps(desired_qty, held_qty)
        if desired_qty <= 0:
            line = {
                "when": _now_iso(), "symbol": sym, "trigger": trigger,
                "decision": decision, "action": "SUGGEST_BUY",
                "reason": reason + " (blocked: share cap reached)"
            }
            _append_run(line); return line

        # execute buy (shares)
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
        _append_run(line); return line

    # 6) HOLD
    line = {
        "when": _now_iso(), "symbol": sym, "trigger": trigger,
        "decision": decision, "action": "HOLD", "reason": reason
    }
    _append_run(line)
    return line
