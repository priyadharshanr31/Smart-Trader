from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv, find_dotenv

from config import settings
from core.data_manager import DataManager
from core.semantic_memory import SemanticMemory
from core.finnhub_client import FinnhubClient
from core.llm import LCTraderLLM
from core.debate import Debate
from core.trader import AlpacaTrader
from core.policy import (
    compute_allowed_notional,
    clamp_qty_by_share_caps,
    too_soon_since_last_buy,
    hit_daily_buy_limit,
)
from core.positions import read_ledger, write_ledger

# Ensure .env is loaded for GEMINI_API_KEY, Alpaca, etc.
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv(override=True)


STATE_DIR = "state"
RUN_LOG = os.path.join(STATE_DIR, "auto_runs.jsonl")
os.makedirs(STATE_DIR, exist_ok=True)


def _extract_rationale(raw: str) -> str:
    """
    Best-effort extraction of 'rationale' from the LLM raw string.

    raw looks like:
      [model=models/gemini-2.5-flash] ```json
      {
        "vote": "BUY",
        "confidence": 0.8,
        "rationale": "Price is breaking out..."
      }
      ```
    We just search for "rationale" and grab that field.
    """
    if not raw:
        return ""
    text = raw
    # crude search for "rationale"
    import re, json

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            rat = obj.get("rationale") or obj.get("reason") or ""
            if isinstance(rat, str):
                return rat.strip()
        except Exception:
            pass

    # fallback: just return a truncated version of the whole raw
    return text.replace("\n", " ")[:400]


def _append_run_log(entry: Dict[str, Any]) -> None:
    try:
        with open(RUN_LOG, "a", encoding="utf-8") as f:
            import json as _json

            f.write(_json.dumps(entry) + "\n")
    except Exception:
        pass


def run_once(
    symbol: str,
    is_crypto: bool,
    trigger: str,
    news_boost: bool = False,
) -> Dict[str, Any]:
    """
    Single autonomous run for one symbol (stock or crypto).

    - Builds data snapshot (short/mid/long)
    - Builds semantic memory (news for stocks, crypto news for crypto)
    - Asks Short/Mid/Long agents to vote
    - Debates the votes into one decision (action + horizon)
    - Applies risk policy to turn it into an executable action
    - Logs to state/auto_runs.jsonl
    - RETURNS the final record (dict) which run_scheduler prints
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    print(f"[run_once] ===== RUN START for {symbol} (is_crypto={is_crypto}, trigger={trigger}) =====")

    # --- debug: show which Gemini key is being used this run ---
    from os import getenv

    k = (getenv("GEMINI_API_KEY") or "").strip()
    if k:
        print(f"[run_once] Gemini key fingerprint: {k[:4]}...{k[-4:]}")

    # --- shared tools ---
    dm = DataManager()
    sm = SemanticMemory()
    fh = FinnhubClient(api_key=settings.finnhub_key) if settings.finnhub_key else None
    llm = LCTraderLLM(api_key=settings.gemini_key or getenv("GEMINI_API_KEY"))
    debate = Debate(enter_th=settings.mean_confidence_to_act,
                    exit_th=settings.exit_confidence_to_act)

    # --- build price snapshot ---
    if is_crypto:
        snapshot = dm.layered_snapshot_crypto(symbol)
    else:
        snapshot = dm.layered_snapshot(symbol)

    if snapshot["mid_term"].empty:
        record = {
            "when": now,
            "symbol": symbol,
            "trigger": trigger,
            "decision": {
                "action": "HOLD",
                "target_horizon": None,
                "confidence": 0.0,
                "scores": {"short": 0.0, "mid": 0.0, "long": 0.0},
            },
            "action": "HOLD",
            "reason": "No data available.",
        }
        _append_run_log(record)
        return record

    # --- build semantic memory ---
    print("[run_once] Building semantic memory…")
    try:
        if is_crypto:
            if fh:
                sm.add((fh.crypto_news(max_items=50) or [])[:30])
        else:
            if fh:
                sm.add((fh.company_news(symbol, days=45) or [])[:30])
    except Exception as e:
        print(f"[run_once] News unavailable: {e}")

    # --- agents ---
    from agents.short_term_agent import ShortTermAgent
    from agents.mid_term_agent import MidTermAgent
    from agents.long_term_agent import LongTermAgent

    short = ShortTermAgent("ShortTerm", llm, {})
    mid = MidTermAgent("MidTerm", llm, {})
    long = LongTermAgent("LongTerm", llm, {}, sm)

    votes: List[Dict[str, Any]] = []

    # Short-term vote
    s_dec, s_conf, s_raw = short.vote(snapshot)
    votes.append({"agent": "ShortTerm", "decision": s_dec, "confidence": s_conf, "raw": s_raw})

    # Mid-term vote
    m_dec, m_conf, m_raw = mid.vote(snapshot)
    votes.append({"agent": "MidTerm", "decision": m_dec, "confidence": m_conf, "raw": m_raw})

    # Long-term vote
    l_dec, l_conf, l_raw = long.vote(snapshot)
    votes.append({"agent": "LongTerm", "decision": l_dec, "confidence": l_conf, "raw": l_raw})

    # --- DEBUG: print what the LLM is thinking for each agent ---
    print(f"[run_once] Votes for {symbol} (trigger={trigger}):")
    for v in votes:
        rat = _extract_rationale(v.get("raw", "") or "")
        print(f"  - {v['agent']}: {v['decision']} (conf={v['confidence']:.2f})")
        if rat:
            # indent rationale nicely
            from textwrap import fill

            wrapped = fill(rat, width=100, subsequent_indent=" " * 8)
            print(f"    rationale: {wrapped}")
        else:
            print("    rationale: (none / LLM unavailable)")

    # --- debate ---
    decision_obj = debate.horizon_decide(votes)
    # decision_obj = {"action","target_horizon","confidence","scores":{short,mid,long}}

    # extra logging: explain horizon choice + scores
    scores = decision_obj.get("scores", {}) or {}
    print(
        f"[run_once] Debate result for {symbol}: "
        f"{decision_obj.get('action','HOLD')} "
        f"(horizon={decision_obj.get('target_horizon')}, "
        f"conf={float(decision_obj.get('confidence', 0.0)):.3f}, "
        f"scores={scores})"
    )

    # --- apply risk policy / position logic ---
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    ledger = read_ledger()
    sym_key = symbol  # ledger key uses display symbol (e.g. BTC/USD)

    # current position (ledger + broker)
    broker_pos = trader.position_qty(symbol)
    ledger_pos = float(ledger.get(sym_key, 0.0))
    combined_pos = broker_pos + ledger_pos

    final_action = decision_obj.get("action", "HOLD")
    horizon = decision_obj.get("target_horizon")
    conf = float(decision_obj.get("confidence", 0.0))

    # compute buy/sell quantity based on risk limits
    qty = 0.0
    order_id: Optional[str] = None
    reason_tail = ""

    if final_action == "BUY":
        max_notional = compute_allowed_notional(trader, symbol, horizon or "short")
        qty = trader.notional_to_qty(symbol, max_notional)

        qty = clamp_qty_by_share_caps(trader, symbol, qty)
        if qty <= 0:
            final_action = "HOLD"
            reason_tail = " (qty calculated as 0 after caps)"
        elif too_soon_since_last_buy(sym_key):
            final_action = "HOLD"
            reason_tail = " (rebuy cooldown not elapsed)"
        elif hit_daily_buy_limit(sym_key):
            final_action = "HOLD"
            reason_tail = " (daily buy limit reached)"
        else:
            try:
                order_id = trader.market_buy(symbol, qty)
                ledger[sym_key] = ledger_pos + qty
            except Exception as e:
                final_action = "HOLD"
                reason_tail = f" (BUY failed: {e})"

    elif final_action == "SELL":
        if combined_pos <= 0:
            # we don't own it – log as SELL_NO_POSITION
            final_action = "SELL_NO_POSITION"
        else:
            qty = combined_pos
            try:
                order_id = trader.market_sell(symbol, qty)
                ledger[sym_key] = max(0.0, ledger_pos - qty)
            except Exception as e:
                final_action = "HOLD"
                reason_tail = f" (SELL failed: {e})"

    write_ledger(ledger)

    # --- build reason string for log/DB ---
    reason = (
        f"Final: {final_action if final_action != 'SELL_NO_POSITION' else 'SELL'} "
        f"(horizon={horizon or '-'}, conf={conf:.2f}). "
        f"Votes: "
        f"S:{s_dec}({s_conf:.2f}) | "
        f"M:{m_dec}({m_conf:.2f}) | "
        f"L:{l_dec}({l_conf:.2f})"
        f"{reason_tail}"
    )

    record: Dict[str, Any] = {
        "when": now,
        "symbol": symbol,
        "trigger": trigger,
        "decision": {
            "action": final_action if final_action != "SELL_NO_POSITION" else "SELL",
            "target_horizon": horizon,
            "confidence": round(conf, 3),
            "scores": scores,
        },
        "action": final_action,
        "reason": reason,
        "qty": qty if qty else None,
        "order_id": order_id or None,
    }

    _append_run_log(record)
    print(record)
    return record
