# core/policy.py
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd

# Enter vs Exit thresholds (you can tune these)
ENTER_TH = 0.60   # stricter to open positions
EXIT_TH  = 0.45   # easier to exit (protect capital)

def _macro_vote(votes: list[dict]) -> dict | None:
    return next((v for v in votes if v["agent"] == "LongTerm"), None)

def entry_policy(final_side: str,
                 final_conf: float,
                 df_30m: pd.DataFrame,
                 df_daily: pd.DataFrame,
                 votes: list[Dict]) -> Tuple[bool, str]:
    """
    Confluence gates before opening a position:
      - debate confidence >= ENTER_TH
      - daily trend filter (20DMA) + RSI(14)
      - 30m momentum (MACD vs signal)
      - macro veto if strongly opposed
      - avoid buying/selling right at extreme Bollinger bands
    """
    if final_side not in ("BUY", "SELL"):
        return False, "no_action"

    if final_conf < ENTER_TH:
        return False, "low_consensus"

    # Daily trend context
    close_d = df_daily['close'].iloc[-1]
    ma20_d  = df_daily['close'].rolling(20).mean().iloc[-1]
    rsi_d   = df_daily['rsi'].iloc[-1]

    # 30m momentum + stretch
    macd_30 = df_30m['macd'].iloc[-1]
    sig_30  = df_30m['macd_signal'].iloc[-1]
    px_30   = df_30m['close'].iloc[-1]
    up_30   = df_30m['upper_band'].iloc[-1]
    lo_30   = df_30m['lower_band'].iloc[-1]

    # Macro veto (blocks fighting the regime)
    macro = _macro_vote(votes)
    if macro:
        if final_side == "BUY" and macro["decision"] in ("HOLD", "SELL") and macro["confidence"] >= 0.70:
            return False, "macro_veto"
        if final_side == "SELL" and macro["decision"] in ("HOLD", "BUY") and macro["confidence"] >= 0.70:
            return False, "macro_veto"

    if final_side == "BUY":
        if not (close_d > ma20_d and rsi_d > 50 and macd_30 > sig_30):
            return False, "no_confluence"
        if px_30 >= up_30:
            return False, "stretched_upper"
    else:  # SELL
        if not (close_d < ma20_d and rsi_d < 50 and macd_30 < sig_30):
            return False, "no_confluence"
        if px_30 <= lo_30:
            return False, "stretched_lower"

    return True, "ok"

def exit_policy(final_decision: str,
                final_conf: float,
                bars_held: int | None = None) -> Tuple[bool, str]:
    """
    Minimal exit logic for now: allow agent-driven SELLs with a lower threshold,
    plus an optional time-based exit (e.g., stale trades).
    """
    if final_decision == "SELL" and final_conf >= EXIT_TH:
        return True, "agent_sell"

    if bars_held is not None and bars_held >= 20:
        return True, "time_exit"

    return False, "hold"
