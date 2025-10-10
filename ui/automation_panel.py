# ui/automation_panel.py
from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

from core.positions import read_ledger
from core.trader import AlpacaTrader
from config import settings

RUN_LOG_PATH = os.path.join("state", "auto_runs.jsonl")

# ---------- utils ----------
def _to_local(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_iso

def _read_last_jsonl(path: str, max_lines: int = 500) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 4096
        data = b""
        while size > 0 and len(out) < max_lines:
            read_size = block if size >= block else size
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
            lines = data.split(b"\n")
            if len(lines) > max_lines + 1:
                lines = lines[-(max_lines+1):]
                data = b"\n".join(lines)
                break
        for line in data.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line.decode("utf-8")))
            except Exception:
                continue
    return out[-max_lines:]

def _positions_df(trader: AlpacaTrader) -> pd.DataFrame:
    ledger = read_ledger()
    rows = []
    for sym, meta in ledger.items():
        last = trader.last_price(sym) or None
        qty = float(meta.get("qty", 0))
        entry = float(meta.get("entry_price", 0))
        mv = (last * qty) if (last and qty) else None
        notional = float(meta.get("notional", 0))
        pnl = (mv - notional) if (mv is not None and notional) else None

        timebox_until = meta.get("timebox_until")
        time_remaining = None
        if timebox_until:
            try:
                until = datetime.fromisoformat(timebox_until.replace("Z", "+00:00")).astimezone()
                delta = until - datetime.now().astimezone()
                time_remaining = f"{int(delta.total_seconds()//3600)}h {int((delta.total_seconds()%3600)//60)}m"
            except Exception:
                time_remaining = "-"

        rows.append({
            "Symbol": sym,
            "Horizon": meta.get("horizon"),
            "Qty": qty,
            "Entry Price": round(entry, 4) if entry else None,
            "Last Price": round(last, 4) if last else None,
            "Notional": round(notional, 2) if notional else None,
            "Market Value": round(mv, 2) if mv is not None else None,
            "P&L": round(pnl, 2) if pnl is not None else None,
            "Entered At": _to_local(meta.get("entered_at", "")),
            "Timebox Until": _to_local(timebox_until) if timebox_until else "-",
            "Time Remaining": time_remaining or "-",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Horizon","Symbol"]).reset_index(drop=True)
    return df

def _runs_df(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        dec = r.get("decision", {})
        votes = r.get("votes", [])
        vote_map = {"ShortTerm":"S", "MidTerm":"M", "LongTerm":"L"}
        vstrs = []
        for v in votes:
            tag = vote_map.get(v.get("agent"), v.get("agent"))
            vstrs.append(f"{tag}:{v.get('decision')}({float(v.get('confidence',0)):.2f})")
        rows.append({
            "When": _to_local(r.get("when","")),
            "Symbol": r.get("symbol"),
            "Trigger": r.get("trigger"),
            "Action": r.get("action"),
            "Decision": dec.get("action"),
            "Horizon": dec.get("target_horizon"),
            "Conf": float(dec.get("confidence",0.0)) if isinstance(dec.get("confidence"), (int,float)) else None,
            "Scores": dec.get("scores"),
            "Qty": r.get("qty"),
            "Entry": r.get("entry_price"),
            "Order ID": r.get("order_id"),
            "Reason": r.get("reason"),
            "Timebox Until": _to_local(r.get("timebox_until","")) if r.get("timebox_until") else "-",
            "Votes": " | ".join(vstrs),
            "Cash": r.get("account",{}).get("cash"),
            "Equity": r.get("account",{}).get("equity"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("When", ascending=False).reset_index(drop=True)
    return df

# ---------- main renderer ----------
def render_automation_tab():
    st.subheader("⚙️ Automation — Loops, Decisions & Positions")

    # Account glance
    trader = AlpacaTrader(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_base_url)
    acct = trader.account_balances()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", f"${acct['cash']:.2f}")
    c2.metric("Equity", f"${acct['equity']:.2f}")
    c3.metric("Buying Power", f"${acct['buying_power']:.2f}")
    reserve = settings.CASH_FLOOR_PCT * acct["equity"]
    c4.metric(f"Cash Reserve Target ({int(settings.CASH_FLOOR_PCT*100)}%)", f"${reserve:.2f}")

    st.divider()

    # Open positions (from ledger)
    st.markdown("### 📦 Open Positions (Horizon-tagged)")
    pos_df = _positions_df(trader)
    if pos_df.empty:
        st.info("No horizon-tagged positions in the ledger yet.")
    else:
        st.dataframe(pos_df, use_container_width=True, height=280)

    st.divider()

    # Recent runs
    st.markdown("### 📜 Recent Automation Runs")
    cols = st.columns([1,1,1,1,2])
    with cols[0]:
        max_rows = st.selectbox("Rows", [50, 100, 200, 500], index=1)
    with cols[1]:
        symbol_filter = st.text_input("Filter by symbol (optional)", value="").upper().strip()
    with cols[2]:
        trigger_filter = st.selectbox("Trigger", ["All","bar_close_30m","price_event","news_event","timebox_expired"], index=0)
    with cols[3]:
        _ = st.button("Refresh")

    runs = _read_last_jsonl(RUN_LOG_PATH, max_rows)
    df = _runs_df(runs)

    if not df.empty:
        if symbol_filter:
            df = df[df["Symbol"] == symbol_filter]
        if trigger_filter != "All":
            df = df[df["Trigger"] == trigger_filter]

    if df.empty:
        st.warning("No runs yet — start the scheduler to see loop activity.")
    else:
        view_cols = ["When","Symbol","Trigger","Decision","Horizon","Conf","Action","Qty","Entry","Order ID","Reason","Timebox Until","Votes","Cash","Equity"]
        view_cols = [c for c in view_cols if c in df.columns]
        st.dataframe(df[view_cols], use_container_width=True, height=360)
        with st.expander("Show all columns (including raw scores)"):
            st.dataframe(df, use_container_width=True, height=360)
