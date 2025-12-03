# app.py
# Streamlit UI with: Dashboard, Stocks, Crypto, Suggestions, News, Automation (NEW)

import os
from datetime import datetime, timezone
from typing import List, Dict
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ui.automation_panel import render_automation_tab

from config import settings
from core.data_manager import DataManager
from core.finnhub_client import FinnhubClient
from core.semantic_memory import SemanticMemory
from core.llm import LCTraderLLM
from core.debate import Debate
from core.trader import AlpacaTrader

from agents.short_term_agent import ShortTermAgent
from agents.mid_term_agent import MidTermAgent
from agents.long_term_agent import LongTermAgent
from agents.suggestions_agent import SuggestionsAgent  # separate suggestions agent

STATE_DIR = "state"
RUN_LOG = os.path.join(STATE_DIR, "auto_runs.jsonl")

# ------------------------- App bootstrap -------------------------
load_dotenv()
st.set_page_config(page_title="Three-Agent Trader", page_icon="🤖", layout="wide")

# ------------------------- Sidebar: keys & settings -------------------------
st.sidebar.header("🔐 API Keys")
#
# Instead of pre-populating API key inputs with potentially stale values from the
# environment, we start with empty strings (password fields) and fall back to
# environment variables or config values only after the user has had an
# opportunity to supply new keys.  This prevents an expired or incorrect
# `GEMINI_API_KEY` from being displayed or inadvertently used by default.

# Alpaca keys: leave blank to use values from .env or environment
alpaca_key_input    = st.sidebar.text_input(
    "Alpaca API Key ID",
    value="",
    type="password",
)
alpaca_secret_input = st.sidebar.text_input(
    "Alpaca Secret Key",
    value="",
    type="password",
)
alpaca_base   = st.sidebar.text_input(
    "Alpaca Paper Base URL",
    value=settings.alpaca_base_url,
)

# Optional API keys: leave blank to use values from .env or environment
finnhub_key_input   = st.sidebar.text_input(
    "Finnhub API Key",
    value="",
    type="password",
)
gemini_key_input    = st.sidebar.text_input(
    "Gemini API Key",
    value="",
    type="password",
)

st.sidebar.header("⚙️ Strategy Settings")
mean_conf = st.sidebar.slider(
    "Min confidence to act",
    0.0,
    1.0,
    settings.mean_confidence_to_act,
    0.05,
)

# ---------------------------------------------------------------------------
# After collecting user inputs, determine the actual keys to use.  If the user
# leaves a field blank, we fall back to the corresponding environment
# variable or the value defined in `config.py`.  This allows the application
# to honour values from `.env` without showing them in the UI and avoids
# pre-populating stale keys.  We then update the runtime environment and
# `settings` so that subsequent calls within this session use the resolved
# keys consistently.

import os as _os  # local import to avoid clobbering the top-level os

# Resolve Alpaca keys
alpaca_key = (
    alpaca_key_input
    or _os.getenv("ALPACA_API_KEY_ID")
    or settings.alpaca_key
    or ""
)
alpaca_secret = (
    alpaca_secret_input
    or _os.getenv("ALPACA_API_SECRET_KEY")
    or settings.alpaca_secret
    or ""
)

# Resolve optional Finnhub key
finnhub_key = (
    finnhub_key_input
    or _os.getenv("FINNHUB_API_KEY")
    or settings.finnhub_key
    or ""
)

# Resolve Gemini key
gemini_key = (
    gemini_key_input
    or _os.getenv("GEMINI_API_KEY")
    or settings.gemini_key
    or ""
)

# Persist the resolved keys into the environment for this session.  This
# ensures that downstream modules (e.g., `core.llm.LCTraderLLM`) which rely
# on `os.getenv()` see the correct values.  Use `.env` fallback only if
# user did not override them in the sidebar.
_os.environ["ALPACA_API_KEY_ID"] = alpaca_key
_os.environ["ALPACA_API_SECRET_KEY"] = alpaca_secret
_os.environ["ALPACA_PAPER_BASE_URL"] = alpaca_base
_os.environ["FINNHUB_API_KEY"] = finnhub_key
_os.environ["GEMINI_API_KEY"] = gemini_key

# Update the global settings object to reflect the new keys.  This is
# important because other parts of the application use `settings.*` to
# initialise clients.  Without this update, those modules may continue
# referencing stale values.
settings.alpaca_key = alpaca_key
settings.alpaca_secret = alpaca_secret
settings.alpaca_base_url = alpaca_base
settings.finnhub_key = finnhub_key
settings.gemini_key = gemini_key
settings.mean_confidence_to_act = mean_conf

st.sidebar.header("📈 Watchlist")
default_watch = os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,NVDA,AMZN,GOOG,META,TSLA")
default_crypto = os.getenv("WATCHLIST_CRYPTO", "BTC/USD,ETH/USD,SOL/USD")
watchlist_stocks = st.sidebar.text_input("Stocks (comma-separated)", value=default_watch)
watchlist_crypto = st.sidebar.text_input("Crypto (comma-separated)", value=default_crypto)

# ---- Persist keys/settings across reruns ----
if "keys" not in st.session_state:
    st.session_state["keys"] = {}
st.session_state["keys"].update({
    "alpaca_key": alpaca_key,
    "alpaca_secret": alpaca_secret,
    "alpaca_base": alpaca_base,
    "finnhub_key": finnhub_key,
    "gemini_key": gemini_key,
    "mean_conf": mean_conf,
})

# ✅ Initialize top-level session_state keys used later
st.session_state["stocks_watch"] = watchlist_stocks
st.session_state["crypto_watch"] = watchlist_crypto

# Session state for last analyses per tab (now stores horizon-aware decision objects)
for key in ("analysis_stocks", "analysis_crypto"):
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------- Header -------------------------
st.title("🤖 Three-Agent Alpaca Trader")
st.caption("Short-term (30m), Swing (daily), Macro (weekly + news). Paper trading via Alpaca.")

tab_dash, tab_stocks, tab_crypto, tab_suggest, tab_news, tab_auto = st.tabs(
    ["Dashboard", "Stocks", "Crypto", "Suggestions", "News", "Automation"]
)

# ------------------------- Shared helpers -------------------------
def _ensure_keys(require_finnhub: bool = False):
    if not (alpaca_key and alpaca_secret and gemini_key):
        st.error("Please provide Alpaca and Gemini API keys in the sidebar.")
        st.stop()
    if require_finnhub and not finnhub_key:
        st.error("Finnhub API key is required for this feature.")
        st.stop()

def _agent_pack():
    dm = DataManager()
    sm = SemanticMemory()
    fh = FinnhubClient(api_key=finnhub_key) if finnhub_key else None
    llm = LCTraderLLM(api_key=gemini_key)  # your current constructor
    debate = Debate()
    return dm, sm, fh, llm, debate

def _read_last_runs(n: int = 5) -> List[Dict]:
    if not os.path.exists(RUN_LOG):
        return []
    try:
        with open(RUN_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = [pd.json.loads(l) if hasattr(pd, "json") else json.loads(l) for l in lines]
    except Exception:
        entries = []
        with open(RUN_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(eval_json(line))
                except Exception:
                    continue
    try:
        with open(RUN_LOG, "r", encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
    except Exception:
        entries = []
    return list(reversed(entries[-n:]))

def eval_json(s: str) -> Dict:
    import json as _json
    return _json.loads(s)

def _fmt_when(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return iso_ts or ""

# =================================================================
#                           DASHBOARD
# =================================================================
with tab_dash:
    st.subheader("Portfolio Overview")
    _ensure_keys(require_finnhub=False)
    broker = AlpacaTrader(alpaca_key, alpaca_secret, alpaca_base)

    try:
        acct = broker.get_account()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", f"${acct['equity']:,.2f}")
        c2.metric("Cash", f"${acct['cash']:,.2f}")
        c3.metric("Buying Power", f"${acct['buying_power']:,.2f}")
        c4.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
        st.caption(f"Account status: **{acct.get('status', 'unknown')}** • Multiplier: {acct.get('multiplier', '—')}")
    except Exception as e:
        st.warning(f"Could not load account metrics: {e}")

    st.markdown("### Open Positions (Stocks & Crypto)")
    try:
        pos = broker.list_positions()
        if pos:
            df = pd.DataFrame(pos).rename(columns={
                "symbol": "Symbol", "asset_class": "Asset Class", "qty": "Qty",
                "avg_entry_price": "Avg Entry", "current_price": "Last",
                "market_value": "Mkt Value", "cost_basis": "Cost Basis",
                "unrealized_pl": "Unrealized P/L", "unrealized_plpc": "Unrealized P/L %",
                "exchange": "Exchange",
            })
            df["Unrealized P/L %"] = (df["Unrealized P/L %"] * 100.0).round(2)
            st.dataframe(df, use_container_width=True, height=380)
        else:
            st.info("No open positions.")
    except Exception as e:
        st.warning(f"Could not load positions: {e}")

# =================================================================
#                           STOCKS TAB
# =================================================================
with tab_stocks:
    st.subheader("Stocks")
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker_stk = st.text_input("Stock ticker (e.g., AAPL, MSFT, NVDA)", key="stk_ticker").upper().strip()
    with col2:
        analyze_stk = st.button("Analyze (Stocks)", use_container_width=True, key="stk_analyze_btn")

    if analyze_stk and ticker_stk:
        _ensure_keys(require_finnhub=False)
        dm, sm, fh, llm, debate = _agent_pack()
        with st.spinner("Fetching price history…"):
            snapshot = dm.layered_snapshot(ticker_stk)
            if snapshot["mid_term"].empty:
                st.error("Could not fetch data for this ticker. Check the symbol or try later.")
                st.stop()

        # --- Debug: what the model sees
        with st.expander("🔎 Debug — last 10 rows per horizon (NaN check)"):
            for label in ("short_term", "mid_term", "long_term"):
                st.markdown(f"**{label}**")
                df_dbg = snapshot.get(label)
                if df_dbg is None or df_dbg.empty:
                    st.write("(empty)")
                else:
                    st.dataframe(df_dbg.tail(10), use_container_width=True, height=220)
                    needed = ["close","rsi","macd","macd_signal","upper_band","lower_band"]
                    present = [c for c in needed if c in df_dbg.columns]
                    if present:
                        st.caption(f"NaNs present in {label}: {df_dbg.tail(10)[present].isna().any().any()}")

        with st.spinner("Fetching news & building semantic memory…"):
            try:
                if fh:
                    sm.add((fh.company_news(ticker_stk, days=45) or [])[:30])
            except Exception as e:
                st.warning(f"Finnhub news unavailable: {e}")

        # agents + horizon-aware debate (🔁 CHANGED: use horizon_decide instead of run)
        short = ShortTermAgent("ShortTerm", llm, {})
        mid   = MidTermAgent("MidTerm", llm, {})
        long  = LongTermAgent("LongTerm", llm, {}, sm)

        votes = []
        d, c, raw = short.vote(snapshot); votes.append({"agent": "ShortTerm", "decision": d, "confidence": c, "raw": raw})
        d, c, raw = mid.vote(snapshot);   votes.append({"agent": "MidTerm",  "decision": d, "confidence": c, "raw": raw})
        d, c, raw = long.vote(snapshot);  votes.append({"agent": "LongTerm", "decision": d, "confidence": c, "raw": raw})

        decision_obj = Debate(enter_th=mean_conf, exit_th=0.45).horizon_decide(votes)
        # decision_obj = {"action","target_horizon","confidence","scores":{short,mid,long}}

        st.session_state["analysis_stocks"] = {
            "ticker": ticker_stk,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "votes": votes,
            "decision": decision_obj,
        }

    analysis = st.session_state["analysis_stocks"]
    if analysis:
        st.markdown(f"**Agent Votes — {analysis['ticker']}**  \n_{analysis['timestamp']}_")
        for v in analysis["votes"]:
            st.markdown(f"• **{v['agent']}** → **{v['decision']}** (conf={v['confidence']:.2f})")
            with st.expander("LLM rationale / raw"):
                st.write(v["raw"])

        dec = analysis["decision"]
        horizon_text = dec.get("target_horizon") or "—"
        st.markdown(
            f"### Final: **{dec.get('action','HOLD')}** "
            f"(confidence **{float(dec.get('confidence',0.0)):.2f}**) "
            f"• Target horizon: **{horizon_text}**"
        )
        if isinstance(dec.get("scores"), dict):
            scores = dec["scores"]
            st.caption(f"Per-horizon scores: short={scores.get('short',0):.2f}, mid={scores.get('mid',0):.2f}, long={scores.get('long',0):.2f}")

        st.divider()
        st.subheader("Place Paper Trade (Stocks)")
        # Default side follows horizon-aware action, but you can override
        default_side = dec.get("action", "HOLD")
        if default_side not in ("BUY", "SELL"):
            default_side = "BUY"
        side = st.radio("Order side", ("BUY", "SELL"), index=0 if default_side == "BUY" else 1, horizontal=True, key="stk_side_radio")
        if side != dec.get("action"):
            st.caption("⚠️ You are overriding the debate decision.")
        qty = st.number_input("Quantity (shares)", min_value=1, value=1, step=1, key="stk_qty_input")
        confirm = st.checkbox(f"I confirm a MARKET {side} for {analysis['ticker']} x {qty}.", key="stk_confirm_checkbox")
        if st.button("Place Order (Stocks)", disabled=not confirm, key="stk_place_btn"):
            try:
                broker = AlpacaTrader(alpaca_key, alpaca_secret, alpaca_base)
                last_px = broker.last_price(analysis["ticker"])
                oid = broker.market_buy(analysis["ticker"], int(qty)) if side == "BUY" else broker.market_sell(analysis["ticker"], int(qty))
                px_msg = f" at ~${last_px:.2f}" if last_px is not None else ""
                st.success(f"✅ Order placed: {side} {analysis['ticker']} x {qty}{px_msg}  \n**Order ID:** `{oid}`")
            except Exception as e:
                st.error(f"Order failed: {e}")

# =================================================================
#                           CRYPTO TAB
# =================================================================
with tab_crypto:
    st.subheader("Crypto")
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker_c = st.text_input("Crypto pair (e.g., BTC/USD, ETH/USD)", key="c_ticker").upper().strip()
    with col2:
        analyze_c = st.button("Analyze (Crypto)", use_container_width=True, key="c_analyze_btn")

    if analyze_c and ticker_c:
        _ensure_keys(require_finnhub=False)
        dm, sm, fh, llm, debate = _agent_pack()
        with st.spinner("Fetching crypto price history…"):
            snapshot = dm.layered_snapshot_crypto(ticker_c)
            if snapshot["mid_term"].empty:
                st.error("Could not fetch data for this pair. Try BTC/USD or ETH/USD.")
                st.stop()

        # --- Debug: what the model sees
        with st.expander("🔎 Debug — last 10 rows per horizon (NaN check)"):
            for label in ("short_term", "mid_term", "long_term"):
                st.markdown(f"**{label}**")
                df_dbg = snapshot.get(label)
                if df_dbg is None or df_dbg.empty:
                    st.write("(empty)")
                else:
                    st.dataframe(df_dbg.tail(10), use_container_width=True, height=220)
                    needed = ["close","rsi","macd","macd_signal","upper_band","lower_band"]
                    present = [c for c in needed if c in df_dbg.columns]
                    if present:
                        st.caption(f"NaNs present in {label}: {df_dbg.tail(10)[present].isna().any().any()}")

        with st.spinner("Fetching crypto news & building semantic memory…"):
            try:
                if fh:
                    sm.add((fh.crypto_news(max_items=50) or [])[:30])
            except Exception as e:
                st.warning(f"Crypto news unavailable: {e}")

        # agents + horizon-aware debate (🔁 CHANGED: use horizon_decide instead of run)
        short = ShortTermAgent("ShortTerm", llm, {})
        mid   = MidTermAgent("MidTerm", llm, {})
        long  = LongTermAgent("LongTerm", llm, {}, sm)

        votes = []
        d, c, raw = short.vote(snapshot); votes.append({"agent": "ShortTerm", "decision": d, "confidence": c, "raw": raw})
        d, c, raw = mid.vote(snapshot);   votes.append({"agent": "MidTerm",  "decision": d, "confidence": c, "raw": raw})
        d, c, raw = long.vote(snapshot);  votes.append({"agent": "LongTerm", "decision": d, "confidence": c, "raw": raw})

        decision_obj = Debate(enter_th=mean_conf, exit_th=0.45).horizon_decide(votes)
        display_ticker = snapshot["short_term"]["ticker"].iloc[-1] if not snapshot["short_term"].empty else ticker_c
        st.session_state["analysis_crypto"] = {
            "ticker": display_ticker,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "votes": votes,
            "decision": decision_obj,
        }

    analysis_c = st.session_state["analysis_crypto"]
    if analysis_c:
        st.markdown(f"**Agent Votes — {analysis_c['ticker']}**  \n_{analysis_c['timestamp']}_")
        for v in analysis_c["votes"]:
            st.markdown(f"• **{v['agent']}** → **{v['decision']}** (conf={v['confidence']:.2f})")
            with st.expander("LLM rationale / raw"):
                st.write(v["raw"])

        dec = analysis_c["decision"]
        horizon_text = dec.get("target_horizon") or "—"
        st.markdown(
            f"### Final: **{dec.get('action','HOLD')}** "
            f"(confidence **{float(dec.get('confidence',0.0)):.2f}**) "
            f"• Target horizon: **{horizon_text}**"
        )
        if isinstance(dec.get("scores"), dict):
            scores = dec["scores"]
            st.caption(f"Per-horizon scores: short={scores.get('short',0):.2f}, mid={scores.get('mid',0):.2f}, long={scores.get('long',0):.2f}")

        st.divider()
        st.subheader("Place Paper Trade (Crypto)")
        default_side_c = dec.get("action", "HOLD")
        if default_side_c not in ("BUY", "SELL"):
            default_side_c = "BUY"
        side_c = st.radio("Order side", ("BUY", "SELL"), index=0 if default_side_c == "BUY" else 1, horizontal=True, key="c_side_radio")
        if side_c != dec.get("action"):
            st.caption("⚠️ You are overriding the debate decision.")
        qty_c = st.number_input("Quantity (crypto units)", min_value=0.0001, value=0.001, step=0.0001, format="%.6f", key="c_qty_input")
        confirm_c = st.checkbox(f"I confirm a MARKET {side_c} for {analysis_c['ticker']} x {qty_c}.", key="c_confirm_checkbox")
        if st.button("Place Order (Crypto)", disabled=not confirm_c, key="c_place_btn"):
            try:
                broker = AlpacaTrader(alpaca_key, alpaca_secret, alpaca_base)
                last_px = broker.last_price(analysis_c["ticker"])
                oid = broker.market_buy(analysis_c["ticker"], float(qty_c)) if side_c == "BUY" else broker.market_sell(analysis_c["ticker"], float(qty_c))
                px_msg = f" at ~${last_px:.2f}" if last_px is not None else ""
                st.success(f"✅ Order placed: {side_c} {analysis_c['ticker']} x {qty_c}{px_msg}  \n**Order ID:** `{oid}`")
            except Exception as e:
                st.error(f"Order failed: {e}")

# =================================================================
#                         NEWS TAB (Finnhub)
# =================================================================
with tab_news:
    st.subheader("Latest News (Finnhub)")
    if not finnhub_key:
        st.warning("Enter Finnhub API key in the sidebar.")
    else:
        fh = FinnhubClient(api_key=finnhub_key)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**General Market News**")
            try:
                items = fh.general_news_struct(max_items=25)
                if items:
                    df = pd.DataFrame(items)
                    if "datetime" in df.columns and pd.notna(df["datetime"]).any():
                        df["time"] = pd.to_datetime(df["datetime"], unit="s")
                    show = df[["time", "headline", "source", "url"]].rename(columns={"headline":"Headline","source":"Source","url":"URL","time":"Time"}) if "time" in df.columns else df[["headline","source","url"]]
                    st.dataframe(show, use_container_width=True, height=360)
                else:
                    st.info("No general news available.")
            except Exception as e:
                st.warning(f"General news error: {e}")
        with c2:
            st.markdown("**Crypto News**")
            try:
                items = fh.crypto_news_struct(max_items=25)
                if items:
                    df = pd.DataFrame(items)
                    if "datetime" in df.columns and pd.notna(df["datetime"]).any():
                        df["time"] = pd.to_datetime(df["datetime"], unit="s")
                    show = df[["time", "headline", "source", "url"]].rename(columns={"headline":"Headline","source":"Source","url":"URL","time":"Time"}) if "time" in df.columns else df[["headline","source","url"]]
                    st.dataframe(show, use_container_width=True, height=360)
                else:
                    st.info("No crypto news available.")
            except Exception as e:
                st.warning(f"Crypto news error: {e}")
        st.markdown("---")
        st.markdown("**Company / Ticker News**")
        sym = st.text_input("Ticker (e.g., AAPL, MSFT, NVDA)", value="AAPL").upper().strip()
        days = st.slider("Lookback (days)", 1, 60, 14)
        if st.button("Fetch Company News"):
            try:
                items = fh.company_news_struct(sym, days=days, max_items=50)
                if items:
                    df = pd.DataFrame(items)
                    if "datetime" in df.columns and pd.notna(df["datetime"]).any():
                        df["time"] = pd.to_datetime(df["datetime"], unit="s")
                    show = df[["time", "symbol", "headline", "source", "url"]].rename(
                        columns={"symbol":"Symbol","headline":"Headline","source":"Source","url":"URL","time":"Time"}
                    ) if "time" in df.columns else df[["symbol","headline","source","url"]]
                    st.dataframe(show, use_container_width=True, height=380)
                else:
                    st.info("No company news found for that period.")
            except Exception as e:
                st.warning(f"Company news error: {e}")

# =================================================================
#                        AUTOMATION TAB (NEW)
# =================================================================
with tab_auto:
    render_automation_tab()
