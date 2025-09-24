# app.py
# Streamlit UI with: Dashboard, Stocks, Crypto, Suggestions, News, Automation (NEW)

import os
from datetime import datetime, timezone
from typing import List, Dict
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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
RUN_LOG = os.path.join(STATE_DIR, "auto_runs.jsonl")  # produced by autonomous_runner

# ------------------------- App bootstrap -------------------------
load_dotenv()
st.set_page_config(page_title="Three-Agent Trader", page_icon="🤖", layout="wide")

# ------------------------- Sidebar: keys & settings -------------------------
st.sidebar.header("🔐 API Keys")
alpaca_key    = st.sidebar.text_input("Alpaca API Key ID", value=settings.alpaca_key or "", type="password")
alpaca_secret = st.sidebar.text_input("Alpaca Secret Key", value=settings.alpaca_secret or "", type="password")
alpaca_base   = st.sidebar.text_input("Alpaca Paper Base URL", value=settings.alpaca_base_url)

finnhub_key   = st.sidebar.text_input("Finnhub API Key", value=settings.finnhub_key or "", type="password")
gemini_key    = st.sidebar.text_input("Gemini API Key", value=settings.gemini_key or "", type="password")

st.sidebar.header("⚙️ Strategy Settings")
mean_conf = st.sidebar.slider("Min confidence to act", 0.0, 1.0, settings.mean_confidence_to_act, 0.05)

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

# Session state for last analyses per tab
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
    llm = LCTraderLLM(gemini_key=gemini_key)
    debate = Debate(mean_conf_to_act=mean_conf)
    return dm, sm, fh, llm, debate

def _run_agents_on_snapshot(snapshot, sm, llm, debate):
    short = ShortTermAgent("ShortTerm", llm, {})
    mid   = MidTermAgent("MidTerm", llm, {})
    long  = LongTermAgent("LongTerm", llm, {}, sm)

    votes = []
    d, c, raw = short.vote(snapshot); votes.append({"agent": "ShortTerm", "decision": d, "confidence": c, "raw": raw})
    d, c, raw = mid.vote(snapshot);   votes.append({"agent": "MidTerm",  "decision": d, "confidence": c, "raw": raw})
    d, c, raw = long.vote(snapshot);  votes.append({"agent": "LongTerm", "decision": d, "confidence": c, "raw": raw})

    final_decision, final_conf = debate.run(votes)
    return votes, final_decision, final_conf

def _read_last_runs(n: int = 5) -> List[Dict]:
    """Read last N entries from the JSONL log written by the autonomous runner."""
    if not os.path.exists(RUN_LOG):
        return []
    try:
        with open(RUN_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = [pd.json.loads(l) if hasattr(pd, "json") else json.loads(l) for l in lines]  # fallback below
    except Exception:
        # simple parser
        entries = []
        with open(RUN_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(eval_json(line))
                except Exception:
                    continue
    # Fast path using std lib json
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
        with st.spinner("Fetching news & building semantic memory…"):
            try:
                if fh:
                    sm.add((fh.company_news(ticker_stk, days=45) or [])[:30])
            except Exception as e:
                st.warning(f"Finnhub news unavailable: {e}")
        votes, final_decision, final_conf = _run_agents_on_snapshot(snapshot, sm, llm, debate)
        st.session_state["analysis_stocks"] = {
            "ticker": ticker_stk,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "votes": votes,
            "final_decision": final_decision,
            "final_confidence": final_conf,
        }

    analysis = st.session_state["analysis_stocks"]
    if analysis:
        st.markdown(f"**Agent Votes — {analysis['ticker']}**  \n_{analysis['timestamp']}_")
        for v in analysis["votes"]:
            st.markdown(f"• **{v['agent']}** → **{v['decision']}** (conf={v['confidence']:.2f})")
            with st.expander("LLM rationale / raw"):
                st.write(v["raw"])
        st.markdown(f"### Final: **{analysis['final_decision']}** (confidence **{analysis['final_confidence']:.2f}**)")
        st.divider()
        st.subheader("Place Paper Trade (Stocks)")
        default_side = analysis["final_decision"] if analysis["final_decision"] in ("BUY", "SELL") else "BUY"
        side = st.radio("Order side", ("BUY", "SELL"), index=0 if default_side == "BUY" else 1, horizontal=True, key="stk_side_radio")
        if side != analysis["final_decision"]:
            st.caption("⚠️ You are overriding the agents' final decision.")
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
        with st.spinner("Fetching crypto news & building semantic memory…"):
            try:
                if fh:
                    sm.add((fh.crypto_news(max_items=50) or [])[:30])
            except Exception as e:
                st.warning(f"Crypto news unavailable: {e}")
        votes, final_decision, final_conf = _run_agents_on_snapshot(snapshot, sm, llm, debate)
        display_ticker = snapshot["short_term"]["ticker"].iloc[-1] if not snapshot["short_term"].empty else ticker_c
        st.session_state["analysis_crypto"] = {
            "ticker": display_ticker,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "votes": votes,
            "final_decision": final_decision,
            "final_confidence": final_conf,
        }

    analysis_c = st.session_state["analysis_crypto"]
    if analysis_c:
        st.markdown(f"**Agent Votes — {analysis_c['ticker']}**  \n_{analysis_c['timestamp']}_")
        for v in analysis_c["votes"]:
            st.markdown(f"• **{v['agent']}** → **{v['decision']}** (conf={v['confidence']:.2f})")
        with st.expander("LLM rationale / raw"):
            for v in analysis_c["votes"]:
                st.markdown(f"**{v['agent']}**")
                st.write(v["raw"])
        st.markdown(f"### Final: **{analysis_c['final_decision']}** (confidence **{analysis_c['final_confidence']:.2f}**)")
        st.divider()
        st.subheader("Place Paper Trade (Crypto)")
        default_side_c = analysis_c["final_decision"] if analysis_c["final_decision"] in ("BUY", "SELL") else "BUY"
        side_c = st.radio("Order side", ("BUY", "SELL"), index=0 if default_side_c == "BUY" else 1, horizontal=True, key="c_side_radio")
        if side_c != analysis_c["final_decision"]:
            st.caption("⚠️ You are overriding the agents' final decision.")
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
#                     SUGGESTIONS TAB (separate agent)
# =================================================================
with tab_suggest:
    st.subheader("Suggestions (News-Sentiment Agent)")
    if not (alpaca_key and alpaca_secret and gemini_key and finnhub_key):
        st.warning("Please provide Alpaca, Gemini and Finnhub API keys to generate suggestions.")
    else:
        broker = AlpacaTrader(alpaca_key, alpaca_secret, alpaca_base)
        fh = FinnhubClient(api_key=finnhub_key)
        llm = LCTraderLLM(gemini_key=gemini_key)

        wl_stocks = [s.strip().upper() for s in st.session_state["stocks_watch"].split(",") if s.strip()]
        try:
            held_syms = set([p["symbol"] for p in broker.list_positions()])
        except Exception:
            held_syms = set()

        max_syms = st.slider("Max stocks to scan", 1, 30, 10)
        min_conf = st.slider("Min confidence to recommend (BUY)", 0.0, 1.0, st.session_state["keys"]["mean_conf"], 0.05)

        if st.button("Generate Suggestions", type="primary"):
            agent = SuggestionsAgent(llm=llm, finnhub_client=fh, min_conf=min_conf)
            rows, details_map = [], {}
            with st.spinner("Analyzing news sentiment and headlines…"):
                for sym in wl_stocks[:max_syms]:
                    if sym in held_syms:
                        continue
                    try:
                        res = agent.analyze_symbol(sym)
                        if res["recommendation"] == "BUY" and res["confidence"] >= min_conf:
                            rows.append({
                                "Symbol": res["symbol"],
                                "Recommendation": res["recommendation"],
                                "Confidence": round(res["confidence"], 3),
                                "NewsScore": round(res["companyNewsScore"], 3) if res["companyNewsScore"] is not None else None,
                                "Bullish%": round(res["bullishPercent"] * 100.0, 2) if res["bullishPercent"] is not None else None,
                                "Bearish%": round(res["bearishPercent"] * 100.0, 2) if res["bearishPercent"] is not None else None,
                                "Buzz": round(res["buzz"], 3) if res["buzz"] is not None else None,
                                "Articles(7d)": res["articlesInLastWeek"],
                            })
                            details_map[res["symbol"]] = res
                    except Exception:
                        continue

            if rows:
                st.success(f"Found {len(rows)} BUY suggestions.")
                df = pd.DataFrame(rows).sort_values(["Confidence", "NewsScore"], ascending=False)
                st.dataframe(df, use_container_width=True, height=360)
                sel = st.selectbox("Show details for:", options=["(choose)"] + [r["Symbol"] for r in rows])
                if sel and sel != "(choose)":
                    info = details_map.get(sel, {})
                    st.markdown(f"#### {sel} — Why it’s suggested")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("companyNewsScore", f"{info.get('companyNewsScore', 0):.3f}")
                    c2.metric("Bullish %", f"{(info.get('bullishPercent', 0)*100):.2f}%")
                    c3.metric("Bearish %", f"{(info.get('bearishPercent', 0)*100):.2f}%")
                    c1, c2 = st.columns(2)
                    c1.metric("Buzz", f"{info.get('buzz', 0):.3f}")
                    c2.metric("Articles (last week)", f"{info.get('articlesInLastWeek', 0)}")
                    with st.expander("Recent headlines"):
                        for h in (info.get("headlines") or []):
                            headline = h.get("headline") or ""
                            url = h.get("url") or ""
                            source = h.get("source") or ""
                            ts = h.get("datetime")
                            tstr = pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M") if ts else ""
                            st.markdown(f"- [{headline}]({url})  \n  _{source} • {tstr}_")
                    with st.expander("LLM raw reasoning"):
                        st.write(info.get("raw") or "(none)")
            else:
                st.info("No strong BUY ideas right now. Try again later or widen your watchlist.")

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
    st.subheader("Last 5 Automation Runs")
    st.caption("Shows the most recent activity from the autonomous scheduler (stocks & crypto).")
    if st.button("Refresh"):
        st.rerun()

    if not os.path.exists(RUN_LOG):
        st.info("No automation log yet. Start your scheduler (run_scheduler.py) to generate entries.")
    else:
        # Read last 5 entries (most recent first)
        try:
            with open(RUN_LOG, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
        except Exception:
            lines = []
            with open(RUN_LOG, "r", encoding="utf-8") as f:
                for l in f:
                    try:
                        lines.append(eval_json(l))
                    except Exception:
                        pass

        last = list(reversed(lines[-5:])) if lines else []
        if not last:
            st.info("Log exists but contains no entries yet.")
        else:
            # Summary table
            rows = []
            for e in last:
                final = (e.get("final") or {})
                rows.append({
                    "When": _fmt_when(e.get("_ts", "")),
                    "Kind": e.get("kind", ""),
                    "Symbol": e.get("symbol", ""),
                    "Status": e.get("status", ""),
                    "Decision": final.get("decision", ""),
                    "Conf": round(final.get("confidence", 0), 3) if isinstance(final.get("confidence", 0), (int, float)) else final.get("confidence", ""),
                    "Reason": e.get("reason", ""),
                    "Order ID": e.get("order_id", ""),
                    "Qty": e.get("qty", ""),
                    "Bar Time": e.get("bar_ts", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=300)

            # Details expanders
            st.markdown("### Details")
            for i, e in enumerate(last, 1):
                with st.expander(f"{i}. {e.get('kind','').upper()} {e.get('symbol','')} — {e.get('status','').upper()} @ { _fmt_when(e.get('_ts','')) }"):
                    st.json({
                        "status": e.get("status"),
                        "symbol": e.get("symbol"),
                        "kind": e.get("kind"),
                        "bar_ts": e.get("bar_ts"),
                        "final": e.get("final"),
                        "reason": e.get("reason"),
                        "order_id": e.get("order_id"),
                        "qty": e.get("qty"),
                        "votes": e.get("votes"),
                    })

# Footer tip
st.caption("Pro tip: Update your Watchlist in the sidebar to drive suggestions and news.")
