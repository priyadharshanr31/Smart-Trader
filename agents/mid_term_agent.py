from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a MID-TERM (swing) trend analyst. You consider DAILY bars with "
    "5/20-day moving averages, RSI, MACD (and its signal), and Bollinger Bands. "
    "Return a STRICT JSON object ONLY (no prose) with keys: "
    '{"vote":"BUY|SELL|HOLD","confidence":0..1,"rationale":"one or two short lines"}.'
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence in [0,1] using the daily context below.\n\n"
    "Daily data (last 20 rows):\n{table}\n\n"
    "5DMA (tail=5):\n{ma5}\n\n"
    "20DMA (tail=5):\n{ma20}\n"
    "Return ONLY JSON as specified."
)

REQ_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

class MidTermAgent(BaseAgent):
    # Minimum number of rows for mid‑term agent.  Lowered from 120 to 80 to
    # accommodate assets with shorter daily histories while still giving
    # adequate data for moving average calculations.
    MIN_ROWS = 80
    TAIL_N = 20

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('mid_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "(no mid-term data)"
        if len(df) < self.MIN_ROWS:
            return "HOLD", 0.5, f"(mid-term rows < {self.MIN_ROWS})"

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            return "HOLD", 0.5, f"(missing cols: {missing})"

        tail = df.tail(self.TAIL_N)[REQ_COLS]
        if tail.isna().any().any():
            return "HOLD", 0.5, "(NaNs in mid-term tail window)"

        ticker = df['ticker'].iloc[-1]
        table = tail.to_string(index=False)
        ma5 = df['close'].rolling(window=5).mean().tail(5).to_string(index=False)
        ma20 = df['close'].rolling(window=20).mean().tail(5).to_string(index=False)

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table, "ma5": ma5, "ma20": ma20},
        )

        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        # ---------------------------------------------------------------
        # Fallback heuristic for mid‑term: If the LLM returns HOLD with
        # confidence ≤0.5, derive a basic trend signal using moving
        # averages.  A simple 5‑day vs 20‑day moving average crossover is
        # computed on the full mid‑term dataframe.  If the short MA is
        # above the long MA and the slope is positive, we BUY; if the
        # short MA is below the long MA and the slope is negative, we
        # SELL; otherwise we HOLD with lower confidence.  Confidence
        # scales with the relative distance between the MAs.
        if decision.upper() == "HOLD" and conf <= 0.5:
            try:
                close = df['close']
                ma5_series = close.rolling(window=5).mean()
                ma20_series = close.rolling(window=20).mean()
                ma5_last = float(ma5_series.iloc[-1])
                ma20_last = float(ma20_series.iloc[-1])
                # compute simple slope over last 5 periods
                ma5_prev = float(ma5_series.iloc[-5]) if len(ma5_series) >= 5 else ma5_last
                ma20_prev = float(ma20_series.iloc[-5]) if len(ma20_series) >= 5 else ma20_last
                slope5 = ma5_last - ma5_prev
                slope20 = ma20_last - ma20_prev
                buy_signal = False
                sell_signal = False
                if (ma5_last > ma20_last) and (slope5 > 0 or slope20 > 0):
                    buy_signal = True
                if (ma5_last < ma20_last) and (slope5 < 0 or slope20 < 0):
                    sell_signal = True
                if buy_signal and not sell_signal:
                    decision = "BUY"
                    distance = abs(ma5_last - ma20_last) / (ma20_last + 1e-9)
                    conf = min(1.0, 0.5 + distance * 2.0)
                elif sell_signal and not buy_signal:
                    decision = "SELL"
                    distance = abs(ma5_last - ma20_last) / (ma20_last + 1e-9)
                    conf = min(1.0, 0.5 + distance * 2.0)
                else:
                    decision = "HOLD"
                    conf = 0.4
                raw = f"{raw} [fallback_heuristic_midterm]"
            except Exception:
                pass

        return decision, conf, raw
