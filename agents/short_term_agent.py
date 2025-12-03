from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

SYSTEM_MSG = (
    "You are a SHORT-TERM momentum trader. You analyze recent 30-minute bars "
    "using RSI, MACD (and its signal), and Bollinger Bands. "
    "Return a STRICT JSON object ONLY (no prose) with keys: "
    '{"vote":"BUY|SELL|HOLD","confidence":0..1,"rationale":"one or two short lines"}.'
)

USER_TMPL = (
    "Ticker: {ticker}\n"
    "Task: Decide BUY/SELL/HOLD with confidence in [0,1] based on the last 10 rows.\n"
    "Short-term data (tail=10):\n{table}\n"
    "Return ONLY JSON as specified."
)

REQ_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

class ShortTermAgent(BaseAgent):
    # Minimum number of rows required for the short‑term agent to operate.  The
    # previous value of 60 excluded many symbols with shorter histories (e.g.
    # recently listed cryptos).  Reducing this to 30 enables the agent to
    # function with less historical context while still maintaining enough
    # bars for indicator calculations.
    MIN_ROWS = 30
    TAIL_N = 10

    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        df = snapshot.get('short_term')
        if df is None or df.empty:
            return "HOLD", 0.5, "(no short-term data)"
        if len(df) < self.MIN_ROWS:
            return "HOLD", 0.5, f"(short-term rows < {self.MIN_ROWS})"

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            return "HOLD", 0.5, f"(missing cols: {missing})"

        tail = df.tail(self.TAIL_N)[REQ_COLS]
        if tail.isna().any().any():
            return "HOLD", 0.5, "(NaNs in short-term tail window)"

        ticker = df['ticker'].iloc[-1]
        table = tail.to_string(index=False)

        decision, conf, raw = self.llm.vote_structured(
            system_msg=SYSTEM_MSG,
            user_template=USER_TMPL,
            variables={"ticker": ticker, "table": table},
        )

        # ✅ ensure conf is propagated correctly
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        # ---------------------------------------------------------------
        # Fallback heuristic: If the LLM cannot provide a decisive signal
        # (manifested as a HOLD decision with <=0.5 confidence), compute a
        # basic momentum signal using RSI and MACD.  This ensures the agent
        # can still produce actionable BUY/SELL recommendations when the
        # generative model is unavailable or declines to take a stance.  The
        # heuristic looks at the last 10 bars: trending up/down, RSI
        # overbought/oversold, and MACD crossing its signal line.  A
        # moderately strong trend with RSI extreme or MACD crossover will
        # trigger a BUY or SELL; otherwise it retains HOLD with lower
        # confidence.
        if decision.upper() == "HOLD" and conf <= 0.5:
            try:
                # Use the same tail used for the LLM (TAIL_N rows, REQ_COLS)
                tail_df = df.tail(self.TAIL_N)[REQ_COLS].copy()
                # compute simple trend: difference between last and first close
                last_close = float(tail_df['close'].iloc[-1])
                first_close = float(tail_df['close'].iloc[0])
                trend = last_close - first_close
                # Latest indicator values
                rsi_last = float(tail_df['rsi'].iloc[-1])
                macd_last = float(tail_df['macd'].iloc[-1])
                macd_signal_last = float(tail_df['macd_signal'].iloc[-1])
                # Determine heuristics
                buy_signal = False
                sell_signal = False
                # Oversold RSI or positive MACD crossover with upward trend
                if (rsi_last < 35) or (macd_last > macd_signal_last and trend > 0):
                    buy_signal = True
                # Overbought RSI or negative MACD crossover with downward trend
                if (rsi_last > 65) or (macd_last < macd_signal_last and trend < 0):
                    sell_signal = True
                # Assign decision & confidence
                if buy_signal and not sell_signal:
                    decision = "BUY"
                    # confidence scales with RSI distance from neutral (50) and magnitude of trend
                    conf = min(1.0, 0.5 + abs(50 - rsi_last) / 50.0 + abs(trend / last_close))
                elif sell_signal and not buy_signal:
                    decision = "SELL"
                    conf = min(1.0, 0.5 + abs(rsi_last - 50) / 50.0 + abs(trend / last_close))
                else:
                    # ambiguous: retain HOLD but assign modest confidence
                    decision = "HOLD"
                    conf = 0.4
                raw = f"{raw} [fallback_heuristic_shortterm]"
            except Exception:
                # if any error occurs during heuristics, leave decision unchanged
                pass

        return decision, conf, raw
