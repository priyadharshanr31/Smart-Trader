import os, math, requests, pandas as pd
from datetime import datetime
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
SEC = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
if not (KEY and SEC):
    raise SystemExit("Missing keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (or use a .env).")

HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}
PAPER = "https://paper-api.alpaca.markets"
LIVE  = "https://api.alpaca.markets"

def pick_base():
    for base in (PAPER, LIVE):
        r = requests.get(f"{base}/v2/account", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return base
    # if we get here, show the last response for debugging
    raise SystemExit(f"Auth failed. Last response {r.status_code}: {r.text}")

def fetch_equity_history(base, period="1M", timeframe="1D", extended_hours=False):
    r = requests.get(
        f"{base}/v2/account/portfolio/history",
        params={"period": period, "timeframe": timeframe,
                "extended_hours": str(extended_hours).lower()},
        headers=HEADERS, timeout=30
    )
    if r.status_code != 200:
        raise SystemExit(f"portfolio/history {r.status_code}: {r.text}")
    d = r.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(d["timestamp"], unit="s"),
        "equity": pd.to_numeric(d["equity"])
    }).dropna().sort_values("date")
    return df

def sharpe_from_equity(df, rf_annual=0.06, periods_per_year=252):
    ret = df["equity"].pct_change().dropna()
    rf_period = (1 + rf_annual)**(1/periods_per_year) - 1
    excess = ret - rf_period
    sharpe = excess.mean() / ret.std(ddof=1) * math.sqrt(periods_per_year)
    return sharpe, {
        "n": len(ret),
        "mean_daily": ret.mean(),
        "vol_daily": ret.std(ddof=1),
        "mean_ann": ret.mean()*periods_per_year,
        "vol_ann": ret.std(ddof=1)*math.sqrt(periods_per_year),
        "rf_annual": rf_annual
    }

if __name__ == "__main__":
    base = pick_base()  # auto paper/live
    df = fetch_equity_history(base, period="1M", timeframe="1D")
    df.to_csv("alpaca_equity_history.csv", index=False)
    sharpe, s = sharpe_from_equity(df, rf_annual=0.06)
    print(f"Base: {base}")
    print(f"Window: {df['date'].min().date()} → {df['date'].max().date()} (n={s['n']})")
    print(f"Mean daily: {s['mean_daily']:.4%} | Vol daily: {s['vol_daily']:.4%}")
    print(f"Mean ann:  {s['mean_ann']:.2%} | Vol ann:  {s['vol_ann']:.2%}")
    print(f"Sharpe (annualized): {sharpe:.2f} | Rf used: {s['rf_annual']:.2%}")
    print("Saved: alpaca_equity_history.csv")
