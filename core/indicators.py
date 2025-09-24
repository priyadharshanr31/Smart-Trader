import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Columns present: {list(df.columns)}")

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    _require_cols(df, ["close"])
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    _require_cols(df, ["close"])
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    _require_cols(df, ["close"])
    ma = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # normalize common alt names before computing
    cols_lower = [c.lower().strip() for c in df.columns]
    df = df.copy()
    df.columns = cols_lower
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']
    if 'time' not in df.columns and df.index.name:
        df = df.reset_index().rename(columns={df.index.name: 'time'})

    _require_cols(df, ["close"])  # fail fast with clear message

    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df)
    return df
