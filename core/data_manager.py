# core/data_manager.py
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from core.indicators import enrich_indicators
from config import settings

_REQUIRED_INDICATOR_COLS = ["close", "rsi", "macd", "macd_signal", "upper_band", "lower_band"]

def _drop_indicator_nans(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Only drop on columns that actually exist, always include 'close'
    cols = ["close"] + [c for c in _REQUIRED_INDICATOR_COLS if c in df.columns]
    cols = list(dict.fromkeys(cols))  # unique, keep order
    return df.dropna(subset=[c for c in cols if c in df.columns])

class DataManager:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or settings.data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    # ----------------------- helpers -----------------------
    def _cache_path(self, symbol: str, kind: str) -> str:
        return os.path.join(self.data_dir, f"{symbol.upper().replace('/', '_')}_{kind}.parquet")

    def _is_stale(self, path: str, max_age_minutes: int) -> bool:
        if not os.path.exists(path):
            return True
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) > timedelta(minutes=max_age_minutes)

    def _reset_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        rename_map = {}
        for cand in ["Date", "Datetime", "date", "datetime", df.columns[0]]:
            if cand in df.columns:
                rename_map[cand] = "time"
                break
        df = df.rename(columns=rename_map)
        return df

    def _ensure_ohlcv(self, df: pd.DataFrame, symbol_for_ticker: str) -> pd.DataFrame:
        df.columns = [c.lower().strip() for c in df.columns]
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns and c.title() in df.columns:
                df[c] = df[c.title()]
        required = ["open", "high", "low", "close", "volume", "time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Downloaded data missing columns: {missing}. Have: {list(df.columns)}")
        df["ticker"] = symbol_for_ticker.upper()
        return df[["time", "open", "high", "low", "close", "volume", "ticker"]]

    def _download(self, y_symbol: str, interval: str, period: str, display_ticker: str) -> pd.DataFrame:
        raw = yf.download(y_symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if raw.empty:
            return raw
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw.columns = [str(c) for c in raw.columns]
        df = self._reset_time_column(raw)
        df = self._ensure_ohlcv(df, display_ticker)
        return df

    def _read_parquet_normalized(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if "time" not in df.columns:
            for cand in ["Date", "Datetime", "date", "datetime"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "time"})
                    break
        df.columns = [c.lower().strip() for c in df.columns]
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]
        needed = {"time", "open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            return pd.DataFrame()
        return df

    # ======================= STOCKS (30m / 1d / 1wk) =======================
    def get_intraday_short(self, symbol: str) -> pd.DataFrame:
        kind = f"{settings.short_interval}"
        path = self._cache_path(symbol, kind)
        if self._is_stale(path, max_age_minutes=15):
            df = self._download(symbol, interval=settings.short_interval, period=settings.short_period, display_ticker=symbol)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(symbol, interval=settings.short_interval, period=settings.short_period, display_ticker=symbol)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.short_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def get_daily_mid(self, symbol: str) -> pd.DataFrame:
        path = self._cache_path(symbol, '1d')
        if self._is_stale(path, max_age_minutes=1440):
            df = self._download(symbol, interval='1d', period='5y', display_ticker=symbol)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(symbol, interval='1d', period='5y', display_ticker=symbol)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.mid_daily_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def get_weekly_long(self, symbol: str) -> pd.DataFrame:
        path = self._cache_path(symbol, '1wk')
        if self._is_stale(path, max_age_minutes=1440):
            df = self._download(symbol, interval='1wk', period='10y', display_ticker=symbol)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(symbol, interval='1wk', period='10y', display_ticker=symbol)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.long_weekly_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def layered_snapshot(self, symbol: str) -> dict:
        return {
            'short_term': self.get_intraday_short(symbol),
            'mid_term'  : self.get_daily_mid(symbol),
            'long_term' : self.get_weekly_long(symbol),
        }

    # ======================= CRYPTO (30m / 1d / 1wk) =======================
    @staticmethod
    def _map_crypto_symbol(user_symbol: str) -> tuple[str, str]:
        norm = user_symbol.upper().replace(" ", "")
        if "/" in norm:
            base, quote = norm.split("/", 1)
        elif "-" in norm:
            base, quote = norm.split("-", 1)
        else:
            if norm.endswith("USDT"):
                base, quote = norm[:-4], "USDT"
            elif norm.endswith("USD"):
                base, quote = norm[:-3], "USD"
            else:
                base, quote = norm, "USD"
        y_quote = "USD"
        y_symbol = f"{base}-{y_quote}"
        display_symbol = f"{base}/{quote}"
        return y_symbol, display_symbol

    def get_intraday_short_crypto(self, user_symbol: str) -> pd.DataFrame:
        y_symbol, display = self._map_crypto_symbol(user_symbol)
        kind = f"CRYPTO_{settings.short_interval}"
        path = self._cache_path(display, kind)
        if self._is_stale(path, max_age_minutes=15):
            df = self._download(y_symbol, interval=settings.short_interval, period=settings.short_period, display_ticker=display)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(y_symbol, interval=settings.short_interval, period=settings.short_period, display_ticker=display)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.short_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def get_daily_mid_crypto(self, user_symbol: str) -> pd.DataFrame:
        y_symbol, display = self._map_crypto_symbol(user_symbol)
        path = self._cache_path(display, 'CRYPTO_1d')
        if self._is_stale(path, max_age_minutes=1440):
            df = self._download(y_symbol, interval='1d', period='5y', display_ticker=display)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(y_symbol, interval='1d', period='5y', display_ticker=display)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.mid_daily_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def get_weekly_long_crypto(self, user_symbol: str) -> pd.DataFrame:
        y_symbol, display = self._map_crypto_symbol(user_symbol)
        path = self._cache_path(display, 'CRYPTO_1wk')
        if self._is_stale(path, max_age_minutes=1440):
            df = self._download(y_symbol, interval='1wk', period='10y', display_ticker=display)
            if not df.empty:
                df.to_parquet(path, index=False)
        else:
            df = self._read_parquet_normalized(path)
            if df.empty:
                df = self._download(y_symbol, interval='1wk', period='10y', display_ticker=display)
                if not df.empty:
                    df.to_parquet(path, index=False)
        df = df.tail(settings.long_weekly_lookback) if not df.empty else df
        df = enrich_indicators(df)
        return _drop_indicator_nans(df)

    def layered_snapshot_crypto(self, user_symbol: str) -> dict:
        return {
            'short_term': self.get_intraday_short_crypto(user_symbol),
            'mid_term'  : self.get_daily_mid_crypto(user_symbol),
            'long_term' : self.get_weekly_long_crypto(user_symbol),
        }
