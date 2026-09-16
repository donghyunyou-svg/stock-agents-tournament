"""기술적 지표 계산 유틸리티."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def momentum_return(series: pd.Series, window: int) -> float:
    if len(series) <= window:
        return 0.0
    return series.iloc[-1] / series.iloc[-1 - window] - 1


def volatility(series: pd.Series, window: int = 20) -> float:
    rets = series.pct_change().dropna()
    if len(rets) < window:
        return float(rets.std()) if len(rets) > 1 else 0.0
    return float(rets.iloc[-window:].std())
