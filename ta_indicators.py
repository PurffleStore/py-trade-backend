"""ta_indicators.py

Compute common technical indicators and return per-indicator signals.
Indicators included: RSI, MACD, ATR, EMA crossover, ADX, Fibonacci levels (Fibo), Bollinger Bands (BB), Support/Resistance (SR) placeholder, PA_MS (price action / multi-scale) placeholder.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from pandas import DataFrame


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return pd.DataFrame({'macd': macd_line, 'signal': sig, 'hist': hist})


def atr(df: DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def adx(df: DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
    adx_series.index = df.index
    return adx_series.fillna(0)


def bollinger_bands(series: pd.Series, window: int = 20, n_std: int = 2) -> DataFrame:
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * n_std)
    lower = sma - (std * n_std)
    return pd.DataFrame({'upper': upper, 'middle': sma, 'lower': lower})


def fibo_levels(series: pd.Series, lookback: int = 30) -> Dict[str, float]:
    recent = series[-lookback:]
    high = recent.max()
    low = recent.min()
    diff = high - low if high > low else 0.0
    levels = {
        '0.0': float(low),
        '0.236': float(high - 0.236 * diff),
        '0.382': float(high - 0.382 * diff),
        '0.5': float(high - 0.5 * diff),
        '0.618': float(high - 0.618 * diff),
        '1.0': float(high)
    }
    return levels


def sr_levels(df: DataFrame, lookback: int = 60) -> Dict[str, float]:
    # placeholder: use rolling highs/lows as simple support/resistance
    highs = df['High'][-lookback:]
    lows = df['Low'][-lookback:]
    return {
        'resistance': float(highs.max()),
        'support': float(lows.min())
    }


def pa_ms_signal(df: DataFrame) -> str:
    # placeholder for multi-scale price action; return 'buy' if recent close higher than few-day SMA
    close = df['Close']
    sma5 = close.rolling(5).mean()
    if len(sma5.dropna()) < 2:
        return 'hold'
    if close.iloc[-1] > sma5.iloc[-1]:
        return 'buy'
    if close.iloc[-1] < sma5.iloc[-1]:
        return 'sell'
    return 'hold'


# simple per-indicator signals

def rsi_signal(series: pd.Series) -> str:
    val = float(series.iloc[-1])
    if val > 70:
        return 'sell'
    if val < 30:
        return 'buy'
    return 'hold'


def macd_signal(macd_df: DataFrame) -> str:
    if len(macd_df) < 2:
        return 'hold'
    if macd_df['hist'].iloc[-1] > 0 and macd_df['hist'].iloc[-2] <= 0:
        return 'buy'
    if macd_df['hist'].iloc[-1] < 0 and macd_df['hist'].iloc[-2] >= 0:
        return 'sell'
    return 'hold'


def ema_signal(series: pd.Series) -> str:
    # use 12/26 cross
    e12 = ema(series, 12)
    e26 = ema(series, 26)
    if e12.iloc[-1] > e26.iloc[-1] and e12.iloc[-2] <= e26.iloc[-2]:
        return 'buy'
    if e12.iloc[-1] < e26.iloc[-1] and e12.iloc[-2] >= e26.iloc[-2]:
        return 'sell'
    return 'hold'


def atr_signal(atr_series: pd.Series) -> str:
    # ATR is non-directional; consider high volatility as hold
    v = float(atr_series.iloc[-1])
    if v > atr_series.mean() * 1.6:
        return 'hold'
    return 'hold'


def adx_signal(adx_series: pd.Series) -> str:
    if float(adx_series.iloc[-1]) > 25:
        return 'buy'
    return 'hold'


def bb_signal(series: pd.Series) -> str:
    bb = bollinger_bands(series)
    last = series.iloc[-1]
    if last > bb['upper'].iloc[-1]:
        return 'sell'
    if last < bb['lower'].iloc[-1]:
        return 'buy'
    return 'hold'


def sr_signal(df: DataFrame) -> str:
    s = sr_levels(df)
    last = float(df['Close'].iloc[-1])
    if last > s['resistance'] * 0.995:
        return 'sell'
    if last < s['support'] * 1.005:
        return 'buy'
    return 'hold'


def aggregate_signals(signals: Dict[str, str]) -> str:
    counts = {'buy': 0, 'sell': 0, 'hold': 0}
    for v in signals.values():
        counts[v] = counts.get(v, 0) + 1
    if counts['buy'] > counts['sell'] and counts['buy'] >= counts['hold']:
        return 'buy'
    if counts['sell'] > counts['buy'] and counts['sell'] >= counts['hold']:
        return 'sell'
    return 'hold'


def compute_all(df: DataFrame) -> Dict[str, str]:
    """Compute all indicator signals for the given OHLCV DataFrame and return a dict mapping key->signal."""
    res = {}
    close = df['Close']
    res['RSI'] = rsi_signal(rsi(close))
    macd_df = macd(close)
    res['MACD'] = macd_signal(macd_df)
    res['ATR'] = atr_signal(atr(df))
    res['EMA'] = ema_signal(close)
    res['ADX'] = adx_signal(adx(df))
    res['Fibo'] = 'hold'  # fibo_levels could be used to produce signals
    res['BB'] = bb_signal(close)
    res['SR'] = sr_signal(df)
    res['PA_MS'] = pa_ms_signal(df)

    return res
