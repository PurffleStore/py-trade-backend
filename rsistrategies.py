from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from collections import OrderedDict
import datetime

# --- Strategy Functions ---

def get_overbought_oversold_signal(recent):
    if (recent['RSI_14'] < 30).any():
        return "Bullish"
    elif (recent['RSI_14'] > 70).any():
        return "Bearish"
    else:
        return "Neutral"

def get_rsi_crossover_signal(rsi5, rsi14):    
    
    for i in range(len(rsi5) - 1):
        older_rsi5 = rsi5[i]
        newer_rsi5 = rsi5[i + 1]
        older_rsi14 = rsi14[i]
        newer_rsi14 = rsi14[i + 1]
        
        # Bullish crossover (MACD crosses above Signal)
        if older_rsi5 <= older_rsi14 and newer_rsi5 > newer_rsi14:
            return "Bullish"
        # Bearish crossover (MACD crosses below Signal)
        elif older_rsi5 >= older_rsi14 and newer_rsi5 < newer_rsi14:
            return "Bearish"
    
    return "Neutral"



def get_mean_reversion_signal(df):
    rsi = df['RSI_5']
    if len(rsi) < 6:
        return "Neutral"

    # Check for crossover below 20 in last 5 entries
    buy_signal = ((rsi < 20) & (rsi.shift(1) >= 20)).tail(5).any()
    sell_signal = ((rsi > 80) & (rsi.shift(1) <= 80)).tail(5).any()

    if buy_signal:
        return "Bullish"
    elif sell_signal:
        return "Bearish"
    else:
        return "Neutral"


def get_bollinger_rsi_signal(recent):
    buy = ((recent['close'].to_numpy().flatten() < recent['Lower_BB']) & (recent['RSI_14'] < 30)).any()
    sell = ((recent['close'].to_numpy().flatten() > recent['Upper_BB']) & (recent['RSI_14'] > 70)).any()
    if buy:
        return "Bullish"
    elif sell:
        return "Bearish"
    else:
        return "Neutral"


def get_rsi_with_ma_signal(recent):
    buy = ((recent['close'].to_numpy().flatten() > recent['MA_20']) & (recent['RSI_14'] > 50)).any()
    sell = ((recent['close'].to_numpy().flatten() < recent['MA_20']) & (recent['RSI_14'] < 50)).any()
    if buy:
        return "Bullish"
    elif sell:
        return "Bearish"
    else:
        return "Neutral"

def get_rsi_50_trend_signal(recent):
    if (recent['RSI_14'] > 50).all():
        return "Bullish"
    elif (recent['RSI_14'] < 50).all():
        return "Bearish"
    else:
        return "Neutral"


def get_swing_rejection_signal(rsi14):
    
    r1, r2, r3, r4, r5, r6 = rsi14

    if (
        r1 < 30 and
        r2 > r1 and
        r3 < r2 and r3 > r1 and
        r4 > r3 and
        (r5 > r2 or r6 > r2) and
        r6 > 30
    ):
        return "Bullish"

    elif (
        r1 > 70 and
        r2 < r1 and
        r3 > r2 and r3 < r1 and
        r4 < r3 and
        (r5 < r2 or r6 < r2) and
        r6 < 70
    ):
        return "Bearish"

    return "Neutral"


def is_pivot_low(prices, idx, left=5, right=5):
    """Check if current point is a pivot low"""
    if idx < left or idx + right >= len(prices):
        return False
    return all(prices[idx] < prices[idx - i] and prices[idx] < prices[idx + i] for i in range(1, left + 1))

def is_pivot_high(prices, idx, left=5, right=5):
    """Check if current point is a pivot high"""
    if idx < left or idx + right >= len(prices):
        return False
    return all(prices[idx] > prices[idx - i] and prices[idx] > prices[idx + i] for i in range(1, left + 1))

def get_rsi_divergence_signal(df):
    df = df.dropna().reset_index(drop=True)
    prices = df['close'].values
    rsi = df['RSI_14'].values

    left = 5
    right = 5
    max_range = 20

    recent_idx = len(prices) - 1  # latest candle
    start_idx = max(recent_idx - max_range, left)

    for i in range(recent_idx - 1, start_idx - 1, -1):
        if is_pivot_low(prices, i, left, right) and is_pivot_low(rsi, i, left, right):
            # Regular Bullish Divergence
            if prices[recent_idx] < prices[i] and rsi[recent_idx] > rsi[i]:
                return "Bullish"

        if is_pivot_high(prices, i, left, right) and is_pivot_high(rsi, i, left, right):
            # Regular Bearish Divergence
            if prices[recent_idx] > prices[i] and rsi[recent_idx] < rsi[i]:
                return "Bearish"

    return "Neutral"


# --- Master RSI Strategy Function ---

def rsi_strategies(df):

    close_prices = df['close']
   
    # Calculate all indicators
    df['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
    df['RSI_5'] = talib.RSI(close_prices, timeperiod=5)
    df['MA_20'] = talib.SMA(close_prices, timeperiod=20)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(close_prices, timeperiod=20)

    
    # Ensure all calculations are added to df before slicing
    recent = df.tail(5)
    

    signals = OrderedDict([
    ("RSI 14", round(df[['RSI_14']].iloc[-1][0], 2)),
    ("Overbought/Oversold", get_overbought_oversold_signal(recent)),
    ("RSI Swing Rejection", get_swing_rejection_signal(df['RSI_14'].tail(6))),
    ("RSI Divergence", get_rsi_divergence_signal(df)),
    ("RSI_Bollinger Band", get_bollinger_rsi_signal(recent)),
    ("RSI 5/14 Crossover", get_rsi_crossover_signal(df['RSI_5'].tail(5),df['RSI_14'].tail(5))),
    ("RSI Trend 50 Confirmation", get_rsi_50_trend_signal(recent)),
    ("RSI_MA", get_rsi_with_ma_signal(recent)),
    ("Mean Reversion", get_mean_reversion_signal(df[['RSI_5']].tail(6)))
        ])


    # Weightage for each signal
    rsi_signal_weights = {
        "Overbought/Oversold": 15,
        "RSI Swing Rejection": 15,
        "RSI Divergence": 15,
        "RSI_Bollinger Band": 15,
        "RSI 5/14 Crossover": 10,                 
        "RSI Trend 50 Confirmation": 10,
        "RSI_MA": 10,
        "Mean Reversion": 10    
        
    }
    
    # Calculate weighted score
    total_score = 0
    for strategy, weight in rsi_signal_weights.items():
        signal = signals[strategy]
        if signal == "Bullish":
            total_score += weight
        elif signal == "Neutral":
            total_score += weight * 0.5
        # Bearish gives 0 score

    overall_percentage = round((total_score / sum(rsi_signal_weights.values())) * 100, 2)

    

    # Final output signal
    if overall_percentage >= 60:
        final_signal = "Buy"
    elif overall_percentage <= 40:
        final_signal = "DBuy"
    else:
        final_signal = "Neutral"
    

    return signals, overall_percentage, final_signal

def extract_series(data, column_name, days=100):
    series = data[[column_name]].dropna().tail(days)
    series.index = series.index.strftime('%Y-%m-%d')
    return series[column_name].round(2).to_dict()

def get_rsi_trade_signal(data):    

    rsi_signals, overallscore, final_signal = rsi_strategies(data)

    return {     
        
        "rsi_signals": rsi_signals,
        "rsi_score": overallscore,
        "rsi_final_signal": final_signal,
        "rsi_14_last_2_years": extract_series(data, 'RSI_14'),
        "rsi_5_last_2_years": extract_series(data, 'RSI_5'),
        "ma": extract_series(data, 'MA_20'),
        "close": extract_series(data, 'close'),
        "open": extract_series(data, 'open'),
        "high": extract_series(data, 'high'),
        "low": extract_series(data, 'low'),
        "lowerbb": extract_series(data, 'Lower_BB'),
        "upperbb": extract_series(data, 'Upper_BB')
    }


