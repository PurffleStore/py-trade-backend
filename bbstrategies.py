from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime

# Bollinger Band Calculation
def calculate_bollinger(data, period=20, stddev=2):
    close = data['close']
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=stddev, nbdevdn=stddev, matype=0)
    return upper, middle, lower


#BB Squeeze  breakout/fade after low volatility
def detect_bb_squeeze(close, upper, lower, middle, lookback=20, perc=20):
    bandwidth = (upper - lower) / middle
    # 20th percentile over the lookback window
    thresh = np.percentile(bandwidth.iloc[-lookback:], perc)
    if bandwidth.iloc[-1] < thresh:
        return "Neutral"
    # otherwise fall back to a breakout rule
    if close.iloc[-1] > upper.iloc[-1]:
        return "Bullish"
    elif close.iloc[-1] < lower.iloc[-1]:
        return "Bearish"
    return "Neutral"


# BB Breakout Detection
def detect_bb_breakout(close, upper, lower):
    if close.iloc[-1] > upper.iloc[-1]:
        return "Bullish"
    elif close.iloc[-1] < lower.iloc[-1]:
        return "Bearish"
    return "Neutral"


# BB Breakout Reversal
def detect_bb_breakout_reversal(data, upper, lower, middle, lookahead=3):
    
    i = len(data) - lookahead - 1  

    if i < 0:
        return "Neutral"

    row = data.iloc[i]
    # Bullish Reversal
    if row['close'] > upper.iloc[i]:
        for j in range(1, lookahead + 1):
            next_row = data.iloc[i + j]
            if next_row['close'] < upper.iloc[i + j] and next_row['close'] > middle.iloc[i + j]:
                return "Bullish"

    # Bearish Reversal
    elif row['close'] < lower.iloc[i]:
        for j in range(1, lookahead + 1):
            next_row = data.iloc[i + j]
            if next_row['close'] > lower.iloc[i + j] and next_row['close'] < middle.iloc[i + j]:
                return "Bearish"

    return "Neutral"



# Middle Band Pullback
def detect_middle_band_pullback(close, middle, upper, lower, threshold=0.10, trend_lookback=3):
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if abs(close.iloc[-1] - middle.iloc[-1]) < band_width * threshold:
        trend_above = all(close.iloc[-i] > middle.iloc[-i] for i in range(2, 2 + trend_lookback))
        trend_below = all(close.iloc[-i] < middle.iloc[-i] for i in range(2, 2 + trend_lookback))
        if trend_above:
            return "Bullish"
        elif trend_below:
            return "Bearish"
    return "Neutral"



# Master strategy function
def bollinger_strategies(data):
    
    upper, middle, lower = calculate_bollinger(data)

    signals = {
        "UpperBand": round(upper.iloc[-1], 2),
        "MiddleBand": round(middle.iloc[-1], 2),
        "LowerBand": round(lower.iloc[-1], 2),
        "BB Squeeze": detect_bb_squeeze(data['close'], upper, lower, middle),
        "BB Breakout": detect_bb_breakout(data['close'], upper, lower),
        "BB Breakout Reversal": detect_bb_breakout_reversal(data, upper, lower, middle),
        "Middle Band Pullback": detect_middle_band_pullback(data['close'], middle, upper, lower)       
        
    }

    weights = {
        "BB Squeeze": 30,
        "BB Breakout": 25,
        "BB Breakout Reversal": 25,
        "Middle Band Pullback": 20        
    }

    total_score = 0
    for strategy, weight in weights.items():
        signal = signals[strategy]
        if "Bullish" in signal or "Breakout Up" in signal or "Squeeze" in signal or "Pullback" in signal:
            total_score += weight
        elif "Neutral" in signal or "No Breakout" in signal:
            total_score += weight * 0.5

    overall_percentage = round((total_score / sum(weights.values())) * 100, 2)

    if overall_percentage >= 60:
        final_signal = "Buy"
    elif overall_percentage <= 40:
        final_signal = "DBuy"
    else:
        final_signal = "Neutral"

    return signals, overall_percentage, final_signal

# API-style function
def get_bollinger_trade_signal(data):
    bb_signals, overall_score, final_signal = bollinger_strategies(data)
    return {
        "bollinger_signals": bb_signals,
        "bollinger_score": overall_score,
        "bollinger_final_signal": final_signal
    }
