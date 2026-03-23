from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime



# Calculate EMA values
def calculate_ema(data, short_period=5, medium_period=20, long_period=50):
    close_prices = data['close']
    ema_short = talib.EMA(close_prices, timeperiod=short_period)
    ema_medium = talib.EMA(close_prices, timeperiod=medium_period)
    ema_long = talib.EMA(close_prices, timeperiod=long_period)
    return ema_short,ema_medium, ema_long

def detect_ema_crossover(ema20, ema50):    
    
    for i in range(len(ema20) - 1):
        older_ema20 = ema20[i]
        newer_ema20 = ema20[i + 1]
        older_ema50 = ema50[i]
        newer_ema50 = ema50[i + 1]
        
        # Bullish crossover - EMA 20 above EMA 50
        if older_ema20 <= older_ema50 and newer_ema20 > newer_ema50:
            return "Bullish"
        # Bearish crossover EMA 20 below EMA 50
        elif older_ema20 >= older_ema50 and newer_ema20 < newer_ema50:
            return "Bearish"
    
    return "Neutral"


def detect_ema_price_crossover(ema20, price, days=5):
    bullish_days = 0
    bearish_days = 0

    # Check the last N days
    for i in range(-days, 0):
        if price[i] > ema20[i]:
            bullish_days += 1
        elif price[i] < ema20[i]:
            bearish_days += 1

    # Final decision
    if bullish_days == days:
        return "Bullish"
    elif bearish_days == days:
        return "Bearish"
    else:
        return "Neutral"


def get_ema_average_slope_signal(ema_series, days=5, threshold=0.1):    

    total_slope = 0

    # Calculate slope for each of the last `days`
    for i in range(-days, -1):
        slope = ema_series[i + 1] - ema_series[i]
        total_slope += slope

    # Average slope
    avg_slope = total_slope / (days - 1)

    if avg_slope > threshold:
        return "Bullish"
    elif avg_slope < -threshold:
        return "Bearish"
    else:
        return "Neutral"

def triple_ema_strategy(ema_short, ema_medium, ema_long):
    
    if ema_short > ema_medium and ema_short > ema_long:
        return "Bullish"
    
    # Bearish condition: Short-term EMA crosses below medium and long-term EMAs
    elif ema_short < ema_medium and ema_short < ema_long:
        return "Bearish"
    
    # Neutral condition: EMAs are not aligned
    else:
        return "Neutral"


# Main strategy function using EMA crossover
def ema_strategies(data):
   
    ema5, ema_20, ema_50 = calculate_ema(data, short_period=5, medium_period=20, long_period=50)
    
    signals = {
        "EMA 20": round(ema_20.iloc[-1], 2),
        "EMA 50": round(ema_50.iloc[-1], 2),
        "EMA Crossover": detect_ema_crossover(ema_20[-5:], ema_50[-5:]),
        "EMA Price Crossover": detect_ema_price_crossover(ema_20[-5:], data['close'][-5:]),
        "EMA Slope": get_ema_average_slope_signal(ema_20[-5:]),
        "Triple EMA": triple_ema_strategy(ema5.iloc[-1], ema_20.iloc[-1], ema_50.iloc[-1])
    }

    weights = {
        "EMA Crossover": 30,
        "EMA Price Crossover": 25,
        "EMA Slope": 20,
        "Triple EMA": 25
    }

    total_score = 0
    for strategy, weight in weights.items():
        signal = signals[strategy]
        if signal == "Bullish":
            total_score += weight
        elif signal == "Neutral":
            total_score += weight * 0.5

    overall_percentage = round((total_score / sum(weights.values())) * 100, 2)

    if overall_percentage >= 60:
        final_signal = "Buy"
    elif overall_percentage <= 40:
        final_signal = "DBuy"
    else:
        final_signal = "Neutral"

    return signals, overall_percentage, final_signal,ema5, ema_20, ema_50

# API-style function
def get_ema_trade_signal(data):
    ema_signals, overallscore, final_signal,ema5, ema_20, ema_50 = ema_strategies(data)

    ema5_series = pd.Series(ema5, index=data.index).dropna().tail(100)
    ema5_series.index = ema5_series.index.strftime('%Y-%m-%d')
    ema_20_series = pd.Series(ema_20, index=data.index).dropna().tail(100)
    ema_20_series.index = ema_20_series.index.strftime('%Y-%m-%d')
    ema_50_series = pd.Series(ema_50, index=data.index).dropna().tail(100)
    ema_50_series.index = ema_50_series.index.strftime('%Y-%m-%d')
    return {
        "ema_signals": ema_signals,
        "ema_score": overallscore,
        "ema_final_signal": final_signal,
        "EMA_5": ema5_series.round(2).to_dict(),
        "EMA_20": ema_20_series.round(2).to_dict(),
        "EMA_50": ema_50_series.round(2).to_dict()

    }
