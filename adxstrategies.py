from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime

# Calculate ADX, +DI, and -DI values
def calculate_adx(data, period=14):
    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']

    adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
    plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
    minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)

    return adx, plus_di, minus_di

# Detect ADX crossover (directional indicators +DI and -DI)
def adx_di_crossover_strategy(plus_di, minus_di, adx, threshold=20, lookback_days=5):
    # We will loop over the last `lookback_days` to check for crossovers
    for i in range(-lookback_days, 0):
        # Check if ADX is above the threshold
        if adx[i] > threshold:
            # Bullish condition: +DI crosses above -DI and ADX is above threshold
            if plus_di[i] > minus_di[i] and plus_di[i - 1] <= minus_di[i - 1]:
                return "Bullish"
            # Bearish condition: -DI crosses above +DI and ADX is above threshold
            elif minus_di[i] > plus_di[i] and minus_di[i - 1] <= plus_di[i - 1]:
                return "Bearish"
    
    return "Neutral"

#ADX Breakout strategy
def adx_breakout_strategy(data, adx, threshold=25):
   
    # Detect breakout condition (ADX above 25, price breaking resistance/support)
    if adx[-1] > threshold:
        if data['close'][-1] > data['high'][-2]:  # Bullish breakout
            return "Bullish"
        elif data['close'][-1] < data['low'][-2]:  # Bearish breakout
            return "Bearish"
    
    return "Neutral"

# ADX Slope Strategy
def get_adx_slope_signal(adx, days=5, threshold=0.1):    
    total_slope = 0

    # Calculate slope for each of the last `days`
    for i in range(-days, -1):
        slope = adx[i + 1] - adx[i]
        total_slope += slope

    # Average slope
    avg_slope = total_slope / (days - 1)

    if avg_slope > threshold:
        return "Bullish"
    elif avg_slope < -threshold:
        return "Bearish"
    else:
        return "Neutral"

# ADX Divergence Strategy
def adx_divergence_strategy(data, adx, threshold=25):
    """
    Detects divergence between price and ADX.
    A divergence occurs when price makes a new high/low, but ADX does not follow the same direction.
    """
    price_high = data['high']
    price_low = data['low']

    # Checking for divergence
    price_divergence_bullish = price_high[-1] > price_high[-2] and adx[-1] < adx[-2]
    price_divergence_bearish = price_low[-1] < price_low[-2] and adx[-1] > adx[-2]

    if price_divergence_bullish:
        return "Bullish"
    elif price_divergence_bearish:
        return "Bearish"
    return "Neutral"


# Main ADX strategy function
def adx_strategies(data):
   
    # Calculate ADX, +DI, and -DI
    adx, plus_di, minus_di = calculate_adx(data)

    signals = {
        "ADX": round(adx.iloc[-1], 2),        
        "ADX + DI Crossover": adx_di_crossover_strategy(plus_di, minus_di, adx),
        "ADX Breakout": adx_breakout_strategy(data, adx),       
        "ADX Slope": get_adx_slope_signal(adx[-5:]),
        "ADX Divergence": adx_divergence_strategy(data, adx)
    }

    weights = {
        "ADX + DI Crossover": 35,
        "ADX Breakout": 30,     
        "ADX Slope": 20,
        "ADX Divergence": 15        
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

    return signals, overall_percentage, final_signal,adx, plus_di, minus_di


def extract_series(data, series, days=100):
    series = pd.Series(series).dropna().tail(days)
    series.index = data.index[-len(series):]
    series.index = series.index.strftime('%Y-%m-%d')
    return series.round(2).to_dict()

# API-style function
def get_adx_trade_signal(data):
    adx_signals, overallscore, final_signal,adx, plus_di, minus_di = adx_strategies(data)
    return {
        "adx_signals": adx_signals,
        "adx_score": overallscore,
        "adx_final_signal": final_signal,
        "ADX_Indicator": extract_series(data, adx),
        "PLUS_DI": extract_series(data, plus_di),
        "MINUS_DI": extract_series(data, minus_di) 
    }
