from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime


# Detect recent trend (uptrend or downtrend)
def detect_trend(data, lookback_days=7):
    recent_data = data.iloc[-lookback_days:]
    closes = recent_data['close'].values

    if closes[-1] > closes[0]:
        return "Uptrend"
    elif closes[-1] < closes[0]:
        return "Downtrend"
    else:
        return "Sideways"

# Fibonacci Retracement Pullback Strategy
def fibonacci_retracement_bounce(data, fib_levels=[0.382, 0.5, 0.618], tolerance=0.005, lookback_days=7):
    last_close = data['close'].iloc[-1]

    # Detect recent trend
    trend = detect_trend(data, lookback_days)

    # Use recent high/low
    recent_data = data.iloc[-lookback_days:]
    swing_high = recent_data['high'].max()
    swing_low = recent_data['low'].min()

    retracement_levels = {level: swing_high - (swing_high - swing_low) * level for level in fib_levels}

    nearest_level = None
    min_deviation = float('inf')

    for level, price_level in retracement_levels.items():
        deviation = abs(last_close - price_level) / price_level
        if deviation < min_deviation:
            min_deviation = deviation
            nearest_level = (level, price_level)

    if nearest_level:
        _, level_price = nearest_level
        if min_deviation <= tolerance:
            if trend == "Uptrend" and last_close > level_price:
                return "Bullish"
            elif trend == "Downtrend" and last_close < level_price:
                return "Bearish"

    return "Neutral"

# 2. Fibonacci Breakout (Retracement Break) Strategy

def fibonacci_breakout(data, fib_threshold=0.618, lookback_days=7, check_candles=3, tolerance=0.005):
    recent_data = data.iloc[-lookback_days:]
    last_data = data.iloc[-check_candles:]

    swing_high = recent_data['high'].max()
    swing_low = recent_data['low'].min()

    fib_level_price = swing_high - (swing_high - swing_low) * fib_threshold

    # Check last few candles
    crossed_above = 0
    crossed_below = 0

    for i in range(len(last_data)):
        close_price = last_data['close'].iloc[i]
        if close_price > fib_level_price * (1 + tolerance):
            crossed_above += 1
        elif close_price < fib_level_price * (1 - tolerance):
            crossed_below += 1

    # Decision
    if crossed_above == check_candles:
        return "Bullish"
    elif crossed_below == check_candles:
        return "Bearish"
    else:
        return "Neutral"


# 3. Golden Pocket Reversal Strategy

def calculate_fib_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '65%': high - 0.65 * diff,
        '78.6%': high - 0.786 * diff
    }

def golden_pocket_reversal_strategy(data, lookback_period=7):
    """
    Golden Pocket Reversal Strategy (61.8% - 65%)
    
    Parameters:
    - data: DataFrame with columns ['high', 'low', 'close']
    - lookback_period: Number of periods to consider for swing high/low
    
    Returns:
    - 'bullish' if bullish reversal signal detected
    - 'bearish' if bearish reversal signal detected
    - 'neutral' if no clear signal
    """
    
    if len(data) < lookback_period + 1:
        return "neutral"
    
    # Get recent swing high and low
    recent_high = data['high'].rolling(lookback_period).max().iloc[-1]
    recent_low = data['low'].rolling(lookback_period).min().iloc[-1]
    
    # Calculate Fibonacci levels
    fib_levels = calculate_fib_levels(recent_high, recent_low)
    golden_zone_low = fib_levels['65%']
    golden_zone_high = fib_levels['61.8%']
    
    # Get recent price action
    recent_close = data['close'].iloc[-1]
    prev_close = data['close'].iloc[-2]
    
    # Check if price is in the golden pocket zone (61.8% - 65%)
    in_golden_zone = golden_zone_low <= recent_close <= golden_zone_high
    
    if not in_golden_zone:
        return "Neutral"
    
    # Check for bullish reversal (price coming from below)
    if recent_close > prev_close and prev_close < golden_zone_low:
        # Additional confirmation - price closed above previous candle's high
        if recent_close > data['high'].iloc[-2]:
            return "Bullish"
    
    # Check for bearish reversal (price coming from above)
    elif recent_close < prev_close and prev_close > golden_zone_high:
        # Additional confirmation - price closed below previous candle's low
        if recent_close < data['low'].iloc[-2]:
            return "Bearish"
    
    return "Neutral" 
    
# 4. Fibonacci Confluence Strategy

def fibonacci_confluence_signal(data, fib_level=0.618, lookback_days=5, ema_period=9, tolerance=0.005):
   
    # Calculate EMA9
    data['EMA9'] = talib.EMA(data['close'], timeperiod=ema_period)

    # Get recent data for swing high/low
    recent = data.iloc[-lookback_days:]
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()

    # Calculate Fibonacci level
    fib_price = swing_high - (swing_high - swing_low) * fib_level

    # Get latest candle data
    current_close = data.iloc[-1]['close']
    current_ema = data.iloc[-1]['EMA9']

    # Check if price is within tolerance of Fibonacci level
    if abs(current_close - fib_price) / fib_price <= tolerance:
        if current_close > current_ema:
            return 'Bullish'
        elif current_close < current_ema:
            return 'Bearish'
    
    return 'Neutral'






# ======================
# Main Fibonacci Strategy Aggregator
# ======================

def fibonacci_strategies(data):
   
    signals = {
        "Fibonacci Retracement Bounce": fibonacci_retracement_bounce(data),
        "Fibonacci Breakout": fibonacci_breakout(data),
        "Golden Pocket Reversal": golden_pocket_reversal_strategy(data),
        "Fibonacci Confluence": fibonacci_confluence_signal(data)
    }

    weights = {
        "Fibonacci Retracement Bounce": 30,
        "Fibonacci Breakout": 25,
        "Golden Pocket Reversal": 30,
        "Fibonacci Confluence": 15
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

    return signals, overall_percentage, final_signal



# ======================
# API-style Function
# ======================

def get_fibonacci_trade_signal(data):
    fib_signals, overallscore, final_signal = fibonacci_strategies(data)
    return {
        "fib_signals": fib_signals,
        "fib_score": overallscore,
        "fib_final_signal": final_signal
    }
