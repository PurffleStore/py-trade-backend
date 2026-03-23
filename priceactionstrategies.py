from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import talib


# Candlestick Pattern Detection Strategy
def candlestick_pattern_strategy(data):
    open_ = data['open']
    high = data['high']
    low = data['low']
    close = data['close']

    # Bullish Patterns
    bullish_patterns = [
        talib.CDLENGULFING(open_, high, low, close),
        talib.CDLHAMMER(open_, high, low, close),
        talib.CDLMORNINGSTAR(open_, high, low, close),
        talib.CDLPIERCING(open_, high, low, close),
        talib.CDLINVERTEDHAMMER(open_, high, low, close),
        talib.CDL3WHITESOLDIERS(open_, high, low, close)
    ]

    # Bearish Patterns
    bearish_patterns = [
        talib.CDLENGULFING(open_, high, low, close),
        talib.CDLSHOOTINGSTAR(open_, high, low, close),
        talib.CDLEVENINGSTAR(open_, high, low, close),
        talib.CDLDARKCLOUDCOVER(open_, high, low, close),
        talib.CDLHANGINGMAN(open_, high, low, close),
        talib.CDL3BLACKCROWS(open_, high, low, close)
    ]

    # Neutral Patterns
    neutral_patterns = [
    talib.CDLDOJI(open_, high, low, close),
    talib.CDLSPINNINGTOP(open_, high, low, close),
    talib.CDLLONGLEGGEDDOJI(open_, high, low, close),
    talib.CDLHIGHWAVE(open_, high, low, close)
    ]


    # Check Bullish
    for pattern in bullish_patterns:
        if pattern.iloc[-1] > 0:
            return "Bullish"

    # Check Bearish
    for pattern in bearish_patterns:
        if pattern.iloc[-1] < 0:
            return "Bearish"

    # Check Neutral
    for pattern in neutral_patterns:
        if pattern.iloc[-1] != 0:
            return "Neutral"

    return "Neutral"

#first day high value is higher than the second day and third day should be higher than second day
def three_bar_triangle_breakout(data):   

    high = data['high']
    low = data['low']
    close = data['close']
    open_ = data['open']

    # Bullish entry condition
    ENTRYLONG = (
        close.iloc[-1] > open_.iloc[-1] and
        close.iloc[-1] > close.iloc[-2] and
        close.iloc[-1] > high.iloc[-2] and
        low.iloc[-2] > low.iloc[-4] and
        low.iloc[-3] > low.iloc[-4] and
        high.iloc[-2] < high.iloc[-4] and
        high.iloc[-3] < high.iloc[-4]
    )

    # Bearish entry condition
    ENTRYSHORT = (
        close.iloc[-1] < open_.iloc[-1] and
        close.iloc[-1] < close.iloc[-2] and
        close.iloc[-1] < low.iloc[-2] and
        low.iloc[-2] > low.iloc[-4] and
        low.iloc[-3] > low.iloc[-4] and
        high.iloc[-2] < high.iloc[-4] and
        high.iloc[-3] < high.iloc[-4]
    )

    if ENTRYLONG:
        return "Bullish"
    elif ENTRYSHORT:
        return "Bearish"
    else:
        return "Neutral"

def hh_ll_price_action_strategy(data, lookback_days=5):
   
    data = data.tail(lookback_days)

    is_higher_high = True
    is_higher_low = True
    is_lower_high = True
    is_lower_low = True

    for i in range(1, len(data)):
        if data['high'].iloc[i] <= data['high'].iloc[i - 1]:
            is_higher_high = False
        if data['low'].iloc[i] <= data['low'].iloc[i - 1]:
            is_higher_low = False
        if data['high'].iloc[i] >= data['high'].iloc[i - 1]:
            is_lower_high = False
        if data['low'].iloc[i] >= data['low'].iloc[i - 1]:
            is_lower_low = False

    if is_higher_high and is_higher_low:
        return "Bullish"
    elif is_lower_high and is_lower_low:
        return "Bearish"
    else:
        return "Neutral"


def fvg_strategy(data, lookback_days=5):
    
    data = data.tail(lookback_days)

    for i in range(2, len(data)):
        high_candle1 = data['high'].iloc[i - 2]
        low_candle1 = data['low'].iloc[i - 2]
        high_candle3 = data['high'].iloc[i]
        low_candle3 = data['low'].iloc[i]

        # Bullish FVG: Gap between high of candle 1 and low of candle 3
        if low_candle3 > high_candle1:
            return "Bullish"

        # Bearish FVG: Gap between low of candle 1 and high of candle 3
        if high_candle3 < low_candle1:
            return "Bearish"

    return "Neutral"

def bos_strategy(data, lookback_days=10):
   
    data = data.tail(lookback_days)
    highs = data['high'].tolist()
    lows = data['low'].tolist()

   
    # Recent high/low
    recent_high = highs[-1]
    previous_high = max(highs[:-1])  # Highest in previous candles

    recent_low = lows[-1]
    previous_low = min(lows[:-1])    # Lowest in previous candles

    # Check for bullish BOS (new high formed)
    if recent_high > previous_high:
        return "Bullish"

    # Check for bearish BOS (new low formed)
    if recent_low < previous_low:
        return "Bearish"

    return "Neutral"

def choch_strategy(data, lookback_period=14):   
    data = data.copy()

    # Calculate recent rolling highs and lows
    data['recent_high'] = data['high'].rolling(window=lookback_period).max()
    data['recent_low'] = data['low'].rolling(window=lookback_period).min()   


    # Check for Bullish or Bearish CHoCH using the most recent candle
    if (data['high'].iloc[-1] < data['high'].iloc[-2]) and (data['low'].iloc[-1] < data['recent_low'].iloc[-2]):
        return "Bearish"

    elif (data['low'].iloc[-1] > data['low'].iloc[-2]) and (data['high'].iloc[-1] > data['recent_high'].iloc[-2]):
        return "Bullish"    

    return "Neutral"


def order_block_strategy(data, lookback_days=2):
   
    data = data.tail(lookback_days)
    
    previous = data.iloc[-2]
    current = data.iloc[-1]

        # Bullish Order Block: last bearish candle followed by a strong bullish move
    if previous['close'] < previous['open'] and current['close'] > current['open'] and current['close'] > previous['high']:
            return "Bullish"

        # Bearish Order Block: last bullish candle followed by a strong bearish move
    elif previous['close'] > previous['open'] and current['close'] < current['open'] and current['close'] < previous['low']:
            return "Bearish"

    return "Neutral"


# Main strategy function using Candlestick
def priceaction_strategies(data):
   
    candlestick_signal = candlestick_pattern_strategy(data)
    
    hh_hl = hh_ll_price_action_strategy(data)

    triangle_breakout = three_bar_triangle_breakout(data)

    fvg = fvg_strategy(data)

    bos = bos_strategy(data)

    choch = choch_strategy(data)

    order_block = order_block_strategy(data)

    signals = {
        "Candlestick Pattern": candlestick_signal,       
        "HH_HL_LL_LH" : hh_hl,
        "Triangle Breakout": triangle_breakout,
        "Fair Value Gap": fvg,
        "BOS": bos,
        "CHoCH": choch,
        "Order_Block": order_block

    }

    weights = {
        "Candlestick Pattern": 15,       
        "HH_HL_LL_LH": 15,
        "Triangle Breakout": 15,
        "Fair Value Gap": 10,
        "BOS": 20,
        "CHoCH": 15,
        "Order_Block": 10
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

# API-style function for Candlestick Strategy
def get_priceaction_trade_signal(data):
    priceaction_signals, overallscore, final_signal = priceaction_strategies(data)
    return {
        "priceaction_signals": priceaction_signals,
        "priceaction_score": overallscore,
        "priceaction_final_signal": final_signal
    }

