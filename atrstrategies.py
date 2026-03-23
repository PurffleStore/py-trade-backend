from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import talib


# Calculate ATR values
def calculate_atr(data):
    atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)      
    return atr

# ATR Breakout Strategy (Price crossing ATR threshold)
def atr_breakout_strategy(data, atr, multiplier=2):
    latest_close = data['close'].iloc[-2]
    previous_close = data['close'].iloc[-1]
    latest_atr = atr.iloc[-2]
    threshold_up = latest_close + (multiplier * latest_atr)
    threshold_down = latest_close - (multiplier * latest_atr)

    # Bullish breakout (price moves above ATR threshold)
    if previous_close > threshold_up:
        return "Bullish"
    # Bearish breakout (price moves below ATR threshold)
    elif previous_close < threshold_down:
        return "Bearish"
    # No breakout, neutral
    else:
        return "Neutral"

def calculate_dynamic_threshold(atr, period=14):
   
    atr_change = atr.pct_change(periods=period)  
    atr_std = atr_change.std()  
    dynamic_threshold = 2 * atr_std  
    return dynamic_threshold

# ATR Expansion Strategy (Confirming trend continuation)
def atr_expansion_strategy(data, atr, period=14, days_to_check=5):
   
   
    dynamicthreshold = calculate_dynamic_threshold(atr, period)
   
 
    atr_last = atr.iloc[-days_to_check:]  
    atr_expansion = atr_last[-1] > atr_last.mean() + dynamicthreshold  
    
    if atr_expansion:
        
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return "Bullish"  
        elif data['close'].iloc[-1] < data['close'].iloc[-2]:
            return "Bearish"  
        else:
            return "Neutral" 

  
    return "Neutral"  

# ATR Squeeze/Compression Strategy (Confirming trend continuation)
def atr_squeeze_strategy(data, atr, period=14, days_to_check=5):   
   
    dynamicthreshold = calculate_dynamic_threshold(atr, period)   
 
    atr_last = atr.iloc[-days_to_check:]  
    atr_compression = atr_last[-1] < atr_last.mean() - dynamicthreshold  
    resistance = data['high'].iloc[-days_to_check:].max() 
    support = data['low'].iloc[-days_to_check:].min() 

    if atr_compression:

        if data['close'].iloc[-1] > resistance:
            return "Bullish"            
       
        elif data['close'].iloc[-1] < support:
            return "Bearish"       
       
        else:
            return "Neutral" 

  
    return "Neutral" 

# ATR Trend Reversal Strategy (ATR rising during price reversal)
def atr_trend_reversal_strategy(atr, price, days=5):
    # Look at the change in price and ATR
    price_diff = price.iloc[-1] - price.iloc[-days]
    atr_diff = atr.iloc[-1] - atr.iloc[-days]
    
    # If price is reversing (uptrend to downtrend or vice versa), and ATR is increasing
    if price_diff > 0 and atr_diff > 0:
        return "Bullish"
    elif price_diff < 0 and atr_diff > 0:
        return "Bearish"
    return "Neutral"

# Main strategy function using ATR strategy
def atr_strategies(data):
    
    atr = calculate_atr(data)
    
    atr_breakout = atr_breakout_strategy(data, atr)    
   
    atr_expansion = atr_expansion_strategy(data,atr)

    atr_squeeze = atr_squeeze_strategy(data,atr)  
    
    atr_trend_reversal = atr_trend_reversal_strategy(atr, data['close'])

    # Collect signals
    signals = {
        "ATR": round(atr.iloc[-1], 2),
        "ATR Breakout": atr_breakout,
        "ATR Expansion": atr_expansion,    
        "ATR Squeeze": atr_squeeze,    
        "ATR Trend Reversal": atr_trend_reversal
    }

    weights = {
        "ATR Breakout": 45,
        "ATR Expansion": 15, 
        "ATR Squeeze": 15,
        "ATR Trend Reversal": 25       
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

    return signals, overall_percentage, final_signal,atr

# API-style function to fetch ATR signals
def get_atr_trade_signal(data):
    atr_signals, overallscore, final_signal,atr = atr_strategies(data)
    atr_series = pd.Series(atr, index=data.index).dropna().tail(100)
    atr_series.index = atr_series.index.strftime('%Y-%m-%d')
    return {
        "atr_signals": atr_signals,
        "atr_score": overallscore,
        "atr_final_signal": final_signal,
        "atr_values": atr_series.round(2).to_dict()

    }
