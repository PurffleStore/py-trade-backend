from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from collections import OrderedDict
import datetime

    
# Calculate MACD, Signal and Histogram
def calculate_macdvalue(data, fast=12, slow=26, signal=9):
    
    close_prices = data['close']

    # Calculate MACD
    macd_line, signal_line, histogram = talib.MACD(
        close_prices,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )

    return macd_line, signal_line, histogram

# MACD Line Crossover - completed
def get_macd_line_crossover_signal(macd, signal):    
    
    for i in range(len(macd) - 1):
        older_macd = macd[i]
        newer_macd = macd[i + 1]
        older_signal = signal[i]
        newer_signal = signal[i + 1]
        
        # Bullish crossover (MACD crosses above Signal)
        if older_macd <= older_signal and newer_macd > newer_signal:
            return "Bullish"
        # Bearish crossover (MACD crosses below Signal)
        elif older_macd >= older_signal and newer_macd < newer_signal:
            return "Bearish"
    
    return "Neutral"

# Zero Line Crossover - completed
def get_macd_zero_line_crossover_signal(macd):   
    
    for i in range(len(macd) - 1):
        older = macd[i]
        newer = macd[i + 1]
        
        if older <= 0 and newer > 0:
            return "Bullish"
        elif older >= 0 and newer < 0:
            return "Bearish"
    
    return "Neutral"

# MACD Momentum Signal - completed
def get_macd_momentum_signal(macd, signal, hist):

    for i in range(len(macd) - 1):
        older_macd = macd[i]
        newer_macd = macd[i + 1]
        older_signal = signal[i]
        newer_signal = signal[i + 1]
        current_hist = hist[i + 1]  # Use the histogram of the newer point

        # Bullish crossover (MACD crosses above Signal) with positive histogram
        if older_macd <= older_signal and newer_macd > newer_signal and current_hist > 0:
            return "Bullish"

        # Bearish crossover (MACD crosses below Signal) with negative histogram
        elif older_macd >= older_signal and newer_macd < newer_signal and current_hist < 0:
            return "Bearish"

    return "Neutral"

# MACD Volume Signal - completed
def get_macd_volume_signal(data, macd, signal):
   
    avg_volume = data['volume'].rolling(window=10).mean()
    recent_volume = data['volume'].values[-1:]
    recent_avg_volume = avg_volume.values[-1:]
    volume_confirm = recent_volume > recent_avg_volume
   
   
    for i in range(len(macd) - 1):
        older_macd = macd[i]
        newer_macd = macd[i + 1]
        older_signal = signal[i]
        newer_signal = signal[i + 1]
        
        if (older_macd <= older_signal and 
            newer_macd > newer_signal and 
            volume_confirm):
            return "Bullish"
        elif (older_macd >= older_signal and 
              newer_macd < newer_signal and 
              volume_confirm):
            return "Bearish"
    
    return "Neutral"

# MACD Multi-Timeframe - completed
def get_macd_multi_timeframe_confirmation(macd, signal,macd_hr, signal_hr):
    for i in range(len(macd) - 1):
        older_macd = macd[i]
        newer_macd = macd[i + 1]
        older_signal = signal[i]
        newer_signal = signal[i + 1]
        older_macd_hr = macd_hr[i]
        newer_macd_hr = macd_hr[i + 1]
        older_signal_hr = signal_hr[i]
        newer_signal_hr = signal_hr[i + 1]
        
        # Bullish crossover (MACD crosses above Signal)
        if older_macd <= older_signal and newer_macd > newer_signal and older_macd_hr <= older_signal_hr and newer_macd_hr > newer_signal_hr:
            return "Bullish"
        # Bearish crossover (MACD crosses below Signal)
        elif older_macd >= older_signal and newer_macd < newer_signal and older_macd_hr >= older_signal_hr and newer_macd_hr < newer_signal_hr:
            return "Bearish"
    
    return "Neutral"

# Price and MACD Divergence - completed
def get_macd_divergence_signal(macd, price):    
    
    # Bullish Divergence: Price makes lower lows, but MACD makes higher lows
    bullish_divergence = None
    for i in range(10, len(price)):  # Look at the last 5 candles
        if price[i] < price[i-1] and macd[i] > macd[i-1]:
            bullish_divergence = "Bullish"
            break

    # Bearish Divergence: Price makes higher highs, but MACD makes lower highs
    bearish_divergence = None
    for i in range(10, len(price)):  # Look at the last 5 candles
        if price[i] > price[i-1] and macd[i] < macd[i-1]:
            bearish_divergence = "Bearish"
            break
    
    if bullish_divergence:
        return bullish_divergence
    elif bearish_divergence:
        return bearish_divergence
    else:
        return "Neutral"

# MACD Hidden Divergence - completed
def get_macd_hidden_divergence_signal(macd, price):    
    
    # Bullish Hidden Divergence: Price makes a higher low, but MACD makes a lower low
    bullish_hidden_divergence = None
    for i in range(10, len(price)):  # Look at the last 5 candles
        if price[i] > price[i-10] and macd[i] < macd[i-10]:
            bullish_hidden_divergence = "Bullish"
            break

    # Bearish Hidden Divergence: Price makes a lower high, but MACD makes a higher high
    bearish_hidden_divergence = None
    for i in range(10, len(price)):  # Look at the last 5 candles
        if price[i] < price[i-10] and macd[i] > macd[i-10]:
            bearish_hidden_divergence = "Bearish"
            break
    
    if bullish_hidden_divergence:
        return bullish_hidden_divergence
    elif bearish_hidden_divergence:
        return bearish_hidden_divergence
    else:
        return "Neutral"


# macd_strategies and get_macd_trade_signal functions
def macd_strategies(data):     

    macd, signal, hist = calculate_macdvalue(data)    
    
    latest_macd = macd[-1]   
    signals = { 
        "MACD": round(latest_macd,2),
        "MACD Line Crossover": get_macd_line_crossover_signal(macd[-5:],signal[-5:]),
        "MACD Zero-Line Crossover": get_macd_zero_line_crossover_signal(macd[-5:]),
        "MACD Divergence": get_macd_divergence_signal(macd[-10:], data['close'][-10:]),
        "Hidden Divergence": get_macd_hidden_divergence_signal(macd[-10:], data['close'][-10:]),
        "MACD Volume": get_macd_volume_signal(data, macd[-5:],signal[-5:]),
        "MACD Momentum": get_macd_momentum_signal(macd[-5:],signal[-5:],hist[-5:]),
        
    }

    macd_signal_weights = {
        "MACD Line Crossover": 25,
        "MACD Zero-Line Crossover": 15, 
        "MACD Divergence": 20,
        "Hidden Divergence": 10,
        "MACD Volume": 15,
        "MACD Momentum": 15,
        
    }
    
    total_score = 0
    for strategy, weight in macd_signal_weights.items():
        signal = signals[strategy]
        if signal == "Bullish":
            total_score += weight
        elif signal == "Neutral":
            total_score += weight * 0.5

    overall_percentage = round((total_score / sum(macd_signal_weights.values())) * 100, 2)

    if overall_percentage >= 60:
        final_signal = "Buy"
    elif overall_percentage <= 40:
        final_signal = "DBuy"
    else:
        final_signal = "Neutral"
    
    return signals, overall_percentage, final_signal


def get_macd_trade_signal(data):    
    macd_signals, overallscore, final_signal = macd_strategies(data)
    macd_line, signal_line, hist = calculate_macdvalue(data)
    # Format and convert MACD and Signal Line for last 100 days
    macd_series = pd.Series(macd_line, index=data.index).dropna().tail(100)
    signal_series = pd.Series(signal_line, index=data.index).dropna().tail(100)

    macd_series.index = macd_series.index.strftime('%Y-%m-%d')
    signal_series.index = signal_series.index.strftime('%Y-%m-%d')
    hist_series = pd.Series(hist, index=data.index).dropna().tail(100)
    hist_series.index = hist_series.index.strftime('%Y-%m-%d')

    return {
        "macd_signals": macd_signals,
        "macd_score": overallscore,
        "macd_final_signal": final_signal,
        "macd_line": macd_series.round(2).to_dict(),
        "macd_signal_line": signal_series.round(2).to_dict(),
        "macd_histogram": hist_series.round(2).to_dict()
    }