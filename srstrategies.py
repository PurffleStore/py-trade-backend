import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import talib


# Calculate Support and Resistance using Pivot Points
def calculate_support_resistance(data):
    # Pivot Point Calculation
    data['Pivot'] = (data['high'] + data['low'] + data['close']) / 3
    # Support and Resistance Calculations
    data['Support1'] = (2 * data['Pivot']) - data['high']
    data['Resistance1'] = (2 * data['Pivot']) - data['low']
   
    return data

#Strategy 1: Reversal strategy - find the difference between close and support/resistance and find the tolerance value based on ATR if it is less than tolernace value and check the candlestick pattern - based on the return bullish/bearish/neutral
def detect_reversal(df, support, resistance):
    df['Signal'] = 'Neutral'  # Default is neutral
    
    close_prices = df['close'].to_numpy().flatten()
    high_prices = df['high'].to_numpy().flatten()
    low_prices = df['low'].to_numpy().flatten()
    open_prices = df['open'].to_numpy().flatten()
    
    # Use common reversal patterns
    hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    shooting_star = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
    doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
    morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
    evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
    piercing_line = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
    harami = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
    df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    tolerance = df['ATR'].iloc[-1] * 2  
    for i in range(1, len(df)):
        close = close_prices[i]
        
        # Detect Bullish Reversal
        if abs(close - support[i]) <= tolerance:
            if hammer[i] > 0 or engulfing[i] > 0 or doji[i] > 0 or morning_star[i] > 0 or piercing_line[i] > 0 or harami[i] > 0:
                df.loc[df.index[i], 'Signal'] = 'Bullish'
        
        # Detect Bearish Reversal
        if abs(close - resistance[i]) <= tolerance:
            if shooting_star[i] < 0 or engulfing[i] < 0 or doji[i] < 0 or evening_star[i] < 0 or piercing_line[i] < 0 or harami[i] < 0:
                df.loc[df.index[i], 'Signal'] = 'Bearish'

    return df


# Strategy 2: Breakout Trading - the previous day should be above to resistance/support and break the support and resistance and crossed
def detect_breakouts(df):    
    support_level = df['Support1'].iloc[-1]
    resistance_level = df['Resistance1'].iloc[-1]    
    
    if df['close'].iloc[-1] > resistance_level and df['close'].iloc[-2] <= resistance_level:
       return "Bullish"

    elif df['close'].iloc[-1] < resistance_level and df['close'].iloc[-2] >= resistance_level:
       return "Bearish"

    elif df['close'].iloc[-1] > support_level and df['close'].iloc[-2] <= support_level:
       return "Bullish"

    elif df['close'].iloc[-1] < support_level and df['close'].iloc[-2] >= support_level:
       return "Bearish"

    return "Neutral"

# Strategy 3: Flip Zone 
def detect_flip_zone(df):
   
    support_level = df['Support1'].iloc[-1]
    resistance_level = df['Resistance1'].iloc[-1]

    if df['close'].iloc[-3] < support_level and df['close'].iloc[-2] >= support_level and df['close'].iloc[-1] > support_level:
      return "Bullish"

    elif df['close'].iloc[-3] > support_level and df['close'].iloc[-2] <= support_level and df['close'].iloc[-1] < support_level:
      return "Bearish"

    elif df['close'].iloc[-3] < resistance_level and df['close'].iloc[-2] >= resistance_level and df['close'].iloc[-1] > resistance_level:
      return "Bullish"

    elif df['close'].iloc[-3] > resistance_level and df['close'].iloc[-2] <= resistance_level and df['close'].iloc[-1] < resistance_level:
      return "Bearish"       
   
    
    return "Neutral"

# Strategy 4: SR RETEST - bounceup and bouncedown
def detect_sr_retest(df):
   
    support_level = df['Support1'].iloc[-1]
    resistance_level = df['Resistance1'].iloc[-1]

    # Retest Strategy: Bullish Retest - Price breaks above resistance and then tests it as support
    if df['close'].iloc[-4] < resistance_level and df['close'].iloc[-3] >= resistance_level and df['close'].iloc[-2] < df['close'].iloc[-3] and df['close'].iloc[-2] < df['close'].iloc[-1]:
        return "Bearish"  

    elif df['close'].iloc[-4] > resistance_level and df['close'].iloc[-3] <= resistance_level and df['close'].iloc[-2] > df['close'].iloc[-3] and df['close'].iloc[-2] > df['close'].iloc[-1]:
        return "Bullish"

    elif df['close'].iloc[-4] < support_level and df['close'].iloc[-3] >= support_level and df['close'].iloc[-2] < df['close'].iloc[-3] and df['close'].iloc[-2] < df['close'].iloc[-1]:
        return "Bullish"  

    elif df['close'].iloc[-4] > support_level and df['close'].iloc[-3] <= support_level and df['close'].iloc[-2] > df['close'].iloc[-3] and df['close'].iloc[-2] > df['close'].iloc[-1]:
        return "Bearish"     
   
    
    return "Neutral"


# Final Signal Calculation
def support_resistance_strategy(data):
   

    # Calculate Support and Resistance levels using Pivot Points
    data = calculate_support_resistance(data)
    # Calculate the market trend based on price and support/resistance
    breakout = detect_breakouts(data)
    reversal = detect_reversal(data, data['Support1'].to_numpy(), data['Resistance1'].to_numpy())
    flip = detect_flip_zone(data)
    
    sr_retest = detect_sr_retest(data)
    

    # Weight the signals for a final decision (example weighting)
    signals = {
        "Support1": round(data['Support1'].iloc[-1], 2),
        "Resistance1": round(data['Resistance1'].iloc[-1], 2),
        "Breakout": breakout,
        "Reversal": reversal['Signal'].iloc[-1],  
        "Flip": flip,        
        "SR_Retest":sr_retest       
        
    }

    weights = {
        "Breakout": 35,
        "Reversal": 25,  
        "Flip": 20,
        "SR_Retest":20       
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

# API-style function
def get_support_resistance_signal(data):
    sr_signals, overallscore, final_signal = support_resistance_strategy(data)
    return {
        "support_resistance_signals": sr_signals,
        "sr_score": overallscore,
        "sr_final_signal": final_signal
    }
