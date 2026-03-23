import yfinance as yf
import pandas as pd
import numpy as np
import talib
import math
import requests
import time
import datetime
import os
from pathlib import Path
from datetime import timedelta
from collections import OrderedDict

from rsistrategies import get_rsi_trade_signal
from macdstrategies import get_macd_trade_signal
from emastrategies import get_ema_trade_signal
from atrstrategies import get_atr_trade_signal
from adxstrategies import get_adx_trade_signal
from fibostrategies import get_fibonacci_trade_signal
from priceactionstrategies import get_priceaction_trade_signal
from srstrategies import get_support_resistance_signal
from bbstrategies import get_bollinger_trade_signal
from fundamental import get_fundamental_details
from news import get_latest_news_with_sentiment
from highlow_forecast import forecast_next_15_high_low

"""analysestock.py

Compute technical/fundamental/news analysis for a ticker and return structured payload.
AI TA predictions are intentionally not included here; they are provided via a separate
backend endpoint `/ai_ta` so the frontend `trading-signal` component can request them
without altering the analysepage output.
"""

import os, numpy as np, pandas as pd

BASE_DIR = Path(__file__).resolve().parent 

# ===================== TA scoring =====================
def calculate_technical_analysis_score(indicator_scores):
    indicator_weights = {
        'RSI': 13,
        'MACD': 13,
        'ATR': 5,
        'ADX': 4,
        'EMA': 13,
        'PriceAction': 14,
        'Bollinger': 10,
        'Fibonacci': 4,
        'SR': 9
    }
    weight_values = list(indicator_weights.values())
    weighted_score = sum(score * weight for score, weight in zip(indicator_scores, weight_values))
    total_weight = sum(weight_values)
    technical_analysis_score = (weighted_score / (total_weight * 100)) * 85  
    overall_ta_signal_100 = np.where(
        technical_analysis_score > 65, 'Buy',
        np.where(technical_analysis_score > 40, 'Neutral', 'DBuy')
    )
    return technical_analysis_score, overall_ta_signal_100
  
def signal_from_score(score, max_points, buy_frac=0.65, neutral_frac=0.40):
    
    buy_cutoff = buy_frac * max_points
    neutral_cutoff = neutral_frac * max_points

    if score > buy_cutoff:
        return "Buy"
    elif score > neutral_cutoff:
        return "Neutral"
    else:
        return "DBuy"

# ================== Pivot levels & trade ==================
def calculate_pivot_points(ticker, score, live_price, atr_period=14):
    data = yf.download(ticker, period="2mo", interval="1wk")
    df = yf.download(ticker, period="2mo", interval="1d")

    if score < 50:
        return {
            "remarks": "Score is below 50%, avoid trading. No trade recommendation",
            "pivot_point": "N/A", "resistance1": "N/A", "support1": "N/A",
            "resistance2": "N/A", "support2": "N/A",
            "resistance3": "N/A", "support3": "N/A",
            "entry_point": "N/A", "stop_loss": "N/A", "target_price": "N/A",
            "s1_pect": "N/A", "s2_pect": "N/A", "s3_pect": "N/A",
            "r1_pect": "N/A", "r2_pect": "N/A", "r3_pect": "N/A", "p1_pect": "N/A"
        }

    if 50 <= score < 65:
        stoploss_multiplier, risk_reward_ratio = 1.2, 1.5
        remarks = "Neutral confidence - Monitor the price for further confirmation."
    elif 65 <= score < 70:
        stoploss_multiplier, risk_reward_ratio = 1.5, 2.0
        remarks = "Moderate confidence - Conservative stop loss and reward."
    elif 70 <= score < 80:
        stoploss_multiplier, risk_reward_ratio = 1.8, 2.5
        remarks = "Good confidence - Balanced approach."
    else:
        stoploss_multiplier, risk_reward_ratio = 2.0, 3.0
        remarks = "High confidence - Aggressive approach."

    close_prices = df['Close'].to_numpy().flatten()
    high_prices = df['High'].to_numpy().flatten()
    low_prices = df['Low'].to_numpy().flatten()
    df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)

    latest_atr = df['ATR'].iloc[-1]
    # ensure latest_atr is numeric
    try:
        latest_atr = float(latest_atr)
    except Exception:
        latest_atr = 0.0

    entry_point = live_price

    # 1) Indicator stop-loss (based on configured multiplier)
    indicator_stop_loss = entry_point - (stoploss_multiplier * latest_atr)

    # 2) Last swing low: use the minimum low of the recent window (fallback to overall min)
    swing_window = 10
    try:
        if len(df) >= swing_window:
            last_swing_low = float(df['Low'].tail(swing_window).min())
        else:
            last_swing_low = float(df['Low'].min())
    except Exception:
        last_swing_low = indicator_stop_loss

    # 3) Entry minus 1.2 * ATR
    entry_minus_atr = entry_point - (1.2 * latest_atr)

    # Final stop-loss is the maximum of the three candidates
    # (choose the highest price among candidates so stop-loss is the tightest allowed by rules)
    stop_loss = max(indicator_stop_loss, last_swing_low, entry_minus_atr)

    # Recompute target using the finalized stop_loss
    target_price = entry_point + ((entry_point - stop_loss) * risk_reward_ratio)

    previous_week = data.iloc[-2]
    high, low, close = previous_week["High"], previous_week["Low"], previous_week["Close"]

    P = (high + low + close) / 3
    R1 = (2 * P) - low
    S1 = (2 * P) - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2 * (P - low)
    S3 = low - 2 * (high - P)

    p1_pect = ((P - live_price) / P) * 100
    s1_pect = ((S1 - live_price) / S1) * 100
    s2_pect = ((S2 - live_price) / S2) * 100
    s3_pect = ((S3 - live_price) / S3) * 100
    r1_pect = ((R1 - live_price) / R1) * 100
    r2_pect = ((R2 - live_price) / R2) * 100
    r3_pect = ((R3 - live_price) / R3) * 100

    return {
        "pivot_point": round(float(P), 2),
        "resistance1": round(float(R1), 2),
        "support1": round(float(S1), 2),
        "resistance2": round(float(R2), 2),
        "support2": round(float(S2), 2),
        "resistance3": round(float(R3), 2),
        "support3": round(float(S3), 2),
        "entry_point": round(float(entry_point), 2),
        "stop_loss": round(float(stop_loss), 2),
        "target_price": round(float(target_price), 2),
        "s1_pect": round(float(s1_pect), 2),
        "s2_pect": round(float(s2_pect), 2),
        "s3_pect": round(float(s3_pect), 2),
        "r1_pect": round(float(r1_pect), 2),
        "r2_pect": round(float(r2_pect), 2),
        "r3_pect": round(float(r3_pect), 2),
        "p1_pect": round(float(p1_pect), 2),
        "remarks": remarks
    }



# =================== Main: short-term swing ===================
def analysestock(ticker):
    now = datetime.datetime.now()
    formatted_datetime = now.strftime('%Y-%m-%d %H:%M:%S.%f')

    threshold_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
    end_date = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d') if now >= threshold_time else now.strftime('%Y-%m-%d')

    stock_data = yf.download(ticker, start="2023-01-01", end=end_date, interval="1d") 
    stock_data.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in stock_data.columns]
    lasttradingdate = stock_data.index[-1].strftime('%d-%m-%Y')
    stockdetail = yf.Ticker(ticker)
    company_name = stockdetail.info.get("longName", "Company name not found")
    live_price = stockdetail.info["regularMarketPrice"]
    price_change = stockdetail.info['regularMarketChange']
    percentage_change = stockdetail.info['regularMarketChangePercent']

    recentdays = stock_data.tail(30)
    ohlc_data = []
    for index, row in recentdays.iterrows():
        ohlc_data.append({
            "x": index.strftime('%Y-%m-%d'),
            "y": [round(row['open'], 2), round(row['high'], 2), round(row['low'], 2), round(row['close'], 2)]
        })

    # TA Strategy signals
    rsi_trade_signal = get_rsi_trade_signal(stock_data)
    macd_trade_signal = get_macd_trade_signal(stock_data)
    ema_trade_signal = get_ema_trade_signal(stock_data)
    atr_trade_signal = get_atr_trade_signal(stock_data)
    adx_trade_signal = get_adx_trade_signal(stock_data)
    fibo_trade_signal = get_fibonacci_trade_signal(stock_data)
    priceaction_trade_signal = get_priceaction_trade_signal(stock_data)
    bb_trade_signal = get_bollinger_trade_signal(stock_data)
    sr_trade_signal = get_support_resistance_signal(stock_data)

    final_trade_signal = OrderedDict([
        ("RSI", rsi_trade_signal['rsi_final_signal']),
        ("MACD", macd_trade_signal['macd_final_signal']),
        ("ATR", atr_trade_signal['atr_final_signal']),
        ("EMA", ema_trade_signal['ema_final_signal']),
        ("ADX", adx_trade_signal['adx_final_signal']),
        ("Fibo", fibo_trade_signal['fib_final_signal']),
        ("BB", bb_trade_signal['bollinger_final_signal']),
        ("SR", sr_trade_signal['sr_final_signal']),
        ("PA_MS", priceaction_trade_signal['priceaction_final_signal']),
    ])

    indicator_score = [
        rsi_trade_signal["rsi_score"],
        macd_trade_signal['macd_score'],
        atr_trade_signal['atr_score'],
        adx_trade_signal['adx_score'],
        ema_trade_signal['ema_score'],
        priceaction_trade_signal['priceaction_score'],
        bb_trade_signal['bollinger_score'],
        fibo_trade_signal['fib_score'],
        sr_trade_signal['sr_score']
    ]

    overall_ta_score,overall_ta_signal = calculate_technical_analysis_score(indicator_score)

    #FA signals

    fundamental_analysis = get_fundamental_details(ticker)

    #news   
    
    news_payload = get_latest_news_with_sentiment(
        company_name,
        period="1d",      
        max_results=10,
        language="en",
        country="US"      
    )

    #overallscore

    overall_fa_score = fundamental_analysis["overall_fa_score"]     
    overall_news_score = news_payload['overall_news_score']
    overall_fa_signal = signal_from_score(overall_fa_score,15)
    overall_news_signal = signal_from_score(overall_news_score,5)
    combined_overall_score = overall_ta_score + overall_fa_score + overall_news_score
    combined_overall_signal = np.where(combined_overall_score > 65, 'Buy',
                                       np.where(combined_overall_score > 50, 'Neutral', 'DBuy'))


    #trade recommendation

    pivot_levels = calculate_pivot_points(ticker, combined_overall_score, live_price)


    #prediiction
    forecast_15 = None
    try:
        forecast_15 = forecast_next_15_high_low(
            ticker=ticker,
            stock_data=stock_data
        )
    except Exception as ex:
        forecast_15 = {"error": f"{type(ex).__name__}: {ex}"}

    # Summaries for 15-day forecast (max high, min low) + range series for charts
    max_high_15 = None
    max_high_15_date = None
    min_low_15 = None
    min_low_15_date = None
    highlow_range_15 = None

    if isinstance(forecast_15, dict) and all(k in forecast_15 for k in ("pred_high", "pred_low", "dates")):
        highs = np.asarray(forecast_15["pred_high"], dtype=float)
        lows = np.asarray(forecast_15["pred_low"], dtype=float)
        dates = forecast_15["dates"]

        if highs.size and lows.size and highs.size == lows.size == len(dates):
            hi_idx = int(np.nanargmax(highs))
            lo_idx = int(np.nanargmin(lows))

            max_high_15 = round(float(highs[hi_idx]), 2)
            max_high_15_date = dates[hi_idx]
            min_low_15 = round(float(lows[lo_idx]), 2)
            min_low_15_date = dates[lo_idx]

            # Precomputed rangeBar data: [{x: date, y: [low, high]}]
            highlow_range_15 = [
                {"x": d, "y": [round(float(l), 2), round(float(h), 2)]}
                for d, h, l in zip(dates, highs.tolist(), lows.tolist())
            ]




    response = {
        "ticker": ticker,
        "company_name": company_name,
        "lasttradingdate": lasttradingdate,
        "currentdatetime": formatted_datetime,
        "live_price": round(live_price, 2),
        "price_change": round(price_change, 2),
        "percentage_change": round(percentage_change, 2),
        "ohlc_data":ohlc_data,
        "RSI": rsi_trade_signal['rsi_signals'],
        "MACD": macd_trade_signal['macd_signals'],
        "EMA": ema_trade_signal['ema_signals'],
        "ATR": atr_trade_signal['atr_signals'],
        "ADX": adx_trade_signal['adx_signals'],
        "Fibo": fibo_trade_signal['fib_signals'],
        "SR": sr_trade_signal['support_resistance_signals'],
        "BB": bb_trade_signal['bollinger_signals'],
        "PA_MS": priceaction_trade_signal['priceaction_signals'],
        "final_trade_signal": final_trade_signal,
        "overall_ta_score": round(overall_ta_score, 2),  
        "overall_ta_signal": str(overall_ta_signal),
        "fundamental_analysis": fundamental_analysis,
        "overall_fa_score": overall_fa_score,
        "overall_fa_signal": str(overall_fa_signal),
        "overall_news_signal": str(overall_news_signal),
        "news_overall_score": overall_news_score,
        "news": news_payload["items"],
        "combined_overall_score": round(combined_overall_score, 2),
        "combined_overall_signal": str(combined_overall_signal),
        "tradingInfo": pivot_levels,

        "RSI 14": rsi_trade_signal['rsi_14_last_2_years'],
        "RSI 5": rsi_trade_signal['rsi_5_last_2_years'],
        "MA_20": rsi_trade_signal['ma'],
        "Close": rsi_trade_signal['close'],
        "LowerBB": rsi_trade_signal['lowerbb'],
        "UpperBB": rsi_trade_signal['upperbb'],
        "MACDLine": macd_trade_signal['macd_line'],
        "MACDSignalLine": macd_trade_signal['macd_signal_line'],
        "MACDHistogram": macd_trade_signal['macd_histogram'],
        "ATRValue": atr_trade_signal['atr_values'],
        "EMA 5": ema_trade_signal['EMA_5'],
        "EMA 20": ema_trade_signal['EMA_20'],
        "EMA 50": ema_trade_signal['EMA_50'],
        "ADX_Indicator": adx_trade_signal['ADX_Indicator'],
        "PLUS_DI": adx_trade_signal['PLUS_DI'],
        "MINUS_DI": adx_trade_signal['MINUS_DI']       
    }  

    # attach AI forecast and AI TA results
    response.update({
        "ai_predicted_daily_high_15": (forecast_15.get("pred_high") if isinstance(forecast_15, dict) and "pred_high" in forecast_15 else None),
        "ai_predicted_daily_low_15": (forecast_15.get("pred_low") if isinstance(forecast_15, dict) and "pred_low" in forecast_15 else None),
        "ai_predicted_dates_15": (forecast_15.get("dates") if isinstance(forecast_15, dict) and "dates" in forecast_15 else None),
        "ai_model_meta_15d": (forecast_15.get("bundle_meta") if isinstance(forecast_15, dict) and "bundle_meta" in forecast_15 else None),
        "ai_model_error_15d": (forecast_15.get("error") if isinstance(forecast_15, dict) and "error" in forecast_15 else None),
    })

    # Note: AI TA model predictions are not attached here to avoid changing analysepage output.
    # The trading-signal UI should call the dedicated `/ai_ta` endpoint to obtain per-symbol
    # ExtraTrees-based TA predictions.

    response.update({
        "ai_predicted_max_high_15": max_high_15,
        "ai_predicted_max_high_15_date": max_high_15_date,
        "ai_predicted_min_low_15": min_low_15,
        "ai_predicted_min_low_15_date": min_low_15_date,
        "ai_predicted_highlow_range_15": highlow_range_15
    })

    return response
