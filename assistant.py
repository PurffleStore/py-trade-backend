"""
assistant.py — PyTrade AI chatbot engine.

Provider priority:
  1. Groq  (free tier, fast) — set GROQ_API_KEY
  2. HuggingFace router      — set HF_TOKEN
  3. Smart rule-based fallback (always works, no API key needed)
"""
import os
import re
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PyTrade AI, an expert stock market and trading assistant built into the Py-Trade platform.

You specialise in:
- Technical analysis: RSI, MACD, EMA, Bollinger Bands, ATR, ADX, support/resistance, candlestick patterns, Fibonacci levels
- Fundamental analysis: P/E ratio, P/B ratio, EPS, revenue growth, debt-to-equity, ROE, balance sheet reading
- Indian markets: NSE, BSE, NIFTY 50, SENSEX, SEBI regulations, F&O, SGX Nifty
- Global markets: S&P 500, NASDAQ, Dow Jones, Nikkei, DAX, FTSE, currency pairs, commodities
- Trading strategies: swing trading, momentum, positional, intraday, value investing, options strategies
- Risk management: stop-loss placement, position sizing, risk-reward ratios, portfolio diversification

Rules:
- Give clear, structured answers with bullet points or numbered lists where helpful
- Use simple language — explain jargon when you use it
- For market forecasts always add: "This is educational analysis, not financial advice."
- Keep answers focused and concise — no unnecessary padding
- When asked about a specific stock or index, provide a balanced view with both bullish and bearish factors"""

# ── Environment ──────────────────────────────────────────────────────────────
GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")
HF_TOKEN:     Optional[str] = os.environ.get("HF_TOKEN")
TEMP: float = float(os.environ.get("TEMPERATURE", "0.3"))

# ── Build chain helper ────────────────────────────────────────────────────────
def _build_chain(api_key: str, base_url: str, model: str):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=TEMP,
        timeout=40,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{message}")
    ])
    return prompt | llm | StrOutputParser()

# ── Initialise best available provider ───────────────────────────────────────
_chain = None
_provider = "none"

# 1. Try Groq (free tier — https://console.groq.com)
if GROQ_API_KEY:
    try:
        _chain = _build_chain(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )
        _provider = "groq"
        log.info("PyTrade assistant: using Groq provider (llama-3.3-70b-versatile)")
    except Exception as e:
        log.warning("PyTrade assistant: Groq init failed — %s", e)

# 2. Fallback: HuggingFace Inference Router
if _chain is None and HF_TOKEN:
    try:
        _chain = _build_chain(
            api_key=HF_TOKEN,
            base_url=os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.environ.get("HF_MODEL", "openai/gpt-oss-20b:nebius"),
        )
        _provider = "huggingface"
        log.info("PyTrade assistant: using HuggingFace provider")
    except Exception as e:
        log.warning("PyTrade assistant: HuggingFace init failed — %s", e)

if _chain is None:
    log.warning("PyTrade assistant: no LLM provider configured — using rule-based fallback")

# ── Rule-based fallback (always available, no API key needed) ─────────────────
_RULES: list[tuple[list[str], str]] = [
    (
        ["rsi", "relative strength"],
        """**RSI (Relative Strength Index)**

RSI is a momentum oscillator that measures the speed and magnitude of price changes on a scale of 0 to 100.

**Key levels:**
- **Above 70** → Overbought — potential reversal or pullback signal
- **Below 30** → Oversold — potential bounce or buying opportunity
- **50 level** → Acts as support/resistance for trend direction

**How to use in swing trading:**
1. Wait for RSI to enter overbought/oversold zone
2. Look for a reversal candlestick pattern as confirmation
3. Enter trade when RSI crosses back through 70 (sell) or 30 (buy)
4. Use RSI divergence (price makes new high but RSI doesn't) as an early reversal warning

*Standard period: 14 days. Common on NSE stocks and indices.*"""
    ),
    (
        ["macd", "moving average convergence"],
        """**MACD (Moving Average Convergence Divergence)**

MACD is a trend-following momentum indicator using two exponential moving averages.

**Components:**
- **MACD Line** = 12-period EMA − 26-period EMA
- **Signal Line** = 9-period EMA of the MACD Line
- **Histogram** = MACD Line − Signal Line

**Trading signals:**
1. **Bullish crossover** → MACD line crosses above Signal line → potential buy
2. **Bearish crossover** → MACD line crosses below Signal line → potential sell
3. **Zero line cross** → MACD crossing above 0 confirms uptrend
4. **Histogram shrinking** → momentum weakening, possible reversal ahead
5. **Divergence** → strongest signal; price and MACD moving in opposite directions

*Best used on daily charts for swing trades. Combine with RSI for confirmation.*"""
    ),
    (
        ["ema", "exponential moving average", "moving average"],
        """**EMA (Exponential Moving Average)**

EMA gives more weight to recent prices, making it more responsive than a simple moving average.

**Common EMA periods for traders:**
- **EMA 20** → Short-term trend (swing trading)
- **EMA 50** → Medium-term trend
- **EMA 200** → Long-term trend; major support/resistance level

**Trading rules:**
1. **Price above EMA 20** → short-term uptrend; look for buy setups
2. **Price below EMA 200** → avoid buying; market is in long-term downtrend
3. **EMA 20 crosses EMA 50** → Golden/Death cross — trend change signal
4. **EMA acting as support** → bounce from EMA is a common entry point

*Tip: On NSE, EMA 20 and EMA 50 on daily charts are widely watched by institutional traders.*"""
    ),
    (
        ["swing trading", "swing trade"],
        """**Swing Trading Strategy**

Swing trading captures price moves over 2 to 10 days, holding positions through short-term fluctuations.

**Core approach:**
1. **Identify the trend** → Use EMA 20/50 to confirm direction
2. **Wait for a pullback** → Price pulls back to support in an uptrend
3. **Look for entry signal** → Bullish candlestick + RSI near 40-50 + MACD crossover
4. **Set stop-loss** → Below the recent swing low (typically 2-4% below entry)
5. **Target** → Previous resistance level (risk-reward minimum 1:2)

**Best setups for NSE swing trading:**
- Breakout above consolidation with volume surge
- Bounce from EMA 20 in uptrending stock
- RSI oversold + bullish engulfing candle

**Risk management:** Never risk more than 1-2% of capital per trade."""
    ),
    (
        ["support", "resistance", "support and resistance"],
        """**Support and Resistance Levels**

Support is a price level where buying interest is strong enough to prevent further decline.
Resistance is where selling pressure prevents further rise.

**How to identify them:**
1. **Previous highs and lows** → Most reliable levels
2. **Round numbers** → e.g. ₹1000, ₹1500, ₹2000 (psychological levels)
3. **Moving averages** → EMA 20, 50, 200 act as dynamic S/R
4. **Fibonacci levels** → 38.2%, 50%, 61.8% retracements
5. **Volume nodes** → Areas of high historical trading activity

**Trading rules:**
- Buy near support with stop below it
- Sell near resistance with stop above it
- When support breaks → it becomes resistance (and vice versa)
- Stronger the level (more times tested) → more significant the breakout"""
    ),
    (
        ["bollinger", "bollinger band"],
        """**Bollinger Bands**

Bollinger Bands plot two standard deviations above and below a 20-period moving average, creating a dynamic price channel.

**Reading the bands:**
- **Band squeeze** → Bands narrow = low volatility = big move incoming
- **Band expansion** → Bands widen = high volatility, trend in motion
- **Price at upper band** → Overbought in a ranging market
- **Price at lower band** → Oversold in a ranging market
- **Walking the bands** → Price hugging upper band = strong uptrend (don't sell short)

**Strategy — Bollinger Band Bounce:**
1. Wait for price to touch the lower band
2. Confirm with RSI below 35 and a bullish candle
3. Enter long with target at the middle band (20 EMA) or upper band
4. Stop-loss: 1% below the lower band"""
    ),
    (
        ["nse", "national stock exchange", "sensex", "nifty", "bse", "indian market", "india"],
        """**NSE & BSE — Indian Stock Markets**

**NSE (National Stock Exchange):**
- India's largest stock exchange by volume
- Key indices: NIFTY 50, BANK NIFTY, NIFTY MIDCAP 100, NIFTY IT
- Trading hours: 9:15 AM – 3:30 PM IST (Mon–Fri)
- Derivatives (F&O) are actively traded here

**BSE (Bombay Stock Exchange):**
- Asia's oldest stock exchange
- Key index: SENSEX (top 30 companies)
- Same trading hours as NSE

**Key facts for traders:**
- Both exchanges are regulated by SEBI
- Settlement: T+1 (trade day + 1 working day)
- Circuit breakers: 10%, 15%, 20% market-wide halt limits
- SGX Nifty (now Gift Nifty) indicates opening trend before market opens
- FII/DII activity heavily influences market direction"""
    ),
    (
        ["stop loss", "stoploss", "stop-loss", "risk management", "position size"],
        """**Risk Management & Stop-Loss**

Proper risk management is what separates profitable traders from those who lose capital.

**Stop-loss placement:**
- **Technical stop** → Below support level, swing low, or EMA
- **Percentage stop** → Fixed % below entry (e.g. 2-3% for swing trades)
- **ATR stop** → Entry − (1.5 × ATR) — accounts for volatility

**Position sizing formula:**
```
Position Size = (Capital × Risk %) ÷ (Entry Price − Stop Price)
```
Example: ₹1,00,000 capital, 1% risk = ₹1,000 max loss
If entry ₹500, stop ₹480 → Risk per share = ₹20
Position size = ₹1,000 ÷ ₹20 = **50 shares**

**Golden rules:**
- Never risk more than 1-2% of total capital per trade
- Minimum risk-reward ratio: 1:2 (risk ₹1 to make ₹2)
- Move stop to breakeven once trade moves in your favour by 1R"""
    ),
    (
        ["fundamental", "pe ratio", "eps", "balance sheet", "p/e"],
        """**Fundamental Analysis for Traders**

Even swing traders should check basic fundamentals to avoid weak companies.

**Key metrics to check:**
- **P/E Ratio** → Price ÷ EPS. Lower than industry average = potentially undervalued
- **P/B Ratio** → Price ÷ Book Value. Below 1 = trading below asset value
- **Debt/Equity** → Below 1 is generally safer for most sectors
- **ROE** → Return on Equity. Above 15% = efficiently using shareholder capital
- **Revenue growth** → Consistent YoY growth = business momentum
- **EPS growth** → Earnings per share growing = company becoming more profitable

**Quick checklist before a swing trade:**
1. ✅ P/E not excessively above sector average
2. ✅ No major debt concerns
3. ✅ No upcoming negative events (quarterly results, legal issues)
4. ✅ Promoter holding stable or increasing (check NSE/BSE filings)"""
    ),
]

def _rule_based_answer(message: str) -> Optional[str]:
    msg = message.lower()
    for keywords, answer in _RULES:
        if any(kw in msg for kw in keywords):
            return answer
    return None

# ── Public API ────────────────────────────────────────────────────────────────
def get_answer(message: str) -> str:
    if not message or not message.strip():
        raise ValueError("message cannot be empty.")

    msg = message.strip()

    # Try LLM first
    if _chain is not None:
        try:
            return _chain.invoke({"message": msg})
        except Exception as e:
            log.error("LLM chain error (%s): %s", _provider, e)
            # Fall through to rule-based

    # Try rule-based fallback
    rule_answer = _rule_based_answer(msg)
    if rule_answer:
        return rule_answer

    # Generic fallback
    return (
        "I can help you with questions about **technical analysis** (RSI, MACD, EMA, Bollinger Bands), "
        "**fundamental analysis** (P/E ratio, earnings, balance sheet), **Indian markets** (NSE, BSE, NIFTY), "
        "**trading strategies** (swing trading, momentum), and **risk management** (stop-loss, position sizing).\n\n"
        "Try asking something like:\n"
        "- *What is RSI and how do I use it?*\n"
        "- *How to pick stocks for swing trading?*\n"
        "- *Explain support and resistance levels*\n"
        "- *How to place a stop-loss?*\n\n"
        "**To enable full AI responses:** Set the `GROQ_API_KEY` environment variable on the server "
        "(free at [console.groq.com](https://console.groq.com))."
    )
