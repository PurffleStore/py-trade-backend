"""ai_ta_routes.py

Provides endpoints to manage and serve AI TA ExtraTrees predictions per symbol,
plus a comprehensive /ai_analysis endpoint combining TA + FA + News Sentiment.
"""
from flask import Blueprint, request, jsonify
import os
import traceback
from pathlib import Path
import time
import numpy as np

from ai_ta_model import train_and_save_model, predict_score_for_symbol

bp = Blueprint('ai_ta', __name__)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# simple in-memory cache: symbol -> (timestamp, result)
_AI_TA_CACHE = {}
_CACHE_TTL = int(os.environ.get('AI_TA_CACHE_TTL', '60'))  # seconds

def _get_cached(symbol: str):
    now = time.time()
    entry = _AI_TA_CACHE.get(symbol)
    if entry:
        ts, val = entry
        if now - ts < _CACHE_TTL:
            return val
        else:
            try:
                del _AI_TA_CACHE[symbol]
            except Exception:
                pass
    return None

def _set_cached(symbol: str, val):
    _AI_TA_CACHE[symbol] = (time.time(), val)

@bp.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or '').strip()
    if not symbol:
        return jsonify({'error': 'symbol is required'}), 400
    out_path = str(MODELS_DIR / f"{symbol.replace('/', '_')}.pkl")
    try:
        meta = train_and_save_model(symbol, out_path, model_type='extra_trees')
        return jsonify({'ok': True, 'meta': meta}), 200
    except Exception as e:
        # Do not return raw exception to client; log and return friendly message
        return jsonify({'ok': False, 'error': 'Training failed. See server logs for details.'}), 500

@bp.route('/ai_ta', methods=['GET'])
def get_ai_ta():
    symbol = (request.args.get('symbol') or '').strip()
    if not symbol:
        return jsonify({'error': 'symbol is required'}), 400
    # check cache
    cached = _get_cached(symbol)
    if cached is not None:
        return jsonify(cached), 200
    model_path = str(MODELS_DIR / f"{symbol.replace('/', '_')}.pkl")
    try:
        res = predict_score_for_symbol(symbol, model_path=model_path)
        _set_cached(symbol, res)
        return jsonify(res), 200
    except Exception as e:
        # friendly error
        return jsonify({'error': 'Analysis temporarily unavailable'}), 500

@bp.route('/ai_ta_bulk', methods=['POST'])
def ai_ta_bulk():
    """POST /ai_ta_bulk
    Body: { "symbols": ["ASIANPAINT.NS", "INFY.NS"] }
    Returns: { results: { symbol: { score,label,features,indicators, model_type } } }
    """
    data = request.get_json(silent=True) or {}
    symbols = data.get('symbols') or data.get('tickers') or []
    if isinstance(symbols, str):
        # allow comma-separated string
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]

    if not symbols or not isinstance(symbols, (list, tuple)):
        return jsonify({'error': 'symbols is required (array)'}), 400

    out = {}
    for sym in symbols:
        try:
            sym_clean = (sym or '').strip()
            # check cache first
            cached = _get_cached(sym_clean)
            if cached is not None:
                out[sym_clean] = cached
                continue

            model_path = str(MODELS_DIR / f"{sym_clean.replace('/', '_')}.pkl")
            # predict_score_for_symbol will fall back to heuristic if model missing
            res = predict_score_for_symbol(sym_clean, model_path=model_path)
            _set_cached(sym_clean, res)
            out[sym_clean] = res
        except Exception:
            # mask raw error
            out[sym] = {'error': 'Analysis temporarily unavailable'}

    return jsonify({'results': out}), 200


# ---------------------------------------------------------------------------
# Comprehensive AI Analysis: TA + FA + News Sentiment combined
# ---------------------------------------------------------------------------
_AI_ANALYSIS_CACHE: dict = {}
_AI_ANALYSIS_TTL = int(os.environ.get('AI_ANALYSIS_CACHE_TTL', '120'))


def _safe_float(v, default=None):
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return round(f, 4)
    except Exception:
        return default


def _safe_round(v, decimals=2):
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None


@bp.route('/ai_analysis', methods=['POST'])
def ai_analysis():
    """POST /ai_analysis
    Body: { "symbol": "RELIANCE.NS" }
    Returns comprehensive analysis combining TA model + fundamental + news sentiment.
    """
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or '').strip()
    if not symbol:
        return jsonify({'error': 'symbol is required'}), 400

    # check cache
    now = time.time()
    cached = _AI_ANALYSIS_CACHE.get(symbol)
    if cached:
        ts, val = cached
        if now - ts < _AI_ANALYSIS_TTL:
            return jsonify(val), 200

    result = {
        'symbol': symbol,
        'ta': None,
        'fa': None,
        'news': None,
        'forecast': None,
        'combined': None,
    }

    # --- 1. Technical Analysis (AI model) ---
    try:
        model_path = str(MODELS_DIR / f"{symbol.replace('/', '_')}.pkl")
        ta_res = predict_score_for_symbol(symbol, model_path=model_path)
        result['ta'] = ta_res
    except Exception:
        result['ta'] = {'error': 'TA analysis unavailable', 'score': 0, 'label': 'HOLD'}

    # --- 2. Fundamental Analysis ---
    try:
        from fundamental import get_fundamental_details
        fa_res = get_fundamental_details(symbol)
        # extract key metrics for frontend
        fa_score = _safe_float(fa_res.get('overall_fa_score', 0), 0)
        investor_metrics = fa_res.get('Investor Insight Metrics', {})
        result['fa'] = {
            'score': fa_score,
            'max_score': 10,
            'signal': 'Good' if fa_score >= 7 else ('Fair' if fa_score >= 4 else 'Weak'),
            'metrics': {
                'eps': _safe_round(investor_metrics.get('EPS')),
                'pe_ratio': _safe_round(investor_metrics.get('P/E Ratio')),
                'revenue_growth': _safe_round(investor_metrics.get('Revenue Growth'), 4),
                'debt_to_equity': _safe_round(investor_metrics.get('Debt-to-Equity Ratio')),
                'earnings_growth_yoy': _safe_round(investor_metrics.get('Earnings Growth(YoY)'), 2),
            },
            'strategy': fa_res.get('fa_strategy', {}),
            'company': fa_res.get('Company Overview', {}),
            'profitability': fa_res.get('Profitability Indicators', {}),
            'risk': fa_res.get('Risk Indicators', {}),
        }
    except Exception:
        traceback.print_exc()
        result['fa'] = {'error': 'FA analysis unavailable', 'score': 0, 'signal': 'N/A'}

    # --- 3. News Sentiment ---
    try:
        from news import get_latest_news_with_sentiment
        import yfinance as yf
        ticker_obj = yf.Ticker(symbol)
        company_name = ticker_obj.info.get('longName', symbol)
        news_res = get_latest_news_with_sentiment(
            company_name, period='7d', max_results=10, language='en', country='US'
        )
        news_score = _safe_float(news_res.get('overall_news_score', 0), 0)
        items = news_res.get('items', [])
        positive = sum(1 for i in items if i.get('sentiment') == 'Positive')
        negative = sum(1 for i in items if i.get('sentiment') == 'Negative')
        neutral = sum(1 for i in items if i.get('sentiment') == 'Neutral')
        result['news'] = {
            'score': news_score,
            'max_score': 5,
            'signal': 'Positive' if news_score >= 3.0 else ('Neutral' if news_score >= 2.0 else 'Negative'),
            'count': len(items),
            'breakdown': {'positive': positive, 'negative': negative, 'neutral': neutral},
            'items': items[:6],
        }
    except Exception:
        traceback.print_exc()
        result['news'] = {'error': 'News analysis unavailable', 'score': 0, 'signal': 'N/A'}

    # --- 4. 15-day High/Low Forecast ---
    try:
        from highlow_forecast import forecast_next_15_high_low
        forecast = forecast_next_15_high_low(ticker=symbol)
        if isinstance(forecast, dict) and 'pred_high' in forecast:
            result['forecast'] = {
                'dates': forecast.get('dates', []),
                'pred_high': [_safe_round(h) for h in forecast.get('pred_high', [])],
                'pred_low': [_safe_round(l) for l in forecast.get('pred_low', [])],
            }
    except Exception:
        traceback.print_exc()
        result['forecast'] = None

    # --- 5. Combined Score ---
    ta_score = _safe_float((result['ta'] or {}).get('score', 0), 0)
    fa_score = _safe_float((result['fa'] or {}).get('score', 0), 0)
    news_score = _safe_float((result['news'] or {}).get('score', 0), 0)

    # Weighted combination: TA 60%, FA 25%, News 15%
    # Normalize: TA is 0-100, FA is 0-10 (scale to 100), News is 0-5 (scale to 100)
    ta_norm = min(ta_score, 100)
    fa_norm = min((fa_score / 10) * 100, 100)
    news_norm = min((news_score / 5) * 100, 100)
    combined = round(ta_norm * 0.60 + fa_norm * 0.25 + news_norm * 0.15, 2)

    if combined >= 70:
        combined_signal = 'BUY'
    elif combined >= 45:
        combined_signal = 'HOLD'
    else:
        combined_signal = 'SELL'

    if combined >= 80:
        confidence = 'High'
    elif combined >= 60:
        confidence = 'Moderate'
    elif combined >= 40:
        confidence = 'Low'
    else:
        confidence = 'Very Low'

    result['combined'] = {
        'score': combined,
        'signal': combined_signal,
        'confidence': confidence,
        'weights': {'ta': 0.60, 'fa': 0.25, 'news': 0.15},
        'normalized': {'ta': round(ta_norm, 2), 'fa': round(fa_norm, 2), 'news': round(news_norm, 2)},
    }

    _AI_ANALYSIS_CACHE[symbol] = (time.time(), result)
    return jsonify(result), 200


@bp.route('/ai_analysis_bulk', methods=['POST'])
def ai_analysis_bulk():
    """POST /ai_analysis_bulk
    Body: { "symbols": ["RELIANCE.NS", "TCS.NS"] }
    Returns quick TA-only bulk results (for the stock list overview).
    Full analysis should be requested per-symbol via /ai_analysis.
    """
    data = request.get_json(silent=True) or {}
    symbols = data.get('symbols') or []
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    if not symbols:
        return jsonify({'error': 'symbols is required'}), 400

    out = {}
    for sym in symbols:
        try:
            sym_clean = (sym or '').strip()
            cached = _get_cached(sym_clean)
            if cached is not None:
                out[sym_clean] = cached
                continue
            model_path = str(MODELS_DIR / f"{sym_clean.replace('/', '_')}.pkl")
            res = predict_score_for_symbol(sym_clean, model_path=model_path)
            _set_cached(sym_clean, res)
            out[sym_clean] = res
        except Exception:
            out[sym] = {'error': 'Analysis temporarily unavailable'}

    return jsonify({'results': out}), 200


def _run_single_full_analysis(symbol):
    """Run full TA + FA + News analysis for a single symbol (no forecast for speed).
    Returns the analysis result dict."""
    result = {
        'symbol': symbol,
        'ta': None,
        'fa': None,
        'news': None,
        'forecast': None,
        'combined': None,
    }

    # --- 1. Technical Analysis ---
    try:
        model_path = str(MODELS_DIR / f"{symbol.replace('/', '_')}.pkl")
        ta_res = predict_score_for_symbol(symbol, model_path=model_path)
        result['ta'] = ta_res
    except Exception:
        result['ta'] = {'error': 'TA analysis unavailable', 'score': 0, 'label': 'HOLD'}

    # --- 2. Fundamental Analysis ---
    try:
        from fundamental import get_fundamental_details
        fa_res = get_fundamental_details(symbol)
        fa_score = _safe_float(fa_res.get('overall_fa_score', 0), 0)
        investor_metrics = fa_res.get('Investor Insight Metrics', {})
        result['fa'] = {
            'score': fa_score,
            'max_score': 10,
            'signal': 'Good' if fa_score >= 7 else ('Fair' if fa_score >= 4 else 'Weak'),
            'metrics': {
                'eps': _safe_round(investor_metrics.get('EPS')),
                'pe_ratio': _safe_round(investor_metrics.get('P/E Ratio')),
                'revenue_growth': _safe_round(investor_metrics.get('Revenue Growth'), 4),
                'debt_to_equity': _safe_round(investor_metrics.get('Debt-to-Equity Ratio')),
                'earnings_growth_yoy': _safe_round(investor_metrics.get('Earnings Growth(YoY)'), 2),
            },
            'strategy': fa_res.get('fa_strategy', {}),
            'company': fa_res.get('Company Overview', {}),
            'profitability': fa_res.get('Profitability Indicators', {}),
            'risk': fa_res.get('Risk Indicators', {}),
        }
    except Exception:
        result['fa'] = {'error': 'FA analysis unavailable', 'score': 0, 'signal': 'N/A'}

    # --- 3. News Sentiment ---
    try:
        from news import get_latest_news_with_sentiment
        import yfinance as yf
        ticker_obj = yf.Ticker(symbol)
        company_name = ticker_obj.info.get('longName', symbol)
        news_res = get_latest_news_with_sentiment(
            company_name, period='7d', max_results=10, language='en', country='US'
        )
        news_score = _safe_float(news_res.get('overall_news_score', 0), 0)
        items = news_res.get('items', [])
        positive = sum(1 for i in items if i.get('sentiment') == 'Positive')
        negative = sum(1 for i in items if i.get('sentiment') == 'Negative')
        neutral = sum(1 for i in items if i.get('sentiment') == 'Neutral')
        result['news'] = {
            'score': news_score,
            'max_score': 5,
            'signal': 'Positive' if news_score >= 3.0 else ('Neutral' if news_score >= 2.0 else 'Negative'),
            'count': len(items),
            'breakdown': {'positive': positive, 'negative': negative, 'neutral': neutral},
            'items': items[:6],
        }
    except Exception:
        result['news'] = {'error': 'News analysis unavailable', 'score': 0, 'signal': 'N/A'}

    # --- 4. Combined Score ---
    ta_score = _safe_float((result['ta'] or {}).get('score', 0), 0)
    fa_score = _safe_float((result['fa'] or {}).get('score', 0), 0)
    news_score = _safe_float((result['news'] or {}).get('score', 0), 0)

    ta_norm = min(ta_score, 100)
    fa_norm = min((fa_score / 10) * 100, 100)
    news_norm = min((news_score / 5) * 100, 100)
    combined = round(ta_norm * 0.60 + fa_norm * 0.25 + news_norm * 0.15, 2)

    if combined >= 70:
        combined_signal = 'BUY'
    elif combined >= 45:
        combined_signal = 'HOLD'
    else:
        combined_signal = 'SELL'

    if combined >= 80:
        confidence = 'High'
    elif combined >= 60:
        confidence = 'Moderate'
    elif combined >= 40:
        confidence = 'Low'
    else:
        confidence = 'Very Low'

    result['combined'] = {
        'score': combined,
        'signal': combined_signal,
        'confidence': confidence,
        'weights': {'ta': 0.60, 'fa': 0.25, 'news': 0.15},
        'normalized': {'ta': round(ta_norm, 2), 'fa': round(fa_norm, 2), 'news': round(news_norm, 2)},
    }

    # Cache the result
    _AI_ANALYSIS_CACHE[symbol] = (time.time(), result)
    return result


@bp.route('/ai_analysis_all', methods=['POST'])
def ai_analysis_all():
    """POST /ai_analysis_all
    Body: { "symbols": ["RELIANCE.NS", "TCS.NS", ...] }
    Runs full TA + FA + News analysis for ALL symbols (no forecast for speed).
    Returns: { "results": { "RELIANCE.NS": {...}, "TCS.NS": {...} } }
    """
    data = request.get_json(silent=True) or {}
    symbols = data.get('symbols') or []
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    if not symbols:
        return jsonify({'error': 'symbols is required'}), 400

    out = {}
    for sym in symbols:
        sym_clean = (sym or '').strip()
        if not sym_clean:
            continue

        # Check full analysis cache first
        now = time.time()
        cached = _AI_ANALYSIS_CACHE.get(sym_clean)
        if cached:
            ts, val = cached
            if now - ts < _AI_ANALYSIS_TTL:
                out[sym_clean] = val
                continue

        try:
            result = _run_single_full_analysis(sym_clean)
            out[sym_clean] = result
        except Exception:
            traceback.print_exc()
            out[sym_clean] = {
                'symbol': sym_clean,
                'error': 'Analysis failed',
                'ta': {'score': 0, 'label': 'HOLD'},
                'fa': {'score': 0, 'signal': 'N/A'},
                'news': {'score': 0, 'signal': 'N/A'},
                'combined': {
                    'score': 0, 'signal': 'HOLD', 'confidence': 'Very Low',
                    'weights': {'ta': 0.60, 'fa': 0.25, 'news': 0.15},
                    'normalized': {'ta': 0, 'fa': 0, 'news': 0},
                },
            }

    return jsonify({'results': out}), 200
