"""ai_ta_model.py

Train / save / predict AI TA score based on TA features computed from historical OHLCV.

- train_and_save_model(symbol, out_path): downloads 1y data, computes features for rolling dates, builds binary target (future max return > threshold), trains a tree ensemble (ExtraTrees by default) and saves model + feature names + weights.
- predict_score_for_symbol(symbol, model_path): downloads recent data (last 14 days), computes features, loads model and returns score 0-100 (probability * 100) plus per-indicator breakdown and weights.

If sklearn is not available the module falls back to a SimpleHeuristicModel which produces a deterministic score.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import os
import logging

from market_data import get_ohlcv
import ta_indicators as ta

# optional sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    _SKLEARN = True
except Exception:
    _SKLEARN = False


_logger = logging.getLogger(__name__)


class SimpleHeuristicModel:
    def predict_score(self, features: Dict[str, float]) -> float:
        score = 50.0
        score += (features.get('rsi', 50) - 50) * 0.4
        score += np.tanh(features.get('macd_hist', 0.0)) * 12
        score += (features.get('adx', 20) - 20) * 0.2
        # clamp
        return float(np.clip(score, 0, 100))


def _get_last_value(obj, default: float = 0.0) -> float:
    """Safely extract last numeric value from a pandas Series/DataFrame column or scalar.
    Returns default on failure.
    """
    try:
        # pandas Series / Index-like
        if hasattr(obj, 'iloc'):
            if len(obj) > 0:
                v = obj.iloc[-1]
                return float(v) if v is not None and not (pd.isna(v)) else default
            return default
        # numpy array or scalar
        if np.isscalar(obj):
            return float(obj)
        try:
            arr = np.asarray(obj)
            if arr.size:
                return float(arr[-1])
        except Exception:
            pass
    except Exception:
        pass
    return float(default)


def build_feature_vector(df: pd.DataFrame, last_n_days: int = 14) -> Dict[str, float]:
    """Compute numeric feature vector from an OHLCV DataFrame for the last available rows.

    For runtime predictions we only use the last `last_n_days` rows to compute indicators (deterministic short-term swing trading).
    """
    # ensure we operate on a copy and only last_n_days
    if df is None or df.empty:
        raise RuntimeError('Empty OHLCV data')
    if last_n_days is not None and last_n_days > 0:
        df = df.tail(last_n_days).copy()

    close = df['Close']
    features: Dict[str, float] = {}

    # RSI
    try:
        rsi_series = ta.rsi(close, period=14)
        features['rsi'] = _get_last_value(rsi_series, 50.0)
    except Exception:
        features['rsi'] = 50.0

    # MACD hist
    try:
        macd_df = ta.macd(close)
        features['macd_hist'] = _get_last_value(macd_df['hist'], 0.0)
        features['macd'] = _get_last_value(macd_df['macd'], 0.0)
    except Exception:
        features['macd_hist'] = 0.0
        features['macd'] = 0.0

    # ATR
    try:
        atr_series = ta.atr(df)
        features['atr'] = _get_last_value(atr_series, 0.0)
    except Exception:
        features['atr'] = 0.0

    # EMA spread (12-26)
    try:
        ema12 = ta.ema(close, span=12)
        ema26 = ta.ema(close, span=26)
        features['ema_spread'] = _get_last_value(ema12, 0.0) - _get_last_value(ema26, 0.0)
    except Exception:
        features['ema_spread'] = 0.0

    # ADX
    try:
        adx_series = ta.adx(df)
        features['adx'] = _get_last_value(adx_series, 20.0)
    except Exception:
        features['adx'] = 20.0

    # Bollinger distance from middle (z-score-like)
    try:
        bb = ta.bollinger_bands(close)
        if not bb['middle'].isna().all():
            middle = _get_last_value(bb['middle'], 0.0)
            upper = _get_last_value(bb['upper'], middle)
            lower = _get_last_value(bb['lower'], middle)
            last = _get_last_value(close, middle)
            width = max(upper - lower, 1e-8)
            features['bb_dist'] = float((last - middle) / width)
        else:
            features['bb_dist'] = 0.0
    except Exception:
        features['bb_dist'] = 0.0

    # Fibonacci proximity to 0.618 level
    try:
        fib = ta.fibo_levels(close, lookback=60)
        f618 = fib.get('0.618', None)
        if f618 is not None and f618 != 0:
            features['fibo_dist'] = float((_get_last_value(close, 0.0) - f618) / f618)
        else:
            features['fibo_dist'] = 0.0
    except Exception:
        features['fibo_dist'] = 0.0

    # SR distance
    try:
        sr = ta.sr_levels(df, lookback=60)
        res = sr.get('resistance', None)
        sup = sr.get('support', None)
        if res is None or sup is None:
            features['dist_to_res'] = 0.0
            features['dist_to_sup'] = 0.0
        else:
            last_close = _get_last_value(close, 0.0)
            features['dist_to_res'] = float((res - last_close) / max(1e-8, res))
            features['dist_to_sup'] = float((last_close - sup) / max(1e-8, sup))
    except Exception:
        features['dist_to_res'] = 0.0
        features['dist_to_sup'] = 0.0

    # PA_MS: encode as 1 buy, -1 sell, 0 hold
    try:
        pa = ta.pa_ms_signal(df)
        features['pa_ms'] = 1.0 if pa == 'buy' else (-1.0 if pa == 'sell' else 0.0)
    except Exception:
        features['pa_ms'] = 0.0

    return features


def make_dataset_from_symbol(symbol: str, lookback_days: int = 120, horizon: int = 5, threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
    """Build features and binary target from historical data for a single symbol.

    - lookback_days: how many days of history used for features (per sample we use available history up to that date)
    - horizon: days ahead to look for target
    - threshold: minimum future return to label positive (e.g., 0.02 => 2%)
    Returns X dataframe and y series aligned.
    """
    df = get_ohlcv(symbol, days=365)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol}")

    X_list = []
    y_list = []

    # for each candidate date i where we have at least `lookback_days` historical rows and `horizon` future rows
    for i in range(lookback_days, len(df) - horizon):
        window = df.iloc[i - lookback_days + 1:i + 1].copy()
        # compute features using window
        try:
            feats = build_feature_vector(window, last_n_days=None)  # use full window when building dataset
        except Exception:
            continue
        # compute target using future prices
        current_close = df['Close'].iloc[i]
        future_max = df['Close'].iloc[i + 1:i + 1 + horizon].max()
        future_return = (future_max - current_close) / max(1e-9, current_close)
        target = 1 if future_return >= threshold else 0
        X_list.append(feats)
        y_list.append(target)

    # Diagnostic logging and guards before constructing DataFrame
    try:
        _logger.info('make_dataset_from_symbol: built X_list type=%s len=%d', type(X_list), len(X_list))
        # Print small sample for debugging (avoid huge logs)
        if len(X_list) > 0:
            sample = X_list[:3]
            _logger.debug('make_dataset_from_symbol: sample X_list (up to 3): %s', sample)
    except Exception:
        pass

    # If user accidentally produced a single dict instead of a list, wrap it
    if isinstance(X_list, dict):
        _logger.warning('make_dataset_from_symbol: X_list is dict, wrapping into list to avoid pandas error')
        X_list = [X_list]

    if not X_list:
        raise RuntimeError('Not enough samples to build dataset')

    # Ensure all feature entries are plain dicts with scalar floats
    sanitized = []
    for item in X_list:
        if isinstance(item, dict):
            clean = {}
            for k, v in item.items():
                try:
                    clean[k] = float(v) if v is not None and not (pd.isna(v)) else 0.0
                except Exception:
                    clean[k] = 0.0
            sanitized.append(clean)
        else:
            # unexpected item type -> try to coerce
            try:
                sanitized.append(dict(item))
            except Exception:
                sanitized.append({})

    X = pd.DataFrame(sanitized)
    y = pd.Series(y_list)
    return X, y


def train_and_save_model(symbol: str, out_path: str = 'ai_ta_model.pkl', lookback_days: int = 120, horizon: int = 5, threshold: float = 0.02, model_type: str = 'extra_trees') -> Dict[str, Any]:
    """Train model for given symbol and persist to out_path. Returns metadata including scores and normalized feature weights.

    model_type: 'extra_trees' (default) or 'random_forest'
    """
    if not _SKLEARN:
        raise RuntimeError('sklearn is required to train model. Install scikit-learn and joblib.')

    X, y = make_dataset_from_symbol(symbol, lookback_days=lookback_days, horizon=horizon, threshold=threshold)

    # simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42, stratify=y)

    clf = None
    mt = (model_type or '').lower()
    if mt in ('extra_trees', 'extratrees', 'extra-trees'):
        clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)

    # compute normalized feature importances and store in bundle
    try:
        importances = getattr(clf, 'feature_importances_', None)
        if importances is None:
            importances = np.zeros(len(X.columns), dtype=float)
        # ensure non-negative
        importances = np.maximum(importances, 0.0)
        total = float(np.sum(importances))
        if total <= 0:
            # fallback to uniform
            weights = {f: 1.0 / max(1, len(X.columns)) for f in list(X.columns)}
        else:
            weights = {f: float(imp) / total for f, imp in zip(list(X.columns), importances)}
    except Exception:
        weights = {f: 1.0 / max(1, len(X.columns)) for f in list(X.columns)}

    # save model and feature columns and model type and weights
    model_bundle = {'model': clf, 'features': list(X.columns), 'model_type': mt, 'weights': weights}
    joblib.dump(model_bundle, out_path)

    # compute simple validation metrics
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == y_test.values).mean())

    return {'path': out_path, 'accuracy': acc, 'n_samples': len(X), 'features': list(X.columns), 'model_type': mt, 'weights': weights}


def load_model(path: str = 'ai_ta_model.pkl'):
    if not _SKLEARN:
        return None
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    return bundle


def _indicator_mapping_and_score(feats: Dict[str, float]) -> List[Dict[str, Any]]:
    """Produce per-indicator signal (BUY/HOLD/SELL) and a deterministic numeric score 0-10 based on feature values.

    The score mapping is coarse but deterministic and explainable. BUY -> 7-10, HOLD -> 4-6, SELL -> 0-3.
    """
    inds = []
    # helper to map signal to base numeric score
    def score_for_signal(sig: str) -> float:
        if sig == 'BUY':
            return 8.0
        if sig == 'HOLD':
            return 5.0
        return 2.0

    # RSI
    rsi = float(feats.get('rsi', 50.0))
    rsi_sig = 'BUY' if rsi > 60 else ('SELL' if rsi < 40 else 'HOLD')
    inds.append({'key': 'RSI', 'signal': rsi_sig, 'score': score_for_signal(rsi_sig)})

    # MACD (hist)
    macd_hist = float(feats.get('macd_hist', 0.0))
    macd_sig = 'BUY' if macd_hist > 0 else ('SELL' if macd_hist < 0 else 'HOLD')
    inds.append({'key': 'MACD', 'signal': macd_sig, 'score': score_for_signal(macd_sig)})

    # ATR - we treat ATR as volatility; neutral signal
    atr_sig = 'HOLD'
    inds.append({'key': 'ATR', 'signal': atr_sig, 'score': score_for_signal(atr_sig)})

    # EMA spread
    ema_spread = float(feats.get('ema_spread', 0.0))
    ema_sig = 'BUY' if ema_spread > 0 else ('SELL' if ema_spread < 0 else 'HOLD')
    inds.append({'key': 'EMA', 'signal': ema_sig, 'score': score_for_signal(ema_sig)})

    # ADX
    adx = float(feats.get('adx', 20.0))
    adx_sig = 'BUY' if adx > 25 else 'HOLD'
    inds.append({'key': 'ADX', 'signal': adx_sig, 'score': score_for_signal(adx_sig)})

    # Fibo
    fibo_dist = float(feats.get('fibo_dist', 0.0))
    fibo_sig = 'BUY' if abs(fibo_dist) < 0.02 else 'HOLD'
    inds.append({'key': 'Fibo', 'signal': fibo_sig, 'score': score_for_signal(fibo_sig)})

    # BB
    bb_dist = float(feats.get('bb_dist', 0.0))
    if abs(bb_dist) < 1:
        bb_sig = 'HOLD'
    else:
        bb_sig = 'BUY' if bb_dist > 0 else 'SELL'
    inds.append({'key': 'BB', 'signal': bb_sig, 'score': score_for_signal(bb_sig)})

    # SR (support/resistance) - use dist_to_sup/dist_to_res
    dist_to_res = float(feats.get('dist_to_res', 0.0))
    dist_to_sup = float(feats.get('dist_to_sup', 0.0))
    # if close to support -> BUY, close to resistance -> SELL, else HOLD
    if dist_to_sup >= 0 and dist_to_sup < 0.02:
        sr_sig = 'BUY'
    elif dist_to_res >= 0 and dist_to_res < 0.02:
        sr_sig = 'SELL'
    else:
        sr_sig = 'HOLD'
    inds.append({'key': 'SR', 'signal': sr_sig, 'score': score_for_signal(sr_sig)})

    # PA_MS
    pa_ms = float(feats.get('pa_ms', 0.0))
    pa_sig = 'BUY' if pa_ms == 1.0 else ('SELL' if pa_ms == -1.0 else 'HOLD')
    inds.append({'key': 'PA_MS', 'signal': pa_sig, 'score': score_for_signal(pa_sig)})

    return inds


def predict_score_for_symbol(symbol: str, model_path: str = 'ai_ta_model.pkl') -> Dict[str, Any]:
    """Compute AI TA score for a symbol using trained model if available, else use heuristic.
    Returns dict with keys: score (0-100), label (BUY/HOLD/SELL), features, indicators (per-indicator), model_type.
    """
    df = get_ohlcv(symbol, days=60)
    if df is None or df.empty:
        # consistent friendly error
        raise RuntimeError('No market data available')

    # compute features using last 14 days only (runtime rule)
    feats = build_feature_vector(df, last_n_days=14)

    # default uniform weights and model_type None
    weights = None
    model_type = None

    if _SKLEARN and os.path.exists(model_path):
        bundle = load_model(model_path)
        if bundle is not None:
            model = bundle['model']
            feature_order = bundle.get('features', [])
            model_type = bundle.get('model_type')
            weights = bundle.get('weights', None)

            # ensure feature_order exists and build Xrow accordingly
            try:
                Xrow = np.array([feats.get(f, 0.0) for f in feature_order]).reshape(1, -1)
                try:
                    prob = float(model.predict_proba(Xrow)[0, 1])
                except Exception:
                    pred = int(model.predict(Xrow)[0])
                    prob = float(pred)
                # base score from model probability
                base_score = round(prob * 100, 1)
            except Exception:
                # fallback to heuristic if unexpected model error
                base_score = SimpleHeuristicModel().predict_score(feats)
                prob = base_score / 100.0
        else:
            base_score = SimpleHeuristicModel().predict_score(feats)
            prob = base_score / 100.0
    else:
        base_score = SimpleHeuristicModel().predict_score(feats)
        prob = base_score / 100.0

    # compute per-indicator signals and scores
    try:
        indicators = _indicator_mapping_and_score(feats)
    except Exception:
        indicators = []

    # derive indicator weights: map model feature weights (per-feature) to indicators (some indicators are composed of multiple features)
    # define mapping from indicator keys to list of feature names
    indicator_to_features = {
        'RSI': ['rsi'],
        'MACD': ['macd_hist'],
        'ATR': ['atr'],
        'EMA': ['ema_spread'],
        'ADX': ['adx'],
        'Fibo': ['fibo_dist'],
        'BB': ['bb_dist'],
        'SR': ['dist_to_res', 'dist_to_sup'],
        'PA_MS': ['pa_ms']
    }

    # if no model weights, use uniform weights across indicators
    if not weights:
        num_inds = len(indicator_to_features)
        ind_weights = {k: 1.0 / max(1, num_inds) for k in indicator_to_features.keys()}
    else:
        # sum feature weights per indicator
        ind_weights = {}
        for ind, feats_list in indicator_to_features.items():
            w_sum = 0.0
            for f in feats_list:
                w_sum += float(weights.get(f, 0.0))
            ind_weights[ind] = float(w_sum)
        # normalize to sum 1 (avoid zero)
        total_w = sum(ind_weights.values())
        if total_w <= 0:
            num_inds = len(ind_weights)
            ind_weights = {k: 1.0 / max(1, num_inds) for k in ind_weights.keys()}
        else:
            ind_weights = {k: float(v) / total_w for k, v in ind_weights.items()}

    # attach weights to indicators list and compute weighted sum
    weighted_sum = 0.0
    for ind in indicators:
        key = ind.get('key')
        score_0_10 = float(ind.get('score', 0.0))
        w = float(ind_weights.get(key, 0.0))
        ind['weight'] = w
        weighted_sum += score_0_10 * w

    # weighted_sum is in 0-10 range (since score 0-10 and weights sum to 1). Convert to 0-100
    final_score = float(np.clip(weighted_sum * 10.0, 0.0, 100.0))

    # final label mapping per provided thresholds
    label = 'BUY' if final_score >= 70 else ('HOLD' if final_score >= 45 else 'SELL')

    return {'score': round(final_score, 1), 'label': label, 'features': feats, 'indicators': indicators, 'model_type': model_type}
