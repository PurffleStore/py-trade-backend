import numpy as np
import pandas as pd
import talib

# Optional ML imports (graceful fallback if scikit-learn is not installed)
try:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    _SKLEARN_AVAILABLE = True
except Exception:
    ExtraTreesRegressor = None
    TimeSeriesSplit = None
    mean_absolute_error = None
    _SKLEARN_AVAILABLE = False

# Optional: HistGradientBoostingRegressor for quantile regression
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HGBR_AVAILABLE = True
except Exception:
    HistGradientBoostingRegressor = None
    _HGBR_AVAILABLE = False

# --------------------- Configuration ---------------------

# Prefer quantile gradient boosting for extreme values (better for High/Low)
_USE_HGBR_QUANTILE = True  # auto-fallback to ExtraTrees when unavailable

# Quantiles for high/low tails (in log-ratio space)
_Q_HIGH = 0.80  # upper-tail for High
_Q_LOW = 0.20   # lower-tail for Low

# Blend ML predictions with TA fallback (in log-return space)
# Set to 0.0 to disable blending
_BLEND_TA_WEIGHT = 0.20

# Log-ratio target winsorization to reduce outlier impact: [q_low, q_high] (ExtraTrees path)
_WINSOR_Q_LOW = 0.005
_WINSOR_Q_HIGH = 0.995

# Exponential recency weighting: larger = faster decay (0.0 to disable)
_RECENCY_DECAY = 0.003  # per-sample step

# ExtraTrees hyperparameters tuned for generalization
_ETR_PARAMS_CV = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=0.6,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)
_ETR_PARAMS_FINAL = dict(
    n_estimators=1200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=0.6,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)

# HistGradientBoosting hyperparameters for quantile regression
_HGBR_PARAMS = dict(
    loss="quantile",
    learning_rate=0.05,
    max_iter=600,
    max_depth=3,
    max_leaf_nodes=31,
    max_bins=255,
    l2_regularization=0.0,
    early_stopping=False,  # avoid random holdout leaking time
    random_state=42,
)

# In-memory per-ticker model cache (no disk I/O)
_MEM_CACHE = {}  # key: ticker.upper(), value: bundle dict

# --------------------- OHLC Utilities ---------------------

def _ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    mapping = {}
    for n in need:
        if n in cols:
            mapping[cols[n]] = n
        else:
            # try MultiIndex column cases from yfinance
            for c in df.columns:
                name = c[0].lower() if isinstance(c, tuple) and len(c) > 0 else str(c).lower()
                if name == n:
                    mapping[c] = n
                    break
    out = df.rename(columns=mapping).copy()
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after normalization: {missing}")
    return out[["open", "high", "low", "close", "volume"]]

# --------------------- Business day helper ---------------------

def _next_business_days(last_date: pd.Timestamp, periods: int, exchange: str = "XNYS") -> pd.DatetimeIndex:
    """
    Return next 'periods' business sessions after last_date.
    Tries exchange calendar via pandas_market_calendars (holidays-aware), fallback to weekdays-only.
    exchange examples: 'XNYS' (NYSE), 'XBOM' (BSE), 'XNAS' (NASDAQ), 'XNSE' (NSE).
    """
    last_date = pd.Timestamp(last_date).tz_localize(None)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar(exchange)
        # buffer long enough to cover holidays
        schedule = cal.schedule(start_date=last_date + pd.Timedelta(days=1),
                                end_date=last_date + pd.Timedelta(days=180))
        sessions = schedule.index.tz_localize(None)
        if len(sessions) >= periods:
            return sessions[:periods]
        # If for some reason not enough sessions, extend with weekday fallback
        needed = periods - len(sessions)
        tail = pd.bdate_range(sessions[-1] + pd.offsets.BDay(1) if len(sessions) else last_date + pd.offsets.BDay(1),
                              periods=needed)
        return sessions.append(tail)
    except Exception:
        # Weekdays-only fallback
        return pd.bdate_range(last_date + pd.offsets.BDay(1), periods=periods)

# --------------------- TA Heuristic (Fallback, No ML) ---------------------

def _last_finite(values: np.ndarray, default: float = np.nan) -> float:
    for x in values[::-1]:
        if np.isfinite(x):
            return float(x)
    return float(default)

def _ta_fallback_forecast(ohlc: pd.DataFrame, horizons: int = 15):
    h = ohlc["high"].astype(float).values
    l = ohlc["low"].astype(float).values
    c = ohlc["close"].astype(float).values

    if len(c) < 60:
        raise ValueError("Not enough history for TA fallback (need >=60 rows).")

    base_close = _last_finite(ohlc["close"].replace(0.0, np.nan).values)
    if not np.isfinite(base_close) or base_close <= 0:
        raise ValueError("Invalid last close after cleaning.")

    atr14 = talib.ATR(h, l, c, timeperiod=14)
    atr_last = _last_finite(atr14, default=np.nan)
    atr_pct = (atr_last / base_close) if np.isfinite(atr_last) and base_close > 0 else np.nan

    ema20 = talib.EMA(c, timeperiod=20)
    ema50 = talib.EMA(c, timeperiod=50)
    ema20_last = _last_finite(ema20, default=np.nan)
    ema50_last = _last_finite(ema50, default=np.nan)

    trend_strength = 0.0
    if np.isfinite(ema20_last) and np.isfinite(ema50_last) and ema50_last > 0:
        trend_strength = np.clip(ema20_last / ema50_last - 1.0, -0.05, 0.05)
    ema20_slope = 0.0
    if len(ema20) >= 2 and np.isfinite(ema20[-1]) and np.isfinite(ema20[-2]) and ema20[-2] > 0:
        ema20_slope = np.clip((ema20[-1] / ema20[-2]) - 1.0, -0.05, 0.05)

    adx14 = talib.ADX(h, l, c, timeperiod=14)
    adx = _last_finite(adx14, default=20.0) / 100.0
    adx = float(np.clip(adx, 0.0, 1.0))

    rsi14 = talib.RSI(c, timeperiod=14)
    rsi = _last_finite(rsi14, default=50.0)
    tilt = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

    logret = np.diff(np.log(np.maximum(c, 1e-12)))
    if len(logret) >= 20 and np.isfinite(logret[-20:]).sum() >= 10:
        sigma20 = float(pd.Series(logret).rolling(20).std().iloc[-1])
    else:
        sigma20 = float(np.nan)

    components = []
    if np.isfinite(sigma20):
        components.append(sigma20)
    if np.isfinite(atr_pct):
        components.append(atr_pct)
    daily_vol = 0.0
    if components:
        daily_vol = 0.6 * components[0] + (0.4 * components[1] if len(components) > 1 else 0.0)
    daily_vol = float(np.clip(daily_vol if np.isfinite(daily_vol) else 0.02, 0.004, 0.08))

    drift_per_day = float(np.clip(0.5 * trend_strength + 0.5 * ema20_slope, -0.02, 0.02))

    up_weight = 1.0 - 0.3 * tilt
    dn_weight = 1.0 + 0.3 * tilt
    up_weight = float(np.clip(up_weight, 0.5, 1.5))
    dn_weight = float(np.clip(dn_weight, 0.5, 1.5))
    trend_amp = 0.75 + 0.5 * adx

    pred_high, pred_low = [], []
    for k in range(1, horizons + 1):
        amp = daily_vol * np.sqrt(k) * trend_amp
        drift = drift_per_day * k
        up_move = amp * up_weight
        dn_move = amp * dn_weight
        hi = base_close * (1.0 + drift + up_move)
        lo = base_close * (1.0 + drift - dn_move)
        hi = max(0.0, hi)
        lo = max(0.0, lo)
        if lo > hi:
            lo, hi = hi, lo
        pred_high.append(hi)
        pred_low.append(lo)

    return base_close, np.array(pred_high), np.array(pred_low)

# --------------------- Feature Engineering for ML ---------------------

def _compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_ohlc_columns(df).copy()
    o, h, l, c, v = [df[k].astype(float).values for k in ("open", "high", "low", "close", "volume")]

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    df_feat = pd.DataFrame(index=df.index)

    # Basic price features
    df_feat["ret_1"] = close.pct_change(1)
    df_feat["logret_1"] = np.log(close.replace(0.0, np.nan)).diff(1)
    df_feat["ret_5"] = close.pct_change(5)
    df_feat["ret_10"] = close.pct_change(10)
    df_feat["roll_mean_5"] = close.rolling(5).mean() / close - 1.0
    df_feat["roll_mean_20"] = close.rolling(20).mean() / close - 1.0
    df_feat["roll_std_10"] = close.pct_change().rolling(10).std()
    df_feat["range_pct"] = (high - low) / close.replace(0.0, np.nan)

    # Candle features (normalized)
    with np.errstate(divide="ignore", invalid="ignore"):
        body = (close - open_) / close
        upper_shadow = (high - np.maximum(close, open_)) / close
        lower_shadow = (np.minimum(close, open_) - low) / close
    df_feat["candle_body"] = body
    df_feat["candle_upper"] = upper_shadow
    df_feat["candle_lower"] = lower_shadow
    df_feat["gap_open"] = open_.shift(0) / close.shift(1) - 1.0

    # EMAs and distances
    ema5 = talib.EMA(close.values, timeperiod=5)
    ema20 = talib.EMA(close.values, timeperiod=20)
    ema50 = talib.EMA(close.values, timeperiod=50)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["ema5_dist"] = (ema5 / close.values) - 1.0
        df_feat["ema20_dist"] = (ema20 / close.values) - 1.0
        df_feat["ema50_dist"] = (ema50 / close.values) - 1.0
        # EMA slopes (1-day change)
        df_feat["ema20_slope"] = (pd.Series(ema20, index=df.index).pct_change(1))

    # RSI family
    df_feat["rsi14"] = talib.RSI(close.values, timeperiod=14) / 100.0
    df_feat["rsi5"] = talib.RSI(close.values, timeperiod=5) / 100.0

    # MACD
    macd, macdsig, macdhist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    df_feat["macd"] = macd
    df_feat["macdsig"] = macdsig
    df_feat["macdhist"] = macdhist

    # Bollinger Bands width
    upper, middle, lower = talib.BBANDS(close.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["bb_width"] = (upper - lower) / middle

    # Volatility/Trend
    atr = talib.ATR(h, l, c, timeperiod=14)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["atr14"] = atr / close.values
    df_feat["adx14"] = talib.ADX(h, l, c, timeperiod=14) / 100.0

    # Additional momentum/oscillators
    df_feat["roc10"] = talib.ROC(close.values, timeperiod=10) / 100.0
    df_feat["cci14"] = talib.CCI(h, l, c, timeperiod=14) / 100.0
    df_feat["mfi14"] = talib.MFI(h, l, c, v, timeperiod=14) / 100.0
    df_feat["willr14"] = talib.WILLR(h, l, c, timeperiod=14) / 100.0  # [-1, 0]

    # Stochastic
    slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df_feat["stoch_k"] = slowk / 100.0
    df_feat["stoch_d"] = slowd / 100.0

    # OBV normalized (robust to missing/flat volume)
    finite_vol = np.isfinite(vol.values)
    if finite_vol.sum() >= max(30, int(0.5 * len(vol))):
        obv = talib.OBV(close.values, vol.values)
        df_feat["obv_z"] = pd.Series(obv, index=df.index).pct_change(5)
    else:
        df_feat["obv_z"] = 0.0

    # Volume z-score and turnover proxies
    vol_roll_mean = vol.rolling(20).mean()
    vol_roll_std = vol.rolling(20).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["vol_z20"] = (vol - vol_roll_mean) / vol_roll_std
        df_feat["turnover_z20"] = ((vol * close) - (vol * close).rolling(20).mean()) / (vol * close).rolling(20).std()

    # Distance to rolling extremes
    roll_max_20 = close.rolling(20).max()
    roll_min_20 = close.rolling(20).min()
    roll_max_55 = close.rolling(55).max()
    roll_min_55 = close.rolling(55).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["dist_max20"] = roll_max_20 / close - 1.0
        df_feat["dist_min20"] = close / roll_min_20 - 1.0
        df_feat["dist_max55"] = roll_max_55 / close - 1.0
        df_feat["dist_min55"] = close / roll_min_55 - 1.0

    # Realized volatility features
    logret = np.log(close.replace(0.0, np.nan)).diff(1)
    df_feat["rv5"] = logret.rolling(5).std()
    df_feat["rv20"] = logret.rolling(20).std()
    df_feat["avg_range5"] = ((high - low) / close.replace(0.0, np.nan)).rolling(5).mean()

    # Calendar (cyclical day-of-week, month-of-year)
    dow = pd.Series(df.index).map(lambda d: d.weekday() if hasattr(d, "weekday") else pd.Timestamp(d).weekday())
    df_feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df_feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    moy = pd.Series(df.index).map(lambda d: (d.month if hasattr(d, "month") else pd.Timestamp(d).month))
    df_feat["moy_sin"] = np.sin(2 * np.pi * (moy.astype(float) - 1.0) / 12.0)
    df_feat["moy_cos"] = np.cos(2 * np.pi * (moy.astype(float) - 1.0) / 12.0)

    # Lags of basic signals
    df_feat["ret_1_lag1"] = df_feat["ret_1"].shift(1)
    df_feat["ret_1_lag2"] = df_feat["ret_1"].shift(2)
    df_feat["range_pct_lag1"] = df_feat["range_pct"].shift(1)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.loc[:, df_feat.notna().any(axis=0)]
    return df_feat

def _clean_features_for_training(feats: pd.DataFrame, warmup: int = 60) -> pd.DataFrame:
    if feats.empty:
        return feats
    clean = feats.copy()
    clean = clean.fillna(method="ffill").fillna(method="bfill")
    if len(clean) > warmup:
        clean = clean.iloc[warmup:]
    clean = clean.dropna()
    return clean

def _winsorize_targets(Y: np.ndarray, horizons: int, q_low: float, q_high: float) -> tuple[np.ndarray, dict]:
    """
    Winsorize concatenated targets Y = [highs(0:h), lows(h:2h)] row-wise using global quantiles.
    Returns clipped Y and thresholds used.
    """
    h = horizons
    Yh = Y[:, :h].ravel()
    Yl = Y[:, h:].ravel()

    lo_h, hi_h = np.quantile(Yh, [q_low, q_high]) if Yh.size else (-np.inf, np.inf)
    lo_l, hi_l = np.quantile(Yl, [q_low, q_high]) if Yl.size else (-np.inf, np.inf)

    Y_clip = Y.copy()
    Y_clip[:, :h] = np.clip(Y_clip[:, :h], lo_h, hi_h)
    Y_clip[:, h:] = np.clip(Y_clip[:, h:], lo_l, hi_l)

    return Y_clip, {"high": (float(lo_h), float(hi_h)), "low": (float(lo_l), float(hi_l))}

def _sample_weights(n: int, decay: float) -> np.ndarray:
    """
    Exponential recency weights. Newer samples get higher weight.
    w_i = exp(-decay * (n-1-i)), i in [0..n-1]
    """
    if decay <= 0 or n <= 0:
        return np.ones(n, dtype=float)
    idx = np.arange(n, dtype=float)
    w = np.exp(-decay * (n - 1 - idx))
    w /= np.average(w)  # normalize to mean 1.0
    return w

def _make_supervised(df: pd.DataFrame, horizons: int = 15):
    """
    Build X, Y for multi-horizon high/low forecast.
    Targets (log-ratio): y_high_h = log(High[t+h]/Close[t]), y_low_h = log(Low[t+h]/Close[t])
    Log transform stabilizes variance and reduces skew.
    """
    ohlc = _ensure_ohlc_columns(df)
    feats = _compute_ta_features(df)
    feat_df = _clean_features_for_training(feats, warmup=60)

    # Align to cleaned feature index
    ohlc = ohlc.loc[feat_df.index]

    highs = ohlc["high"].astype(float).values
    lows = ohlc["low"].astype(float).values
    closes = ohlc["close"].astype(float).values
    X_all = feat_df.values

    n = len(feat_df)
    if n < horizons + 30:
        raise ValueError(f"Not enough rows after feature warm-up for {horizons}-day training. Have: {n}")

    X_list, Y_list = [], []
    for i in range(n - horizons):
        base_c = closes[i]
        if not np.isfinite(base_c) or base_c <= 0:
            continue

        future_highs = highs[i + 1:i + horizons + 1]
        future_lows = lows[i + 1:i + horizons + 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            yh = np.log(np.maximum(future_highs, 1e-12) / base_c)
            yl = np.log(np.maximum(future_lows, 1e-12) / base_c)

        if np.any(~np.isfinite(yh)) or np.any(~np.isfinite(yl)):
            continue

        X_list.append(X_all[i, :])
        Y_list.append(np.concatenate([yh, yl], axis=0))

    X = np.asarray(X_list)
    Y = np.asarray(Y_list)
    if X.size == 0 or Y.size == 0:
        raise ValueError("No valid supervised samples after cleaning. Check data quality (NaNs/zeros).")
    feature_names = feat_df.columns.tolist()
    return X, Y, feature_names, feat_df.index[:len(X)]

def _get_sklearn_version():
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return None

# --------------------- Model Train/Load (In-Memory Only) ---------------------

def train_or_load_highlow_15d(df: pd.DataFrame, ticker: str, horizons: int = 15):
    key = ticker.upper()
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]

    # If sklearn is not available at all, keep TA fallback metadata
    if not _SKLEARN_AVAILABLE:
        bundle = {
            "model": None,
            "feature_names": None,
            "horizons": horizons,
            "trained_rows": int(len(df)),
            "metrics": None,
            "sklearn_version": None,
            "ticker": key,
            "model_path": None,
            "winsor": None,
            "blend_weight": _BLEND_TA_WEIGHT,
            "transform": "logratio",
            "feature_importances": None,
            "algo": "NONE",
        }
        _MEM_CACHE[key] = bundle
        return bundle

    # Build supervised set
    X, Y_raw, feature_names, _ = _make_supervised(df, horizons=horizons)
    sw = _sample_weights(X.shape[0], _RECENCY_DECAY)

    # Prefer quantile gradient boosting if available
    if _USE_HGBR_QUANTILE and _HGBR_AVAILABLE and HistGradientBoostingRegressor is not None:
        q_models_high, q_models_low = [], []
        for k in range(horizons):
            # High models (upper quantile)
            mh = HistGradientBoostingRegressor(**_HGBR_PARAMS, quantile=_Q_HIGH)
            mh.fit(X, Y_raw[:, k], sample_weight=sw)
            q_models_high.append(mh)

            # Low models (lower quantile)
            ml = HistGradientBoostingRegressor(**_HGBR_PARAMS, quantile=_Q_LOW)
            ml.fit(X, Y_raw[:, horizons + k], sample_weight=sw)
            q_models_low.append(ml)

        bundle = {
            "model": None,  # not used in quantile path
            "q_models_high": q_models_high,
            "q_models_low": q_models_low,
            "feature_names": feature_names,
            "horizons": horizons,
            "trained_rows": int(X.shape[0]),
            "metrics": None,  # optional: add custom CV if desired
            "sklearn_version": _get_sklearn_version(),
            "ticker": key,
            "model_path": None,
            "winsor": None,
            "blend_weight": _BLEND_TA_WEIGHT,
            "transform": "logratio",
            "feature_importances": None,
            "algo": f"HGBR_QUANTILE(high={_Q_HIGH}, low={_Q_LOW})",
        }
        _MEM_CACHE[key] = bundle
        return bundle

    # Else fall back to ExtraTrees mean-regression (existing path)
    Y_clip, winsor_info = _winsorize_targets(Y_raw, horizons, _WINSOR_Q_LOW, _WINSOR_Q_HIGH)

    fold_metrics = []
    feature_importances = None

    if TimeSeriesSplit is not None:
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xvl = X[train_idx], X[val_idx]
            Ytr_clipped = Y_clip[train_idx]
            Yvl_true = Y_raw[val_idx]  # evaluate on true (unclipped) targets
            w_tr = sw[train_idx] if sw is not None else None

            model_cv = ExtraTreesRegressor(**_ETR_PARAMS_CV)
            model_cv.fit(Xtr, Ytr_clipped, sample_weight=w_tr)
            Yhat = model_cv.predict(Xvl)

            # Convert log-ratio back to percentage move for reporting
            h = horizons
            if mean_absolute_error is not None:
                yh_pct = (np.exp(Yvl_true[:, :h]) - 1.0) * 100.0
                yl_pct = (np.exp(Yvl_true[:, h:]) - 1.0) * 100.0
                yhat_h_pct = (np.exp(Yhat[:, :h]) - 1.0) * 100.0
                yhat_l_pct = (np.exp(Yhat[:, h:]) - 1.0) * 100.0
                high_mae = mean_absolute_error(yh_pct, yhat_h_pct)
                low_mae = mean_absolute_error(yl_pct, yhat_l_pct)
                fold_metrics.append({"high_mae_pct": round(float(high_mae), 4),
                                     "low_mae_pct": round(float(low_mae), 4)})

    final_model = ExtraTreesRegressor(**_ETR_PARAMS_FINAL)
    final_model.fit(X, Y_clip, sample_weight=sw)

    try:
        fi = final_model.feature_importances_
        feature_importances = sorted(
            zip(feature_names, fi),
            key=lambda t: t[1],
            reverse=True
        )[:30]
        feature_importances = [(str(n), float(v)) for n, v in feature_importances]
    except Exception:
        feature_importances = None

    bundle = {
        "model": final_model,
        "feature_names": feature_names,
        "horizons": horizons,
        "trained_rows": int(X.shape[0]),
        "metrics": fold_metrics or None,
        "sklearn_version": _get_sklearn_version(),
        "ticker": key,
        "model_path": None,
        "winsor": winsor_info,
        "blend_weight": _BLEND_TA_WEIGHT,
        "transform": "logratio",
        "feature_importances": feature_importances,
        "algo": "EXTRATREES_MEAN",
    }

    _MEM_CACHE[key] = bundle
    return bundle

# --------------------- Forecast ---------------------

def forecast_next_15_high_low(ticker: str, stock_data: pd.DataFrame):
    """
    Train/load from memory and forecast next 15 business days' High/Low.
    If no ML available or insufficient data, uses TA fallback.
    Returns dict: dates, pred_high, pred_low, base_close, bundle_meta
    """
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data = stock_data.copy()
        stock_data.index = pd.to_datetime(stock_data.index)

    ohlc = _ensure_ohlc_columns(stock_data)

    try:
        bundle = train_or_load_highlow_15d(stock_data, ticker, horizons=15)
        model = bundle.get("model", None)
        horizons = bundle.get("horizons", 15)

        # Build latest feature row
        feats_full = _compute_ta_features(stock_data)
        feats_full = feats_full.replace([np.inf, -np.inf], np.nan)
        feats_full = feats_full.loc[:, feats_full.notna().any(axis=0)]
        feats_full = feats_full.fillna(method="ffill").fillna(method="bfill")
        if len(feats_full) > 60:
            feats_full = feats_full.iloc[60:]
        if feats_full.empty:
            raise ValueError("No features available for inference after cleaning.")

        feature_names = bundle["feature_names"]
        for col in feature_names:
            if col not in feats_full.columns:
                feats_full[col] = 0.0
        feats_full = feats_full[feature_names]
        X_t = feats_full.iloc[[-1]].values

        base_close = float(ohlc.iloc[-1]["close"])
        if not np.isfinite(base_close) or base_close <= 0:
            base_close = float(ohlc["close"].replace(0.0, np.nan).dropna().iloc[-1])

        y_pred_log = None

        # Path 1: ExtraTrees multi-output mean-regression
        if model is not None:
            y_pred_log = model.predict(X_t).reshape(-1)

        # Path 2: Quantile gradient boosting per-horizon
        elif "q_models_high" in bundle and "q_models_low" in bundle:
            qh = bundle["q_models_high"]
            ql = bundle["q_models_low"]
            yh = np.array([qh[k].predict(X_t)[0] for k in range(horizons)], dtype=float)
            yl = np.array([ql[k].predict(X_t)[0] for k in range(horizons)], dtype=float)
            y_pred_log = np.concatenate([yh, yl], axis=0)

        if y_pred_log is not None:
            # Optional hybrid blend with TA fallback in log space for stability
            blend_w = float(bundle.get("blend_weight", _BLEND_TA_WEIGHT) or 0.0)
            if blend_w > 0.0:
                try:
                    _, hi_ta, lo_ta = _ta_fallback_forecast(ohlc, horizons=horizons)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        yh_ta_log = np.log(np.maximum(hi_ta, 1e-12) / base_close)
                        yl_ta_log = np.log(np.maximum(lo_ta, 1e-12) / base_close)
                    yh_ml_log = y_pred_log[:horizons]
                    yl_ml_log = y_pred_log[horizons:]
                    yh_blend_log = (1.0 - blend_w) * yh_ml_log + blend_w * yh_ta_log
                    yl_blend_log = (1.0 - blend_w) * yl_ml_log + blend_w * yl_ta_log
                    y_pred_log = np.concatenate([yh_blend_log, yl_blend_log], axis=0)
                except Exception:
                    pass

            # Convert back from log-ratio to price
            yh = y_pred_log[:horizons]
            yl = y_pred_log[horizons:]
            pred_high = np.exp(yh) * base_close
            pred_low = np.exp(yl) * base_close

            pred_high = np.maximum(pred_high, 0.0)
            pred_low = np.maximum(pred_low, 0.0)
            swp = pred_low > pred_high
            if np.any(swp):
                tmp = pred_high.copy()
                pred_high[swp] = pred_low[swp]
                pred_low[swp] = tmp[swp]

            last_date = feats_full.index[-1]
            future_dates = _next_business_days(last_date, horizons)
            date_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]

            return {
                "dates": date_str,
                "pred_high": [round(float(x), 2) for x in pred_high],
                "pred_low": [round(float(x), 2) for x in pred_low],
                "base_close": round(float(base_close), 4),
                "bundle_meta": {
                    "model": bundle.get("algo", "UNKNOWN"),
                    "trained_rows": bundle.get("trained_rows"),
                    "sklearn_version": bundle.get("sklearn_version"),
                    "metrics": bundle.get("metrics"),
                    "bundle_path": None,
                    "ticker": bundle.get("ticker"),
                    "winsor": bundle.get("winsor"),
                    "blend_weight": bundle.get("blend_weight"),
                    "transform": bundle.get("transform"),
                    "feature_importances_top30": bundle.get("feature_importances"),
                    "quantiles": {"high": _Q_HIGH, "low": _Q_LOW} if "q_models_high" in bundle else None,
                },
            }
    except Exception:
        pass

    base_close, pred_high, pred_low = _ta_fallback_forecast(ohlc, horizons=15)
    last_date = ohlc.index[-1]
    future_dates = _next_business_days(last_date, 15)
    date_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]

    return {
        "dates": date_str,
        "pred_high": [round(float(x), 2) for x in pred_high],
        "pred_low": [round(float(x), 2) for x in pred_low],
        "base_close": round(float(base_close), 4),
        "bundle_meta": {
            "model": "TA heuristic fallback (ATR/EMA/RSI/ADX), no ML",
            "trained_rows": int(len(ohlc)),
            "sklearn_version": _get_sklearn_version(),
            "metrics": None,
            "bundle_path": None,
            "ticker": ticker.upper(),
        },
    }