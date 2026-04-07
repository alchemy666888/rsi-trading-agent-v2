from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone
from typing import Any

import ccxt  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import shap
import talib
import vectorbt as vbt

from agents.state import AgentState

LOGGER = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rolling_corr(series_a: np.ndarray, series_b: np.ndarray, window: int) -> np.ndarray:
    out = np.full(series_a.shape[0], np.nan, dtype=float)
    if window <= 1:
        return out

    for idx in range(window - 1, len(series_a)):
        a_slice = series_a[idx - window + 1 : idx + 1]
        b_slice = series_b[idx - window + 1 : idx + 1]
        a_std = float(np.nanstd(a_slice))
        b_std = float(np.nanstd(b_slice))
        if a_std < 1e-12 or b_std < 1e-12:
            out[idx] = 0.0
        else:
            out[idx] = float(np.corrcoef(a_slice, b_slice)[0, 1])
    return out


def _metric_float(metric: Any) -> float:
    arr = np.asarray(metric, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.nan_to_num(arr[0], nan=0.0, posinf=0.0, neginf=0.0))


def _compute_indicator_bundle(
    prefix: str,
    open_: np.ndarray,
    high_: np.ndarray,
    low_: np.ndarray,
    close_: np.ndarray,
    volume_: np.ndarray,
) -> dict[str, np.ndarray]:
    """# Week 1 upgrade: dense TA-Lib bundle used on 1m and resampled timeframes."""
    feature_map: dict[str, np.ndarray] = {}

    # Price transforms
    feature_map[f"{prefix}hl2"] = (high_ + low_) / 2.0
    feature_map[f"{prefix}hlc3"] = (high_ + low_ + close_) / 3.0
    feature_map[f"{prefix}ohlc4"] = (open_ + high_ + low_ + close_) / 4.0

    periods = [5, 8, 10, 14, 20, 30, 50, 100, 200]
    for period in periods:
        feature_map[f"{prefix}sma_{period}"] = talib.SMA(close_, timeperiod=period)
        feature_map[f"{prefix}ema_{period}"] = talib.EMA(close_, timeperiod=period)
        feature_map[f"{prefix}wma_{period}"] = talib.WMA(close_, timeperiod=period)
        feature_map[f"{prefix}dema_{period}"] = talib.DEMA(close_, timeperiod=period)
        feature_map[f"{prefix}tema_{period}"] = talib.TEMA(close_, timeperiod=period)
        feature_map[f"{prefix}kama_{period}"] = talib.KAMA(close_, timeperiod=period)
        feature_map[f"{prefix}rsi_{period}"] = talib.RSI(close_, timeperiod=period)
        feature_map[f"{prefix}roc_{period}"] = talib.ROC(close_, timeperiod=period)
        feature_map[f"{prefix}mom_{period}"] = talib.MOM(close_, timeperiod=period)
        feature_map[f"{prefix}cmo_{period}"] = talib.CMO(close_, timeperiod=period)
        feature_map[f"{prefix}cci_{period}"] = talib.CCI(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}atr_{period}"] = talib.ATR(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}natr_{period}"] = talib.NATR(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}willr_{period}"] = talib.WILLR(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}mfi_{period}"] = talib.MFI(
            high_, low_, close_, volume_, timeperiod=period
        )

    for period in [7, 14, 28]:
        feature_map[f"{prefix}adx_{period}"] = talib.ADX(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}adxr_{period}"] = talib.ADXR(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}plus_di_{period}"] = talib.PLUS_DI(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}minus_di_{period}"] = talib.MINUS_DI(
            high_, low_, close_, timeperiod=period
        )
        feature_map[f"{prefix}plus_dm_{period}"] = talib.PLUS_DM(
            high_, low_, timeperiod=period
        )
        feature_map[f"{prefix}minus_dm_{period}"] = talib.MINUS_DM(
            high_, low_, timeperiod=period
        )

    for bb_period in [10, 20, 50]:
        upper, middle, lower = talib.BBANDS(
            close_, timeperiod=bb_period, nbdevup=2, nbdevdn=2
        )
        spread = upper - lower
        middle_safe = np.where(np.abs(middle) < 1e-9, 1e-9, middle)
        spread_safe = np.where(np.abs(spread) < 1e-9, 1e-9, spread)
        feature_map[f"{prefix}bb_upper_{bb_period}"] = upper
        feature_map[f"{prefix}bb_middle_{bb_period}"] = middle
        feature_map[f"{prefix}bb_lower_{bb_period}"] = lower
        feature_map[f"{prefix}bb_width_{bb_period}"] = spread / middle_safe
        feature_map[f"{prefix}bb_pctb_{bb_period}"] = (close_ - lower) / spread_safe

    for fast, slow, signal in [(12, 26, 9), (6, 19, 9), (24, 52, 18)]:
        macd, macd_signal, macd_hist = talib.MACD(
            close_,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
        )
        feature_map[f"{prefix}macd_{fast}_{slow}_{signal}"] = macd
        feature_map[f"{prefix}macd_signal_{fast}_{slow}_{signal}"] = macd_signal
        feature_map[f"{prefix}macd_hist_{fast}_{slow}_{signal}"] = macd_hist

    feature_map[f"{prefix}apo_12_26"] = talib.APO(close_, fastperiod=12, slowperiod=26)
    feature_map[f"{prefix}ppo_12_26"] = talib.PPO(close_, fastperiod=12, slowperiod=26)
    feature_map[f"{prefix}trix_30"] = talib.TRIX(close_, timeperiod=30)

    stoch_k, stoch_d = talib.STOCH(
        high_,
        low_,
        close_,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    feature_map[f"{prefix}stoch_k_14_3_3"] = stoch_k
    feature_map[f"{prefix}stoch_d_14_3_3"] = stoch_d

    stochf_k, stochf_d = talib.STOCHF(
        high_,
        low_,
        close_,
        fastk_period=14,
        fastd_period=3,
        fastd_matype=0,
    )
    feature_map[f"{prefix}stochf_k_14_3"] = stochf_k
    feature_map[f"{prefix}stochf_d_14_3"] = stochf_d

    stochrsi_k, stochrsi_d = talib.STOCHRSI(
        close_,
        timeperiod=14,
        fastk_period=5,
        fastd_period=3,
        fastd_matype=0,
    )
    feature_map[f"{prefix}stochrsi_k_14"] = stochrsi_k
    feature_map[f"{prefix}stochrsi_d_14"] = stochrsi_d

    aroondown, aroonup = talib.AROON(high_, low_, timeperiod=14)
    feature_map[f"{prefix}aroon_down_14"] = aroondown
    feature_map[f"{prefix}aroon_up_14"] = aroonup
    feature_map[f"{prefix}aroon_osc_14"] = talib.AROONOSC(high_, low_, timeperiod=14)
    feature_map[f"{prefix}ultosc"] = talib.ULTOSC(
        high_, low_, close_, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )

    # Volume and directional flow indicators
    feature_map[f"{prefix}obv"] = talib.OBV(close_, volume_)
    feature_map[f"{prefix}ad"] = talib.AD(high_, low_, close_, volume_)
    feature_map[f"{prefix}adosc"] = talib.ADOSC(
        high_, low_, close_, volume_, fastperiod=3, slowperiod=10
    )
    feature_map[f"{prefix}bop"] = talib.BOP(open_, high_, low_, close_)

    # Hilbert transform features
    feature_map[f"{prefix}ht_dcperiod"] = talib.HT_DCPERIOD(close_)
    feature_map[f"{prefix}ht_dcphase"] = talib.HT_DCPHASE(close_)
    inphase, quadrature = talib.HT_PHASOR(close_)
    feature_map[f"{prefix}ht_inphase"] = inphase
    feature_map[f"{prefix}ht_quadrature"] = quadrature
    sine, leadsine = talib.HT_SINE(close_)
    feature_map[f"{prefix}ht_sine"] = sine
    feature_map[f"{prefix}ht_leadsine"] = leadsine
    feature_map[f"{prefix}ht_trendmode"] = talib.HT_TRENDMODE(close_)
    feature_map[f"{prefix}ht_trendline"] = talib.HT_TRENDLINE(close_)

    # Candlestick patterns
    feature_map[f"{prefix}cdl_doji"] = talib.CDLDOJI(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_hammer"] = talib.CDLHAMMER(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_engulfing"] = talib.CDLENGULFING(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_shootingstar"] = talib.CDLSHOOTINGSTAR(
        open_, high_, low_, close_
    )
    feature_map[f"{prefix}cdl_morningstar"] = talib.CDLMORNINGSTAR(
        open_, high_, low_, close_
    )
    feature_map[f"{prefix}cdl_eveningstar"] = talib.CDLEVENINGSTAR(
        open_, high_, low_, close_
    )

    return feature_map


def _build_multi_timeframe_features(
    source_df: pl.DataFrame,
    every: str,
    prefix: str,
) -> pl.DataFrame:
    """# Week 1 upgrade: resample 1m bars into higher timeframes and map indicators back."""
    resampled = (
        source_df.group_by_dynamic("dt", every=every, period=every, closed="right", label="right")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls("close")
        .sort("dt")
    )

    if resampled.height == 0:
        return resampled.select(["dt"])

    open_ = resampled["open"].to_numpy()
    high_ = resampled["high"].to_numpy()
    low_ = resampled["low"].to_numpy()
    close_ = resampled["close"].to_numpy()
    volume_ = resampled["volume"].to_numpy()

    tf_bundle = _compute_indicator_bundle(prefix, open_, high_, low_, close_, volume_)
    tf_features = [pl.Series(name, values) for name, values in tf_bundle.items()]

    resampled = resampled.with_columns(
        [
            pl.col("close").pct_change().alias(f"{prefix}ret_1"),
            pl.col("close").pct_change(3).alias(f"{prefix}ret_3"),
            pl.col("volume").pct_change().alias(f"{prefix}vol_chg_1"),
        ]
        + tf_features
    )

    return resampled.select(["dt"] + [col for col in resampled.columns if col.startswith(prefix)])


def _build_features(raw_df: pl.DataFrame) -> pl.DataFrame:
    """# Week 1 upgrade: 300+ engineered features for LightGBM baseline."""
    feature_df = raw_df.with_columns(
        [
            pl.from_epoch("timestamp", time_unit="ms").alias("dt"),
            pl.col("close").pct_change().alias("ret_1"),
            pl.col("close").pct_change(5).alias("ret_5"),
            pl.col("close").pct_change(15).alias("ret_15"),
            pl.col("close").pct_change(60).alias("ret_60"),
            pl.col("close").log().diff().alias("log_ret_1"),
            pl.col("volume").pct_change().alias("vol_chg_1"),
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-9)).alias("hl_range"),
            ((pl.col("close") - pl.col("open")) / (pl.col("open") + 1e-9)).alias("co_return"),
        ]
    )

    open_ = feature_df["open"].to_numpy()
    high_ = feature_df["high"].to_numpy()
    low_ = feature_df["low"].to_numpy()
    close_ = feature_df["close"].to_numpy()
    volume_ = feature_df["volume"].to_numpy()

    # Dense 1m TA-Lib bundle
    ta_bundle = _compute_indicator_bundle("", open_, high_, low_, close_, volume_)
    feature_df = feature_df.with_columns(
        [pl.Series(name, values) for name, values in ta_bundle.items()]
    )

    # Multi-timeframe overlays: 5m, 15m, 1h
    mtf_source = feature_df.select(["dt", "open", "high", "low", "close", "volume"])
    for every, prefix in [("5m", "tf_5m_"), ("15m", "tf_15m_"), ("1h", "tf_1h_")]:
        tf_df = _build_multi_timeframe_features(mtf_source, every=every, prefix=prefix)
        feature_df = feature_df.join_asof(tf_df.sort("dt"), on="dt", strategy="backward")

    # Lag stack (1..60 bars) on key signal drivers
    lag_columns = [
        col
        for col in ["close", "volume", "ret_1", "rsi_14", "macd_hist_12_26_9"]
        if col in feature_df.columns
    ]
    lag_exprs: list[pl.Expr] = []
    for base_col in lag_columns:
        for lag in range(1, 61):
            lag_exprs.append(pl.col(base_col).shift(lag).alias(f"{base_col}_lag_{lag}"))
    feature_df = feature_df.with_columns(lag_exprs)

    # Rolling statistics at different windows
    rolling_exprs: list[pl.Expr] = []
    for window in [5, 10, 20, 30, 60, 120]:
        rolling_exprs.extend(
            [
                pl.col("ret_1").rolling_mean(window_size=window).alias(f"ret_mean_{window}"),
                pl.col("ret_1").rolling_std(window_size=window).alias(f"ret_std_{window}"),
                pl.col("ret_1").rolling_min(window_size=window).alias(f"ret_min_{window}"),
                pl.col("ret_1").rolling_max(window_size=window).alias(f"ret_max_{window}"),
                pl.col("close").rolling_mean(window_size=window).alias(f"close_mean_{window}"),
                pl.col("close").rolling_std(window_size=window).alias(f"close_std_{window}"),
                (
                    (pl.col("close") - pl.col("close").rolling_mean(window_size=window))
                    / (pl.col("close").rolling_std(window_size=window) + 1e-9)
                ).alias(f"close_z_{window}"),
                pl.col("volume").rolling_mean(window_size=window).alias(f"volume_mean_{window}"),
                pl.col("volume").rolling_std(window_size=window).alias(f"volume_std_{window}"),
                (
                    (pl.col("volume") - pl.col("volume").rolling_mean(window_size=window))
                    / (pl.col("volume").rolling_std(window_size=window) + 1e-9)
                ).alias(f"volume_z_{window}"),
                (pl.col("log_ret_1").rolling_std(window_size=window) * math.sqrt(window)).alias(
                    f"realized_vol_{window}"
                ),
            ]
        )
    feature_df = feature_df.with_columns(rolling_exprs)

    # Cross-asset proxy (if ETH data is available)
    if "eth_close" in feature_df.columns:
        feature_df = feature_df.with_columns(
            [
                pl.col("eth_close").pct_change().alias("eth_ret_1"),
                pl.col("eth_volume").pct_change().alias("eth_vol_chg_1"),
                (pl.col("close") / (pl.col("eth_close") + 1e-9)).alias("btc_eth_spread"),
                (pl.col("ret_1") - pl.col("eth_close").pct_change()).alias("btc_eth_ret_spread"),
            ]
        )

        btc_ret = feature_df["ret_1"].to_numpy()
        eth_ret = feature_df["eth_ret_1"].to_numpy()
        corr_features = [
            pl.Series(f"corr_btc_eth_{window}", _rolling_corr(btc_ret, eth_ret, window))
            for window in [15, 30, 60, 120]
        ]
        feature_df = feature_df.with_columns(corr_features)

    # Volatility regime classifier feature
    feature_df = feature_df.with_columns(
        [
            pl.col("realized_vol_60").alias("realized_vol_60_base"),
            (
                pl.col("realized_vol_60")
                > (pl.col("realized_vol_60").rolling_mean(window_size=240) + 1e-9)
            )
            .cast(pl.Int8)
            .alias("volatility_regime"),
        ]
    )

    # Binary next-bar target for LightGBM
    feature_df = feature_df.with_columns(
        (
            (pl.col("close").shift(-1) > pl.col("close"))
            .cast(pl.Int8)
            .fill_null(0)
            .alias("target_up")
        )
    )

    # Clean NaN/nulls while preserving timestamp and dt
    float_cols = [
        col
        for col, dtype in feature_df.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    int_cols = [
        col
        for col, dtype in feature_df.schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        and col not in {"timestamp"}
    ]

    feature_df = feature_df.with_columns(
        [pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col) for col in float_cols]
        + [pl.col(col).fill_null(0).alias(col) for col in int_cols]
    )

    model_feature_count = len(_get_model_feature_columns(feature_df))
    LOGGER.info("Week 1 upgrade: built %s model features.", model_feature_count)

    return feature_df


def _get_model_feature_columns(feature_df: pl.DataFrame) -> list[str]:
    excluded_cols = {"timestamp", "dt", "target_up"}
    return [col for col in feature_df.columns if col not in excluded_cols]


def _train_lightgbm_baseline(
    feature_df: pl.DataFrame,
    config: dict[str, Any],
) -> tuple[lgb.LGBMClassifier, list[str], list[dict[str, Any]]]:
    """# Week 1 upgrade: quick baseline model train-once for live cycle predictions."""
    model_cfg = config.get("model", {})
    feature_cols = _get_model_feature_columns(feature_df)

    train_split = float(model_cfg.get("train_split_ratio", 0.7))
    min_train_rows = int(model_cfg.get("min_train_rows", 400))
    split_idx = max(int(feature_df.height * train_split), min_train_rows)
    split_idx = min(split_idx, feature_df.height - 100)

    if split_idx < 100:
        raise RuntimeError("Not enough data to train LightGBM baseline.")

    x_train = feature_df.select(feature_cols).slice(0, split_idx).to_numpy()
    y_train = feature_df["target_up"].to_numpy()[:split_idx].astype(int)

    # Keep training robust on tiny/imbalanced windows
    if np.unique(y_train).size < 2:
        y_train[-1] = 1 - y_train[-1]

    model = lgb.LGBMClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        num_leaves=int(model_cfg.get("num_leaves", 63)),
        max_depth=int(model_cfg.get("max_depth", -1)),
        min_child_samples=int(model_cfg.get("min_child_samples", 20)),
        subsample=float(model_cfg.get("subsample", 0.85)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.85)),
        random_state=int(model_cfg.get("random_state", 42)),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
        objective="binary",
        verbose=-1,
    )
    model.fit(x_train, y_train)

    gains = model.booster_.feature_importance(importance_type="gain")
    feature_importances = sorted(
        [
            {"feature": feature_cols[idx], "importance": float(gain)}
            for idx, gain in enumerate(gains)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )

    LOGGER.info(
        "Week 1 upgrade: LightGBM trained on %s rows with %s features.",
        split_idx,
        len(feature_cols),
    )
    return model, feature_cols, feature_importances


def _load_historical_btc_data(config: dict[str, Any]) -> pl.DataFrame:
    asset_cfg = config["asset"]
    api_cfg = config.get("api_keys", {})

    exchange_name = asset_cfg["exchange"]
    exchange_cls = getattr(ccxt, exchange_name)
    exchange_params = {
        "enableRateLimit": True,
        "apiKey": api_cfg.get("ccxt_api_key", ""),
        "secret": api_cfg.get("ccxt_api_secret", ""),
    }
    exchange = exchange_cls(exchange_params)

    ohlcv = exchange.fetch_ohlcv(
        symbol=asset_cfg["symbol"],
        timeframe=asset_cfg["timeframe"],
        limit=int(asset_cfg["fetch_limit"]),
    )
    if not ohlcv:
        raise RuntimeError("No OHLCV rows returned from exchange.")

    raw_df = pl.DataFrame(
        {
            "timestamp": [int(row[0]) for row in ohlcv],
            "open": [float(row[1]) for row in ohlcv],
            "high": [float(row[2]) for row in ohlcv],
            "low": [float(row[3]) for row in ohlcv],
            "close": [float(row[4]) for row in ohlcv],
            "volume": [float(row[5]) for row in ohlcv],
        }
    ).sort("timestamp")

    # Cross-asset correlation proxy for Week 1 if available
    try:
        quote = asset_cfg["symbol"].split("/")[1]
        eth_symbol = f"ETH/{quote}"
        eth_ohlcv = exchange.fetch_ohlcv(
            symbol=eth_symbol,
            timeframe=asset_cfg["timeframe"],
            limit=int(asset_cfg["fetch_limit"]),
        )
        if eth_ohlcv:
            eth_df = pl.DataFrame(
                {
                    "timestamp": [int(row[0]) for row in eth_ohlcv],
                    "eth_close": [float(row[4]) for row in eth_ohlcv],
                    "eth_volume": [float(row[5]) for row in eth_ohlcv],
                }
            )
            raw_df = raw_df.join(eth_df, on="timestamp", how="left")
            LOGGER.info("Week 1 upgrade: ETH proxy series joined for cross-asset features.")
    except Exception as exc:  # pragma: no cover - best effort enhancement
        LOGGER.warning("Could not load ETH proxy data, continuing without it: %s", exc)

    LOGGER.info("Loaded %s BTC candles from %s", raw_df.height, exchange_name)
    return _build_features(raw_df)


def _prediction_probability(row: dict[str, Any]) -> float:
    """Fallback probability model if LightGBM is unavailable."""
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    macd_hist = _safe_float(row.get("macd_hist_12_26_9"), 0.0)
    close = max(_safe_float(row.get("close"), 1.0), 1e-9)
    bb_middle = _safe_float(row.get("bb_middle_20"), close)
    atr = _safe_float(row.get("atr_14"), 0.0)

    rsi_signal = (rsi - 50.0) / 50.0
    macd_signal = (macd_hist / close) * 1000.0
    mean_reversion_signal = (close - bb_middle) / close
    atr_signal = atr / close

    linear_score = (
        (1.35 * rsi_signal)
        + (0.90 * macd_signal)
        - (0.65 * mean_reversion_signal)
        - (0.25 * atr_signal)
    )
    return float(np.clip(_sigmoid(linear_score), 0.0, 1.0))


def _compute_performance(
    returns: list[float],
    equity_curve: list[float],
    initial_equity: float,
) -> dict[str, Any]:
    if not returns:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }

    returns_array = np.array(returns, dtype=float)
    mean_return = float(returns_array.mean())
    std_return = float(returns_array.std(ddof=1)) if len(returns_array) > 1 else 0.0
    annualization = math.sqrt(365 * 24 * 60)  # 1m BTC bars
    sharpe = (mean_return / std_return) * annualization if std_return > 1e-12 else 0.0

    equity_array = np.array([initial_equity] + equity_curve, dtype=float)
    rolling_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / rolling_max) - 1.0
    max_drawdown = float(abs(np.min(drawdowns)))

    win_rate = float((returns_array > 0.0).mean())
    total_return = float((equity_array[-1] / initial_equity) - 1.0)
    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
    }


def _run_optuna_tune(state: AgentState) -> tuple[dict[str, float], float]:
    """Tune long/short thresholds on top of fixed model probabilities."""
    data = state["historical_data"]
    cfg = state["config"]
    simulation_cfg = cfg["simulation"]
    cursor = int(state["cursor"])

    start = max(0, cursor - 400)
    window = data.slice(start, cursor - start + 1)
    closes = window["close"].to_numpy()

    if len(closes) < 20:
        default_params = {
            "long_threshold": float(simulation_cfg["long_threshold"]),
            "short_threshold": float(simulation_cfg["short_threshold"]),
        }
        return default_params, 0.0

    model = state.get("lightgbm_model")
    feature_cols = state.get("feature_columns", [])
    if model is not None and feature_cols:
        probs = model.predict_proba(window.select(feature_cols).to_numpy())[:, 1]
    else:
        probs = np.array(
            [_prediction_probability(window.row(idx, named=True)) for idx in range(window.height)],
            dtype=float,
        )

    slippage = float(simulation_cfg["slippage_bps"]) / 10000.0

    def objective(trial: optuna.Trial) -> float:
        long_threshold = trial.suggest_float("long_threshold", 0.52, 0.72)
        short_threshold = trial.suggest_float("short_threshold", 0.28, 0.48)
        if short_threshold >= long_threshold:
            return -10.0

        position = 0
        previous_position = 0
        strategy_returns: list[float] = []

        for idx in range(len(closes) - 1):
            prob_up = float(probs[idx])
            if prob_up > long_threshold:
                position = 1
            elif prob_up < short_threshold:
                position = -1

            market_return = (float(closes[idx + 1]) - float(closes[idx])) / float(
                closes[idx]
            )
            transaction_cost = slippage * abs(position - previous_position)
            strategy_returns.append((position * market_return) - transaction_cost)
            previous_position = position

        if len(strategy_returns) < 5:
            return -10.0
        arr = np.array(strategy_returns, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        if std <= 1e-12:
            return -10.0
        return float((arr.mean() / std) * math.sqrt(365 * 24 * 60))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(simulation_cfg["optuna_trials"]))
    return {
        "long_threshold": float(study.best_params["long_threshold"]),
        "short_threshold": float(study.best_params["short_threshold"]),
    }, float(study.best_value)


def _derive_shap_rule(state: AgentState) -> str:
    data = state["historical_data"]
    cursor = int(state["cursor"])
    feature_cols = state.get("feature_columns", [])
    model = state.get("lightgbm_model")

    if model is None or not feature_cols:
        return "If RSI rises above 55, long probability tends to increase."

    lookback = min(300, cursor + 1)
    segment = data.slice(cursor - lookback + 1, lookback)
    if segment.height < 20:
        return "If RSI rises above 55, long probability tends to increase."

    x_segment = segment.select(feature_cols).to_numpy()
    background = x_segment[: min(200, len(x_segment) - 1)]
    target = x_segment[-1:]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(target)
        if isinstance(shap_values, list):
            shap_vector = np.asarray(shap_values[-1], dtype=float).reshape(-1)
        else:
            shap_arr = np.asarray(shap_values, dtype=float)
            shap_vector = shap_arr.reshape(-1) if shap_arr.ndim <= 2 else shap_arr[0]

        top_idx = int(np.argmax(np.abs(shap_vector)))
        direction = "increases" if shap_vector[top_idx] >= 0 else "decreases"
        feature = feature_cols[top_idx]
        feature_value = float(target[0, top_idx])
        return (
            f"If {feature} is elevated (current={feature_value:.4f}), "
            f"the model estimate for long probability typically {direction}."
        )
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("SHAP rule extraction failed, using fallback rule: %s", exc)
        return "If RSI rises above 55, long probability tends to increase."


def _run_walk_forward_backtest(state: AgentState) -> dict[str, Any]:
    """# Week 1 upgrade: expanding + rolling walk-forward with VectorBT."""
    cfg = state["config"]
    data = state["historical_data"]
    feature_cols = state.get("feature_columns", [])

    if not feature_cols:
        return {
            "settings": {},
            "expanding": {"folds": [], "mean_sharpe": 0.0, "mean_max_drawdown": 0.0, "mean_total_return": 0.0, "mean_accuracy": 0.0},
            "rolling": {"folds": [], "mean_sharpe": 0.0, "mean_max_drawdown": 0.0, "mean_total_return": 0.0, "mean_accuracy": 0.0},
            "overall": {"mean_sharpe": 0.0, "mean_max_drawdown": 0.0, "mean_total_return": 0.0, "mean_accuracy": 0.0},
        }

    walk_cfg = cfg.get("walk_forward", {})
    model_cfg = cfg.get("model", {})
    sim_cfg = cfg["simulation"]

    train_bars = int(walk_cfg.get("train_bars", 400))
    test_bars = int(walk_cfg.get("test_bars", 100))
    step_bars = int(walk_cfg.get("step_bars", 50))

    x_all = data.select(feature_cols).to_numpy()
    y_all = data["target_up"].to_numpy().astype(int)
    close_all = data["close"].to_numpy()

    def run_mode(mode: str) -> dict[str, Any]:
        folds: list[dict[str, Any]] = []
        split_start = train_bars
        fold_idx = 0

        while split_start + test_bars < len(data) - 1:
            if mode == "expanding":
                train_start = 0
            else:
                train_start = max(0, split_start - train_bars)

            train_end = split_start
            test_end = split_start + test_bars

            x_train = x_all[train_start:train_end]
            y_train = y_all[train_start:train_end]
            x_test = x_all[split_start:test_end]
            y_test = y_all[split_start:test_end]
            close_test = close_all[split_start:test_end]

            if x_train.shape[0] < 50 or x_test.shape[0] < 10:
                break
            if np.unique(y_train).size < 2:
                y_train[-1] = 1 - y_train[-1]

            model = lgb.LGBMClassifier(
                n_estimators=int(model_cfg.get("walk_forward_n_estimators", 120)),
                learning_rate=float(model_cfg.get("learning_rate", 0.05)),
                num_leaves=int(model_cfg.get("num_leaves", 63)),
                max_depth=int(model_cfg.get("max_depth", -1)),
                min_child_samples=int(model_cfg.get("min_child_samples", 20)),
                subsample=float(model_cfg.get("subsample", 0.85)),
                colsample_bytree=float(model_cfg.get("colsample_bytree", 0.85)),
                random_state=int(model_cfg.get("random_state", 42)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
                objective="binary",
                verbose=-1,
            )
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_test)[:, 1]

            long_threshold = float(state.get("strategy_params", {}).get("long_threshold", sim_cfg["long_threshold"]))
            short_threshold = float(state.get("strategy_params", {}).get("short_threshold", sim_cfg["short_threshold"]))

            entries = probs > long_threshold
            exits = probs < 0.5
            short_entries = probs < short_threshold
            short_exits = probs > 0.5

            pf = vbt.Portfolio.from_signals(
                close=close_test,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=float(sim_cfg["initial_equity"]),
                fees=float(sim_cfg["slippage_bps"]) / 10000.0,
                freq="1min",
            )

            fold = {
                "fold": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": split_start,
                "test_end": test_end,
                "sharpe": _metric_float(pf.sharpe_ratio()),
                "max_drawdown": abs(_metric_float(pf.max_drawdown())),
                "total_return": _metric_float(pf.total_return()),
                "accuracy": float(((probs > 0.5).astype(int) == y_test).mean()),
            }
            folds.append(fold)
            fold_idx += 1
            split_start += step_bars

        if not folds:
            return {
                "folds": [],
                "mean_sharpe": 0.0,
                "mean_max_drawdown": 0.0,
                "mean_total_return": 0.0,
                "mean_accuracy": 0.0,
            }

        return {
            "folds": folds,
            "mean_sharpe": float(np.mean([fold["sharpe"] for fold in folds])),
            "mean_max_drawdown": float(np.mean([fold["max_drawdown"] for fold in folds])),
            "mean_total_return": float(np.mean([fold["total_return"] for fold in folds])),
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
        }

    expanding = run_mode("expanding")
    rolling = run_mode("rolling")

    overall = {
        "mean_sharpe": float(np.mean([expanding["mean_sharpe"], rolling["mean_sharpe"]])),
        "mean_max_drawdown": float(
            np.mean([expanding["mean_max_drawdown"], rolling["mean_max_drawdown"]])
        ),
        "mean_total_return": float(
            np.mean([expanding["mean_total_return"], rolling["mean_total_return"]])
        ),
        "mean_accuracy": float(np.mean([expanding["mean_accuracy"], rolling["mean_accuracy"]])),
    }

    return {
        "settings": {
            "train_bars": train_bars,
            "test_bars": test_bars,
            "step_bars": step_bars,
        },
        "expanding": expanding,
        "rolling": rolling,
        "overall": overall,
    }


def update_lora_adapter(_: AgentState) -> None:
    # MVP stub: keeps the LoRA update hook in place from day one.
    LOGGER.info("LoRA adapter update hook executed (MVP dummy).")


def append_to_replay_buffer(state: AgentState) -> list[dict[str, Any]]:
    replay = list(state.get("replay_buffer", []))
    replay.append(
        {
            "cycle": int(state["cycle_count"]),
            "return": float(state["returns"][-1]),
            "action": state["last_action"],
            "position": int(state["position"]),
            "regime": state["prediction"]["regime"],
            "model_confidence": float(state["prediction"]["prob_up"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    limit = int(state["config"]["simulation"]["replay_buffer_limit"])
    return replay[-limit:]


def data_node(state: AgentState) -> AgentState:
    config = state["config"]
    historical_data = state.get("historical_data")
    if historical_data is None:
        historical_data = _load_historical_btc_data(config)
        model, feature_columns, feature_importances = _train_lightgbm_baseline(
            historical_data, config
        )

        cursor = int(config["runtime"]["warmup_bars"])
        current_row = historical_data.row(cursor, named=True)
        features_df = historical_data.slice(cursor, 1)
        LOGGER.info(
            "Data node initialized with %s rows; warmup cursor=%s.",
            historical_data.height,
            cursor,
        )
        return {
            "historical_data": historical_data,
            "lightgbm_model": model,
            "feature_columns": feature_columns,
            "feature_importances": feature_importances,
            "cursor": cursor,
            "current_row": current_row,
            "features_df": features_df,
            "done": False,
        }

    cursor = int(state["cursor"])
    max_cycles = int(config["runtime"]["max_cycles"])
    cycle_count = int(state.get("cycle_count", 0))
    done = cycle_count >= max_cycles or cursor >= (historical_data.height - 2)
    if done:
        return {"done": True}

    current_row = historical_data.row(cursor, named=True)
    features_df = historical_data.slice(cursor, 1)
    return {
        "current_row": current_row,
        "features_df": features_df,
        "done": False,
    }


def predict_node(state: AgentState) -> AgentState:
    row = state["current_row"]
    model = state.get("lightgbm_model")
    feature_columns = state.get("feature_columns", [])

    if model is not None and feature_columns:
        x_live = np.array(
            [[_safe_float(row.get(feature_name), 0.0) for feature_name in feature_columns]],
            dtype=float,
        )
        prob_up = float(model.predict_proba(x_live)[0, 1])
    else:
        prob_up = _prediction_probability(row)

    regime_value = _safe_float(row.get("volatility_regime"), 0.0)
    regime = "high_volatility" if regime_value >= 0.5 else "normal"
    return {
        "prediction": {
            "prob_up": prob_up,
            "regime": regime,
        }
    }


def decision_node(state: AgentState) -> AgentState:
    prediction = state["prediction"]
    config = state["config"]
    params = state.get(
        "strategy_params",
        {
            "long_threshold": float(config["simulation"]["long_threshold"]),
            "short_threshold": float(config["simulation"]["short_threshold"]),
        },
    )

    prob_up = float(prediction["prob_up"])
    previous_position = int(state.get("position", 0))
    position = previous_position
    action = "HOLD"

    if prob_up > float(params["long_threshold"]):
        position = 1
        action = "LONG"
    elif prob_up < float(params["short_threshold"]):
        position = -1
        action = "SHORT"

    trades = list(state.get("trades", []))
    if position != previous_position:
        trades.append(
            {
                "cycle": int(state.get("cycle_count", 0)) + 1,
                "timestamp": int(state["current_row"]["timestamp"]),
                "action": action,
                "new_position": position,
                "price": float(state["current_row"]["close"]),
                "prob_up": prob_up,
            }
        )

    return {
        "last_action": action,
        "previous_position": previous_position,
        "position": position,
        "trades": trades,
    }


def evaluate_node(state: AgentState) -> AgentState:
    config = state["config"]
    data = state["historical_data"]
    cursor = int(state["cursor"])
    previous_position = int(state["previous_position"])
    position = int(state["position"])

    close_now = float(data["close"][cursor])
    close_next = float(data["close"][cursor + 1])
    market_return = (close_next - close_now) / close_now

    slippage = float(config["simulation"]["slippage_bps"]) / 10000.0
    transaction_cost = slippage * abs(position - previous_position)
    strategy_return = (position * market_return) - transaction_cost

    returns = list(state.get("returns", []))
    returns.append(strategy_return)

    initial_equity = float(config["simulation"]["initial_equity"])
    equity = float(state.get("equity", initial_equity)) * (1.0 + strategy_return)
    equity_curve = list(state.get("equity_curve", []))
    equity_curve.append(equity)

    cycle_count = int(state.get("cycle_count", 0)) + 1
    performance = _compute_performance(returns, equity_curve, initial_equity)

    walk_forward_interval = int(config.get("walk_forward", {}).get("interval_cycles", 10))
    previous_walk_forward = state.get("performance", {}).get("walk_forward_metrics")
    if cycle_count == 1 or cycle_count % walk_forward_interval == 0:
        performance["walk_forward_metrics"] = _run_walk_forward_backtest(state)
    elif previous_walk_forward is not None:
        performance["walk_forward_metrics"] = previous_walk_forward

    done = (
        cycle_count >= int(config["runtime"]["max_cycles"])
        or (cursor + 1) >= (data.height - 1)
    )
    return {
        "cycle_count": cycle_count,
        "cursor": cursor + 1,
        "equity": equity,
        "returns": returns,
        "equity_curve": equity_curve,
        "performance": performance,
        "done": done,
    }


def optimize_node(state: AgentState) -> AgentState:
    config = state["config"]
    cycle_count = int(state["cycle_count"])
    performance = state["performance"]
    optimize_every = int(config["simulation"]["optimization_interval_cycles"])
    sharpe_trigger = float(config["simulation"]["sharpe_trigger"])

    should_optimize = (cycle_count % optimize_every == 0) or (
        float(performance["sharpe"]) < sharpe_trigger
    )
    if not should_optimize:
        return {}

    best_params, best_score = _run_optuna_tune(state)
    shap_rule = _derive_shap_rule(state)
    update_lora_adapter(state)
    replay_buffer = append_to_replay_buffer(state)

    optimization_events = list(state.get("optimization_events", []))
    optimization_events.append(
        {
            "cycle": cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_params": best_params,
            "optuna_objective": best_score,
            "shap_rule": shap_rule,
        }
    )
    LOGGER.info(
        "Optimization triggered on cycle=%s (sharpe=%.4f).",
        cycle_count,
        float(performance["sharpe"]),
    )
    return {
        "strategy_params": best_params,
        "optimization_events": optimization_events,
        "replay_buffer": replay_buffer,
        "shap_rule": shap_rule,
    }
