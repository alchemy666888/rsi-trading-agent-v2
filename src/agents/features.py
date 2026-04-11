from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import polars as pl
import talib

LOGGER = logging.getLogger(__name__)


def rolling_corr(series_a: np.ndarray, series_b: np.ndarray, window: int) -> np.ndarray:
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


def compute_indicator_bundle(
    prefix: str,
    open_: np.ndarray,
    high_: np.ndarray,
    low_: np.ndarray,
    close_: np.ndarray,
    volume_: np.ndarray,
) -> dict[str, np.ndarray]:
    feature_map: dict[str, np.ndarray] = {}

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
        feature_map[f"{prefix}cci_{period}"] = talib.CCI(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}atr_{period}"] = talib.ATR(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}natr_{period}"] = talib.NATR(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}willr_{period}"] = talib.WILLR(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}mfi_{period}"] = talib.MFI(high_, low_, close_, volume_, timeperiod=period)

    for period in [7, 14, 28]:
        feature_map[f"{prefix}adx_{period}"] = talib.ADX(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}adxr_{period}"] = talib.ADXR(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}plus_di_{period}"] = talib.PLUS_DI(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}minus_di_{period}"] = talib.MINUS_DI(high_, low_, close_, timeperiod=period)
        feature_map[f"{prefix}plus_dm_{period}"] = talib.PLUS_DM(high_, low_, timeperiod=period)
        feature_map[f"{prefix}minus_dm_{period}"] = talib.MINUS_DM(high_, low_, timeperiod=period)

    for bb_period in [10, 20, 50]:
        upper, middle, lower = talib.BBANDS(close_, timeperiod=bb_period, nbdevup=2, nbdevdn=2)
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
        high_,
        low_,
        close_,
        timeperiod1=7,
        timeperiod2=14,
        timeperiod3=28,
    )

    feature_map[f"{prefix}obv"] = talib.OBV(close_, volume_)
    feature_map[f"{prefix}ad"] = talib.AD(high_, low_, close_, volume_)
    feature_map[f"{prefix}adosc"] = talib.ADOSC(high_, low_, close_, volume_, fastperiod=3, slowperiod=10)
    feature_map[f"{prefix}bop"] = talib.BOP(open_, high_, low_, close_)

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

    feature_map[f"{prefix}cdl_doji"] = talib.CDLDOJI(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_hammer"] = talib.CDLHAMMER(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_engulfing"] = talib.CDLENGULFING(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_shootingstar"] = talib.CDLSHOOTINGSTAR(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_morningstar"] = talib.CDLMORNINGSTAR(open_, high_, low_, close_)
    feature_map[f"{prefix}cdl_eveningstar"] = talib.CDLEVENINGSTAR(open_, high_, low_, close_)

    return feature_map


def build_multi_timeframe_features(source_df: pl.DataFrame, every: str, prefix: str) -> pl.DataFrame:
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

    tf_bundle = compute_indicator_bundle(prefix, open_, high_, low_, close_, volume_)
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


def get_model_feature_columns(feature_df: pl.DataFrame) -> list[str]:
    excluded_cols = {"timestamp", "dt", "target_up"}
    return [col for col in feature_df.columns if col not in excluded_cols]


def build_features(raw_df: pl.DataFrame, feature_cfg: dict[str, Any] | None = None) -> pl.DataFrame:
    feature_cfg = feature_cfg or {}
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

    feature_df = feature_df.with_columns(
        [pl.Series(name, values) for name, values in compute_indicator_bundle("", open_, high_, low_, close_, volume_).items()]
    )

    if bool(feature_cfg.get("include_multi_timeframe", True)):
        mtf_source = feature_df.select(["dt", "open", "high", "low", "close", "volume"])
        # Default MTF intervals depend on base timeframe.  At 15m base the
        # original 5m/15m levels are impossible or redundant, so we step up
        # to 1h/4h/1d.  Backward-compatible: at 1m the original levels apply.
        _default_mtf: dict[str, list[str]] = {
            "1m": ["5m", "15m", "1h"],
            "5m": ["15m", "1h", "4h"],
            "15m": ["1h", "4h", "1d"],
            "30m": ["1h", "4h", "1d"],
            "1h": ["4h", "1d"],
        }
        tf_str = feature_cfg.get("timeframe", "15m")
        mtf_intervals = feature_cfg.get(
            "multi_timeframe_intervals",
            _default_mtf.get(tf_str, ["1h", "4h", "1d"]),
        )
        for every in mtf_intervals:
            prefix = f"tf_{every}_"
            tf_df = build_multi_timeframe_features(mtf_source, every=every, prefix=prefix)
            feature_df = feature_df.join_asof(tf_df.sort("dt"), on="dt", strategy="backward")

    lag_columns = [
        col
        for col in ["close", "volume", "ret_1", "rsi_14", "macd_hist_12_26_9"]
        if col in feature_df.columns
    ]
    max_lag = int(feature_cfg.get("max_lag_bars", 60))
    lag_exprs: list[pl.Expr] = []
    for base_col in lag_columns:
        for lag in range(1, max_lag + 1):
            lag_exprs.append(pl.col(base_col).shift(lag).alias(f"{base_col}_lag_{lag}"))
    feature_df = feature_df.with_columns(lag_exprs)

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
                (pl.col("log_ret_1").rolling_std(window_size=window) * math.sqrt(window)).alias(f"realized_vol_{window}"),
            ]
        )
    feature_df = feature_df.with_columns(rolling_exprs)

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
        feature_df = feature_df.with_columns(
            [pl.Series(f"corr_btc_eth_{window}", rolling_corr(btc_ret, eth_ret, window)) for window in [15, 30, 60, 120]]
        )

    # Volatility regime: compare recent realized vol (~1 h) to its longer-term
    # rolling mean (~4 h) to flag elevated-volatility bars.
    # Window sizes adapt to the base timeframe so the real-time horizons stay
    # consistent (at 1m: 60/240 bars; at 15m: 5/20 bars).
    _tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(
        feature_cfg.get("timeframe", "15m"), 15,
    )
    _available_windows = [5, 10, 20, 30, 60, 120]
    _vol_target = max(5, int(60 / _tf_minutes))          # ~1 h in bars, min 5
    _vol_window = min(_available_windows, key=lambda w: abs(w - _vol_target))
    _regime_window = max(16, int(240 / _tf_minutes))      # ~4 h in bars, min 16
    _vol_col = f"realized_vol_{_vol_window}"

    feature_df = feature_df.with_columns(
        [
            pl.col(_vol_col).alias("realized_vol_base"),
            (
                pl.col(_vol_col)
                > (pl.col(_vol_col).rolling_mean(window_size=_regime_window) + 1e-9)
            ).cast(pl.Int8).alias("volatility_regime"),
        ]
    )

    feature_df = feature_df.with_columns(
        ((pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int8).fill_null(0).alias("target_up"))
    )

    float_cols = [col for col, dtype in feature_df.schema.items() if dtype in (pl.Float32, pl.Float64)]
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

    LOGGER.info("Built %s model features.", len(get_model_feature_columns(feature_df)))
    return feature_df
