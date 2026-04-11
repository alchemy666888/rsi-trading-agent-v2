from __future__ import annotations

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import shap

from agents.features import get_model_feature_columns

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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_lgbm_params(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    return {
        "n_estimators": int(model_cfg.get("n_estimators", 200)),
        "learning_rate": float(model_cfg.get("learning_rate", 0.05)),
        "num_leaves": int(model_cfg.get("num_leaves", 63)),
        "max_depth": int(model_cfg.get("max_depth", -1)),
        "min_child_samples": int(model_cfg.get("min_child_samples", 20)),
        "subsample": float(model_cfg.get("subsample", 0.85)),
        "colsample_bytree": float(model_cfg.get("colsample_bytree", 0.85)),
        "random_state": int(model_cfg.get("random_state", 42)),
        "n_jobs": int(model_cfg.get("n_jobs", -1)),
        "objective": "binary",
        "verbose": -1,
    }


def train_lightgbm_baseline(
    feature_df: pl.DataFrame,
    split_metadata: dict[str, int],
    config: dict[str, Any],
) -> tuple[lgb.LGBMClassifier, list[str], list[dict[str, Any]]]:
    feature_cols = get_model_feature_columns(feature_df)
    train_start = int(split_metadata["train_start"])
    train_end = int(split_metadata["train_end"])

    train_df = feature_df.slice(train_start, train_end - train_start)
    x_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["target_up"].to_numpy().astype(int)

    if np.unique(y_train).size < 2:
        y_train[-1] = 1 - y_train[-1]

    model = lgb.LGBMClassifier(**build_lgbm_params(config))
    model.fit(x_train, y_train)

    gains = model.booster_.feature_importance(importance_type="gain")
    feature_importances = sorted(
        [{"feature": feature_cols[idx], "importance": float(gain)} for idx, gain in enumerate(gains)],
        key=lambda row: row["importance"],
        reverse=True,
    )
    LOGGER.info(
        "LightGBM baseline trained on rows [%s, %s) with %s features.",
        train_start,
        train_end,
        len(feature_cols),
    )
    return model, feature_cols, feature_importances


def predict_probability(row: dict[str, Any], model: Any, feature_columns: list[str]) -> float:
    if model is not None and feature_columns:
        x_live = np.array([[safe_float(row.get(feature_name), 0.0) for feature_name in feature_columns]], dtype=float)
        return float(model.predict_proba(x_live)[0, 1])
    return fallback_probability(row)


def fallback_probability(row: dict[str, Any]) -> float:
    rsi = safe_float(row.get("rsi_14"), 50.0)
    macd_hist = safe_float(row.get("macd_hist_12_26_9"), 0.0)
    close = max(safe_float(row.get("close"), 1.0), 1e-9)
    bb_middle = safe_float(row.get("bb_middle_20"), close)
    atr = safe_float(row.get("atr_14"), 0.0)

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
    return float(np.clip(1.0 / (1.0 + np.exp(-linear_score)), 0.0, 1.0))


def derive_shap_rule(
    data: pl.DataFrame,
    cursor: int,
    model: Any,
    feature_cols: list[str],
) -> str:
    if model is None or not feature_cols:
        return "Most recent calibration used the baseline model without SHAP-ready feature importance."

    lookback = min(300, cursor + 1)
    segment = data.slice(cursor - lookback + 1, lookback)
    if segment.height < 20:
        return "Not enough held-out observations yet to derive a stable SHAP rule."

    x_segment = segment.select(feature_cols).to_numpy()
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
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.warning("SHAP rule extraction failed: %s", exc)
        return "SHAP rule extraction failed for this run."
