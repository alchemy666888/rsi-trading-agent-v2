from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt  # type: ignore[import-untyped]
import polars as pl

from agents.features import build_features
from agents.state import SplitMetadata

LOGGER = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_market_frame(raw_df: pl.DataFrame) -> str:
    hash_columns = [column for column in ["timestamp", "open", "high", "low", "close", "volume", "eth_close", "eth_volume"] if column in raw_df.columns]
    canonical = raw_df.select(hash_columns).sort("timestamp")
    payload = json.dumps(canonical.to_dicts(), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_snapshot(snapshot_path: Path) -> pl.DataFrame:
    if snapshot_path.suffix == ".parquet":
        return pl.read_parquet(snapshot_path)
    if snapshot_path.suffix == ".csv":
        return pl.read_csv(snapshot_path)
    raise ValueError(f"Unsupported snapshot format: {snapshot_path.suffix}")


def _write_snapshot(raw_df: pl.DataFrame, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.suffix == ".parquet":
        raw_df.write_parquet(snapshot_path)
        return
    if snapshot_path.suffix == ".csv":
        raw_df.write_csv(snapshot_path)
        return
    raise ValueError(f"Unsupported snapshot format: {snapshot_path.suffix}")


def fetch_exchange_snapshot(config: dict[str, Any]) -> pl.DataFrame:
    asset_cfg = config["asset"]
    api_cfg = config.get("api_keys", {})
    exchange_name = asset_cfg["exchange"]
    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls(
        {
            "enableRateLimit": True,
            "apiKey": api_cfg.get("ccxt_api_key", ""),
            "secret": api_cfg.get("ccxt_api_secret", ""),
        }
    )

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
    except Exception as exc:  # pragma: no cover - exchange best effort
        LOGGER.warning("Could not fetch ETH proxy data: %s", exc)

    snapshot_cfg = config.get("snapshot", {})
    if bool(snapshot_cfg.get("auto_write", False)):
        snapshot_path = Path(str(snapshot_cfg["path"]))
        _write_snapshot(raw_df, snapshot_path)
        LOGGER.info("Wrote market snapshot to %s", snapshot_path)

    return raw_df


def load_historical_data(config: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    source_mode = str(dataset_cfg.get("source_mode", "exchange"))
    snapshot_cfg = config.get("snapshot", {})
    snapshot_path = Path(str(snapshot_cfg.get("path", "snapshots/btcusdt_15m.parquet")))

    snapshot_hash: str | None = None
    if source_mode == "snapshot":
        raw_df = _read_snapshot(snapshot_path)
        source_ref = str(snapshot_path)
        snapshot_hash = _hash_file(snapshot_path)
    elif source_mode == "exchange":
        raw_df = fetch_exchange_snapshot(config)
        source_ref = f"exchange:{config['asset']['exchange']}"
    else:
        raise ValueError(f"Unsupported dataset source mode: {source_mode}")

    raw_df = raw_df.sort("timestamp")
    raw_data_hash = _hash_market_frame(raw_df)
    feature_cfg = dict(config.get("features", {}))
    feature_cfg.setdefault("timeframe", config.get("asset", {}).get("timeframe", "15m"))
    feature_df = build_features(raw_df, feature_cfg=feature_cfg)
    metadata = build_dataset_metadata(
        feature_df,
        source_mode=source_mode,
        source_ref=source_ref,
        raw_data_hash=raw_data_hash,
        snapshot_hash=snapshot_hash,
        snapshot_path=str(snapshot_path) if source_mode == "snapshot" else None,
    )
    return feature_df, metadata


def build_dataset_metadata(
    feature_df: pl.DataFrame,
    source_mode: str,
    source_ref: str,
    *,
    raw_data_hash: str | None = None,
    snapshot_hash: str | None = None,
    snapshot_path: str | None = None,
) -> dict[str, Any]:
    timestamps = feature_df["timestamp"].to_list()
    missing_values_total = 0
    missing_by_column: dict[str, int] = {}
    for column in feature_df.columns:
        null_count = int(feature_df[column].null_count())
        if null_count > 0:
            missing_by_column[column] = null_count
            missing_values_total += null_count

    timestamp_start = int(timestamps[0]) if timestamps else None
    timestamp_end = int(timestamps[-1]) if timestamps else None
    payload = {
        "rows": feature_df.height,
        "columns": feature_df.width,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "timestamp_start_utc": datetime.fromtimestamp(timestamp_start / 1000.0, tz=timezone.utc).isoformat()
        if timestamp_start is not None
        else None,
        "timestamp_end_utc": datetime.fromtimestamp(timestamp_end / 1000.0, tz=timezone.utc).isoformat()
        if timestamp_end is not None
        else None,
        "source_mode": source_mode,
        "source_ref": source_ref,
        "raw_data_hash": raw_data_hash,
        "snapshot_path": snapshot_path,
        "snapshot_hash": snapshot_hash,
        "benchmark_eligible": source_mode == "snapshot",
        "missing_values_total": missing_values_total,
        "missing_values_by_column": missing_by_column,
    }
    payload["dataset_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return payload


def _required_embargo_gap(config: dict[str, Any]) -> int:
    feature_cfg = config.get("features", {})
    walk_cfg = config.get("walk_forward", {})
    max_feature_lag = int(feature_cfg.get("max_lag_bars", 60))
    purge_bars = int(walk_cfg.get("purge_bars", 5))
    # Label horizon is one bar (target_up uses close.shift(-1)).
    return max(purge_bars, max_feature_lag + 1)


def compute_split_metadata(config: dict[str, Any], total_rows: int) -> SplitMetadata:
    split_cfg = config.get("splits", {})
    walk_cfg = config.get("walk_forward", {})
    feature_cfg = config.get("features", {})
    warmup_bars = int(config["runtime"].get("warmup_bars", 200))
    train_ratio = float(split_cfg.get("train_ratio", 0.6))
    validation_ratio = float(split_cfg.get("validation_ratio", 0.2))
    minimum_oos_bars = int(split_cfg.get("minimum_oos_bars", 100))
    purge_bars = int(walk_cfg.get("purge_bars", 5))
    max_feature_lag = int(feature_cfg.get("max_lag_bars", 60))
    required_embargo_gap = _required_embargo_gap(config)

    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    train_end = max(train_end, warmup_bars + 50)
    validation_end = max(validation_end, train_end + 50)
    validation_end = min(validation_end, total_rows - minimum_oos_bars)
    oos_start = max(validation_end + required_embargo_gap, warmup_bars)
    oos_end = min(total_rows - 1, oos_start + int(config["runtime"]["max_cycles"]))

    if oos_start >= total_rows - 2:
        raise RuntimeError("Not enough rows to create train/validation/out-of-sample splits.")

    return {
        "train_start": 0,
        "train_end": train_end,
        "validation_start": train_end,
        "validation_end": validation_end,
        "oos_start": oos_start,
        "oos_end": min(total_rows - 1, max(oos_end, oos_start + 2)),
        "purge_bars": purge_bars,
        "max_feature_lag": max_feature_lag,
        "required_embargo_gap": required_embargo_gap,
    }
