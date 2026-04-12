from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated run_mvp executions across distinct snapshot row windows "
            "to accumulate multi-window regression evidence."
        )
    )
    parser.add_argument("--config", default="config/config.yaml", help="Base YAML config path.")
    parser.add_argument("--windows", type=int, default=None, help="Number of historical windows to run.")
    parser.add_argument("--window-bars", type=int, default=None, help="Rows per window (default auto).")
    parser.add_argument(
        "--stride-bars",
        type=int,
        default=None,
        help="Stride between window starts. Default distributes windows evenly.",
    )
    parser.add_argument(
        "--keep-console-logs",
        action="store_true",
        help="Keep per-step console logs from run_mvp (disabled by default for concise runs).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned windows without executing runs.")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config payload must be a mapping. Got: {type(payload).__name__}")
    return payload


def _resolve_snapshot_path(config: dict[str, Any]) -> Path:
    dataset_cfg = dict(config.get("dataset", {}))
    source_mode = str(dataset_cfg.get("source_mode", "snapshot"))
    if source_mode != "snapshot":
        raise ValueError(
            "run_regression_windows requires dataset.source_mode='snapshot' so windows are reproducible."
        )
    snapshot_cfg = dict(config.get("snapshot", {}))
    snapshot_raw = str(snapshot_cfg.get("path", ""))
    if not snapshot_raw:
        raise ValueError("snapshot.path must be set for snapshot mode.")
    snapshot_path = Path(snapshot_raw).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = (PROJECT_ROOT / snapshot_path).resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
    return snapshot_path


def _default_window_bars(total_rows: int, config: dict[str, Any]) -> int:
    configured_cycles = int(config.get("runtime", {}).get("max_cycles", 200))
    minimum_oos = int(config.get("splits", {}).get("minimum_oos_bars", 200))
    floor = max(1200, (configured_cycles + minimum_oos) * 3)
    target = max(floor, int(total_rows * 0.6))
    return min(total_rows, target)


def _compute_window_starts(
    *,
    total_rows: int,
    windows: int,
    window_bars: int,
    stride_bars: int | None,
) -> list[int]:
    if windows <= 0:
        raise ValueError(f"--windows must be >= 1. Got {windows}.")
    if window_bars <= 1:
        raise ValueError(f"--window-bars must be >= 2. Got {window_bars}.")
    if window_bars > total_rows:
        raise ValueError(
            f"--window-bars ({window_bars}) cannot exceed dataset rows ({total_rows})."
        )
    max_start = total_rows - window_bars
    if windows == 1:
        return [max_start]

    if stride_bars is None:
        stride = max(1, max_start // (windows - 1))
    else:
        stride = int(stride_bars)
        if stride <= 0:
            raise ValueError(f"--stride-bars must be >= 1 when provided. Got {stride}.")

    starts: list[int] = []
    for idx in range(windows):
        start = min(idx * stride, max_start)
        starts.append(start)
    starts = sorted(set(starts))
    if not starts:
        starts = [max_start]
    return starts


def _load_regression_ledger(artifact_root: Path) -> list[dict[str, Any]]:
    ledger_path = artifact_root / "regression_ledger.json"
    if not ledger_path.exists():
        return []
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _historical_window_key(entry: dict[str, Any]) -> str:
    window = entry.get("dataset_window", {})
    if not isinstance(window, dict):
        window = {}
    return "|".join(
        [
            str(window.get("snapshot_hash", "n/a")),
            str(window.get("raw_data_hash", "n/a")),
            str(window.get("dataset_hash", "n/a")),
            str(window.get("timestamp_start", "n/a")),
            str(window.get("timestamp_end", "n/a")),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_config_path(str(args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    base_config = _load_yaml(config_path)
    snapshot_path = _resolve_snapshot_path(base_config)
    artifact_root = PROJECT_ROOT / str(base_config.get("runtime", {}).get("artifact_output_dir", "artifacts"))
    total_rows = pl.read_parquet(snapshot_path).height if snapshot_path.suffix == ".parquet" else pl.read_csv(snapshot_path).height

    windows = int(args.windows) if args.windows is not None else int(base_config.get("readiness", {}).get("min_historical_windows", 5))
    window_bars = int(args.window_bars) if args.window_bars is not None else _default_window_bars(total_rows, base_config)
    starts = _compute_window_starts(
        total_rows=total_rows,
        windows=windows,
        window_bars=window_bars,
        stride_bars=args.stride_bars,
    )
    jobs = [{"start": start, "end": start + window_bars} for start in starts]

    print(f"[regression-windows] config={config_path}")
    print(f"[regression-windows] snapshot={snapshot_path} rows={total_rows}")
    print(f"[regression-windows] planned_jobs={len(jobs)} window_bars={window_bars}")
    for idx, job in enumerate(jobs, start=1):
        print(f"  - job {idx}: start={job['start']} end={job['end']}")
    if args.dry_run:
        return

    with tempfile.TemporaryDirectory(prefix="rsi-regression-windows-") as temp_dir:
        temp_root = Path(temp_dir)
        for idx, job in enumerate(jobs, start=1):
            run_config = copy.deepcopy(base_config)
            dataset_cfg = dict(run_config.get("dataset", {}))
            dataset_cfg["source_mode"] = "snapshot"
            dataset_cfg["row_slice"] = {"start": int(job["start"]), "end": int(job["end"])}
            run_config["dataset"] = dataset_cfg

            logging_cfg = dict(run_config.get("logging", {}))
            if not args.keep_console_logs:
                logging_cfg["enable_console_logs"] = False
            run_config["logging"] = logging_cfg

            config_out = temp_root / f"window_{idx:02d}.yaml"
            config_out.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            print(
                f"[regression-windows] running job {idx}/{len(jobs)} "
                f"(start={job['start']}, end={job['end']})"
            )
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "run_mvp.py"), "--config", str(config_out)],
                cwd=PROJECT_ROOT,
                check=True,
            )

    ledger = _load_regression_ledger(artifact_root)
    historical_keys = {_historical_window_key(entry) for entry in ledger}
    print(
        "[regression-windows] complete "
        f"ledger_entries={len(ledger)} unique_windows={len(historical_keys)}"
    )


if __name__ == "__main__":
    main()
