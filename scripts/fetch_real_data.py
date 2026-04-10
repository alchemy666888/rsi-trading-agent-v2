"""Fetch real BTC/USDT 1m OHLCV data from Binance via CCXT with pagination.

Usage:
    uv run python scripts/fetch_real_data.py [--months 6] [--symbol BTC/USDT] [--out data/btc_1m_real.csv]

The script pages backwards through Binance history using the `since` parameter,
collecting up to `months` of 1-minute candles, then saves to a CSV file.
Only needs to run once (or weekly to refresh).
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

BARS_PER_PAGE = 1_000          # Binance hard cap per fetch_ohlcv call
MS_PER_BAR = 60_000            # 1-minute bars → 60,000 ms each


def fetch_ohlcv_paginated(
    symbol: str,
    months: int,
    out_path: Path,
) -> int:
    """Download `months` of 1m OHLCV from Binance and write to `out_path`.

    Returns the number of rows written.
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_bars = months * 30 * 24 * 60          # approximate; enough for training
    since_ms = now_ms - target_bars * MS_PER_BAR

    LOGGER.info(
        "Fetching %s 1m bars for %s from %s",
        target_bars,
        symbol,
        datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat(),
    )

    all_rows: list[list] = []
    seen_timestamps: set[int] = set()
    cursor = since_ms

    while cursor < now_ms - MS_PER_BAR:
        try:
            batch = exchange.fetch_ohlcv(symbol, "1m", since=cursor, limit=BARS_PER_PAGE)
        except ccxt.NetworkError as exc:
            LOGGER.warning("Network error, retrying in 5s: %s", exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            LOGGER.error("Exchange error: %s — aborting", exc)
            break

        if not batch:
            LOGGER.info("Empty batch at cursor %s — reached end of available history.", cursor)
            break

        new_rows = 0
        for row in batch:
            ts = int(row[0])
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                all_rows.append([ts, row[1], row[2], row[3], row[4], row[5]])
                new_rows += 1

        latest_ts = int(batch[-1][0])
        LOGGER.info(
            "Fetched %d new bars (total %d), latest=%s",
            new_rows,
            len(all_rows),
            datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).isoformat(),
        )

        if new_rows == 0:
            break  # no forward progress; avoid infinite loop

        # Advance cursor to one bar past the latest received timestamp
        cursor = latest_ts + MS_PER_BAR

    if not all_rows:
        LOGGER.error("No data fetched — check exchange connectivity and API limits.")
        return 0

    # Sort by timestamp ascending and write
    all_rows.sort(key=lambda r: r[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(all_rows)

    LOGGER.info("Saved %d rows to %s", len(all_rows), out_path)
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real BTC 1m OHLCV data from Binance.")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch (default: 6)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument(
        "--out",
        default="data/btc_1m_real.csv",
        help="Output CSV path (default: data/btc_1m_real.csv)",
    )
    args = parser.parse_args()

    n = fetch_ohlcv_paginated(
        symbol=args.symbol,
        months=args.months,
        out_path=Path(args.out),
    )
    if n < 1000:
        raise SystemExit(f"Too few bars fetched ({n}). Check connectivity and try again.")
    print(f"Done. {n:,} rows written to {args.out}")


if __name__ == "__main__":
    main()
