"""
NASDAQ OHLC Indicator Calculator (TA-Lib, vectorised)
======================================================
Reads a single parquet file containing all NASDAQ stocks (with a 'Ticker'
column), calculates the selected technical indicators for every ticker using
TA-Lib, and writes the enriched data back to a parquet file.

Performance strategy
--------------------
Rather than slicing the DataFrame per ticker and appending chunks, this
version:
  1. Builds a (ticker_codes, col_t) index once from groupby.cumcount().
  2. Pre-allocates one float64 output array per indicator column.
  3. Loops over tickers only for the TA-Lib calls (unavoidable — TA-Lib is
     not vectorised across tickers), writing results directly into the
     pre-allocated arrays via fancy indexing.
  4. Assigns all output arrays to the DataFrame in one shot at the end —
     no per-ticker DataFrame copies, no pd.concat.

Dependencies:
    pip install pandas pyarrow ta-lib

Usage:
    python calculate_indicators.py -i market_data_full.parquet -o market_data_n_ind.parquet
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# INDICATOR CONFIGURATION
# Toggle any indicator on/off by setting it to True / False.
# Adjust parameters to taste — they are all in one place here.
# =============================================================================

INDICATORS = {
    # --- Trend ---
    "SMA": {
        "enabled": True,
        "params": {
            "periods": [20, 50, 200],   # one output column per period
        },
    },
    "EMA": {
        "enabled": True,
        "params": {
            "periods": [12, 26, 50],    # one output column per period
        },
    },
    "MACD": {
        "enabled": True,
        "params": {
            "fast_period":   12,
            "slow_period":   26,
            "signal_period":  9,
        },
    },

    # --- Momentum ---
    "RSI": {
        "enabled": True,
        "params": {
            "period": 14,
        },
    },
    "STOCH": {
        "enabled": True,
        "params": {
            "fastk_period":  5,
            "slowk_period":  3,
            "slowk_matype":  0,   # 0 = SMA
            "slowd_period":  3,
            "slowd_matype":  0,
        },
    },
    "CCI": {
        "enabled": True,
        "params": {
            "period": 20,
        },
    },
}

# =============================================================================
# COLUMN NAME MAPPING
# Change these if your parquet file uses different column names.
# =============================================================================

COL_TICKER = "Ticker"
COL_DATE   = "Date"       # set to None to skip sorting
COL_OPEN   = "Open"
COL_HIGH   = "High"
COL_LOW    = "Low"
COL_CLOSE  = "Close"      # swap to "Adj Close" for split/dividend-adjusted prices
COL_VOLUME = "Volume"


# =============================================================================
# HELPERS
# =============================================================================

def _safe(arr: np.ndarray) -> np.ndarray:
    """Clean float64 C-contiguous array required by TA-Lib."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _output_names() -> list[str]:
    """Return the ordered list of all indicator output column names."""
    names = []
    cfg = INDICATORS["SMA"]
    if cfg["enabled"]:
        names += [f"SMA_{p}" for p in cfg["params"]["periods"]]
    cfg = INDICATORS["EMA"]
    if cfg["enabled"]:
        names += [f"EMA_{p}" for p in cfg["params"]["periods"]]
    if INDICATORS["MACD"]["enabled"]:
        names += ["MACD", "MACD_signal", "MACD_hist"]
    cfg = INDICATORS["RSI"]
    if cfg["enabled"]:
        names.append(f"RSI_{cfg['params']['period']}")
    if INDICATORS["STOCH"]["enabled"]:
        names += ["STOCH_K", "STOCH_D"]
    cfg = INDICATORS["CCI"]
    if cfg["enabled"]:
        names.append(f"CCI_{cfg['params']['period']}")
    return names


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(input_path: Path, output_path: Path) -> None:
    t0 = time.perf_counter()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading {input_path} …")
    df = pd.read_parquet(input_path)
    n_rows = len(df)
    print(f"      {n_rows:,} rows | {df[COL_TICKER].nunique():,} tickers")

    # ── Sort ─────────────────────────────────────────────────────────────────
    if COL_DATE:
        print(f"[2/4] Sorting by [{COL_TICKER}, {COL_DATE}] …")
        df = df.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)
    else:
        print("[2/4] Skipping sort (COL_DATE is None) …")

    # ── Build flat index ──────────────────────────────────────────────────────
    # ticker_codes[i] = which ticker group row i belongs to (0..N-1)
    # col_t[i]        = position within that ticker's time series (0, 1, 2, …)
    # Built with groupby.cumcount() — no Python loop, no per-ticker slice.
    tickers       = df[COL_TICKER].unique()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    ticker_codes  = df[COL_TICKER].map(ticker_to_idx).values.astype(np.int32)
    col_t         = df.groupby(COL_TICKER, sort=False).cumcount().values.astype(np.int32)
    ticker_sizes  = df.groupby(COL_TICKER, sort=False).size().values  # (N,)

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    out_names = _output_names()
    # One NaN-filled float64 array per indicator output, length = total rows
    outputs   = {name: np.full(n_rows, np.nan, dtype=np.float64)
                 for name in out_names}

    # Extract raw numpy columns once (avoids repeated DataFrame access)
    arr_close  = df[COL_CLOSE].values.astype(np.float64)
    arr_high   = df[COL_HIGH].values.astype(np.float64)
    arr_low    = df[COL_LOW].values.astype(np.float64)

    # ── Calculate per ticker, write into pre-allocated arrays ─────────────────
    enabled = [k for k, v in INDICATORS.items() if v["enabled"]]
    print(f"[3/4] Calculating indicators: {', '.join(enabled)} …")

    # Build per-ticker row slices from ticker_sizes (df is sorted, so each
    # ticker occupies a contiguous block of rows).
    offsets = np.concatenate([[0], ticker_sizes.cumsum()])
    errors  = []
    N       = len(tickers)

    for i, ticker in enumerate(tickers):
        if i % 500 == 0 or i == N - 1:
            pct     = (i + 1) / N * 100
            elapsed = time.perf_counter() - t0
            print(f"      {i+1:>5}/{N} ({pct:5.1f}%)  [{elapsed:6.1f}s]", end="\r")

        start = offsets[i]
        end   = offsets[i + 1]

        # Direct slice into the pre-extracted numpy arrays — no DataFrame copy
        c = _safe(arr_close[start:end])
        h = _safe(arr_high[start:end])
        l = _safe(arr_low[start:end])

        try:
            # ── SMA ──────────────────────────────────────────────────────────
            cfg = INDICATORS["SMA"]
            if cfg["enabled"]:
                for p in cfg["params"]["periods"]:
                    outputs[f"SMA_{p}"][start:end] = talib.SMA(c, timeperiod=p)

            # ── EMA ──────────────────────────────────────────────────────────
            cfg = INDICATORS["EMA"]
            if cfg["enabled"]:
                for p in cfg["params"]["periods"]:
                    outputs[f"EMA_{p}"][start:end] = talib.EMA(c, timeperiod=p)

            # ── MACD ─────────────────────────────────────────────────────────
            cfg = INDICATORS["MACD"]
            if cfg["enabled"]:
                p = cfg["params"]
                macd_line, signal, hist = talib.MACD(
                    c,
                    fastperiod   = p["fast_period"],
                    slowperiod   = p["slow_period"],
                    signalperiod = p["signal_period"],
                )
                outputs["MACD"][start:end]        = macd_line
                outputs["MACD_signal"][start:end] = signal
                outputs["MACD_hist"][start:end]   = hist

            # ── RSI ──────────────────────────────────────────────────────────
            cfg = INDICATORS["RSI"]
            if cfg["enabled"]:
                period = cfg["params"]["period"]
                outputs[f"RSI_{period}"][start:end] = talib.RSI(c, timeperiod=period)

            # ── Stochastic ───────────────────────────────────────────────────
            cfg = INDICATORS["STOCH"]
            if cfg["enabled"]:
                p = cfg["params"]
                slowk, slowd = talib.STOCH(
                    h, l, c,
                    fastk_period = p["fastk_period"],
                    slowk_period = p["slowk_period"],
                    slowk_matype = p["slowk_matype"],
                    slowd_period = p["slowd_period"],
                    slowd_matype = p["slowd_matype"],
                )
                outputs["STOCH_K"][start:end] = slowk
                outputs["STOCH_D"][start:end] = slowd

            # ── CCI ──────────────────────────────────────────────────────────
            cfg = INDICATORS["CCI"]
            if cfg["enabled"]:
                period = cfg["params"]["period"]
                outputs[f"CCI_{period}"][start:end] = talib.CCI(
                    h, l, c, timeperiod=period
                )

        except Exception as exc:
            errors.append((ticker, str(exc)))

    print()  # newline after \r

    if errors:
        print(f"  ⚠  {len(errors)} ticker(s) failed:")
        for ticker, msg in errors[:10]:
            print(f"      {ticker}: {msg}")
        if len(errors) > 10:
            print(f"      … and {len(errors) - 10} more")

    # ── Assign all outputs + write parquet in one shot ────────────────────────
    print(f"[4/4] Assigning {len(out_names)} columns and writing {output_path} …")
    for name, arr in outputs.items():
        df[name] = arr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")

    elapsed = time.perf_counter() - t0
    print(f"\n✓  Done in {elapsed:.1f}s")
    print(f"   Output rows : {len(df):,}")
    print(f"   New columns : {len(out_names)}  →  {out_names}")
    print(f"   Saved to    : {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate TA-Lib indicators for all NASDAQ tickers in a parquet file."
    )
    parser.add_argument("--input",  "-i", required=True, type=Path,
                        help="Path to the input parquet file.")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Path for the output parquet file.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    run(args.input, args.output)


if __name__ == "__main__":
    main()