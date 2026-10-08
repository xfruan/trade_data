"""
calc_indicators_torch.py
========================
GPU-accelerated indicator calculator using torch_indicators.py.

Instead of looping ticker-by-ticker (as the TA-Lib version does), this
script packs all tickers into padded (N, T) tensors and sends the entire
batch to the GPU in one shot — dramatically faster on large universes.

Dependencies:
    pip install pandas pyarrow torch

    torch_indicators.py must be in the same directory (or on PYTHONPATH).

Usage:
    uv run calc_indicators_torch.py -i market_data_full.parquet -o market_data_n_ind.parquet

    # Force CPU even when a GPU is present
    uv run calc_indicators_torch.py -i ... -o ... --device cpu

    # Tune batch size if you run out of VRAM (default: all tickers at once)
    uv run calc_indicators_torch.py -i ... -o ... --batch-size 500
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_indicators import IndicatorEngine, resolve_device

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
            "periods": [20, 50, 200],   # one column per period
        },
    },
    "EMA": {
        "enabled": True,
        "params": {
            "periods": [12, 26, 50],    # one column per period
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
            "fastk_period": 5,
            "slowk_period": 3,
            "slowd_period": 3,
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
# =============================================================================

COL_TICKER = "Ticker"
COL_DATE   = "Date"
COL_OPEN   = "Open"
COL_HIGH   = "High"
COL_LOW    = "Low"
COL_CLOSE  = "Close"       # swap to "Adj Close" for split-adjusted prices
COL_VOLUME = "Volume"


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_index(
    df: pd.DataFrame,
    ticker_to_idx: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build two integer arrays of length len(df) in one vectorised pass:
        ticker_codes  — matrix row  (0..N-1) for each DataFrame row
        col_t         — time-step   (0..T-1) for each DataFrame row

    The DataFrame must already be sorted by [COL_TICKER, COL_DATE].
    These arrays are reused for both packing and unpacking, so we only
    pay this cost once.
    """
    ticker_codes = df[COL_TICKER].map(ticker_to_idx).values.astype(np.int32)
    # cumcount gives the within-group position (0, 1, 2, …) without any loop
    col_t = df.groupby(COL_TICKER, sort=False).cumcount().values.astype(np.int32)
    return ticker_codes, col_t


# =============================================================================
# PACKING  (DataFrame column → (N, T) matrix)
# =============================================================================

def pack_tickers(
    df: pd.DataFrame,
    ticker_codes: np.ndarray,
    col_t: np.ndarray,
    col: str,
    N: int,
    T: int,
) -> np.ndarray:
    """
    Vectorised pack: place each value directly into its (ticker_row, time_col)
    cell using numpy fancy indexing — no Python loop over tickers.
    """
    mat = np.full((N, T), np.nan, dtype=np.float32)
    mat[ticker_codes, col_t] = df[col].values.astype(np.float32)
    return mat


# =============================================================================
# UNPACKING  ((N, T) result matrices → DataFrame columns)
# =============================================================================

def unpack_results(
    indicator_tensors: dict[str, torch.Tensor],
    df: pd.DataFrame,
    ticker_codes: np.ndarray,
    col_t: np.ndarray,
) -> pd.DataFrame:
    """
    Vectorised unpack: read every indicator value with a single numpy
    fancy-index gather per indicator — no Python loop over tickers.

    All (N, T) result tensors are stacked into one (n_ind, N, T) array and
    gathered in one shot, then split back into named DataFrame columns.
    """
    result  = df.copy()
    names   = list(indicator_tensors.keys())

    # Move all tensors to CPU numpy and stack: (n_indicators, N, T)
    matrices = np.stack([t.numpy() for t in indicator_tensors.values()], axis=0)

    # Single fancy-index gather for every indicator at once: (n_indicators, n_rows)
    gathered = matrices[:, ticker_codes, col_t]

    for i, name in enumerate(names):
        result[name] = gathered[i]

    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(input_path: Path, output_path: Path, device_str: str | None,
        batch_size: int | None) -> None:
    t0 = time.perf_counter()

    # ── Device ───────────────────────────────────────────────────────────────
    device = resolve_device(device_str)
    print(f"[0/5] Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram  = props.total_memory / 1024**3
        print(f"      GPU : {props.name}  ({vram:.1f} GB VRAM)")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"[1/5] Loading {input_path} …")
    df = pd.read_parquet(input_path)
    print(f"      {len(df):,} rows | {df[COL_TICKER].nunique():,} tickers")

    # ── Sort ─────────────────────────────────────────────────────────────────
    print(f"[2/5] Sorting by [{COL_TICKER}, {COL_DATE}] …")
    df = df.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)

    # ── Build index + padded tensors ─────────────────────────────────────────
    print("[3/5] Building index and packing tensors …")
    tickers       = df[COL_TICKER].unique()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    N             = len(tickers)
    T             = int(df.groupby(COL_TICKER).size().max())

    # Build (ticker_codes, col_t) once — reused for pack AND unpack
    ticker_codes, col_t = build_index(df, ticker_to_idx)

    print(f"      N={N} tickers  T={T} max time steps  →  tensor shape ({N}, {T})")

    close_mat = pack_tickers(df, ticker_codes, col_t, COL_CLOSE, N, T)
    high_mat  = pack_tickers(df, ticker_codes, col_t, COL_HIGH,  N, T)
    low_mat   = pack_tickers(df, ticker_codes, col_t, COL_LOW,   N, T)

    # ── Calculate indicators ──────────────────────────────────────────────────
    enabled = [k for k, v in INDICATORS.items() if v["enabled"]]
    print(f"[4/5] Calculating indicators: {', '.join(enabled)} …")

    engine = IndicatorEngine(config=INDICATORS, device=device)

    all_results: dict[str, torch.Tensor] = {}
    bs = batch_size or N

    for start in range(0, N, bs):
        end   = min(start + bs, N)
        chunk = slice(start, end)
        pct   = end / N * 100

        print(f"      Batch {start}:{end}  ({pct:.1f}%) …", end="\r")

        c_t = torch.tensor(close_mat[chunk], device=device)
        h_t = torch.tensor(high_mat[chunk],  device=device)
        l_t = torch.tensor(low_mat[chunk],   device=device)

        with torch.no_grad():
            batch_out = engine.compute(close=c_t, high=h_t, low=l_t)

        # Accumulate on CPU to avoid holding everything in VRAM
        if not all_results:
            all_results = {k: v.cpu() for k, v in batch_out.items()}
        else:
            for k, v in batch_out.items():
                all_results[k] = torch.cat([all_results[k], v.cpu()], dim=0)

    print()  # newline after \r

    # ── Unpack + write ────────────────────────────────────────────────────────
    print("[5/5] Unpacking results and writing parquet …")
    result = unpack_results(all_results, df, ticker_codes, col_t)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="snappy")

    elapsed  = time.perf_counter() - t0
    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\n✓  Done in {elapsed:.1f}s")
    print(f"   Output rows : {len(result):,}")
    print(f"   New columns : {len(new_cols)}  →  {new_cols}")
    print(f"   Saved to    : {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated indicator calculator using PyTorch."
    )
    parser.add_argument("--input",  "-i", required=True, type=Path,
                        help="Input parquet file.")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output parquet file.")
    parser.add_argument("--device", "-d", default=None,
                        help="Torch device: 'cuda', 'mps', 'cpu'. "
                             "Auto-detects best available when omitted.")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Number of tickers per GPU batch. "
                             "Reduce if you hit VRAM limits. "
                             "Default: all tickers in one batch.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    run(args.input, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()