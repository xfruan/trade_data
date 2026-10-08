"""
torch_indicators.py
====================
Vectorised technical indicators implemented in pure PyTorch.

All functions operate on **batched, padded tensors** of shape (N, T) where
    N = number of tickers processed in one batch
    T = time-series length (all tickers padded to the same length)

NaN padding is preserved: masked positions propagate NaN through every
calculation exactly as TA-Lib does.

Device handling
---------------
Pass ``device="cuda"`` (or ``"mps"`` on Apple Silicon) to run entirely on
GPU.  CPU is the automatic fallback when no accelerator is available.

Typical usage
-------------
    from torch_indicators import IndicatorEngine

    engine = IndicatorEngine(device="cuda")

    # prices: torch.Tensor of shape (N, T)  — one row per ticker
    # high / low / volume: same shape
    results: dict[str, torch.Tensor] = engine.compute(
        close  = close,
        high   = high,
        low    = low,
        open_  = open_,
        volume = volume,
    )
    # results["SMA_20"] → (N, T) tensor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """
    Return the best available device.
    Priority: requested → CUDA → MPS → CPU.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Low-level primitives  (all operate on float32 (N, T) tensors)
# ---------------------------------------------------------------------------

def _nan_to_num(x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.full_like(x, val), x)


def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Causal rolling sum over the time axis (dim=1).
    Output positions 0 … window-2 are NaN (insufficient history).
    Uses conv1d with a ones kernel — O(N·T) regardless of window size.
    """
    N, T = x.shape
    # Replace NaN with 0 for convolution; we'll re-mask afterwards
    mask   = ~torch.isnan(x)                          # (N, T)
    x_fill = _nan_to_num(x, 0.0)

    # conv1d expects (batch, channels, length)
    x_3d    = x_fill.unsqueeze(1)                     # (N, 1, T)
    mask_3d = mask.float().unsqueeze(1)

    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    pad    = window - 1

    rsum      = F.conv1d(x_3d,    kernel, padding=pad)[:, 0, :T]   # (N, T)
    valid_cnt = F.conv1d(mask_3d, kernel, padding=pad)[:, 0, :T]

    # Positions where fewer than `window` valid values exist → NaN
    out = rsum.clone()
    out[valid_cnt < window] = float("nan")
    return out


def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    return _rolling_sum(x, window) / window


def _ema_single(x: torch.Tensor, period: int) -> torch.Tensor:
    """
    Wilder / exponential moving average along dim=1.
    k = 2/(period+1).  First valid output is the SMA of the first `period`
    values (matching TA-Lib's behaviour).

    This is inherently sequential so we use a compiled Python loop, but it
    runs entirely in GPU memory — no CPU↔GPU transfers per step.
    """
    N, T   = x.shape
    k      = 2.0 / (period + 1)
    out    = torch.full((N, T), float("nan"), device=x.device, dtype=x.dtype)

    # Seed: SMA of first `period` values per ticker
    seed_sum   = torch.zeros(N, device=x.device, dtype=x.dtype)
    seed_count = torch.zeros(N, device=x.device, dtype=x.dtype)

    prev   = torch.full((N,), float("nan"), device=x.device, dtype=x.dtype)
    seeded = torch.zeros(N, dtype=torch.bool, device=x.device)

    for t in range(T):
        col   = x[:, t]                       # (N,)
        valid = ~torch.isnan(col)

        # Accumulate toward seed SMA
        not_seeded = ~seeded
        accumulate = valid & not_seeded
        seed_sum   = torch.where(accumulate, seed_sum + col, seed_sum)
        seed_count = torch.where(accumulate, seed_count + 1, seed_count)

        # Seed tickers that just completed their first `period` values
        ready = not_seeded & (seed_count >= period)
        if ready.any():
            seed_val = seed_sum[ready] / seed_count[ready]
            prev     = prev.clone()
            prev[ready] = seed_val
            seeded[ready] = True
            out[:, t] = torch.where(ready, prev, out[:, t])

        # EMA update for already-seeded tickers
        running = seeded & valid & ~ready
        if running.any():
            prev = prev.clone()
            prev[running] = col[running] * k + prev[running] * (1 - k)
            out[:, t] = torch.where(running, prev, out[:, t])

    return out


# ---------------------------------------------------------------------------
# Public indicator functions
# ---------------------------------------------------------------------------

def sma(close: torch.Tensor, period: int) -> torch.Tensor:
    """Simple Moving Average — (N, T) → (N, T)."""
    return _rolling_mean(close, period)


def ema(close: torch.Tensor, period: int) -> torch.Tensor:
    """Exponential Moving Average — (N, T) → (N, T)."""
    return _ema_single(close, period)


def macd(
    close: torch.Tensor,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MACD, Signal, and Histogram.
    Returns three (N, T) tensors: (macd_line, signal_line, histogram).
    """
    fast_ema   = _ema_single(close, fast_period)
    slow_ema   = _ema_single(close, slow_period)
    macd_line  = fast_ema - slow_ema
    signal     = _ema_single(macd_line, signal_period)
    histogram  = macd_line - signal
    return macd_line, signal, histogram


def rsi(close: torch.Tensor, period: int = 14) -> torch.Tensor:
    """
    Relative Strength Index using Wilder smoothing — (N, T) → (N, T).
    Matches TA-Lib's implementation.
    """
    N, T  = close.shape
    delta = torch.diff(close, dim=1)                       # (N, T-1)
    delta = torch.cat([torch.full((N, 1), float("nan"),
                                  device=close.device, dtype=close.dtype),
                       delta], dim=1)                       # (N, T)
    gain  = torch.clamp(delta,  min=0.0)
    loss  = torch.clamp(-delta, min=0.0)

    # Wilder EMA (period-day)
    avg_gain = _ema_single(gain, period)
    avg_loss = _ema_single(loss, period)

    rs  = avg_gain / (avg_loss + 1e-10)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss ≈ 0 and avg_gain > 0, RSI → 100
    rsi_val = torch.where((avg_loss < 1e-10) & (avg_gain > 0),
                          torch.full_like(rsi_val, 100.0), rsi_val)
    # Both zero → NaN
    rsi_val = torch.where((avg_loss < 1e-10) & (avg_gain < 1e-10),
                          torch.full_like(rsi_val, float("nan")), rsi_val)
    return rsi_val


def stochastic(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slow Stochastic Oscillator — returns (SlowK, SlowD) each (N, T).
    SlowK = SMA(fastK, slowk_period)
    SlowD = SMA(SlowK, slowd_period)
    """
    # Rolling highest-high and lowest-low over fastk_period
    N, T   = close.shape
    hh = torch.full_like(close, float("nan"))
    ll = torch.full_like(close, float("nan"))

    # Unfold: (N, T, fastk_period) — sliding windows
    pad_h = F.pad(high,  (fastk_period - 1, 0), value=float("nan"))
    pad_l = F.pad(low,   (fastk_period - 1, 0), value=float("nan"))

    h_win = pad_h.unfold(1, fastk_period, 1)   # (N, T, fastk_period)
    l_win = pad_l.unfold(1, fastk_period, 1)

    # Replace NaN with -inf / +inf so max/min ignores padding
    h_win_filled = torch.where(torch.isnan(h_win),
                               torch.full_like(h_win, float("-inf")), h_win)
    l_win_filled = torch.where(torch.isnan(l_win),
                               torch.full_like(l_win, float("inf")),  l_win)
    hh = h_win_filled.max(dim=2).values         # (N, T)
    ll = l_win_filled.min(dim=2).values

    # Positions where the entire window was NaN → restore NaN
    all_nan_h = torch.isnan(h_win).all(dim=2)
    all_nan_l = torch.isnan(l_win).all(dim=2)
    hh = torch.where(all_nan_h, torch.full_like(hh, float("nan")), hh)
    ll = torch.where(all_nan_l, torch.full_like(ll, float("nan")), ll)

    denom  = hh - ll
    fastk  = torch.where(
        denom.abs() < 1e-10,
        torch.full_like(denom, float("nan")),
        (close - ll) / denom * 100.0,
    )

    slowk = _rolling_mean(fastk, slowk_period)
    slowd = _rolling_mean(slowk, slowd_period)
    return slowk, slowd


def cci(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    period: int = 20,
) -> torch.Tensor:
    """
    Commodity Channel Index — (N, T) → (N, T).
    CCI = (TP - SMA(TP, n)) / (0.015 * MeanAbsDev(TP, n))
    """
    tp   = (high + low + close) / 3.0            # typical price
    sma_ = _rolling_mean(tp, period)

    # Mean absolute deviation via unfolded windows
    pad    = F.pad(tp, (period - 1, 0), value=float("nan"))
    win    = pad.unfold(1, period, 1)             # (N, T, period)
    sma_ex = sma_.unsqueeze(2)                    # (N, T, 1)
    abs_dev  = torch.abs(win - sma_ex)
    nan_mask = torch.isnan(abs_dev)
    abs_dev  = torch.where(nan_mask, torch.zeros_like(abs_dev), abs_dev)
    valid_n  = (~nan_mask).sum(dim=2).clamp(min=1)
    mad      = abs_dev.sum(dim=2) / valid_n

    return torch.where(
        mad.abs() < 1e-10,
        torch.full_like(mad, float("nan")),
        (tp - sma_) / (0.015 * mad),
    )


# ---------------------------------------------------------------------------
# IndicatorEngine — high-level, batched, GPU-aware
# ---------------------------------------------------------------------------

class IndicatorEngine:
    """
    High-level engine that accepts a (N, T) padded tensor for each price
    series, runs all enabled indicators on the chosen device, and returns
    a dictionary of named (N, T) result tensors.

    Parameters
    ----------
    config : dict
        Same INDICATORS dict used by the calc script.  If None, all
        indicators run with default parameters.
    device : str | torch.device | None
        Target device.  Auto-selects CUDA → MPS → CPU when None.
    dtype : torch.dtype
        Working precision.  float32 is the default (fast on GPU).
        Use float64 for higher precision at the cost of memory/speed.
    """

    def __init__(
        self,
        config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype  = dtype
        self.config = config or _default_config()
        print(f"[IndicatorEngine] device={self.device}  dtype={self.dtype}")

    # ------------------------------------------------------------------
    def _t(self, arr) -> torch.Tensor:
        """Convert numpy array or tensor to the engine's device/dtype."""
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=self.dtype)
        import numpy as np
        return torch.tensor(arr, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    def compute(
        self,
        close:  "np.ndarray | torch.Tensor",
        high:   "np.ndarray | torch.Tensor",
        low:    "np.ndarray | torch.Tensor",
        open_:  "np.ndarray | torch.Tensor | None" = None,
        volume: "np.ndarray | torch.Tensor | None" = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run all enabled indicators.

        Inputs are (N, T) arrays/tensors — N tickers, T time steps.
        Pad shorter tickers with float('nan') before passing in.

        Returns
        -------
        dict mapping indicator name → (N, T) tensor on self.device.
        Tensors still on GPU — call .cpu().numpy() to bring back to host.
        """
        c = self._t(close)
        h = self._t(high)
        l = self._t(low)

        out: dict[str, torch.Tensor] = {}

        # ── SMA ────────────────────────────────────────────────────────
        cfg = self.config.get("SMA", {})
        if cfg.get("enabled", True):
            for p in cfg.get("params", {}).get("periods", [20, 50, 200]):
                out[f"SMA_{p}"] = sma(c, p)

        # ── EMA ────────────────────────────────────────────────────────
        cfg = self.config.get("EMA", {})
        if cfg.get("enabled", True):
            for p in cfg.get("params", {}).get("periods", [12, 26, 50]):
                out[f"EMA_{p}"] = ema(c, p)

        # ── MACD ───────────────────────────────────────────────────────
        cfg = self.config.get("MACD", {})
        if cfg.get("enabled", True):
            p = cfg.get("params", {})
            m, sig, hist = macd(
                c,
                fast_period   = p.get("fast_period",   12),
                slow_period   = p.get("slow_period",   26),
                signal_period = p.get("signal_period",  9),
            )
            out["MACD"]        = m
            out["MACD_signal"] = sig
            out["MACD_hist"]   = hist

        # ── RSI ────────────────────────────────────────────────────────
        cfg = self.config.get("RSI", {})
        if cfg.get("enabled", True):
            period = cfg.get("params", {}).get("period", 14)
            out[f"RSI_{period}"] = rsi(c, period)

        # ── Stochastic ─────────────────────────────────────────────────
        cfg = self.config.get("STOCH", {})
        if cfg.get("enabled", True):
            p = cfg.get("params", {})
            sk, sd = stochastic(
                h, l, c,
                fastk_period = p.get("fastk_period", 5),
                slowk_period = p.get("slowk_period", 3),
                slowd_period = p.get("slowd_period", 3),
            )
            out["STOCH_K"] = sk
            out["STOCH_D"] = sd

        # ── CCI ────────────────────────────────────────────────────────
        cfg = self.config.get("CCI", {})
        if cfg.get("enabled", True):
            period = cfg.get("params", {}).get("period", 20)
            out[f"CCI_{period}"] = cci(h, l, c, period)

        return out


# ---------------------------------------------------------------------------
# Default config (mirrors the calc script's INDICATORS dict)
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    return {
        "SMA":  {"enabled": True, "params": {"periods": [20, 50, 200]}},
        "EMA":  {"enabled": True, "params": {"periods": [12, 26, 50]}},
        "MACD": {"enabled": True, "params": {"fast_period": 12,
                                              "slow_period": 26,
                                              "signal_period": 9}},
        "RSI":  {"enabled": True, "params": {"period": 14}},
        "STOCH":{"enabled": True, "params": {"fastk_period": 5,
                                              "slowk_period": 3,
                                              "slowd_period": 3}},
        "CCI":  {"enabled": True, "params": {"period": 20}},
    }