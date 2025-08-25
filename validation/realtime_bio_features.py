"""
Realtime BIO feature extraction

- Input: irregular (bio_time_ms, BF, HR)
- Resample to a uniform timeline at FS Hz (linear interpolation)
- Keep 10 s (short) and 40 s (long) sliding buffers
- Every HOP (default 10 s): compute features, optionally stream to CSV, and call on_window(row)
"""

from __future__ import annotations

import csv
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable

import numpy as np

# --- Optional SciPy stack ---
try:
    from scipy.signal import find_peaks, butter, filtfilt, welch, correlate
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import skew, kurtosis, entropy
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

__all__ = ["bio_rt_start", "bio_rt_append", "bio_rt_stop"]

# Config 
FS: float       = 120.0  # Hz, uniform timeline
WIN_SEC: float  = 10.0   # short window length (s)
HOP_SEC: float  = 10.0   # hop (s)
LONG_SEC: float = 40.0   # long window length (s)

# Runtime state 
running: bool = False
bio_queue: "queue.Queue[Tuple[float, float, float]]" = queue.Queue()  # (t_ms, BF, HR)
worker: Optional[threading.Thread] = None

# Raw, irregular time base
t0: Optional[float] = None           # absolute start time (s) from first sample
last_raw_t: Optional[float] = None   # last raw t (s, relative to t0)
last_raw_bf: float = 0.0
last_raw_hr: float = 0.0
same_time_run: int = 0               # consecutive duplicates of the same timestamp

# Uniform time scheduler
next_uniform_t: Optional[float] = None  # next uniform time target (s, relative to t0)
dt: float = 1.0 / FS

# Sliding buffers (short / long) -> for long wndow and short window
L_SHORT = int(WIN_SEC * FS)
L_LONG  = int(LONG_SEC * FS)
short_bf: deque = deque(maxlen=L_SHORT)
short_hr: deque = deque(maxlen=L_SHORT)
long_bf:  deque = deque(maxlen=L_LONG)
long_hr:  deque = deque(maxlen=L_LONG)

# Window/hop 
win_idx: int = 0
samples_since_hop: int = 0
HOP_SAMP: int = int(HOP_SEC * FS)

# Output rows kept in memory 
rows: List[Dict[str, float]] = []

# CSV 
out_csv: Optional[Path] = None
live_fp = None
live_writer = None
streaming_enabled: bool = True

on_window_cb: Optional[Callable[[Dict[str, float]], None]] = None


# CSV helpers
def headers() -> List[str]:
    return [
        "t_start_s", "t_center_s", "t_end_s",
        # BF stats (10s)
        "bf_mean", "bf_std", "bf_min", "bf_max", "bf_range", "bf_skew", "bf_kurtosis",
        # HR stats (10s)
        "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_slope", "hr_skew", "hr_kurtosis",
        # HR from peaks
        "hr_low_peaks", "hr_peak_count", "hr_ibi_mean_s", "hr_ibi_std_s", "hr_bpm_from_peaks",
        "hr_sdnn_ms", "hr_rmssd_ms", "hr_pnn50",
        # Long (40s)
        "bf_rr", "hr_bf_corr_long", "hr_slope_long",
        # long-window missing flags
        "missing_bf_rr", "missing_corr", "missing_slope",
        # Respiratory spectral (10s)
        "resp_bp_slow", "resp_bp_fast", "resp_bp_ratio_fast_slow",
        "resp_peak_freq_hz", "resp_spec_centroid_hz", "resp_spec_entropy",
        # HR/BF short coupling (10s)
        "hr_bf_xcorr_max", "hr_bf_xcorr_lag_s",
    ]


def open_live_csv():
    """Open the streaming CSV output (idempotent)."""
    global live_fp, live_writer
    if out_csv is None or live_fp is not None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    live_fp = out_csv.open("w", newline="")
    live_writer = csv.DictWriter(live_fp, fieldnames=headers())
    live_writer.writeheader()
    live_fp.flush()


def write_row(row: Dict[str, float]):
    """Write one feature row to the streaming CSV."""
    if live_writer is None:
        return
    safe = {k: row.get(k, np.nan) for k in headers()}
    live_writer.writerow(safe)
    live_fp.flush()


def close_live_csv():
    """Close the streaming CSV output (safe to call multiple times)."""
    global live_fp, live_writer
    try:
        if live_fp:
            live_fp.flush()
            live_fp.close()
    finally:
        live_fp = None
        live_writer = None


# Public API
def bio_rt_start(output_csv: Path | str,
                 fs: int = 120,
                 win_sec: float = 10.0,
                 hop_sec: float = 10.0,
                 long_sec: float = 40.0,
                 streaming: bool = True,
                 on_window: Optional[Callable[[Dict[str, float]], None]] = None) -> None:
    """
    Start the BIO realtime worker:
      - output_csv: path to write streaming features
      - fs: uniform resampling frequency (Hz)
      - win_sec / hop_sec: short window length & hop (s)
      - long_sec: long window length (s)
      - streaming: if True, write each row immediately to CSV
      - on_window: optional callback(row_dict) invoked per emitted window
    """
    global running, worker, out_csv, FS, WIN_SEC, HOP_SEC, LONG_SEC
    global dt, L_SHORT, L_LONG, HOP_SAMP, streaming_enabled
    global t0, last_raw_t, next_uniform_t
    global short_bf, short_hr, long_bf, long_hr
    global win_idx, samples_since_hop, rows, on_window_cb, same_time_run

    if running:
        return

    out_csv   = Path(output_csv)
    FS        = float(fs)
    WIN_SEC   = float(win_sec)
    HOP_SEC   = float(hop_sec)
    LONG_SEC  = float(long_sec)
    streaming_enabled = bool(streaming)

    # Derived sizes
    dt       = 1.0 / FS
    L_SHORT  = int(WIN_SEC * FS)
    L_LONG   = int(LONG_SEC * FS)
    HOP_SAMP = int(HOP_SEC * FS)

    # Reset timebase & buffers
    t0 = None
    last_raw_t = None
    next_uniform_t = None
    same_time_run = 0

    short_bf = deque(maxlen=L_SHORT)
    short_hr = deque(maxlen=L_SHORT)
    long_bf  = deque(maxlen=L_LONG)
    long_hr  = deque(maxlen=L_LONG)

    win_idx = 0
    samples_since_hop = 0
    rows = []

    if streaming_enabled:
        open_live_csv()

    on_window_cb = on_window

    running = True
    worker = threading.Thread(target=worker_loop, name="BioFeatRT", daemon=True)
    worker.start()


def bio_rt_append(bio_time_ms: float, bf: float, hr: float) -> None:
    """Append a raw BIO sample (time in ms, BF, HR) to the worker queue."""
    if not running:
        return
    bio_queue.put((float(bio_time_ms), float(bf), float(hr)))


def bio_rt_stop() -> Optional[Path]:
    """Stop the worker thread, close CSV (if any), and return the output path."""
    global running, worker, on_window_cb
    if running:
        running = False
        if worker is not None:
            worker.join(timeout=None)
        worker = None
    close_live_csv()
    on_window_cb = None
    return out_csv


# Background worker & pipeline
def worker_loop():
    """Consume the queue and feed the resampler while running."""
    while running or not bio_queue.empty():
        try:
            t_ms, bf, hr = bio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        ingest_and_resample(t_ms / 1000.0, bf, hr)  # ms → s


def ingest_and_resample(t_s: float, bf: float, hr: float):
    """
    Keep a uniform timeline by linearly interpolating between consecutive raw samples.
    - t_s is absolute seconds of the incoming sample (wall time)
    - Internally we convert to a relative clock starting at t0
    """
    global t0, last_raw_t, last_raw_bf, last_raw_hr, next_uniform_t, same_time_run

    if t0 is None:
        # First sample initializes the timebase and the next uniform target
        t0 = t_s
        last_raw_t  = 0.0
        last_raw_bf = bf
        last_raw_hr = hr
        next_uniform_t = 0.0
        same_time_run = 0
        return

    prev_t = last_raw_t
    curr_t = t_s - t0

    # Out of order -> drop
    if curr_t < prev_t:
        return

    # Duplicate timestamp handling
    if curr_t == prev_t:
        same_time_run += 1
        # allow at most two packets with the same timestamp-> drop further duplicates
        if same_time_run >= 2:
            return
    else:
        same_time_run = 0

    # Update "last raw"
    prev_bf, prev_hr = last_raw_bf, last_raw_hr
    last_raw_t, last_raw_bf, last_raw_hr = curr_t, bf, hr

    if next_uniform_t is None:
        next_uniform_t = 0.0

    # Generate uniform samples for next_uniform_t in [prev_t, curr_t]
    while next_uniform_t is not None and next_uniform_t <= curr_t:
        if curr_t == prev_t:
            alpha = 1.0
        else:
            alpha = (next_uniform_t - prev_t) / (curr_t - prev_t)
            alpha = float(np.clip(alpha, 0.0, 1.0))
        bf_u = prev_bf + alpha * (bf - prev_bf)
        hr_u = prev_hr + alpha * (hr - prev_hr)
        push_uniform_sample(bf_u, hr_u)
        next_uniform_t += dt


def push_uniform_sample(bf_u: float, hr_u: float):
    """Append one uniform sample to buffers; emit a row every hop."""
    global samples_since_hop, win_idx

    short_bf.append(bf_u)
    short_hr.append(hr_u)
    long_bf.append(bf_u)
    long_hr.append(hr_u)

    samples_since_hop += 1
    if samples_since_hop >= HOP_SAMP:
        samples_since_hop = 0
        emit_window_if_ready()
        win_idx += 1


def emit_window_if_ready():
    """
    Emit a feature row if the short window is full.
    Long-window features are computed only when the long window is full.
    """
    if len(short_bf) < L_SHORT or len(short_hr) < L_SHORT:
        return

    # Window times on the hop grid
    t_start  = win_idx * HOP_SEC
    t_end    = t_start + WIN_SEC
    t_center = 0.5 * (t_start + t_end)

    bf_s = np.asarray(short_bf, dtype=np.float32)
    hr_s = np.asarray(short_hr, dtype=np.float32)

    row: Dict[str, float] = {}
    row.update(short_features(bf_s, hr_s, int(FS)))

    if len(long_bf) >= L_LONG and len(long_hr) >= L_LONG:
        bf_l = np.asarray(long_bf, dtype=np.float32)
        hr_l = np.asarray(long_hr, dtype=np.float32)
        row.update(long_features(bf_l, hr_l, int(FS)))
    else:
        row.update({
            "bf_rr": np.nan, "hr_bf_corr_long": np.nan, "hr_slope_long": np.nan,
            "missing_bf_rr": 1, "missing_corr": 1, "missing_slope": 1,
        })

    row["t_start_s"]  = float(t_start)
    row["t_center_s"] = float(t_center)
    row["t_end_s"]    = float(t_end)

    rows.append(row)
    if streaming_enabled and out_csv is not None:
        write_row(row)

    try:
        if on_window_cb is not None:
            on_window_cb(dict(row))  # pass a copy
    except Exception:
        # callback errors must not break the pipeline
        pass


# Feature helpers
def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min = np.min(x); x_max = np.max(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min:
        return x
    return (x - x_min) / (x_max - x_min)


def estimate_cutoff_from_variance(x: np.ndarray,
                                  low_std=0.02, high_std=0.15,
                                  low_cut=0.2,  high_cut=0.7) -> float:
    s = float(np.std(x))
    if s <= low_std:  return low_cut
    if s >= high_std: return high_cut
    r = (s - low_std) / (high_std - low_std)
    return low_cut + r * (high_cut - low_cut)


def lowpass(x: np.ndarray, fs: int, cutoff: float, order=2) -> np.ndarray:
    """Low-pass filter: SciPy if available, otherwise simple moving average (~0.5 s)."""
    if x.size == 0:
        return x
    if not HAS_SCIPY:
        k = max(3, int(0.5 * fs))
        if k <= 1: return x
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(x, kernel, mode='same')
    ny = 0.5 * fs; wn = cutoff / ny
    try:
        b, a = butter(order, wn, btype='low', analog=False)
        if x.size < 3 * max(len(a), len(b)): return x
        return filtfilt(b, a, x)
    except Exception:
        return x


def bf_pre(bf: np.ndarray, fs: int) -> np.ndarray:
    """BF preproc: adaptive low-pass + min-max (0–1)."""
    bf = np.nan_to_num(bf)
    cutoff = estimate_cutoff_from_variance(bf)
    bf = lowpass(bf, fs, cutoff, 2)
    return minmax(bf)


def hr_pre(hr: np.ndarray, fs: int) -> np.ndarray:
    """HR preproc: fixed low-pass + min-max (0–1)."""
    hr = np.nan_to_num(hr)
    hr = lowpass(hr, fs, 0.8, 2)
    return minmax(hr)


def filter_intervals_by_rate(peak_idx: np.ndarray, fs: int,
                             min_bpm: float = 30, max_bpm: float = 180) -> np.ndarray:
    """Keep only peaks with IBI corresponding to BPM within [min_bpm, max_bpm]."""
    peak_idx = np.asarray(peak_idx, dtype=int)
    if peak_idx.size < 2:
        return peak_idx
    ibi = np.diff(peak_idx) / float(fs)  # s
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    keep = (bpm >= min_bpm) & (bpm <= max_bpm)
    return np.array([peak_idx[0]] + [peak_idx[i + 1] for i, k in enumerate(keep) if k], dtype=int)


def bf_filter(bf: np.ndarray, fs: int) -> np.ndarray:
    """BF filtered (adaptive low-pass), NO normalization—preserves physical ranges for stats."""
    x = np.nan_to_num(bf)
    cutoff = estimate_cutoff_from_variance(x)
    return lowpass(x, fs, cutoff, 2)


def hr_filter(hr: np.ndarray, fs: int) -> np.ndarray:
    """HR filtered (fixed low-pass), NO normalization—preserves physical ranges for stats."""
    x = np.nan_to_num(hr)
    return lowpass(x, fs, 0.8, 2)


def peaks_from_hr(hr, fs):
    """
    Peak detection on HR waveform (ideally 0–1 normalized PPG/ECG-like).
    Returns: (peaks_idx, hr_smooth)
    """
    x = np.asarray(hr, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int), x

    # smoothing
    if HAS_SCIPY:
        xs = gaussian_filter1d(x, sigma=1.0)
    else:
        k = max(3, int(0.03 * fs))  # ~30 ms
        ker = np.ones(k, dtype=float) / k
        xs = np.convolve(x, ker, mode="same")

    hr_range = float(np.max(xs) - np.min(xs))
    hr_std   = float(np.std(xs))

    if hr_range < 0.5 or hr_std < 0.2:
        distance = int(fs * 0.30); prom = 0.05
    elif hr_range < 1.5 or hr_std < 0.5:
        distance = int(fs * 0.50); prom = 0.10
    else:
        distance = int(fs * 0.70); prom = 0.15

    if HAS_SCIPY:
        peaks, _ = find_peaks(xs, distance=distance, prominence=prom)
    else:
        # naive local maxima + distance + relative threshold
        idx = np.where((xs[1:-1] > xs[:-2]) & (xs[1:-1] > xs[2:]))[0] + 1
        thr = float(np.min(xs) + prom * max(1e-6, (np.max(xs) - np.min(xs))))
        idx = idx[xs[idx] > thr]
        sel, last = [], -10**9
        for i in idx:
            if i - last >= distance:
                sel.append(int(i)); last = i
        peaks = np.asarray(sel, dtype=int)

    # physiological limits 30–180 BPM
    peaks = filter_intervals_by_rate(peaks, fs, min_bpm=30, max_bpm=180)
    return peaks, xs


def welch_like(bf: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD if SciPy is available; simple FFT + Hann fallback otherwise."""
    x = np.asarray(bf, dtype=float)
    if x.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    if HAS_SCIPY:
        f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 1024))
        return f.astype(float), Pxx.astype(float)
    # Fallback
    x = x - float(np.mean(x))
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    Pxx = (np.abs(X) ** 2) / (np.sum(w ** 2) * fs)
    return f.astype(float), Pxx.astype(float)


def respiratory_spectral_features(bf: np.ndarray, fs: int) -> Dict[str, float]:
    f, Pxx = welch_like(bf, fs)
    if f.size == 0 or Pxx.size == 0:
        return {
            "resp_bp_slow": 0.0,
            "resp_bp_fast": 0.0,
            "resp_bp_ratio_fast_slow": 0.0,
            "resp_peak_freq_hz": np.nan,
            "resp_spec_centroid_hz": np.nan,
            "resp_spec_entropy": np.nan,
        }

    def bandpower(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0

    bp_slow = bandpower(0.03, 0.15)
    bp_fast = bandpower(0.15, 0.70)
    total   = float(np.trapz(Pxx, f) + 1e-12)

    peak_freq = float(f[int(np.argmax(Pxx))])
    centroid  = float(np.sum(f * Pxx) / total)

    Pn = Pxx / (np.sum(Pxx) + 1e-12)
    if HAS_SCIPY:
        spec_ent = float(entropy(Pn) / np.log(len(Pn)))
    else:
        spec_ent = float(-np.sum(Pn * np.log(Pn + 1e-12)) / np.log(len(Pn)))

    return {
        "resp_bp_slow": bp_slow,
        "resp_bp_fast": bp_fast,
        "resp_bp_ratio_fast_slow": float(bp_fast / (bp_slow + 1e-12)),
        "resp_peak_freq_hz": peak_freq,
        "resp_spec_centroid_hz": centroid,
        "resp_spec_entropy": spec_ent,
    }


def hr_bf_coupling(hr: np.ndarray, bf: np.ndarray, fs: int, max_lag_s: float = 2.0) -> Dict[str, float]:
    """Short coupling (xcorr max + lag) within ±max_lag_s."""
    hr0 = np.asarray(hr, dtype=float) - float(np.mean(hr))
    bf0 = np.asarray(bf, dtype=float) - float(np.mean(bf))
    if hr0.size == 0 or bf0.size == 0:
        return {"hr_bf_xcorr_max": np.nan, "hr_bf_xcorr_lag_s": np.nan}
    max_lag = int(max_lag_s * fs)
    if HAS_SCIPY:
        xcorr = correlate(hr0, bf0, mode="full")
    else:
        xcorr = np.correlate(hr0, bf0, mode="full")
    lags = np.arange(-len(hr0) + 1, len(bf0))
    m = (lags >= -max_lag) & (lags <= max_lag)
    xcorr = xcorr[m]; lags = lags[m]
    denom = (np.std(hr0) * np.std(bf0) * len(hr0) + 1e-12)
    xcorr = xcorr / denom
    idx = int(np.argmax(np.abs(xcorr)))
    return {
        "hr_bf_xcorr_max": float(xcorr[idx]),
        "hr_bf_xcorr_lag_s": float(lags[idx] / float(fs)),
    }


def rr_from_peaks(x: np.ndarray, fs: int) -> float:
    """Respiration rate from peaks (requires SciPy; otherwise returns NaN)."""
    if not HAS_SCIPY or x.size < int(2.0 * fs):
        return float('nan')
    x = gaussian_filter1d(x, sigma=0.8)
    xr = float(np.max(x) - np.min(x))
    xs = float(np.std(x))
    if xr < 0.1 or xs < 0.02:
        distance = int(fs * 0.10); prom = 0.005
    elif xr < 0.2 or xs < 0.04:
        distance = int(fs * 0.18); prom = 0.02
    else:
        distance = int(fs * 0.30); prom = 0.04
    peaks, _ = find_peaks(x, distance=distance, prominence=prom)
    if peaks.size >= 2:
        rr = np.diff(peaks) / float(fs)  # s
        return 60.0 / float(np.mean(rr)) if np.mean(rr) > 0 else float('nan')
    return float('nan')


def rr_from_psd(x: np.ndarray, fs: int) -> float:
    """Respiration rate from PSD peak (FFT fallback)."""
    if x.size < int(8 * fs):
        return float('nan')
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if not np.any(np.isfinite(x)) or np.allclose(np.std(x), 0):
        return float('nan')
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (np.sum(w ** 2) * fs)
    band = (freqs >= 0.03) & (freqs <= 0.8)
    if np.count_nonzero(band) < 3:
        return float('nan')
    band_psd = psd[band]; band_freqs = freqs[band]
    k = int(np.argmax(band_psd))
    f_peak = float(band_freqs[k]); rel_power = float(band_psd[k] / (np.sum(band_psd) + 1e-12))
    if not np.isfinite(f_peak) or rel_power < 0.02:
        return float('nan')
    return 60.0 * f_peak


# Window feature 
def short_features(bf: np.ndarray, hr: np.ndarray, fs: int) -> Dict[str, float]:
    """
    10 s features on filtered (non-normalized) signals:
      - BF/HR stats, skew/kurtosis (if SciPy), HR peaks & HRV
      - respiratory spectrum
      - short HR↔BF coupling (xcorr max + lag)
    """
    out: Dict[str, float] = {}

    bf_f = bf_filter(bf, fs)
    hr_f = hr_filter(hr, fs)

    # BF stats
    if bf_f.size:
        out.update({
            "bf_mean":  float(np.mean(bf_f)),
            "bf_std":   float(np.std(bf_f)),
            "bf_min":   float(np.min(bf_f)),
            "bf_max":   float(np.max(bf_f)),
            "bf_range": float(np.max(bf_f) - np.min(bf_f)),
        })
    else:
        out.update({k: np.nan for k in ["bf_mean", "bf_std", "bf_min", "bf_max", "bf_range"]})

    # HR stats
    if hr_f.size:
        slope = float((hr_f[-1] - hr_f[0]) / max(1, hr_f.size))
        out.update({
            "hr_mean":  float(np.mean(hr_f)),
            "hr_std":   float(np.std(hr_f)),
            "hr_min":   float(np.min(hr_f)),
            "hr_max":   float(np.max(hr_f)),
            "hr_range": float(np.max(hr_f) - np.min(hr_f)),
            "hr_slope": slope,
        })
    else:
        out.update({k: np.nan for k in ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_slope"]})

    # Skew/Kurtosis
    if HAS_SCIPY and bf_f.size > 2 and hr_f.size > 2:
        out.update({
            "bf_skew": float(skew(bf_f)),
            "bf_kurtosis": float(kurtosis(bf_f)),
            "hr_skew": float(skew(hr_f)),
            "hr_kurtosis": float(kurtosis(hr_f)),
        })
    else:
        out.update({k: np.nan for k in ["bf_skew", "bf_kurtosis", "hr_skew", "hr_kurtosis"]})

    # HR peaks -> HRV on raw HR window
    peaks, _ = peaks_from_hr(hr, fs)
    out["hr_low_peaks"] = int(len(peaks) < 2)
    if len(peaks) > 1:
        rr   = np.diff(peaks) / float(fs)   # s
        rrms = rr * 1000.0                  # ms
        out.update({
            "hr_peak_count":      int(len(peaks)),
            "hr_ibi_mean_s":      float(np.mean(rr)),
            "hr_ibi_std_s":       float(np.std(rr)),
            "hr_bpm_from_peaks":  float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
            "hr_sdnn_ms":         float(np.std(rrms, ddof=1)) if rrms.size > 1 else np.nan,
            "hr_rmssd_ms":        float(np.sqrt(np.mean(np.diff(rrms) ** 2))) if rrms.size > 1 else np.nan,
            "hr_pnn50":           float(np.mean(np.abs(np.diff(rrms)) > 50.0)) if rrms.size > 1 else np.nan,
        })
    else:
        out.update({
            "hr_peak_count": 0,
            "hr_ibi_mean_s": np.nan,
            "hr_ibi_std_s":  np.nan,
            "hr_bpm_from_peaks": np.nan,
            "hr_sdnn_ms": np.nan,
            "hr_rmssd_ms": np.nan,
            "hr_pnn50": np.nan,
        })

    # Respiratory spectrum on BF
    out.update(respiratory_spectral_features(bf, fs))

    # Short coupling HR↔BF
    coup = hr_bf_coupling(hr, bf, fs, max_lag_s=2.0)
    out["hr_bf_xcorr_max"] = coup["hr_bf_xcorr_max"]
    out["hr_bf_xcorr_lag_s"] = coup["hr_bf_xcorr_lag_s"]

    return out

def long_features(bf: np.ndarray, hr: np.ndarray, fs: int) -> Dict[str, float]:
    """
    40 s features:
      - BF respiration rate (RR) from peaks (fallback to PSD)
      - Long-range HR↔BF correlation
      - HR slope over the long window
    """
    out: Dict[str, float] = {}

    bf_p = bf_pre(bf, fs)
    hr_p = hr_pre(hr, fs)

    rr = rr_from_peaks(bf_p, fs)
    if np.isnan(rr):
        rr = rr_from_psd(bf_p, fs)
    out["bf_rr"] = rr

    if hr_p.size == bf_p.size and hr_p.size > 1:
        hr_m = hr_p - np.mean(hr_p)
        bf_m = bf_p - np.mean(bf_p)
        denom = np.sqrt(np.sum(hr_m ** 2) * np.sum(bf_m ** 2))
        corr  = float(np.sum(hr_m * bf_m) / denom) if denom > 0 else np.nan
        slope = float((hr_p[-1] - hr_p[0]) / max(1, hr_p.size))
    else:
        corr = np.nan
        slope = np.nan

    out["hr_bf_corr_long"] = corr
    out["hr_slope_long"]   = slope
    out["missing_bf_rr"]   = 0 if not np.isnan(out["bf_rr"]) else 1
    out["missing_corr"]    = 0 if not np.isnan(out["hr_bf_corr_long"]) else 1
    out["missing_slope"]   = 0 if not np.isnan(out["hr_slope_long"]) else 1
    return out
