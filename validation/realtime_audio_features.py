"""
Realtime AUDIO feature extraction

- Input: audio chunks (numpy array, mono or stereo)
- Buffers samples until a full window (WIN_SEC) is available
- Every HOP_SEC seconds, computes a window of features and:
  - streams a CSV row (if streaming=True)
  - invokes an optional callback on_window(row_dict)
- Resamples to SR (default 22050 Hz) for librosa-based features
"""

from __future__ import annotations

import csv
import queue
import threading
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import librosa

try:
    import scipy.signal as sps
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

__all__ = ["rt_start", "rt_append", "rt_stop"]

# Parameters
SR = 22050            # target SR for features (downsample)
WIN_SEC = 10.0
HOP_SEC = 10.0
N_MELS = 32
N_MFCC = 10
FRAME_LENGTH = 1024
HOP_LENGTH = 512
ROLLOFF_PCT = 0.95

# State
running: bool = False
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
worker_thread: Optional[threading.Thread] = None

buffer: np.ndarray = np.zeros(0, dtype=np.float32)
rows: List[Dict[str, float]] = []

out_csv: Optional[Path] = None
capture_sr: int = 44100
L_cap: int = int(WIN_SEC * capture_sr)  # window length in samples (capture rate)
H_cap: int = int(HOP_SEC * capture_sr)  # hop in samples (capture rate)

# Window progress (for consistent timing)
win_idx: int = 0  # 0,1,2,... → t_start = win_idx * H_cap / capture_sr

# CSV streaming 
live_csv_fp = None
live_csv_writer = None
live_headers: Optional[List[str]] = None
streaming_enabled: bool = True
on_window_cb: Optional[Callable[[Dict[str, float]], None]] = None


# Feature
def compute_window_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """Compute per-window audio features (mel/MFCC/chroma stats + centroid/rolloff/bandwidth, ZCR, RMS)."""
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    feat: Dict[str, float] = {}

    # Mel stats
    mel_mean, mel_std = S_mel_db.mean(axis=1), S_mel_db.std(axis=1)
    for i, v in enumerate(mel_mean): feat[f"mel_mean_{i}"] = float(v)
    for i, v in enumerate(mel_std):  feat[f"mel_std_{i}"]  = float(v)

    # MFCC stats
    mfcc = librosa.feature.mfcc(S=S_mel_db, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean, mfcc_std = mfcc.mean(axis=1), mfcc.std(axis=1)
    for i, v in enumerate(mfcc_mean): feat[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_std):  feat[f"mfcc_std_{i}"]  = float(v)

    # Chroma stats
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean, chroma_std = chroma.mean(axis=1), chroma.std(axis=1)
    for i, v in enumerate(chroma_mean): feat[f"chroma_mean_{i}"] = float(v)
    for i, v in enumerate(chroma_std):  feat[f"chroma_std_{i}"]  = float(v)

    feat["spec_centroid"]      = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    feat["spec_rolloff"]       = float(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=ROLLOFF_PCT).mean())
    feat["spec_bandwidth"]     = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    feat["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(y).mean())
    feat["rms"]                = float(librosa.feature.rms(y=y).mean())
    return feat


# CSV
def headers() -> List[str]:
    return (
        ["t_start_s", "t_center_s", "t_end_s"]
        + [f"mel_mean_{i}" for i in range(N_MELS)]
        + [f"mel_std_{i}"  for i in range(N_MELS)]
        + [f"mfcc_mean_{i}" for i in range(N_MFCC)]
        + [f"mfcc_std_{i}"  for i in range(N_MFCC)]
        + [f"chroma_mean_{i}" for i in range(12)]
        + [f"chroma_std_{i}"  for i in range(12)]
        + ["rms", "zero_crossing_rate", "spec_centroid", "spec_rolloff", "spec_bandwidth"]
    )


def open_live_csv():
    """Open streaming CSV (idempotent)."""
    global live_csv_fp, live_csv_writer, live_headers
    if out_csv is None or live_csv_fp is not None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    live_headers = headers()
    live_csv_fp = out_csv.open("w", newline="")
    live_csv_writer = csv.DictWriter(live_csv_fp, fieldnames=live_headers)
    live_csv_writer.writeheader()
    live_csv_fp.flush()


def write_row(row: Dict[str, float]):
    """Write one feature row to the streaming CSV."""
    if live_csv_writer is None:
        return
    safe = {k: row.get(k, np.nan) for k in (live_headers or headers())}
    live_csv_writer.writerow(safe)
    live_csv_fp.flush()


def close_live_csv():
    """Close streaming CSV (safe to call multiple times)."""
    global live_csv_fp, live_csv_writer, live_headers
    try:
        if live_csv_fp:
            live_csv_fp.flush()
            live_csv_fp.close()
    finally:
        live_csv_fp = None
        live_csv_writer = None
        live_headers = None


# API
def rt_start(output_csv: Path | str,
             input_sr: int = 44100,
             streaming: bool = True,
             on_window: Optional[Callable[[Dict[str, float]], None]] = None) -> None:
    """
    Start realtime audio features:
      - output_csv: path to write streaming features (if streaming=True)
      - input_sr: incoming audio sample rate (Hz)
      - streaming: write each window immediately to CSV
      - on_window: optional callback(row_dict) per emitted window
    """
    global running, worker_thread, out_csv, capture_sr, L_cap, H_cap, buffer, rows, win_idx
    global streaming_enabled, on_window_cb

    if running:
        return

    out_csv = Path(output_csv)
    capture_sr = int(input_sr)
    L_cap = int(round(WIN_SEC * capture_sr))
    H_cap = int(round(HOP_SEC * capture_sr))

    buffer = np.zeros(0, dtype=np.float32)
    rows.clear()
    win_idx = 0

    streaming_enabled = bool(streaming)
    on_window_cb = on_window

    if streaming_enabled:
        open_live_csv()

    running = True
    worker_thread = threading.Thread(target=worker_loop, name="AudioFeatRT", daemon=True)
    worker_thread.start()


def rt_append(block: np.ndarray) -> None:
    """Append an audio block (numpy). Accepts mono or (N, C) stereo—mixes to mono."""
    if not running:
        return
    if block.ndim == 2:
        block = block.mean(axis=1)
    audio_queue.put(block.astype(np.float32, copy=False))


def rt_stop() -> Optional[Path]:
    """Stop worker, flush remaining **full** windows, and close CSV."""
    global running, worker_thread, on_window_cb
    if not running:
        return out_csv
    running = False
    if worker_thread is not None:
        worker_thread.join(timeout=None)
    worker_thread = None
    try:
        flush_full_windows()   # no padding -> partial windows are dropped
        if not streaming_enabled:
            write_csv_full()
    except Exception as e:
        print(f"[audio_rt] close error: {e}")
    finally:
        close_live_csv()
        on_window_cb = None
    return out_csv



def worker_loop() -> None:
    """Consume the queue and process chunks while running."""
    while running or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        append_and_process(chunk)


def append_and_process(chunk: np.ndarray) -> None:
    """Append chunk to the rolling buffer and emit any full windows."""
    global buffer
    buffer = np.concatenate([buffer, chunk])
    emit_from_buffer()


def emit_from_buffer() -> None:
    """Emit all complete windows currently in the buffer."""
    global buffer, win_idx
    while buffer.size >= L_cap:
        win_cap = buffer[:L_cap]
        row = process_window(win_cap)

        # Times based on the fixed hop grid
        t_start = win_idx * H_cap / capture_sr
        t_end   = (win_idx * H_cap + L_cap) / capture_sr
        t_center = (t_start + t_end) / 2
        row["t_start_s"]  = float(t_start)
        row["t_center_s"] = float(t_center)
        row["t_end_s"]    = float(t_end)

        rows.append(row)

        # CSV streaming + callback
        if streaming_enabled and out_csv is not None:
            write_row(row)
        if on_window_cb is not None:
            try:
                on_window_cb(dict(row))  # pass a copy
            except Exception as e:
                print("[audio_rt] on_window error:", e)

        win_idx += 1
        buffer = buffer[H_cap:]


def flush_full_windows() -> None:
    """On stop, emit all remaining full windows (drop incomplete tail)."""
    emit_from_buffer()
    # any tail < L_cap is intentionally dropped (no padding)


# Pipeline
def process_window(win_cap: np.ndarray) -> Dict[str, float]:
    """Resample capture-rate audio to SR and compute features on the resampled window."""
    if capture_sr == SR:
        y = win_cap
    elif HAS_SCIPY:
        g = gcd(int(SR), int(capture_sr))
        up = int(SR // g)
        down = int(capture_sr // g)
        y = sps.resample_poly(win_cap, up=up, down=down)
    else:
        try:
            y = librosa.resample(win_cap, orig_sr=capture_sr, target_sr=SR, res_type="kaiser_best")
        except Exception:
            step = max(1, int(round(capture_sr / SR)))
            y = win_cap[::step]
    return compute_window_features(y, sr=SR)


# CSV (when streaming=False)
def write_csv_full() -> None:
    """Write all accumulated rows to CSV (used when streaming=False)."""
    if out_csv is None:
        return
    hdrs = headers()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdrs)
        w.writeheader()
        for r in rows:
            safe = {k: r.get(k, np.nan) for k in hdrs}
            w.writerow(safe)
