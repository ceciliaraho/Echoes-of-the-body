import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def extract_features(df, fs=120, window_s=5):
    window_size = int(fs * window_s)
    features = []
    n_samples = len(df)
    
    for start in range(0, n_samples, window_size):
        end = start + window_size
        if end > n_samples:
            break
        chunk = df.iloc[start:end]
        
        # Skip finestre con label "unlabeled" o assenti
        label_series = chunk["label"]
        if label_series.isna().any():
            continue
        label = label_series.mode()[0]
        if str(label).lower() == "unlabeled":
            continue

        hr = chunk["HR"].values
        bf = chunk["BF"].values

        f = {
            "hr_mean": hr.mean(),
            "hr_std": hr.std(),
            "hr_min": hr.min(),
            "hr_max": hr.max(),
            "hr_range": hr.max() - hr.min(),
            "bf_mean": bf.mean(),
            "bf_std": bf.std(),
            "bf_min": bf.min(),
            "bf_max": bf.max(),
            "bf_range": bf.max() - bf.min(),
            "label": label
        }

        # Applica smoothing al segnale di respiro
        bf_smooth = gaussian_filter1d(bf, sigma=2)

        # Parametri adattivi per find_peaks
        if label == "viparita_swasa":
            distance = fs * 0.2
            prom = 0.015
        elif label == "chanting":
            distance = fs * 0.8
            prom = 0.08
        elif label == "retention":
            distance = fs * 1.0
            prom = 0.02
        else:  # pranayama, meditation, etc.
            distance = fs * 0.5
            prom = 0.03

        # Calcolo della bf_rr solo se non è retention
        peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
        if len(peaks) >= 2:
            rr = 60 / np.mean(np.diff(peaks) / fs)
        else:
            rr = np.nan
        f["bf_rr"] = rr

        features.append(f)

    return pd.DataFrame(features)



