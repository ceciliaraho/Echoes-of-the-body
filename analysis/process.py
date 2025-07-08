
import pandas as pd
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def uniform_amplitude(signal, target_std=0.1):
    signal = signal - np.mean(signal)
    current_std = np.std(signal)
    if current_std < 1e-6:
        return signal  # No division per zero
    return signal * (target_std / current_std)

def estimate_cutoff_from_variance(signal, low_std=0.02, high_std=0.15, low_cutoff=0.2, high_cutoff=0.7):
    std = np.std(signal)
    if std <= low_std:
        return low_cutoff
    elif std >= high_std:
        return high_cutoff
    else:
        ratio = (std - low_std) / (high_std - low_std)
        return low_cutoff + ratio * (high_cutoff - low_cutoff)

def lowpass_filter(data, fs, cutoff=0.5, order=2):
    data = np.nan_to_num(data)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) < 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

def zscore_normalize(signal):
    scaler = StandardScaler()
    if isinstance(signal, pd.Series):
        signal = signal.values
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

def min_max_normalize(signal):
    if isinstance(signal, pd.Series):
        signal = signal.values
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val) if max_val != min_val else signal

def preprocess_breath_signals(df, fs_custom=120):
    """
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError("Colonna 'label' mancante nel DataFrame.")

    processed_segments = []
    for label in df["label"].unique():
        segment = df[df["label"] == label].copy()
        for signal in ["BF"]:
            if signal in segment.columns:
                segment[signal] = segment[signal].interpolate(method='linear', limit_direction='both')
                cutoff_dynamic = estimate_cutoff_from_variance(segment[signal].values)
                print(f"Label: {label} | Cutoff estimated: {cutoff_dynamic:.2f} Hz")
                segment[signal] = lowpass_filter(segment[signal].values, fs=fs_custom, cutoff=cutoff_dynamic)
                #segment[signal] = uniform_amplitude(segment[signal].values)
                segment[signal] = zscore_normalize(segment[signal])
        processed_segments.append(segment)
    return pd.concat(processed_segments).sort_values(by="time_from_start").reset_index(drop=True)
    """
    
    df = df.copy()
    for signal in ["BF"]:
        if signal in df.columns:
            df[signal] = df[signal].interpolate(method='linear', limit_direction='both')
            cutoff_dynamic = estimate_cutoff_from_variance(df[signal].values)
            print(f"[Global] Cutoff estimated: {cutoff_dynamic:.2f} Hz")
            df[signal] = lowpass_filter(df[signal].values, fs=fs_custom, cutoff=cutoff_dynamic)
            #df[signal] = zscore_normalize(df[signal])
            df[signal] = min_max_normalize(df[signal])
    return df


def preprocess_hr_signals(df, columns=["HR"]):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            signal = df[column].copy()
            signal = signal.interpolate(method='linear', limit_direction='both')
            signal = signal.rolling(window=3, center=True, min_periods=1).mean()
            #signal = uniform_amplitude(signal.values)
            #ignal = zscore_normalize(signal)
            signal = min_max_normalize(signal)
            df[column] = signal
    return df
