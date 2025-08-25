import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch, correlate, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy, skew, kurtosis, pearsonr
import matplotlib.pyplot as plt

# Utility
def z_normalize(sig):
    sig = np.asarray(sig)
    s = np.std(sig)
    return (sig - np.mean(sig)) / s if s > 0 else sig

def _filter_intervals_by_rate(peak_idx, fs, min_bpm, max_bpm):
    peak_idx = np.asarray(peak_idx, dtype=int)
    if len(peak_idx) < 2:
        return peak_idx
    ibi = np.diff(peak_idx) / fs  # s
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    keep = (bpm >= min_bpm) & (bpm <= max_bpm)
    return np.array([peak_idx[0]] + [peak_idx[i+1] for i, k in enumerate(keep) if k], dtype=int)


# Peak detection — BREATH
def get_peaks_from_bf(bf, label, fs):
    bf_smooth = gaussian_filter1d(bf, sigma=0.8)  # filter

    bf_range = np.max(bf_smooth) - np.min(bf_smooth)
    bf_std = np.std(bf_smooth)

    # Adaptive parameters
    if bf_range < 0.1 or bf_std < 0.02:
        # (es. viparita)
        distance = int(fs * 0.1)
        prom = 0.01
    elif bf_range < 0.2 or bf_std < 0.04:
        # Signal with intermediate variations
        distance = int(fs * 0.2)
        prom = 0.03
    elif bf_range < 0.05:
        bf_smooth = gaussian_filter1d(z_normalize(bf), sigma=0.5)
        prom = 0.3  
    else:
        # Slow signal 
        distance = int(fs * 0.35)
        prom = 0.05

    print(f"[BF PEAKS] range={bf_range:.3f} | std={bf_std:.3f} | dist={distance} | prom={prom}")

    peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
    return peaks, bf_smooth


# Peak detection — HR
def get_peaks_from_hr(hr, fs):
    hr = np.asarray(hr)
    hr_smooth = gaussian_filter1d(hr, sigma=1.0)

    hr_range = float(np.max(hr_smooth) - np.min(hr_smooth))
    hr_std   = float(np.std(hr_smooth))

    if hr_range < 0.5 or hr_std < 0.2:
        distance = int(fs * 0.30); prom = 0.05
    elif hr_range < 1.5 or hr_std < 0.5:
        distance = int(fs * 0.50); prom = 0.10
    else:
        distance = int(fs * 0.70); prom = 0.15

    peaks, _ = find_peaks(hr_smooth, distance=distance, prominence=prom)
    peaks = _filter_intervals_by_rate(peaks, fs, min_bpm=30, max_bpm=180)
    return peaks, hr_smooth


# HR features (ampliude + time-domain from peaks)
def hr_features_from_wave(hr, fs=120):
    hr = np.asarray(hr)
    feats = {
        "hr_mean": float(np.mean(hr)),
        "hr_std":  float(np.std(hr)),
        "hr_min":  float(np.min(hr)),
        "hr_max":  float(np.max(hr)),
        "hr_range": float(np.max(hr) - np.min(hr)),
        "hr_skew": float(skew(hr)) if len(hr) > 2 else np.nan,
        "hr_kurtosis": float(kurtosis(hr)) if len(hr) > 2 else np.nan,
        "hr_slope": float((hr[-1] - hr[0]) / len(hr)) if len(hr) > 1 else np.nan,
    }
    peaks, _ = get_peaks_from_hr(hr, fs)
    feats["hr_low_peaks"] = int(len(peaks) < 2)
    if len(peaks) > 1:
        rr = np.diff(peaks) / fs  # s
        rr_ms = rr * 1000.0
        feats.update({
            "hr_peak_count": int(len(peaks)),
            "hr_ibi_mean_s": float(np.mean(rr)),
            "hr_ibi_std_s":  float(np.std(rr)),
            "hr_bpm_from_peaks": float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
            "hr_sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan,
            "hr_rmssd_ms": float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 1 else np.nan,
            "hr_pnn50": float(np.mean(np.abs(np.diff(rr_ms)) > 50.0)) if len(rr_ms) > 1 else np.nan,
        })
    else:
        feats.update({
            "hr_peak_count": 0,
            "hr_ibi_mean_s": np.nan,
            "hr_ibi_std_s":  np.nan,
            "hr_bpm_from_peaks": np.nan,
            "hr_sdnn_ms": np.nan, "hr_rmssd_ms": np.nan, "hr_pnn50": np.nan
        })
    return feats


# Respiratory spectral features
def resp_spectral_features(bf, fs):
    bf = np.asarray(bf)
    f, Pxx = welch(bf, fs=fs, nperseg=min(len(bf), 1024))
    def bandpower(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0

    bp_slow = bandpower(0.03, 0.15)
    bp_fast = bandpower(0.15, 0.70)
    total   = float(np.trapz(Pxx, f) + 1e-12)

    peak_freq = float(f[np.argmax(Pxx)])
    centroid  = float(np.sum(f * Pxx) / total)

    Pn = Pxx / (np.sum(Pxx) + 1e-12)
    spec_ent = float(entropy(Pn) / np.log(len(Pn)))

    return {
        "resp_bp_slow": bp_slow,
        "resp_bp_fast": bp_fast,
        "resp_bp_ratio_fast_slow": float(bp_fast / (bp_slow + 1e-12)),
        "resp_peak_freq_hz": peak_freq,
        "resp_spec_centroid_hz": centroid,
        "resp_spec_entropy": spec_ent,
    }


# Coupling HR ↔ BF (cross-corr max + lag)
def hr_bf_coupling(hr, bf, fs, max_lag_s=2.0):
    hr0 = np.asarray(hr) - np.mean(hr)
    bf0 = np.asarray(bf) - np.mean(bf)
    max_lag = int(max_lag_s * fs)
    xcorr = correlate(hr0, bf0, mode="full")
    lags = np.arange(-len(hr0)+1, len(bf0))
    m = (lags >= -max_lag) & (lags <= max_lag)
    xcorr = xcorr[m]; lags = lags[m]
    norm = (np.std(hr0) * np.std(bf0) * len(hr0) + 1e-12)
    xcorr = xcorr / norm
    idx = int(np.argmax(np.abs(xcorr)))
    return {
        "hr_bf_xcorr_max": float(xcorr[idx]),
        "hr_bf_xcorr_lag_s": float(lags[idx] / fs),
    }


# Feature on short windows (10 s)
def extract_features(df, fs=120, window_s=10):
    window = int(fs * window_s)
    out = []
    n = len(df)

    for start in range(0, n, window):
        end = start + window
        if end > n: break
        chunk = df.iloc[start:end]

        label_series = chunk["label"]
        if label_series.mode().empty: continue
        label = label_series.mode()[0]
        if str(label).lower() == "unlabeled": continue

        bf = chunk["BF"].values
        hr = chunk["HR"].values

        feats = {}
        # BF base
        feats.update({
            "bf_mean": float(np.mean(bf)),
            "bf_std":  float(np.std(bf)),
            "bf_min":  float(np.min(bf)),
            "bf_max":  float(np.max(bf)),
            "bf_range": float(np.max(bf) - np.min(bf)),
            "bf_skew": float(skew(bf)) if len(bf) > 2 else np.nan,
            "bf_kurtosis": float(kurtosis(bf)) if len(bf) > 2 else np.nan,
        })
        
        #bf_peaks, _bf_smooth = get_peaks_from_bf(bf, label, fs)
        # feats["bf_low_peaks"] = int(len(bf_peaks) < 2)

        # HR features (amplitude + HRV time-domain)
        feats.update(hr_features_from_wave(hr, fs=fs))

        # Breathing
        feats.update(resp_spectral_features(bf, fs))

        # Coupling 
        feats.update(hr_bf_coupling(hr, bf, fs, max_lag_s=2.0))


        feats["label"] = label
        if "participant" in chunk.columns:
            feats["participant"] = chunk["participant"].mode().iloc[0]

        # Window time
        feats["start"]= float(start/fs)
        feats["time_center"] = float((start / fs) + window_s / 2)
        feats["end"]= float(end/fs)
        out.append(feats)

    df_short = pd.DataFrame(out)
    return df_short


# RR (respiratory rate) & coupling on long windows (40 s, step 10 s)
def extract_rr_and_coupling_long(df, fs=120, window_s=40, step_s=10):
    w = int(fs * window_s); step = int(fs * step_s)
    out = []
    for start in range(0, len(df) - w + 1, step):
        end = start + w
        chunk = df.iloc[start:end]
        if chunk["label"].mode().empty: continue
        label = chunk["label"].mode()[0]
        if str(label).lower() == "unlabeled": continue

        bf = chunk["BF"].values
        hr = chunk["HR"].values

        # RR from BF peks
        bf_peaks, _ = get_peaks_from_bf(bf, label, fs)
        if len(bf_peaks) >= 2:
            rr_s = np.diff(bf_peaks) / fs
            bf_rr = float(60.0 / np.mean(rr_s))
        else:
            bf_rr = np.nan

        # Long coupling
        coup = hr_bf_coupling(hr, bf, fs, max_lag_s=2.0)

        out.append({
            "start": float(start/fs),
            "time_center": float(start / fs + window_s / 2),
            "end": float(end/fs),
            "bf_rr": bf_rr,
            "hr_bf_corr_long": coup["hr_bf_xcorr_max"],
            "hr_slope_long": float((hr[-1] - hr[0]) / len(hr)) if len(hr) > 1 else np.nan,
            "label": label
        })
    return pd.DataFrame(out)

# Merge 
def extract_all_features(df, fs=120):
    # 10s
    f10 = extract_features(df, fs=fs, window_s=10).sort_values("time_center")
    # 40s
    f40 = extract_rr_and_coupling_long(df, fs=fs, window_s=40, step_s=10).sort_values("time_center")

    # Merge asof
    final_df = pd.merge_asof(f10, f40[["time_center","bf_rr","hr_bf_corr_long","hr_slope_long"]],
                             on="time_center", direction="nearest", tolerance=10)

    # Flags
    final_df["missing_bf_rr"] = final_df["bf_rr"].isna().astype(int)
    final_df["bf_rr"] = final_df["bf_rr"].fillna(-1.0)

    final_df["missing_corr"] = final_df["hr_bf_corr_long"].isna().astype(int)
    final_df["hr_bf_corr_long"] = final_df["hr_bf_corr_long"].fillna(0.0)

    final_df["missing_slope"] = final_df["hr_slope_long"].isna().astype(int)
    final_df["hr_slope_long"] = final_df["hr_slope_long"].fillna(0.0)

    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    meta_cols = [c for c in ["participant","label","time_center"] if c in final_df.columns]
    feat_cols = [c for c in final_df.columns if c not in meta_cols]
    final_df = final_df[feat_cols + meta_cols]

    return final_df



# Plot 
def plot_peaks(bf, hr, time, bf_peaks, label, hr_peaks=None):
    
    plt.figure(figsize=(14, 6))
    # BF plot
    plt.subplot(2, 1, 1)
    plt.plot(time, bf, label="BF smoothed", color="blue")
    if bf_peaks is not None and len(bf_peaks) > 0:
        plt.plot(time[bf_peaks], bf[bf_peaks], "x", label="BF Peaks", color="red")
    plt.title(f"{label} - Breathing Signal ({0 if bf_peaks is None else len(bf_peaks)} peaks)")
    plt.xlabel("Time (s)")
    plt.ylabel("BF")
    plt.grid(True)
    plt.legend()

    # HR plot
    plt.subplot(2, 1, 2)
    plt.plot(time, hr, label="HR smoothed", color="green")
    if hr_peaks is not None and len(hr_peaks) > 0:
        plt.plot(time[hr_peaks], hr[hr_peaks], "o", label="HR Peaks", color="orange")
    plt.title(f"{label} - Heart Signal ({0 if hr_peaks is None else len(hr_peaks)} peaks)")
    plt.xlabel("Time (s)")
    plt.ylabel("HR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_peaks_by_section(df, fs=120):
    unique_labels = df["label"].dropna().unique()

    for label in unique_labels:
        if str(label).lower() == "unlabeled":
            continue

        section_df = df[df["label"] == label].copy()
        if section_df.empty:
            continue

        bf = section_df["BF"].values
        hr = section_df["HR"].values
        time = section_df["time_from_start"].values

        bf_peaks, bf_smooth = get_peaks_from_bf(bf, label, fs)
        hr_peaks, hr_smooth = get_peaks_from_hr(hr, fs)

        plot_peaks(bf_smooth, hr_smooth, time, bf_peaks, label, hr_peaks=hr_peaks)
