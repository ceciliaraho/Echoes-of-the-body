import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# Parameters for peaks
def get_peak_params(label, fs):
    if label == "viparita_swasa":
        return int(fs * 0.2), 0.005
    elif label == "chanting":
        return int(fs * 0.9), 0.3
    elif label == "retention":
        return int(fs * 3), 0.1
    elif label == "pranayama":
        return int(fs * 1.1), 0.2
    else:
        return int(fs * 0.9), 0.1
    
def get_peaks_from_bf(bf, label, fs):
    """
    bf_smooth = gaussian_filter1d(bf, sigma=2)
    distance = int(fs*0.3)
    prom = 0.2 * (np.max(bf_smooth) - np.min(bf_smooth))  # 20% del range dinamico
    #distance, prom = get_peak_params(label, fs)
    peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
    return peaks, bf_smooth

     bf_smooth = gaussian_filter1d(bf, sigma=1.2)

    # Stima dinamica del range e dell’energia del segnale
    dynamic_range = np.max(bf_smooth) - np.min(bf_smooth)
    std_dev = np.std(bf_smooth)

    if dynamic_range < 2.5:  # segnale molto piatto
        distance = int(fs * 0.15)
        prom = 0.2
    elif dynamic_range < 4:
        distance = int(fs * 0.25)
        prom = 0.3
    else:
        distance = int(fs * 0.35)
        prom = 0.4

    peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
    return peaks, bf_smooth
    """
    bf_smooth = gaussian_filter1d(bf, sigma=0.8)  # filtraggio moderato

    # Indicatori del segnale
    bf_range = np.max(bf_smooth) - np.min(bf_smooth)

    bf_std = np.std(bf_smooth)

    # Parametri adattivi
    if bf_range < 0.1 or bf_std < 0.02:
        # Segnale piatto o fine (es. viparita)
        distance = int(fs * 0.1)
        prom = 0.01
    elif bf_range < 0.2 or bf_std < 0.04:
        # Segnale con variazioni medie
        distance = int(fs * 0.2)
        prom = 0.03
    elif bf_range < 0.05:
        print("Rinormalizzo con zscore solo per i picchi")
        bf_smooth = gaussian_filter1d(z_normalize(bf), sigma=0.5)
        prom = 0.3  # torna a valori per zscore
    else:
        # Segnale ampio e lento
        distance = int(fs * 0.35)
        prom = 0.05

    # Debug (opzionale)
    print(f"[PEAKS] range={bf_range:.3f} | std={bf_std:.3f} | dist={distance} | prom={prom}")

    peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
    return peaks, bf_smooth

def extract_hr_features(hr):
    hr = np.asarray(hr)
    # RMSSD
    hr_diff = np.diff(hr)
    hr_rmssd = np.sqrt(np.mean(hr_diff ** 2)) if len(hr_diff) > 1 else np.nan
    # Slope
    hr_slope = (hr[-1] - hr[0]) / len(hr) if len(hr) > 1 else np.nan

    return {
        "hr_rmssd": hr_rmssd,
        "hr_slope": hr_slope,
    }
    
#Skewness and kurtosis for windows of hr and bf.
def compute_skew_kurtosis_features(hr, bf):
    return {
        "hr_skew": skew(hr) if len(hr) > 2 else np.nan,
        "hr_kurtosis": kurtosis(hr) if len(hr) > 2 else np.nan,
        "bf_skew": skew(bf) if len(bf) > 2 else np.nan,
        "bf_kurtosis": kurtosis(bf) if len(bf) > 2 else np.nan
    }

# Features extraction with sliding window of 10s
def extract_features(df, fs=120, window_s=10):
    window_size = int(fs * window_s)
    features = []
    n_samples = len(df)

    for start in range(0, n_samples, window_size):
        end = start + window_size
        if end > n_samples:
            break
        chunk = df.iloc[start:end]

        label_series = chunk["label"]
        #if label_series.isna().any():
        #    continue
        label = label_series.mode()[0]
        if str(label).lower() == "unlabeled":
            continue

        hr = chunk["HR"].values
        bf = chunk["BF"].values

        f = {
            "label": label,
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
           
        }

        hrv_features = extract_hr_features(hr)
        f.update(hrv_features)

        # Skewness and Kurtosis features
        shape_features = compute_skew_kurtosis_features(hr, bf)
        f.update(shape_features)

        features.append(f)

    return pd.DataFrame(features)

# RR estimation on longer windows -> 40s
def extract_rr_sliding(df, fs=120, window_s=40, step_s=10):
    window_size = int(fs * window_s)
    step_size = int(fs * step_s)
    rr_list = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        chunk = df.iloc[start:end]
        label_series = chunk["label"]
        if label_series.isna().any() or label_series.mode().empty:
            continue
        label = label_series.mode()[0]
        if str(label).lower() == "unlabeled":
            continue

        bf = chunk["BF"].values
        peaks, _ = get_peaks_from_bf(bf, label, fs)

        if len(peaks) >= 2:
            rr = 60 / np.mean(np.diff(peaks) / fs)
        #elif label == "retention":
        #    rr = 0.0
        else:
            rr = np.nan

        time_center = int(chunk["time_from_start"].iloc[window_size // 2])
        rr_list.append({"time_center": time_center, "bf_rr": rr})

    return pd.DataFrame(rr_list)

def z_normalize(sig):
    return (sig - np.mean(sig)) / np.std(sig) if np.std(sig) > 0 else sig


def extract_hr_corr_and_slope_long(df, fs=120, window_s=40, step_s=10):
    window_size = int(fs * window_s)
    step_size = int(fs * step_s)
    results = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        chunk = df.iloc[start:end]
        label_series = chunk["label"]
        #if label_series.isna().any() or label_series.mode().empty:
        #    continue
        label = label_series.mode()[0]
        if str(label).lower() == "unlabeled":
            continue

        hr = chunk["HR"].values
        bf = chunk["BF"].values

        # Correlation between HR-BF
        if len(hr) == len(bf) and len(hr) > 1:
        
            try:
                corr = pearsonr(hr, bf)[0]
            except:
                corr = np.nan
        else:
            corr = np.nan

        # Slope HR
        slope = (hr[-1] - hr[0]) / len(hr) if len(hr) > 1 else np.nan

        time_center = int(chunk["time_from_start"].iloc[window_size // 2])
        results.append({
            "time_center": time_center,
            "hr_bf_corr_long": corr,
            "hr_slope_long": slope
        })

    return pd.DataFrame(results)

def extract_all_features(df, fs=120):
    window_s = 10
    feature_df = extract_features(df, fs=fs, window_s=window_s)
    feature_df["time_center"] = feature_df.index * window_s

    rr_df = extract_rr_sliding(df, fs=fs, window_s=40, step_s=10)
    corr_slope_df = extract_hr_corr_and_slope_long(df, fs=fs, window_s=40, step_s=10)

    feature_df = feature_df.sort_values("time_center")
    rr_df = rr_df.sort_values("time_center")
    corr_slope_df = corr_slope_df.sort_values("time_center")

    final_df = pd.merge_asof(feature_df, rr_df, on="time_center", direction="nearest", tolerance=10)
    final_df = pd.merge_asof(final_df, corr_slope_df, on="time_center", direction="nearest", tolerance=10)

    return final_df


def normalize_all_versions(df, drop_cols=["label", "participant", "time_center"]):
    """
    Applica tre tipi di normalizzazione: MinMax, Standard (Z-score) e Robust.
    Restituisce un dizionario di DataFrame normalizzati.
    """
    results = {}
    scalers = {
        "minmax": MinMaxScaler(),
        "zscore": StandardScaler(),
        "robust": RobustScaler()
    }

    df = df.copy()
    keep = df[drop_cols]
    features = df.drop(columns=drop_cols)

    for name, scaler in scalers.items():
        scaled = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled, columns=features.columns)
        df_final = pd.concat([df_scaled, keep.reset_index(drop=True)], axis=1)
        results[name] = df_final

    return results

def plot_peaks(bf,hr, time, bf_peaks, label):
    plt.figure(figsize=(14, 6))
    # BF plot
    plt.subplot(2, 1, 1)
    plt.plot(time, bf, label="BF smoothed", color="blue")
    plt.plot(time[bf_peaks], bf[bf_peaks], "x", label="BF Peaks", color="red")
    plt.title(f"{label} - Breathing Signal ({len(bf_peaks)} peaks)")
    plt.xlabel("Time (s)")
    plt.ylabel("BF")
    plt.grid(True)
    plt.legend()

     # HR plot
    plt.subplot(2, 1, 2)
    plt.plot(time, hr, label="HR", color="green")
    plt.title(f"{label} - Heart Signal")
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
        hr=section_df["HR"].values
        time = section_df["time_from_start"].values

        bf_peaks, bf_smooth = get_peaks_from_bf(bf, label, fs)
        plot_peaks(bf_smooth, hr, time, bf_peaks, label)

        