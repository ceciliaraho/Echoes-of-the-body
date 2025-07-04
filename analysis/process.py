import pandas as pd
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from plotty import plot_breathing_and_heart_rate
from detect_peaks import detect_r_peaks_and_valleys
from feature_extraction import extract_respiratory_features_from_peaks, extract_hrv_features_from_peaks


# === FILTER ===
def lowpass_filter(data, fs, cutoff=0.5, order=2):
    data = np.nan_to_num(data)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) < 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)


# === BREATHING PREPROCESS ===
def preprocess_breath_signals(df, fs_custom=120):
    df = df.copy()
    breath_signals = {
        'BF': fs_custom,
    }

    for signal, fs in breath_signals.items():
        if signal in df.columns:
            df[signal] = df[signal].interpolate(method='linear', limit_direction='both')
            df[signal] = lowpass_filter(df[signal], fs=fs)

    return df


# === HEART PREPROCESS ===
def preprocess_hr_signals(df, columns=["HR"]):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            signal = df[column].copy()
            signal = signal.interpolate(method='linear', limit_direction='both')
            signal = signal.rolling(window=3, center=True, min_periods=1).mean()
            df[column] = signal
    return df


# === NORMALIZE ZEPHYR + PLOT ===
#def normalize_physio_signals(df, normalize_range=(0, 1), exclude_if_already_normalized=True, feature_output_path=None):
#    df = df.copy()

    # Find breathing peaks and valleys
#    peaksCustom, _ = find_peaks(df['BF'], height=df['BF'].mean(), distance=int(120))
#    valleysCustom, _ = find_peaks(-df['BF'], height=-df['BF'].mean(), distance=int(120))
    
#    peaksCustomHR, valleysCustomHR = detect_r_peaks_and_valleys(df['HR'].values, fs=120, time=df['time_from_start'].values)

#    plot_breathing_and_heart_rate(df, peaksCustom, valleysCustom, peaksCustomHR, valleysCustomHR)

    # Respirazione
#    features_custom_bf = extract_respiratory_features_from_peaks(df, df['time_from_start'].values, peaksCustom, valleysCustom)
    

    # HR waveform
#    features_custom_hr = extract_hrv_features_from_peaks(df['time_from_start'].values, peaksCustomHR)
   
    # Unione in un unico dizionario
#    all_features = {
#        'BF': features_custom_bf,
        
#        'HR': features_custom_hr
    
#    }
#    df_features = pd.DataFrame.from_dict(all_features, orient='index')
#    print("\nFEATURES:\n")
#    print(df_features.round(4).T)  # trasposto per leggibilità

#  if feature_output_path is None:
#       feature_output_path = "output/features.csv"

#    os.makedirs(os.path.dirname(feature_output_path), exist_ok=True)
#    df_features.to_csv(feature_output_path)
#    print(f"\nSaved features in: {os.path.abspath(feature_output_path)}")


#    return df
