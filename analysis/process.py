import pandas as pd
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def uniform_amplitude(signal, target_std=0.1):
    """
    Uniforma l'ampiezza del segnale in modo dinamico:
    - se troppo piatto (std < min_std), lo amplifica
    - se troppo energico (std > max_std), lo riduce
    - altrimenti lo lascia com'è

    Args:
        signal (np.ndarray): segnale 1D
        target_std (float): deviazione standard desiderata
        min_std (float): soglia inferiore di deviazione standard
        max_std (float): soglia superiore di deviazione standard

    Returns:
        np.ndarray: segnale uniformato
    """
    

    signal = signal - np.mean(signal)
    current_std = np.std(signal)
    if current_std < 1e-6:
        return signal  # Evita divisione per zero
    return signal * (target_std / current_std)


def estimate_cutoff_from_variance(signal, low_std=0.02, high_std=0.15, 
                                  low_cutoff=0.2, high_cutoff=0.7):
    """
    Stima il cutoff del filtro passa-basso in base alla varianza (std) del segnale.
    
    - Se la std è bassa → respiri lenti → cutoff basso
    - Se la std è alta → respiri veloci → cutoff alto

    Args:
        signal (np.ndarray): segnale 1D normalizzato
        low_std (float): soglia inferiore per la std
        high_std (float): soglia superiore per la std
        low_cutoff (float): cutoff corrispondente a low_std
        high_cutoff (float): cutoff corrispondente a high_std

    Returns:
        float: cutoff da usare per il filtro
    """
    std = np.std(signal)

    if std <= low_std:
        return low_cutoff
    elif std >= high_std:
        return high_cutoff
    else:
        # interpolazione lineare tra low_cutoff e high_cutoff
        ratio = (std - low_std) / (high_std - low_std)
        return low_cutoff + ratio * (high_cutoff - low_cutoff)


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
    """
    Applica il preprocessing dei segnali respiratori per ciascuna label (fase) separatamente.
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

                # Calcolo del cutoff basato sulla varianza del singolo segmento
                cutoff_dynamic = estimate_cutoff_from_variance(segment[signal].values)
                print(f"🔹 Label: {label} | Cutoff stimato: {cutoff_dynamic:.2f} Hz")

                segment[signal] = lowpass_filter(segment[signal].values, fs=fs_custom, cutoff=cutoff_dynamic)
                segment[signal] = uniform_amplitude(segment[signal].values)

        processed_segments.append(segment)

    # Ricompongo tutto
    return pd.concat(processed_segments).sort_values(by="time_from_start").reset_index(drop=True)


# === HEART PREPROCESS ===
def preprocess_hr_signals(df, columns=["HR"]):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            signal = df[column].copy()
            signal = signal.interpolate(method='linear', limit_direction='both')
            signal = signal.rolling(window=3, center=True, min_periods=1).mean()
            signal = uniform_amplitude(signal.values) 
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
