import numpy as np

def extract_respiratory_features_from_peaks(df, time, peaks, valleys):
    """
    Estrai feature dal segnale di respiro usando picchi e valli.
    time: array-like dei timestamp (in secondi)
    peaks, valleys: array di indici
    """
    features = {}

    # Durate dei cicli (tempo tra due picchi consecutivi)
    if len(peaks) > 1:
        cycle_durations = np.diff(time[peaks])
        features['Mean_Breath_Cycle_Duration'] = np.mean(cycle_durations)
        features['Std_Breath_Cycle_Duration'] = np.std(cycle_durations)
        features['Breathing_Rate_BPM'] = 60 / np.mean(cycle_durations)
    else:
        features['Mean_Breath_Cycle_Duration'] = np.nan
        features['Std_Breath_Cycle_Duration'] = np.nan
        features['Breathing_Rate_BPM'] = np.nan

    # Ampiezze dei cicli (distanza tra picchi e valli)
    n = min(len(peaks), len(valleys))
    if n > 0:
        amplitudes = np.abs(df['BF'].iloc[peaks[:n]].values - df['BF'].iloc[valleys[:n]].values)
        features['Mean_Amplitude'] = np.mean(amplitudes)
        features['Std_Amplitude'] = np.std(amplitudes)
    else:
        features['Mean_Amplitude'] = np.nan
        features['Std_Amplitude'] = np.nan

    return features

def extract_hrv_features_from_peaks(time, r_peaks):
    """
    Estrai feature HRV (heart rate variability) da picchi R.
    """
    features = {}

    rr_intervals = np.diff(time[r_peaks])  # in secondi

    if len(rr_intervals) < 2:
        features['HRV_SDNN'] = np.nan
        features['HRV_RMSSD'] = np.nan
        features['Mean_HR'] = np.nan
    else:
        features['HRV_SDNN'] = np.std(rr_intervals)
        features['HRV_RMSSD'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        features['Mean_HR'] = np.mean(60 / rr_intervals)

    return features

