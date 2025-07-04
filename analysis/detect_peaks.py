from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

def detect_r_peaks_and_valleys(signal, fs=120, threshold_factor=0.5, plot=False, time=None):
    """
    Rileva picchi (R peaks) e valli da un segnale ECG-like.
    
    Parameters:
        signal: array-like, waveform ECG.
        fs: int, frequenza di campionamento in Hz (default 120).
        threshold_factor: float, moltiplicatore della std per la soglia di altezza.
        plot: bool, se True mostra il plot con picchi e valli.
        time: array-like (opzionale), asse temporale per il plot.
    
    Returns:
        peaks: indici dei picchi (massimi)
        valleys: indici delle valli (minimi)
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_height = mean_val + threshold_factor * std_val
    min_depth = -mean_val + threshold_factor * std_val

    # Picchi (massimi)
    peaks, _ = find_peaks(signal, distance=int(0.5 * fs), height=min_height)

    # Valli (minimi)
    valleys, _ = find_peaks(-signal, distance=int(0.5 * fs), height=min_depth)

    # Plot opzionale
    if plot:
        plt.figure(figsize=(12, 4))
        if time is not None:
            plt.plot(time, signal, label='ECG waveform', color='black')
            plt.plot(time[peaks], signal[peaks], 'ro', label='R peaks')
            plt.plot(time[valleys], signal[valleys], 'gx', label='Valleys')
        else:
            plt.plot(signal, label='ECG waveform', color='black')
            plt.plot(peaks, signal[peaks], 'ro', label='R peaks')
            plt.plot(valleys, signal[valleys], 'gx', label='Valleys')
        plt.title('R Peaks and Valleys Detection')
        plt.xlabel('Time (s)' if time is not None else 'Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return peaks, valleys
