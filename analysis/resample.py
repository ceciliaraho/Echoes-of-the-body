import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def resample_signals(df, fs=120):
    
    df = df.copy()
    if not all(col in df.columns for col in ['local_timestamp','time_from_start', 'BF', 'HR']):
        raise ValueError("Dataset should contain: 'time_from_start', 'BF', 'HR', 'local_timestamp'.")
    
    df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])


    # Regular timeline
    dt = 1.0 / fs
    uniform_time = np.arange(df['time_from_start'].min(), df['time_from_start'].max(), dt)

    interp_BF = interp1d(df['time_from_start'], df['BF'], kind='linear', fill_value='extrapolate')
    interp_HR = interp1d(df['time_from_start'], df['HR'], kind='linear', fill_value='extrapolate')
    interp_TS = interp1d(df['time_from_start'], df['local_timestamp'].astype('int64'), kind='linear', fill_value='extrapolate')

    df_resampled = pd.DataFrame({
        'time_from_start': uniform_time,
        'BF': interp_BF(uniform_time),
        'HR': interp_HR(uniform_time),
        'local_timestamp': pd.to_datetime(interp_TS(uniform_time))
    })



    #  Plot
    #plt.figure(figsize=(14, 6))

    # HR
    #plt.subplot(2, 1, 1)
    #plt.plot(df['time_from_start'], df['HR'], '.', color='gray', label='Original HR', markersize=2)
    #plt.plot(uniform_time, df_resampled['HR'], '-', color='blue', label='Interpolated HR (120 Hz)', linewidth=1)
    #plt.title("Heart Rate (HR) - Original vs Resampled")
    #plt.xlabel("Time (s)")
    #plt.ylabel("HR")
    #plt.legend()
    #plt.grid(True)

    # BF
    #plt.subplot(2, 1, 2)
    #plt.plot(df['time_from_start'], df['BF'], '.', color='gray', label='Original BF', markersize=2)
    #plt.plot(uniform_time, df_resampled['BF'], '-', color='green', label='Interpolated BF (120 Hz)', linewidth=1)
    #plt.title("Breathing Frequency (BF) - Original vs Resampled")
    #plt.xlabel("Time (s)")
    #plt.ylabel("BF")
    #plt.legend()
    #plt.grid(True)

    #plt.tight_layout()
    #plt.show()

    return df_resampled




