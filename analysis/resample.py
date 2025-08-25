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

    return df_resampled




