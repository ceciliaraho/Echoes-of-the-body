# 03_plotting_functions.py

import matplotlib.pyplot as plt

# ----------------------------
# --- Plotting Functions ---
# ----------------------------

def plot_breathing_and_heart_rate(df,
                                  peaksCustom=None, valleysCustom=None,
                                  peaksCustomHR=None, valleysCustomHR=None):
    """
    Plot custom and zephyr breathing and heart rate signals.
    Optionally highlight detected peaks and valleys for each.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))  

    # --- Custom Breathing ---
    axs[0].plot(df['time_from_start'], df['BF'], label='Custom Breathing', color='blue')
    if peaksCustom is not None and valleysCustom is not None:
        axs[0].scatter(df['time_from_start'].iloc[peaksCustom], df['BF'].iloc[peaksCustom], color='red', label='Peaks', marker='o')
        axs[0].scatter(df['time_from_start'].iloc[valleysCustom], df['BF'].iloc[valleysCustom], color='green', label='Valleys', marker='x')
    axs[0].set_ylabel('Breathing')
    axs[0].set_title('Custom Breathing Signal')
    axs[0].legend()
    axs[0].grid(True)



    # --- Custom Heart Rate ---
    axs[1].plot(df['time_from_start'], df['HR'], label='Custom HR', color='orange')
    if peaksCustomHR is not None and valleysCustomHR is not None:
        axs[1].scatter(df['time_from_start'].iloc[peaksCustomHR], df['HR'].iloc[peaksCustomHR], color='red', label='Peaks', marker='o')
        axs[1].scatter(df['time_from_start'].iloc[valleysCustomHR], df['HR'].iloc[valleysCustomHR], color='green', label='Valleys', marker='x')
    axs[1].set_ylabel('Heart Rate (bpm)')
    axs[1].set_title('Custom Heart Rate Signal')
    axs[1].legend()
    axs[1].grid(True)



    plt.tight_layout()
    plt.show()
