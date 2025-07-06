## Script Overview

| File               | Description |
|--------------------|-------------|
| `main.py`          | Main entry point of the analysis pipeline. It iterates over all participant folders, loads `custom_data.csv`, applies resampling, labeling, cleaning and feature extraction, and saves the resulting structured signals and features. |
| `resample.py`      | Contains utility functions to uniformly resample the physiological signals (e.g., BF, HR) to a fixed frequency (default: 120 Hz), using interpolation. Also includes optional plotting to compare raw and resampled signals. |
| `labeling.py`      | Assigns labels to the physiological data based on time markers from annotated CSV files in the `timestamps/` folder. The labels correspond to the different phases of Shambhavi Mahamudra. |
| `process.py`       | Handles the cleaning and preprocessing of signals, including outlier removal, smoothing, normalization, and peak detection (e.g., breathing cycles). It is used before feature extraction. |
| `features.py`      | Extracts both statistical and physiological features from the cleaned signals in a sliding window. The features are saved to a CSV file for each session. |

---

## 🔧 Configuration Files

| File               | Description |
|--------------------|-------------|
| `labels_config.py` | Contains the dictionary of label definitions and phase names used throughout the labeling and feature extraction scripts. Helps ensure label consistency across participants and sessions. |

---

## Output Files

| File               | Description |
|--------------------|-------------|
| `features_dataset.csv` | Final dataset with the extracted features and their corresponding phase labels, used for machine learning and further analysis. |

###  Time-Domain Features (10s windows)

| Feature Name     | Description |
|------------------|-------------|
| `hr_mean`        | Mean heart rate within the window |
| `hr_std`         | Standard deviation of HR |
| `hr_min`         | Minimum HR value |
| `hr_max`         | Maximum HR value |
| `hr_range`       | Difference between max and min HR |
| `bf_mean`        | Mean breathing frequency |
| `bf_std`         | Standard deviation of BF |
| `bf_min`         | Minimum BF value |
| `bf_max`         | Maximum BF value |
| `bf_range`       | Difference between max and min BF |

### Heart Rate Variability (HRV) Features

| Feature Name     | Description |
|------------------|-------------|
| `hr_rmssd`       | Root Mean Square of Successive Differences in HR (HRV index) |
| `hr_slope`       | Linear trend (slope) of HR in the window |

### Shape Features

| Feature Name     | Description |
|------------------|-------------|
| `hr_skew`        | Skewness of the HR signal |
| `hr_kurtosis`    | Kurtosis of the HR signal |
| `bf_skew`        | Skewness of the BF signal |
| `bf_kurtosis`    | Kurtosis of the BF signal |

### Long-Window Features (40s windows, 10s step)

| Feature Name         | Description |
|----------------------|-------------|
| `bf_rr`              | Estimated respiratory rate based on inter-peak distance of BF |
| `hr_bf_corr_long`    | Pearson correlation between z-normalized HR and BF |
| `hr_slope_long`      | Linear slope of HR over the longer window |

These features are merged based on the `time_center` of each window. The output is a rich dataset describing physiological patterns across the five phases of **Shambhavi Mahamudra**.

---
