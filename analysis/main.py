import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
from process import preprocess_breath_signals, preprocess_hr_signals
from resample import resample_signals
from labels_config import labels_info
from labeling import assign_labels
from features1 import extract_all_features, plot_peaks_by_section


def process_folder(participant_path, participant, subfolder):
    folder_path = os.path.join(participant_path, subfolder)

    custom_path = os.path.join(folder_path, "custom_bio_data.csv")

    if not os.path.exists(custom_path):
        print(f"Custom file missing: {custom_path}")
        return
    custom_df = pd.read_csv(custom_path)
    start_index = custom_df[custom_df['bio_time'] == 0.0].index.min()

    # Remove everything before start time
    custom_df = custom_df.loc[start_index:].copy()
        
    # time_from_start -> time in sec
    custom_df['time_from_start'] = custom_df['bio_time'] / 1000.0
    custom_df.reset_index(drop=True, inplace=True)

    # Selection of number columns 
    numeric_cols = custom_df.select_dtypes(include="number").columns

    # Avarage where double value of time
    df_numeric = custom_df.groupby("time_from_start", as_index=False)[numeric_cols].mean()
    df_timestamp = custom_df.groupby("time_from_start", as_index=False).first()[["time_from_start", "local_timestamp"]]

    custom_df= pd.merge(df_numeric, df_timestamp, on="time_from_start")
    

    participant_id = os.path.basename(participant_path)
    label_ranges = labels_info.get(participant_id)

    # Remove last double value 
    # If the different between last two time_from_start is small (0.001 sec), remove the last one
    if abs(custom_df['time_from_start'].iloc[-1] - custom_df['time_from_start'].iloc[-2]) < 0.01:
        custom_df = custom_df.iloc[:-1]


    # Resample -> resample.py
    resample_df = resample_signals(custom_df, fs=120)
    resample_path = os.path.join(folder_path, "resample_signals.csv")
    resample_df.to_csv(resample_path, index=False)
    print(f"\nSave file: {resample_path}\n")

    # Labels -> labeling.py
    if label_ranges:
        resample_df = assign_labels(resample_df, label_ranges)
    else:
        print(f"No founded label for {participant_id}")

    resample_df = resample_df[resample_df["label"].str.lower() != "unlabeled"]

    if resample_df.empty:
        print(f" No valid lines in {folder_path}.")


    # Clean signals -> process.py
    clean_df = preprocess_breath_signals(resample_df)
    clean_df = preprocess_hr_signals(clean_df)

    output_path = os.path.join(folder_path, "clean_signals_filtered.csv")
    clean_df.to_csv(output_path, index=False)
    print(f"\nSave file clean signals: {output_path}\n")

    plot_peaks_by_section(clean_df, fs=120)


    # Features extraction -> features.py
    feats = extract_all_features(clean_df, fs=120)
    feats["participant"] = participant
    # Save features per each participant
    feature_out_path = os.path.join(folder_path, "features.csv")
    feats.to_csv(feature_out_path, index=False)
    
    
    #normalized_versions = normalize_all_versions(feats, drop_cols=["label", "time_center", "participant"])
    #for norm_type, norm_df in normalized_versions.items():
    #    out_path = os.path.join(folder_path, f"features_{norm_type}.csv")
    #    norm_df.to_csv(out_path, index=False)

    #return normalized_versions  # Rest
    return feats

if __name__ == "__main__":

    base_path = os.path.join("..", "dataset")
    participants = [p for p in os.listdir(base_path) if p.startswith("P")]

    #all_versions = { "minmax": [], "zscore": [], "robust": [] }
    all_raw = []

    for participant in participants:
        print(f"\n====\nProcessing {participant}\n====")
        participant_path = os.path.join(base_path, participant)

        for subfolder in ["session"]:
            print(f"\n--- {subfolder.upper()} ---")
            feats = process_folder(participant_path, participant, subfolder)
            
            if feats is None:
                print("process_folder gave None.")
                continue
            if hasattr(feats, "empty") and feats.empty:
                print(" No features")
                continue

            for col in ("participant", "label"):
                if col not in feats.columns:
                    print(f"Missed column '{col}' in {participant}/{subfolder}.")
                    break
            else:
                all_raw.append(feats)

    if all_raw:
        df_all = pd.concat(all_raw, ignore_index=True)
        out_path = "features_dataset.csv"   # Only RAW (no normalitation)
        df_all.to_csv(out_path, index=False)
        print(f"Saved {out_path} — Shape: {df_all.shape}")
    else:
        print("No founded data.")
            
            
            
            #if feats:
            #    for key in all_versions.keys():
            #        all_versions[key].append(feats[key]) 


    #for norm_type, dfs in all_versions.items():
    #    if dfs:
    #        df_all = pd.concat(dfs, ignore_index=True)
    #        out_path = f"features_dataset_{norm_type}.csv"
    #        df_all.to_csv(out_path, index=False)
    #        print(f"Saved {out_path} - Shape: {df_all.shape}")
  
    #base_path = os.path.join("..", "dataset")
    #participant_path = os.path.join(base_path, "P3")

    #for subfolder in ["session"]:
    #    print(f"\n--- {subfolder.upper()} ---")
    #    process_folder(participant_path, subfolder)
