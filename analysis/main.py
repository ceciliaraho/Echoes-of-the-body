import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
from process import preprocess_breath_signals, preprocess_hr_signals
from resample import resample_signals
from labels_config import labels_info
from labeling import assign_labels
from features import extract_all_features, plot_peaks_by_section


def process_folder(participant_path, participant, subfolder):
    folder_path = os.path.join(participant_path, subfolder)

    custom_path = os.path.join(folder_path, "custom_bio_data.csv")

    if not os.path.exists(custom_path):
        print(f"Custom file missing: {custom_path}")
        return
    custom_df = pd.read_csv(custom_path)
    start_index = custom_df[custom_df['bio_time'] == 0.0].index.min()

    # Taglia tutto ciò che viene prima
    custom_df = custom_df.loc[start_index:].copy()
        

    # ✅ Rinomina colonne e assegna time_from_start come time_sec
    custom_df['time_from_start'] = custom_df['bio_time'] / 1000.0
    custom_df.reset_index(drop=True, inplace=True)
    # Seleziona solo le colonne numeriche da mediarsi (esclude timestamp!)
    numeric_cols = custom_df.select_dtypes(include="number").columns

    # Calcola la media solo per le colonne numeriche
    df_numeric = custom_df.groupby("time_from_start", as_index=False)[numeric_cols].mean()

    # Ora salva il primo local_timestamp associato a ciascun time_from_start (opzionale)
    df_timestamp = custom_df.groupby("time_from_start", as_index=False).first()[["time_from_start", "local_timestamp"]]

    # Unisci i due dataframe
    custom_df= pd.merge(df_numeric, df_timestamp, on="time_from_start")
    

    participant_id = os.path.basename(participant_path)
    label_ranges = labels_info.get(participant_id)

    # ✅ Rimuove l’ultimo duplicato se l'ultimo timestamp è uguale al penultimo
    # Se la differenza tra gli ultimi due time_from_start è piccolissima (tipo 0.001 sec), rimuovi l’ultima
    if abs(custom_df['time_from_start'].iloc[-1] - custom_df['time_from_start'].iloc[-2]) < 0.01:
        custom_df = custom_df.iloc[:-1]


    #resample
    resample_df = resample_signals(custom_df, fs=120)
    print("\n Dopo resampling:")
    print(resample_df["BF"].head(5))
    print(resample_df["HR"].head(5))
    print("STD:", np.std(resample_df["BF"].dropna().values))
    resample_path = os.path.join(folder_path, "resample_signals.csv")
    resample_df.to_csv(resample_path, index=False)
    print(f"\nSave file: {resample_path}\n")
    
    if label_ranges:
        resample_df = assign_labels(resample_df, label_ranges)
    else:
        print(f"Nessuna label trovata per {participant_id}")

    resample_df = resample_df[resample_df["label"].str.lower() != "unlabeled"]

    if resample_df.empty:
        print(f"⚠️ Nessuna riga valida in {folder_path} dopo il filtro.")

    # ✅ Pulizia dei segnali
    clean_df = preprocess_breath_signals(resample_df)
    clean_df = preprocess_hr_signals(clean_df)
    print("\n Dopo preprocess:")
    print(clean_df["BF"].head(5))
    print("STD:", np.std(clean_df["BF"].dropna().values))


    # ✅ Salvataggio finale
    output_path = os.path.join(folder_path, "clean_signals_filtered.csv")
    clean_df.to_csv(output_path, index=False)
    print(f"\nSave file: {output_path}\n")

    plot_peaks_by_section(clean_df, fs=120)
    
    all_feats = []
    feats = extract_all_features(clean_df, fs=120)
    feature_out_path = os.path.join(folder_path, "features.csv")
    feats.to_csv(feature_out_path, index=False)
    
    
    feats["participant"] = participant
    #feats["subfolder"] = subfolder
    all_feats.append(feats)

    return all_feats
    # ✅ Estrazione e salvataggio delle feature
    #feature_out_path = os.path.join(folder_path, "features.csv")
    #clean_df = normalize_physio_signals(clean_df, feature_output_path=feature_out_path)

    

if __name__ == "__main__":

    base_path = os.path.join("..", "dataset")
    participants = [p for p in os.listdir(base_path) if p.startswith("P")]

    all_participant_feats = []  # QUI raccogli tutte le feature

    for participant in participants:
        print(f"\n=======================\nProcessing {participant}\n=======================")
        participant_path = os.path.join(base_path, participant)

        for subfolder in ["session"]:
            print(f"\n--- {subfolder.upper()} ---")
            feats = process_folder(participant_path, participant, subfolder)
            if feats:
                all_participant_feats.extend(feats)  # ACCUMULA tutto

    # ✅ Ora salva il dataset completo con tutte le feature di tutti i partecipanti
    df_feats = pd.concat(all_participant_feats, ignore_index=True)
    df_feats.to_csv("features_dataset.csv", index=False)
    print("\n✅ Feature extraction completata:", df_feats.shape)
  
    #base_path = os.path.join("..", "dataset")
    #participant_path = os.path.join(base_path, "P3")

    #for subfolder in ["session"]:
    #    print(f"\n--- {subfolder.upper()} ---")
    #    process_folder(participant_path, subfolder)
