
import librosa
import numpy as np
import pandas as pd
import os
from labels_config import labels_info

# Patch temporaneo per compatibilità con NumPy > 1.24
np.complex = complex

SR = 44100
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 5

def extract_audio_features(audio_path, label_ranges):
    y, sr = librosa.load(audio_path, sr=SR)

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    num_frames = len(rms)
    features = {
        'rms': rms,
        'zcr': zcr,
        'centroid': centroid
    }
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i+1}'] = mfccs[i]

    df = pd.DataFrame(features)
    df['time_sec'] = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=HOP_LENGTH)

    df['label'] = 'unlabeled'
    for label, (start, end) in label_ranges.items():
        df.loc[(df['time_sec'] >= start) & (df['time_sec'] < end), 'label'] = label

    return df

def process_audio_folder(dataset_path):
    all_audio_features = []

    participants = [p for p in os.listdir(dataset_path) if p.startswith("P")]
    for participant in participants:
        print(f"\nProcessing audio for {participant}")
        participant_path = os.path.join(dataset_path, participant, "session")
        audio_file = os.path.join(participant_path, f"{participant}_audio.wav")

        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            continue

        label_ranges = labels_info.get(participant)
        if not label_ranges:
            print(f"No label ranges found for {participant}")
            continue

        df_audio = extract_audio_features(audio_file, label_ranges)
        df_audio['participant'] = participant
        all_audio_features.append(df_audio)

        out_csv = os.path.join(participant_path, f"{participant}_audio_features.csv")
        df_audio.to_csv(out_csv, index=False)
        print(f"Saved features to {out_csv} - Shape: {df_audio.shape}")

    return all_audio_features

if __name__ == "__main__":
    dataset_path = "../dataset"
    all_features = process_audio_folder(dataset_path)
    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)
        df_all.to_csv("all_participants_audio_features.csv", index=False)
        print(f"\nDataset completo salvato: all_participants_audio_features.csv - Shape: {df_all.shape}")
