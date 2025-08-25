# statisticalFeatures.py
import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from labels_config import labels_info  # dict: { "P2": {"pranayama": (start,end), ...}, ... }

# Parameters
SR = 22050
WIN_SEC = 10.0
HOP_SEC = 10.0            # windows 0,10,20,...
N_MELS = 32
FRAME_LENGTH = 1024
HOP_LENGTH = 512
DEFAULT_CLASS = -1

DATASET_PATH = "../dataset"
PARTICIPANTS = [p for p in os.listdir(DATASET_PATH) if p.startswith("P")]

def load_audio(path, target_sr=SR):
    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Mono
    if fs != target_sr:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_sr)
        fs = target_sr
    return audio, fs

def intervals_to_label_vector(total_len_samples, fs, intervals, t0_audio_sec=0.0, default_class=DEFAULT_CLASS):
    """
    Constructs y_seq (per-sample labels) by realigning the timeline to t0_audio_sec
    """
    y = np.full(total_len_samples, default_class, dtype=np.int32)
    for (t_start, t_end, c) in intervals:
        start_sec = max(t_start, t0_audio_sec)
        end_sec = min(t_end, t0_audio_sec + total_len_samples / fs)
        if end_sec <= start_sec:
            continue
        s_idx = int(round((start_sec - t0_audio_sec) * fs))
        e_idx = int(round((end_sec  - t0_audio_sec) * fs))
        y[s_idx:e_idx] = c
    return y

# Feature for each window 
def compute_window_features(y_win_audio, sr=SR):
    """
    Extracts statistical features from a 10 s audio window.
    Returns a dict containing: mel/mfcc/chroma mean+std, centroid/rolloff/bandwidth, zcr, rms.
    """
    # Mel (power→dB)
    S_mel = librosa.feature.melspectrogram(
        y=y_win_audio, sr=sr, n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    mel_mean = S_mel_db.mean(axis=1)
    mel_std  = S_mel_db.std(axis=1)

    # MFCC
    mfcc = librosa.feature.mfcc(S=S_mel_db, sr=sr, n_mfcc=10)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y_win_audio, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_std  = chroma.std(axis=1)

    # Other statistical features 
    spec_centroid  = float(librosa.feature.spectral_centroid(y=y_win_audio, sr=sr).mean())
    spec_rolloff   = float(librosa.feature.spectral_rolloff(y=y_win_audio, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=y_win_audio, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y_win_audio).mean())
    rms = float(librosa.feature.rms(y=y_win_audio).mean())

    # Merge
    feat = {}
    for i, v in enumerate(mel_mean):    feat[f"mel_mean_{i}"]     = float(v)
    for i, v in enumerate(mel_std):     feat[f"mel_std_{i}"]      = float(v)
    for i, v in enumerate(mfcc_mean):   feat[f"mfcc_mean_{i}"]    = float(v)
    for i, v in enumerate(mfcc_std):    feat[f"mfcc_std_{i}"]     = float(v)
    for i, v in enumerate(chroma_mean): feat[f"chroma_mean_{i}"]  = float(v)
    for i, v in enumerate(chroma_std):  feat[f"chroma_std_{i}"]   = float(v)

    feat["spec_centroid"]      = spec_centroid
    feat["spec_rolloff"]       = spec_rolloff
    feat["spec_bandwidth"]     = spec_bandwidth
    feat["zero_crossing_rate"] = zcr
    feat["rms"]                = rms
    return feat

def extract_stat_features_like_train(audio, y_seq, fs, win_sec=WIN_SEC, hop_sec=HOP_SEC, t0_audio_sec=0.0, inv_label_map=None):
    L = int(round(win_sec * fs))
    H = int(round(hop_sec * fs))
    starts = range(0, max(1, len(audio) - L + 1), H)

    rows = []
    for s in starts:
        e = s + L
        seg = audio[s:e]
        if len(seg) < L: 
            seg = np.pad(seg, (0, L - len(seg)))

        # Majority labeling (come train_offline)
        y_win = y_seq[s:min(e, len(y_seq))]
        vals, counts = np.unique(y_win, return_counts=True)
        if len(vals) == 0:
            continue
        maj = vals[np.argmax(counts)]
        if maj == DEFAULT_CLASS:
            continue  # scarta le unlabeled, esattamente come train_offline

        # (extra debug non invasivo)
        coverage = counts[np.argmax(counts)] / len(y_win)
        unlabeled_frac = float(np.mean(y_win == DEFAULT_CLASS))

        # Feature su segment (nessuna normalizzazione qui: train_offline normalizza waveform, ma per statistiche non serve)
        feat = compute_window_features(seg, sr=fs)

        # tempi: relativi all'audio (0,10,20,...) e assoluti su timeline sessione (t0 + relativo)
        rel_start = s / fs
        rel_end   = e / fs
        abs_start = t0_audio_sec + rel_start
        abs_end   = t0_audio_sec + rel_end

        rel_center = (rel_start + rel_end) / 2.0
        abs_center = (abs_start + abs_end) / 2.0

        row = {
            **feat,
            "rel_start_s": rel_start,
            "rel_end_s": rel_end,
            "rel_time_center": rel_center,   
            "abs_start_s": abs_start,
            "abs_end_s": abs_end,
            "abs_time_center": abs_center, 
            "label_id": int(maj),
            "label": inv_label_map.get(int(maj), int(maj)) if inv_label_map else int(maj),
            
        }
        rows.append(row)

    return pd.DataFrame(rows)

# Main 
def main():
    # Map label → id 
    all_labels = {name for labdict in labels_info.values() for name in labdict.keys()}
    label_map = {name: idx for idx, name in enumerate(sorted(all_labels))}
    inv_label_map = {v: k for k, v in label_map.items()}
    print("label_map:", label_map)

    # 
    with open("label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    all_tables = []
    for participant in PARTICIPANTS:
        print(f"\n[StatFeatures] Processing {participant}")
        p_path = os.path.join(DATASET_PATH, participant, "session")
        features_dir = os.path.join(p_path, "features")
        os.makedirs(features_dir, exist_ok=True)    

        wav = os.path.join(p_path, "P_audio.wav")
        if not os.path.exists(wav):
            print("Don't founded audio:", wav); continue

        ld = labels_info.get(participant)
        if not ld:
            print("No label:", participant); continue

        # Intervalli numerici e t0 (come train_offline)
        intervals, starts = [], []
        for label_name, (start_sec, end_sec) in ld.items():
            if label_name not in label_map:
                continue
            class_id = label_map[label_name]
            intervals.append((float(start_sec), float(end_sec), int(class_id)))
            starts.append(float(start_sec))
        if not intervals:
            print("Any valid interval."); continue

        t0_audio_sec = min(starts)

        # Load audio 
        audio, fs = load_audio(wav, target_sr=SR)
        y_seq = intervals_to_label_vector(len(audio), fs, intervals, t0_audio_sec, default_class=DEFAULT_CLASS)

        # Feature extraction
        df = extract_stat_features_like_train(
            audio=audio, y_seq=y_seq, fs=fs,
            win_sec=WIN_SEC, hop_sec=HOP_SEC,
            t0_audio_sec=t0_audio_sec, inv_label_map=inv_label_map
        )

        if df.empty:
            print("Any valid window."); continue

        df.insert(0, "participant", participant)
        out_csv = os.path.join(features_dir, f"{participant}_AudioFeatures_stats.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} | finestre: {len(df)}")

        all_tables.append(df)

    # CSV globale
    if all_tables:
        all_df = pd.concat(all_tables, axis=0, ignore_index=True)
        all_df.to_csv("audioFeatures_stats.csv", index=False)
        print("\nFile audioFeatures_stats.csv saved:", len(all_df))
    else:
        print("\nNo saved file.")

if __name__ == "__main__":
    main()
