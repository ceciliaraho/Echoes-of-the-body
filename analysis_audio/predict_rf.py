# predict_session_offline_rf.py

import numpy as np
import librosa
import joblib

# === CONFIGURAZIONE ===
AUDIO_PATH = "P_audio1.wav"
MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler_rf.pkl"
ENCODER_PATH = "label_encoder_rf.pkl"
WINDOW_SEC = 5
SR = 22050

# === CARICAMENTO ===
rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === FEATURE SET ===
top_features = [
    'chroma_mean_5', 'chroma_mean_4', 'mfcc_mean_4', 'mel_mean_8', 'mel_mean_4',
    'chroma_mean_7', 'mfcc_mean_5', 'mfcc_mean_9', 'mel_mean_12', 'chroma_mean_6',
    'mel_mean_5', 'mel_std_55', 'mel_mean_10', 'mel_mean_11', 'mel_mean_13',
    'mel_mean_9', 'chroma_std_7', 'chroma_mean_8', 'mel_mean_16', 'mfcc_mean_10'
]

# === ESTRAZIONE FEATURE PER OGNI FINESTRA ===
def extract_features_from_segment(y, sr):
    hop_length = 512
    n_fft = 1024

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    feats = {}
    for i in range(len(mel_mean)):
        feats[f"mel_mean_{i}"] = mel_mean[i]
    for i in range(len(mel_std)):
        feats[f"mel_std_{i}"] = mel_std[i]
    for i in range(len(mfcc_mean)):
        feats[f"mfcc_mean_{i}"] = mfcc_mean[i]
    for i in range(len(chroma_mean)):
        feats[f"chroma_mean_{i}"] = chroma_mean[i]
    for i in range(len(chroma_std)):
        feats[f"chroma_std_{i}"] = chroma_std[i]

    return feats

# === ANALISI AUDIO ===
y, sr = librosa.load(AUDIO_PATH, sr=SR)
total_samples = len(y)
window_samples = WINDOW_SEC * SR

print("\n🎧 Inizio analisi sessione lunga con Random Forest...\n")

for start in range(0, total_samples - window_samples + 1, window_samples):
    end = start + window_samples
    segment = y[start:end]
    time_start = start / SR
    time_end = end / SR

    feats = extract_features_from_segment(segment, SR)
    x = np.array([feats[f] for f in top_features]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = rf.predict(x_scaled)[0]
    prob = rf.predict_proba(x_scaled)[0]
    confidence = np.max(prob) * 100
    name = label_encoder.inverse_transform([pred])[0]

    print(f"🕒 {time_start:5.1f}s - {time_end:5.1f}s → 🧘‍♀️ {name.upper()} ({confidence:.1f}%)")
