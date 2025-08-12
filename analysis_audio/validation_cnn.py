# predict_session_offline.py for cnn with all features

import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === CONFIGURAZIONE ===
AUDIO_PATH = "P_audio1.wav"
MODEL_PATH = "cnn_d_deep_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
WINDOW_SEC = 5
SR = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 1024
N_MELS = 64

# === CARICAMENTO MODELLO E STRUMENTI ===
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === ESTRAZIONE DI TUTTE LE FEATURE ===
def extract_all_features(y, sr):
    # Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_LENGTH,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)

    # MFCC
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    # Dizionario completo delle feature
    feats = {}
    for i, v in enumerate(mel_mean): feats[f"mel_mean_{i}"] = v
    for i, v in enumerate(mel_std): feats[f"mel_std_{i}"] = v
    for i, v in enumerate(mfcc_mean): feats[f"mfcc_mean_{i}"] = v
    for i, v in enumerate(mfcc_std): feats[f"mfcc_std_{i}"] = v
    for i, v in enumerate(chroma_mean): feats[f"chroma_mean_{i}"] = v
    for i, v in enumerate(chroma_std): feats[f"chroma_std_{i}"] = v

    feats["spec_centroid"] = spec_centroid
    feats["spec_rolloff"] = spec_rolloff
    feats["spec_bandwidth"] = spec_bandwidth
    feats["zero_crossing_rate"] = zcr
    feats["rms"] = rms

    return feats

# === ANALISI AUDIO ===
y, sr = librosa.load(AUDIO_PATH, sr=SR)
total_samples = len(y)
window_samples = WINDOW_SEC * SR

print("\n🎧 Inizio analisi sessione lunga (CNN con tutte le feature)...\n")

for start in range(0, total_samples - window_samples + 1, window_samples):
    end = start + window_samples
    segment = y[start:end]
    time_start = start / SR
    time_end = end / SR

    feats = extract_all_features(segment, SR)
    x = np.array(list(feats.values())).reshape(1, -1)

    # Normalizza e reshapa per CNN
    x_scaled = scaler.transform(x)
    x_reshaped = x_scaled.reshape((1, x_scaled.shape[1], 1))

    pred_probs = model.predict(x_reshaped, verbose=0)
    pred_label = np.argmax(pred_probs)
    pred_name = label_encoder.inverse_transform([pred_label])[0]

    print(f"🕒 {time_start:5.1f}s - {time_end:5.1f}s → 🧘‍♀️ {pred_name.upper()} ({pred_probs[0][pred_label]*100:.1f}%)")
