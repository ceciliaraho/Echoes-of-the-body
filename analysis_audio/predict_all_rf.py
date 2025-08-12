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
N_MELS = 64
FRAME_LENGTH = 1024
HOP_LENGTH = 512

# === CARICAMENTO ===
rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === FUNZIONE DI ESTRAZIONE FEATURE ===
def extract_features_from_segment(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_LENGTH,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)

    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    feats = {}

    for i, val in enumerate(mel_mean):
        feats[f"mel_mean_{i}"] = val
    for i, val in enumerate(mel_std):
        feats[f"mel_std_{i}"] = val
    for i, val in enumerate(mfcc_mean):
        feats[f"mfcc_mean_{i}"] = val
    for i, val in enumerate(mfcc_std):
        feats[f"mfcc_std_{i}"] = val
    for i, val in enumerate(chroma_mean):
        feats[f"chroma_mean_{i}"] = val
    for i, val in enumerate(chroma_std):
        feats[f"chroma_std_{i}"] = val

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

print("\n🎧 Inizio analisi sessione lunga con Random Forest...\n")

for start in range(0, total_samples - window_samples + 1, window_samples):
    end = start + window_samples
    segment = y[start:end]
    time_start = start / SR
    time_end = end / SR

    feats = extract_features_from_segment(segment, sr)
    x = np.array([list(feats.values())]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = rf.predict(x_scaled)[0]
    prob = rf.predict_proba(x_scaled)[0]
    confidence = np.max(prob) * 100
    name = label_encoder.inverse_transform([pred])[0]

    print(f"🕒 {time_start:5.1f}s - {time_end:5.1f}s → 🧘‍♀️ {name.upper()} ({confidence:.1f}%)")
