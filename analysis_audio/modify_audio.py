import librosa
import soundfile as sf
import numpy as np

# === STEP 1: Carica audio originale di P2 ===
p2_audio_path = "P2.wav"  # Cambia con il path corretto
y, sr = librosa.load(p2_audio_path, sr=None)

# === STEP 2: Aggiungi 50.32 s di silenzio iniziale ===
silence_duration = 54.32 - 4  # 50.32 s
silence = np.zeros(int(silence_duration * sr))
y_shifted = np.concatenate((silence, y))

# === STEP 3: Applica pitch shift leggero (opzionale) ===
y_shifted = librosa.effects.pitch_shift(y_shifted, sr=sr, n_steps=1)

# === STEP 4: Timestamp P2 e P7 ===
p2_labels = {
    'pranayama': (4, 395),
    'chanting': (395, 774),
    'viparita_swasa': (774, 995),
    'breath_retention': (995, 1060),
    'meditation': (1060, 1342)
}

p7_labels = {
    'pranayama': (54.32, 476),
    'chanting': (476, 752),
    'viparita_swasa': (752, 992),
    'breath_retention': (992, 1042),
    'meditation': (1042, 1375)
}

# === STEP 5: Funzione per stretch ===
def stretch_phase(y, sr, start_old, end_old, target_duration):
    y_phase = y[int(start_old * sr):int(end_old * sr)]
    src_duration = end_old - start_old
    rate = src_duration / target_duration
    return librosa.effects.time_stretch(y_phase, rate=rate)

# === STEP 6: Ricostruzione dell’audio simulato ===
phases = ['pranayama', 'chanting', 'viparita_swasa', 'breath_retention', 'meditation']
y_simulated = np.array([])

for phase in phases:
    start_old, end_old = p2_labels[phase]
    # Aggiungi il delay iniziale a tutti i timestamp di P2
    start_old += silence_duration
    end_old += silence_duration
    start_new, end_new = p7_labels[phase]
    target_duration = end_new - start_new
    y_mod = stretch_phase(y_shifted, sr, start_old, end_old, target_duration)
    y_simulated = np.concatenate((y_simulated, y_mod))

# === STEP 7: Salvataggio finale ===
sf.write("P7_simulato1.wav", y_simulated, sr)
print("✔️ File audio simulato salvato come 'P7_simulato.wav'")
