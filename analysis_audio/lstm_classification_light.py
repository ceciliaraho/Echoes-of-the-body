import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf

# === PARAMETRI ===
window_size = 430
feature_cols = ['rms', 'zcr', 'centroid', 'spectral_flatness',
                'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
                'chroma_mean', 'chroma_var']
label_col = 'label'

# === CARICAMENTO DATI ===
df = pd.read_csv("all_participants_audio_features.csv")
df = df[df["label"].str.lower() != "unlabeled"]
df = df.dropna()

# === NORMALIZZAZIONE ===
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# === COSTRUZIONE SEQUENZE ===
X = []
y = []

for i in range(0, len(df) - window_size):
    window = df.iloc[i:i+window_size]
    label = df.iloc[i + window_size][label_col]
    if len(set(window["label"])) == 1:
        sequence = window[feature_cols].values
        X.append(sequence)
        y.append(label)

X = np.array(X)
y = np.array(y)

# === ENCODING LABELS ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# === MODELLO LSTM LIGHT ===
model = Sequential()
model.add(LSTM(32, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# === TRAINING VELOCE ===
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# === VALUTAZIONE ===
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")

# === SALVATAGGIO SE SOPRA SOGLIA ===
#if acc > 0.6:
#    model.save("lstm_saved_model.h5")
#    print("✅ Modello salvato: lstm_saved_model.h5")
#else:
#    print("⚠️ Accuracy bassa, modello NON salvato.")
