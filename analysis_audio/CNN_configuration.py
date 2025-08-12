# cnn_tuning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
print("Load data...")
df = pd.read_csv("all_participants_audio_features.csv")

exclude_cols = ["label", "participant", "start", "end", "time_center", "abs_start","abs_end", "abs_time_center"]
all_features = [col for col in df.columns if col not in exclude_cols]

#top_features = [
#    'chroma_mean_5', 'chroma_mean_4', 'mfcc_mean_4', 'mel_mean_8', 'mel_mean_4',
#    'chroma_mean_7', 'mfcc_mean_5', 'mfcc_mean_9', 'mel_mean_12', 'chroma_mean_6',
#    'mel_mean_5', 'mel_std_55', 'mel_mean_10', 'mel_mean_11', 'mel_mean_13',
#    'mel_mean_9', 'chroma_std_7', 'chroma_mean_8', 'mel_mean_16', 'mfcc_mean_10'
#]

X = df[all_features].values.astype(np.float32)
y = df["label"].astype(str).values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], len(all_features), 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# 2. Options
configs = {
    "A_base":      {"filters": [64, 128],      "dropout": 0.0, "lr": 0.001, "batch": 32},
    "B_dropout":   {"filters": [64, 128],      "dropout": 0.3, "lr": 0.001, "batch": 32},
    "C_light":     {"filters": [32, 64],       "dropout": 0.5, "lr": 0.0005, "batch": 16},
    "D_deep":      {"filters": [64, 128, 256], "dropout": 0.3, "lr": 0.0005, "batch": 32},
}

results = {}

# 3. Training loop
for name, cfg in configs.items():
    print(f"\n Training model {name}...")
    
    model = Sequential()
    model.add(Conv1D(cfg["filters"][0], 3, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(cfg["filters"][1], 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    if len(cfg["filters"]) == 3:
        model.add(Conv1D(cfg["filters"][2], 3, activation='relu'))
        model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    if cfg["dropout"] > 0:
        model.add(Dropout(cfg["dropout"]))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100, batch_size=cfg["batch"],
        callbacks=[early_stop], verbose=0
    )

    used_epochs = len(history.history['loss'])
    print(f"Used epochs {name}: {used_epochs}")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {acc:.3f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predict")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    results[name] = {
        "acc": acc,
        "history": history.history,
        "epochs": used_epochs,
        "model": model
    }

# 4. Accuracy
plt.figure(figsize=(8,4))
for name, res in results.items():
    plt.plot(res["history"]["val_accuracy"], label=f"{name}")
plt.title("Validation accuracy - Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. best model
best_name = max(results, key=lambda x: results[x]['acc'])
print(f"\n Best: {best_name} with accuracy {results[best_name]['acc']:.3f} and {results[best_name]['epochs']} epochs")
