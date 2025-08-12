import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import joblib

# 1. Caricamento dati
df = pd.read_csv("all_participants_audio_features.csv")
#top_features = [
#    'chroma_mean_5', 'chroma_mean_4', 'mfcc_mean_4', 'mel_mean_8', 'mel_mean_4',
#    'chroma_mean_7', 'mfcc_mean_5', 'mfcc_mean_9', 'mel_mean_12', 'chroma_mean_6',
#    'mel_mean_5', 'mel_std_55', 'mel_mean_10', 'mel_mean_11', 'mel_mean_13',
#    'mel_mean_9', 'chroma_std_7', 'chroma_mean_8', 'mel_mean_16', 'mfcc_mean_10'
#]

exclude_cols = ["label", "participant", "start", "end", "time_center", "abs_start","abs_end", "abs_time_center"]
all_features = [col for col in df.columns if col not in exclude_cols]

X = df[all_features].values
y = df["label"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], len(all_features), 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# 2. Calcolo class weights per bilanciare le classi
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

#class_weights_dict = class_weights_dict = {
#    0: 3.0,   # breath_retention
#    1: 1.7,   # chanting
#    2: 0.5,   # meditation
#    3: 1.2,   # pranayama
#    4: 2.5    # viparita_swasa
#}

# 3. Definizione modello D_deep
model = Sequential([
    Input(shape=(X_reshaped.shape[1], 1)),
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(256, 3, activation='relu'),
    BatchNormalization(),

    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 4. Addestramento
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=75,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
    #class_weight=class_weights_dict
)

model.save("cnn_d_deep_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# 5. Valutazione
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Accuracy finale D_deep: {acc:.3f}")

# 6. Grafico accuratezza
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Andamento Accuracy - Modello D_deep")
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predetta")
plt.ylabel("Reale")
plt.title("Confusion Matrix - D_deep")
plt.tight_layout()
plt.show()

# 8. Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

