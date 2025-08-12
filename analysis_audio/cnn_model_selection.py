# cnn_tuning_groups.py
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ---------------- Repro ----------------
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(0); np.random.seed(0); tf.random.set_seed(0)

CSV = "all_participants_audio_features.csv"
EXCLUDE = ["label","participant","start","end","time_center","abs_start","abs_end","abs_time_center"]

# ---------------- Load -----------------
print("🔍 Caricamento dati…")
df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feat_cols].values.astype(np.float32)
y_str = df["label"].astype(str).values
groups = df["participant"].astype(str).values

le = LabelEncoder()
y_int = le.fit_transform(y_str)
y_cat = to_categorical(y_int)
n_classes = y_cat.shape[1]
input_len = X.shape[1]

# ------------- Group splits -----------
# test 20% by participant
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss1.split(X, y_int, groups=groups))

# from remaining train, make validation 25% of it -> 60/20/20 overall
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
sub_groups = groups[train_idx]
sub_y = y_int[train_idx]
sub_idx_train, sub_idx_val = next(gss2.split(X[train_idx], sub_y, groups=sub_groups))
train_idx = train_idx[sub_idx_train]
val_idx   = train_idx[sub_idx_val]  # careful: we reindex into original

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y_cat[train_idx], y_cat[val_idx], y_cat[test_idx]

# ---------- Scale (fit on train only) ----------
scaler = StandardScaler().fit(X_train)
def reshape(a): return scaler.transform(a).reshape((a.shape[0], input_len, 1))
X_train, X_val, X_test = reshape(X_train), reshape(X_val), reshape(X_test)

# ------------- Configs ----------------
configs = {
    "A_base":    {"filters":[64,128],      "dropout":0.0, "batch":32},
    "B_dropout": {"filters":[64,128],      "dropout":0.3, "batch":32},
    "C_light":   {"filters":[32,64],       "dropout":0.5, "batch":16},
    "D_deep":    {"filters":[64,128,256],  "dropout":0.3, "batch":32},
}

def build_model(cfg):
    m = Sequential()
    m.add(Conv1D(cfg["filters"][0], 3, activation="relu", input_shape=(input_len,1)))
    m.add(BatchNormalization()); m.add(MaxPooling1D(2))
    m.add(Conv1D(cfg["filters"][1], 3, activation="relu"))
    m.add(BatchNormalization()); m.add(MaxPooling1D(2))
    if len(cfg["filters"])==3:
        m.add(Conv1D(cfg["filters"][2], 3, activation="relu"))
        m.add(BatchNormalization())
    m.add(GlobalAveragePooling1D())
    m.add(Dense(128, activation="relu"))
    if cfg["dropout"]>0: m.add(Dropout(cfg["dropout"]))
    m.add(Dense(n_classes, activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

results = {}

# ------------- Train loop -------------
for name, cfg in configs.items():
    print(f"\n🚀 Training {name} …")

    model = build_model(cfg)

    early = EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.002,
        restore_best_weights=True
    )
    reduce = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_delta=0.001, min_lr=1e-6, verbose=1
    )
    ckpt = ModelCheckpoint(f"best_{name}.keras", monitor="val_loss", save_best_only=True, verbose=0)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=cfg["batch"],
        callbacks=[early, reduce, ckpt],
        verbose=0, shuffle=True
    )

    used_epochs = len(hist.history["loss"])
    print(f"⏱️ Epoche usate: {used_epochs}")

    # ----- Test evaluation -----
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y_true, y_pred)

    print(f"✅ Test Accuracy: {test_acc:.3f} | Macro‑F1: {macro_f1:.3f} | Balanced Acc: {bal_acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predetta"); plt.ylabel("Reale")
    plt.tight_layout(); plt.savefig(f"cm_{name}.png", dpi=150); plt.close()

    # Store results
    results[name] = {
        "val_acc_curve": hist.history["val_accuracy"],
        "epochs": used_epochs,
        "test_acc": float(test_acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
    }

# --------- Curva di validazione ----------
plt.figure(figsize=(9,4))
for name, res in results.items():
    plt.plot(res["val_acc_curve"], label=name)
plt.title("Validation accuracy - Comparison")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("validation_comparison.png", dpi=150); plt.close()

# --------- Tabella risultati -------------
res_df = pd.DataFrame([
    {"model":k, "epochs":v["epochs"], "test_acc":v["test_acc"],
     "macro_f1":v["macro_f1"], "balanced_acc":v["balanced_acc"]}
    for k,v in results.items()
]).sort_values("test_acc", ascending=False)
print("\n🏁 Risultati finali (ordinati per test_acc):")
print(res_df.to_string(index=False))
res_df.to_csv("results_summary.csv", index=False)

best = res_df.iloc[0]["model"]
print(f"\n🏆 Best: {best}  —  file pesi: best_{best}.keras")
