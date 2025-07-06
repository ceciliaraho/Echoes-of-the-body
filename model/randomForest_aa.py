import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# === Carica i dati ===
df = pd.read_csv("features_dataset_clean.csv")

# === Prepara X e y ===
X = df.drop(columns=["label", "participant", 'time_center'])
y = LabelEncoder().fit_transform(df["label"])
participants = df["participant"].unique()

# === LOPOCV ===
all_true = []
all_pred = []
per_participant_report = defaultdict(dict)

for p in participants:
    train_idx = df["participant"] != p
    test_idx = df["participant"] == p

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    acc = accuracy_score(y_test, y_pred)
    per_participant_report[p]["accuracy"] = acc
    print(f"📍 Participant {p}: Accuracy = {acc:.2f}")

# === Report finale ===
print("\n=== Global Classification Report (LOPOCV) ===")
print(classification_report(all_true, all_pred))
