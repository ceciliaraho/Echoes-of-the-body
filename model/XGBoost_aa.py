import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("features_dataset_minmax_clean.csv")

# Codifica le label numericamente
label_mapping = {label: i for i, label in enumerate(df['label'].unique())}
df['label_encoded'] = df['label'].map(label_mapping)

# Definizione di X, y, groups
X = df.drop(columns=["label", "participant", "label_encoded", "time_center"])
y = df['label_encoded']
groups = df['participant']

# Leave-One-Participant-Out CV
logo = LeaveOneGroupOut()
all_y_true = []
all_y_pred = []

participant_scores = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    participant = groups.iloc[test_idx].iloc[0]

    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    participant_scores.append((participant, acc))

    print(f"📍 Participant {participant}: Accuracy = {acc:.2f}")

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Report globale
print("\n=== Global Classification Report (LOPOCV - XGBoost) ===")
print(classification_report(all_y_true, all_y_pred, target_names=label_mapping.keys()))

# Salva label mapping se necessario
#label_mapping_df = pd.DataFrame.from_dict(label_mapping, orient='index', columns=["encoded"])
#label_mapping_df.to_csv("/mnt/data/label_mapping_xgboost.csv")

# Salva grafico accuracy per partecipante
participants, accuracies = zip(*participant_scores)
plt.figure(figsize=(10, 5))
plt.bar(participants, accuracies, color="teal")
plt.ylabel("Accuracy")
plt.xlabel("Participant")
plt.title("LOPOCV Accuracy per Participant (XGBoost)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

