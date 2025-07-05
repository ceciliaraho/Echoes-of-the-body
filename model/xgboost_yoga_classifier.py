
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# === CARICA IL CSV ===
df = pd.read_csv("features_dataset.csv")

# === RIMUOVI COLONNE NON UTILI (es. 'time_center') ===
if "time_center" in df.columns:
    df = df.drop(columns=["time_center"])

# === SEPARA FEATURE E LABEL ===
X = df.drop(columns=["Label"])
y = df["Label"]

# === ENCODING DELLE LABEL ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === SPLIT TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === MODELLO XGBOOST ===
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False
)
model.fit(X_train, y_train)

# === PREDIZIONI ===
y_pred = model.predict(X_test)

# === REPORT ===
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === SALVA MODELLO E ENCODER ===
joblib.dump(model, "xgboost_model.joblib")
joblib.dump(le, "label_encoder.joblib")
print("\nModello e encoder salvati.")
