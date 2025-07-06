
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# === CARICA IL CSV ===
df = pd.read_csv("features_dataset_clean.csv")

# === RIMUOVI COLONNE NON UTILI (es. 'time_center') ===
#if "time_center" in df.columns:
#    df = df.drop(columns=["time_center"])

# === SEPARA FEATURE E LABEL ===
X = df.drop(columns=["label", 'participant', 'time_center'])
y = df["label"]

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
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# === 6. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.show()

# === SALVA MODELLO E ENCODER ===
#joblib.dump(model, "xgboost_model.joblib")
#joblib.dump(le, "label_encoder.joblib")
#print("\nModello e encoder salvati.")

# === Calcola l'importanza delle feature ===
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

# === Plot ===
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.show()