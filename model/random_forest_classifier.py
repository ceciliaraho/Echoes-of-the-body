import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# === 1. Caricamento dati ===
df = pd.read_csv("features_dataset_clean.csv")

# === 2. Separazione X e y ===
X = df.drop(columns=["label", "participant", "time_center"])
y = df["label"]

# Encoding etichette
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === 4. Addestramento modello Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === 5. Predizione e valutazione ===
y_pred = clf.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === 6. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# === 7. Feature Importance ===
importances = clf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
