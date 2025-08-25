import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Carica il CSV
df = pd.read_csv("audioFeatures_stats.csv")

# Top 20 feature
#top_features = [
#    'chroma_mean_5', 'chroma_mean_4', 'mfcc_mean_4', 'mel_mean_8', 'mel_mean_4',
#    'chroma_mean_7', 'mfcc_mean_5', 'mfcc_mean_9', 'mel_mean_12', 'chroma_mean_6',
#    'mel_mean_5', 'mel_std_55', 'mel_mean_10', 'mel_mean_11', 'mel_mean_13',
#    'mel_mean_9', 'chroma_std_7', 'chroma_mean_8', 'mel_mean_16', 'mfcc_mean_10'
#]

exclude_cols = ["label", "participant", "rel_start_s", "rel_end_s", "rel_time_center", "abs_start_s","abs_end_s", "abs_time_center", "label_id"]
all_features = [col for col in df.columns if col not in exclude_cols]

X = df[all_features].values
y = df["label"].values

# 3. Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Normalizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Calcolo dei pesi
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weights_dict = dict(enumerate(class_weights))
sample_weights = np.array([weights_dict[cls] for cls in y_train])

# Addestramento con pesi
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train, sample_weight=sample_weights)

# Valutazione
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.title("Confusion Matrix - Random Forest con class_weight")
plt.xlabel("Predetta")
plt.ylabel("Reale")
plt.tight_layout()
plt.show()


joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler_rf.pkl")
joblib.dump(label_encoder, "label_encoder_rf.pkl")