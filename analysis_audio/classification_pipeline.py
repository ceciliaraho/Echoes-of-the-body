import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv("all_participants_audio_features.csv")
df = df[df["label"].str.lower() != "unlabeled"]

#Feature and label separation
features = [col for col in df.columns if col not in ["time_sec", "label", "participant"]]
X = df[features].values
y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Classifiers
classifiers = [
    ("KNN", KNeighborsClassifier()),
    ("Random Forest", RandomForestClassifier(n_estimators=30, max_depth=10, class_weight='balanced')),
    ("MLP (1 hidden layer)", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500))
]

# Valuation
print("Classifier Evaluation")
for name, clf in classifiers:
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<25}: Accuracy = {acc:.4f}")
