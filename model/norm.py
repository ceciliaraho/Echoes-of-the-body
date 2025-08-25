import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("features_dataset.csv")

exclude = ["label", "participant", "time_center", "start", "end"]
cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# Assicurati che le feature siano float (evita warning)
df[cols] = df[cols].astype("float32")

scaler = MinMaxScaler()
for pid, group in df.groupby("participant"):
    df.loc[group.index, cols] = scaler.fit_transform(group[cols])

df.to_csv("features_dataset_normalized.csv", index=False)
print("✅ Normalizzazione per participant completata (dtype ok).")
