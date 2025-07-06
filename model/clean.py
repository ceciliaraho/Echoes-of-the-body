import pandas as pd

df = pd.read_csv("features_dataset.csv")

# Conta righe incomplete
print(f"Righe totali: {len(df)}")
print(f"Righe con valori NaN:\n{df.isna().sum()}")

# Rimuove righe con qualsiasi valore mancante
df_clean = df.dropna()
print(f"Righe rimanenti dopo dropna: {len(df_clean)}")

# Salva nuovo dataset
df_clean.to_csv("features_dataset_clean.csv", index=False)
print("✔ File salvato: features_dataset_clean.csv")