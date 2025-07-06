import pandas as pd

df = pd.read_csv("features_dataset.csv")

# Count lines where there is NaN value
print(f"Righe totali: {len(df)}")
print(f"NaN:\n{df.isna().sum()}")

# Delete these lines
df_clean = df.dropna()
print(f"After remove NaN: {len(df_clean)}")

# Save new dataset
df_clean.to_csv("features_dataset_clean.csv", index=False)
print("Saved file: features_dataset_clean.csv")
