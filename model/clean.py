import pandas as pd

df = pd.read_csv("features_dataset_minmax.csv")
df1 = pd.read_csv("features_dataset_robust.csv")
df2 = pd.read_csv("features_dataset_zscore.csv")

# Count lines where there is NaN value
print(f"Righe totali: {len(df)}")
print(f"NaN:\n{df.isna().sum()}")

# Count lines where there is NaN value
print(f"Righe totali: {len(df1)}")
print(f"NaN:\n{df1.isna().sum()}")

# Count lines where there is NaN value
print(f"Righe totali: {len(df2)}")
print(f"NaN:\n{df2.isna().sum()}")
# Delete these lines
df_clean = df.dropna()
print(f"After remove NaN: {len(df_clean)}")
 #Delete these lines
df_clean1 = df1.dropna()
print(f"After remove NaN: {len(df_clean1)}")
# Delete these lines
df_clean2 = df2.dropna()
print(f"After remove NaN: {len(df_clean2)}")

# Save new dataset
df_clean.to_csv("features_dataset_minmax_clean.csv", index=False)
df_clean1.to_csv("features_dataset_robust_clean.csv", index=False)
df_clean2.to_csv("features_dataset_zscore_clean.csv", index=False)
print("Saved file: features_dataset_clean.csv")
