import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il tuo CSV con le feature
df = pd.read_csv("features_dataset_clean.csv")

# Escludi le colonne da conservare
cols_to_keep = ["label", "participant", "time_center"]

# Trova le colonne con NaN (escludendo quelle da conservare)
nan_cols = df.drop(columns=cols_to_keep).columns[df.drop(columns=cols_to_keep).isna().any()]

# Rimuovile
df_clean = df.drop(columns=nan_cols)
# Melt dei dati per avere un formato compatibile con il plot
df_melted = pd.melt(df_clean, id_vars=["label", "participant", "time_center"], 
                    var_name="feature", value_name="value")

# Crea il plot
plt.figure(figsize=(16, 10))
sns.set(style="whitegrid", palette="husl")
ax = sns.stripplot(x="feature", y="value", hue="label", data=df_melted,
                   jitter=True, edgecolor="white", alpha=0.7)

plt.title("Distribuzione delle feature per fase di yoga")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
