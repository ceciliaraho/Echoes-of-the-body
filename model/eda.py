import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("features_dataset_clean.csv")

cols_to_keep = ["label", "participant", "time_center"]

nan_cols = df.drop(columns=cols_to_keep).columns[df.drop(columns=cols_to_keep).isna().any()]

df_clean = df.drop(columns=nan_cols)
df_melted = pd.melt(df_clean, id_vars=["label", "participant", "time_center"], 
                    var_name="feature", value_name="value")

plt.figure(figsize=(16, 10))
sns.set(style="whitegrid", palette="husl")
ax = sns.stripplot(x="feature", y="value", hue="label", data=df_melted,
                   jitter=True, edgecolor="white", alpha=0.7)

plt.title("Distribuzione delle feature per fase di yoga")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
