import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# === CONFIG ===
AUDIO_CSV = "audioFeatures_stats.csv"
BIO_CSV   = "features_dataset_minmax.csv"
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

N_ESTIMATORS = 300
RANDOM_STATE = 42
W_AUDIO = 0.55   # scelto in cross-validation

# === FUNZIONI ===
def strict_pairing_mapping(df_audio_in: pd.DataFrame, df_bio_in: pd.DataFrame) -> pd.DataFrame:
    parts = sorted(set(df_audio_in["participant"]).intersection(df_bio_in["participant"]))
    rows = []
    for p in parts:
        aa = df_audio_in[df_audio_in["participant"]==p].sort_values("rel_start_s").reset_index()
        bb = df_bio_in[df_bio_in["participant"]==p].sort_values("time_center").reset_index()
        n = min(len(aa), len(bb))
        for i in range(n):
            rows.append({
                "participant": p,
                "row_id_audio": int(aa.loc[i, "index"]),
                "row_id_bio":   int(bb.loc[i, "index"]),
                "rel_start_s":  float(aa.loc[i, "rel_start_s"]),
                "time_center":  float(bb.loc[i, "time_center"]),
            })
    return pd.DataFrame(rows)

def prepare_data(audio_csv, bio_csv):
    df_audio = pd.read_csv(audio_csv)
    df_bio   = pd.read_csv(bio_csv)
    exclude_audio = {'participant','label','label_id','abs_start_s','abs_end_s',
                     'abs_time_center','rel_time_center','rel_end_s','rel_start_s'}
    exclude_bio   = {'participant','label','time_center'}
    audio_features = [c for c in df_audio.columns if c not in exclude_audio and pd.api.types.is_numeric_dtype(df_audio[c])]
    bio_features   = [c for c in df_bio.columns   if c not in exclude_bio   and pd.api.types.is_numeric_dtype(df_bio[c])]
    mapping = strict_pairing_mapping(df_audio[["participant","rel_start_s"]],
                                     df_bio[["participant","time_center"]])
    X_audio = df_audio.loc[mapping["row_id_audio"], audio_features].reset_index(drop=True)
    X_bio   = df_bio.loc[mapping["row_id_bio"],   bio_features].reset_index(drop=True)
    y       = df_audio.loc[mapping["row_id_audio"], "label"].reset_index(drop=True)
    return X_audio, X_bio, y

# === MAIN ===
if __name__ == "__main__":
    print("Carico e preparo i dati...")
    Xa, Xb, y = prepare_data(AUDIO_CSV, BIO_CSV)

    print("Alleno Random Forest per audio...")
    rf_audio = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_audio.fit(Xa, y)

    print("Alleno Random Forest per biofeedback...")
    rf_bio = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_bio.fit(Xb, y)

    # Salvataggio modelli
    joblib.dump(rf_audio, MODEL_DIR / "rf_audio.pkl")
    joblib.dump(rf_bio, MODEL_DIR / "rf_bio.pkl")
    print(f"✅ Modelli salvati in {MODEL_DIR}")

    # Salva anche config con i pesi
    config = {
        "n_estimators": N_ESTIMATORS,
        "w_audio": W_AUDIO,
        "random_state": RANDOM_STATE
    }
    joblib.dump(config, MODEL_DIR / "fusion_config.pkl")
    print("✅ Configurazione di fusione salvata")

    print("Tutto pronto. Usa questi modelli + pesi quando avrai i 4 nuovi partecipanti!")
