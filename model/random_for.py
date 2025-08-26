#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio RandomForest (Top-K features + StandardScaler) senza riga di comando.
- Selezione feature sul TRAIN (RF importances) -> Top-K
- Holdout 80/20 (opzionale) con selezione nel fold
- LOSO (Leave-One-Subject-Out) con selezione per fold
- Fit finale su TUTTI i dati (pipeline pronta per produzione)
- Salvataggi: modello, encoder, metriche, predizioni+probabilità, top features

REQUISITI CSV:
- Colonne obbligatorie: 'label' (target), 'participant' (per LOSO)
- Le feature devono essere numeriche; verranno escluse automaticamente:
  ['participant','label','time_center','start','end']
"""

# =========================
# CONFIG — MODIFICA QUI
# =========================
CSV_PATH   = "features_dataset.csv"   # <-- metti il percorso del tuo CSV
OUTDIR     = "out_bio"                # <-- cartella dove salvare tutto
TOP_K      = 30                       # numero di feature da tenere
RF_RANK_EST = 300                     # alberi per ranking feature
RF_FINAL_EST = 200                    # alberi del modello finale
DO_HOLDOUT = True                     # True = esegue anche holdout 80/20

# =========================
# CODICE
# =========================
import os
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from joblib import dump

# (Opzionale) se vuoi anche la confusion matrix in PNG per l'holdout
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

META_COLS = ['participant','label','time_center','start','end']


def load_data(csv_path: str,
              target_col: str = "label",
              participant_col: str = "participant"
              ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Carica CSV, separa X/y, rimuove metadati, tiene solo le colonne numeriche."""
    assert os.path.exists(csv_path), f"CSV non trovato: {csv_path}"
    df = pd.read_csv(csv_path)
    assert target_col in df.columns, f"Target '{target_col}' mancante."
    assert participant_col in df.columns, f"Colonna '{participant_col}' mancante."

    y_raw = df[target_col].copy()
    groups = df[participant_col].copy()

    drop_cols = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    feat_cols = list(X.columns)
    return X, y_raw, groups, feat_cols


def rank_features_rf(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 300) -> pd.DataFrame:
    """Ranking feature con RF importances + imputation (RF non richiede scaling)."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ))
    ])
    pipe.fit(X, y)
    importances = pipe.named_steps["rf"].feature_importances_
    ranking = (pd.DataFrame({"feature": X.columns, "importance": importances})
               .sort_values("importance", ascending=False)
               .reset_index(drop=True))
    return ranking


def build_pipeline(selected_features: List[str],
                   n_estimators: int = 200) -> Pipeline:
    """Pipeline finale: Imputer → StandardScaler → RandomForest."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), selected_features),
        ],
        remainder='drop'
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_features='sqrt'
    )
    return Pipeline([("prep", preprocess), ("rf", rf)])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict:
    """Metriche principali."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def save_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_png: Path, title: str):
    """Salva una confusion matrix come immagine PNG (matplotlib)."""
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45, ha='right')
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    # numeri nelle celle
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def holdout_training(X: pd.DataFrame, y_enc: np.ndarray, le: LabelEncoder,
                     outdir: Path, top_k: int = 30,
                     rf_rank_est: int = 300, rf_final_est: int = 200) -> None:
    """Holdout stratificato 80/20 con selezione feature sul TRAIN."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    # Ranking feature sul solo TRAIN
    ranking = rank_features_rf(X_train, y_train, n_estimators=rf_rank_est)
    K = min(top_k, ranking.shape[0])
    top_features = ranking.head(K)["feature"].tolist()

    # Salva ranking + top features
    ranking.to_csv(outdir/"feature_importances_all.csv", index=False)
    pd.DataFrame({"selected_feature": top_features}).to_csv(outdir/"selected_features.csv", index=False)

    # Fit pipeline
    pipe = build_pipeline(top_features, n_estimators=rf_final_est)
    pipe.fit(X_train[top_features], y_train)

    # Eval
    y_pred = pipe.predict(X_test[top_features])
    y_proba = pipe.predict_proba(X_test[top_features])

    labels_idx = list(range(len(le.classes_)))
    metrics = evaluate_predictions(y_test, y_pred, labels_idx)
    report = classification_report(y_test, y_pred, labels=labels_idx, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)

    # Salvataggi
    with open(outdir/"metrics_holdout.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "classes": le.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
            "report": report
        }, f, indent=2)

    # Predizioni + proba (per debug e fusion)
    proba_cols = [f"proba_{c}" for c in le.classes_]
    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(y_pred),
    })
    pred_df = pd.concat([pred_df, pd.DataFrame(y_proba, columns=proba_cols)], axis=1)
    pred_df.to_csv(outdir/"bio_test_predictions_holdout.csv", index=False)

    # Confusion matrix PNG (opzionale ma utile)
    save_confusion_matrix_png(cm, list(le.classes_), outdir/"confusion_matrix_holdout.png",
                              title="Confusion Matrix — Holdout")


def loso_evaluation(X: pd.DataFrame, y_enc: np.ndarray, groups: pd.Series,
                    le: LabelEncoder, outdir: Path,
                    top_k: int = 30, rf_rank_est: int = 300, rf_final_est: int = 200) -> None:
    """
    Per ogni partecipante: ranking feature sul TRAIN (senza test), fit pipeline, eval sul partecipante escluso.
    Salva metriche aggregate e predizioni/probabilità per istanza.
    """
    out_feats = outdir / "loso_fold_top_features"
    out_feats.mkdir(parents=True, exist_ok=True)

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    proba_rows = []
    labels_idx = list(range(len(le.classes_)))

    for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X, y_enc, groups=groups), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y_enc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y_enc[te_idx]
        test_group = groups.iloc[te_idx].iloc[0]  # tutti i te_idx hanno stesso participant

        # Ranking sul TRAIN del fold
        ranking = rank_features_rf(X_tr, y_tr, n_estimators=rf_rank_est)
        K = min(top_k, ranking.shape[0])
        top_features = ranking.head(K)["feature"].tolist()

        # Salva Top-K per fold
        pd.DataFrame({"selected_feature": top_features}).to_csv(
            out_feats/f"fold_{fold_i:02d}_participant_{test_group}.csv", index=False
        )

        # Fit e predizione
        pipe = build_pipeline(top_features, n_estimators=rf_final_est)
        pipe.fit(X_tr[top_features], y_tr)

        y_pred = pipe.predict(X_te[top_features])
        y_proba = pipe.predict_proba(X_te[top_features])

        y_true_all.append(y_te)
        y_pred_all.append(y_pred)

        proba_cols = [f"proba_{c}" for c in le.classes_]
        fold_df = pd.DataFrame({
            "participant": groups.iloc[te_idx].values,
            "true_label": le.inverse_transform(y_te),
            "pred_label": le.inverse_transform(y_pred),
        })
        fold_df = pd.concat([fold_df, pd.DataFrame(y_proba, columns=proba_cols).reset_index(drop=True)], axis=1)
        proba_rows.append(fold_df)

    # Concatenate
    y_true_concat = np.concatenate(y_true_all, axis=0)
    y_pred_concat = np.concatenate(y_pred_all, axis=0)
    proba_df_all = pd.concat(proba_rows, axis=0).reset_index(drop=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true_concat, y_pred_concat)),
        "f1_macro": float(f1_score(y_true_concat, y_pred_concat, average="macro", labels=labels_idx, zero_division=0)),
        "f1_weighted": float(f1_score(y_true_concat, y_pred_concat, average="weighted", labels=labels_idx, zero_division=0)),
    }
    cm = confusion_matrix(y_true_concat, y_pred_concat, labels=labels_idx)
    report = classification_report(y_true_concat, y_pred_concat, labels=labels_idx, target_names=le.classes_, zero_division=0)

    with open(outdir/"metrics_loso.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "classes": le.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
            "report": report
        }, f, indent=2)

    proba_df_all.to_csv(outdir/"loso_predictions.csv", index=False)


def fit_final_on_all(X: pd.DataFrame, y_enc: np.ndarray, le: LabelEncoder,
                     outdir: Path, top_k: int = 30,
                     rf_rank_est: int = 300, rf_final_est: int = 200) -> None:
    """
    Ranking su TUTTO il dataset -> Top-K -> fit pipeline su TUTTO -> salva artefatti.
    NB: questo è il modello da usare in produzione (scaler "congelato").
    """
    ranking = rank_features_rf(X, y_enc, n_estimators=rf_rank_est)
    K = min(top_k, ranking.shape[0])
    top_features = ranking.head(K)["feature"].tolist()

    ranking.to_csv(outdir/"feature_importances_all.csv", index=False)
    pd.DataFrame({"selected_feature": top_features}).to_csv(outdir/"selected_features.csv", index=False)

    pipe = build_pipeline(top_features, n_estimators=rf_final_est)
    pipe.fit(X[top_features], y_enc)

    dump(pipe, outdir/"bio_rf_model.joblib")
    dump(le, outdir/"label_encoder.joblib")


def run_all():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Carica dati
    X, y_raw, groups, feat_cols = load_data(CSV_PATH)
    print(f"[INFO] Caricati {len(X)} campioni, {len(feat_cols)} feature numeriche.")
    # 2) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"[INFO] Classi: {list(le.classes_)}")

    # 3) Holdout (opzionale)
    if DO_HOLDOUT:
        print("[INFO] Holdout 80/20...")
        holdout_training(X, y_enc, le, outdir, top_k=TOP_K,
                         rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)

    # 4) LOSO
    print("[INFO] LOSO evaluation...")
    loso_evaluation(X, y_enc, groups, le, outdir, top_k=TOP_K,
                    rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)

    # 5) Fit finale su tutti i dati
    print("[INFO] Final fit on ALL data...")
    fit_final_on_all(X, y_enc, le, outdir, top_k=TOP_K,
                     rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)

    print(f"[DONE] Artefatti salvati in: {outdir.resolve()}")
    print(" - bio_rf_model.joblib (pipeline: Imputer+Scaler+RF)")
    print(" - label_encoder.joblib")
    print(" - selected_features.csv, feature_importances_all.csv")
    if DO_HOLDOUT:
        print(" - metrics_holdout.json, bio_test_predictions_holdout.csv, confusion_matrix_holdout.png")
    print(" - metrics_loso.json, loso_predictions.csv, loso_fold_top_features/")


# Esegui tutto quando lanci il file
if __name__ == "__main__":
    run_all()
