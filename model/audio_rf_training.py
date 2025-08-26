
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio RandomForest (Top-K + StandardScaler) — no command line.
- Selezione feature sul TRAIN (RF importances) -> Top-K
- Holdout 80/20 (opzionale)
- LOSO (Leave-One-Subject-Out) su 'participant'
- Fit finale su TUTTI i dati e salvataggio artefatti

CONFIG: modifica CSV_PATH e OUTDIR qui sotto e poi esegui il file.
"""

# =========================
# CONFIG — MODIFICA QUI
# =========================
CSV_PATH   = "audioFeatures_stats.csv"  # percorso al CSV audio
OUTDIR     = "out_audio"                # cartella output
TOP_K      = 30
RF_RANK_EST = 300
RF_FINAL_EST = 200
DO_HOLDOUT = True

# =========================
# CODICE
# =========================
import os, json, warnings
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import dump

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

META_AUDIO = ['participant','label','label_id','abs_start_s','abs_end_s','abs_time_center','rel_time_center','rel_end_s','rel_start_s']

def load_audio(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    assert os.path.exists(csv_path), f"CSV non trovato: {csv_path}"
    df = pd.read_csv(csv_path)
    assert 'label' in df.columns, "Manca la colonna 'label'."
    assert 'participant' in df.columns, "Manca la colonna 'participant'."
    y_raw = df['label'].copy()
    groups = df['participant'].copy()
    X = df.drop(columns=[c for c in META_AUDIO if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y_raw, groups, list(X.columns)

def rank_features_rf(X, y, n_estimators=300):
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
    imp = pipe.named_steps["rf"].feature_importances_
    return pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)

def build_pipe(selected_features: list, n_estimators=200):
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

def save_cm_png(cm, class_names, out_png, title):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45, ha='right')
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def holdout(X, y_enc, le, outdir, top_k=30, rf_rank_est=300, rf_final_est=200):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE)
    rank = rank_features_rf(X_tr, y_tr, n_estimators=rf_rank_est)
    K = min(top_k, rank.shape[0]); top = rank.head(K)["feature"].tolist()
    rank.to_csv(outdir/"audio_feature_importances_all.csv", index=False)
    pd.DataFrame({"selected_feature": top}).to_csv(outdir/"audio_selected_features.csv", index=False)
    pipe = build_pipe(top, n_estimators=rf_final_est); pipe.fit(X_tr[top], y_tr)
    y_pred = pipe.predict(X_te[top]); y_proba = pipe.predict_proba(X_te[top])
    labels_idx = list(range(len(le.classes_)))
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", labels=labels_idx, zero_division=0)),
        "f1_weighted": float(f1_score(y_te, y_pred, average="weighted", labels=labels_idx, zero_division=0)),
    }
    cm = confusion_matrix(y_te, y_pred, labels=labels_idx)
    report = classification_report(y_te, y_pred, labels=labels_idx, target_names=le.classes_, zero_division=0)
    with open(outdir/"audio_metrics_holdout.json", "w") as f:
        json.dump({"metrics": metrics, "classes": le.classes_.tolist(), "confusion_matrix": cm.tolist(), "report": report}, f, indent=2)
    proba_cols = [f"proba_{c}" for c in le.classes_]
    pred_df = pd.DataFrame({"true_label": le.inverse_transform(y_te), "pred_label": le.inverse_transform(y_pred)})
    pred_df = pd.concat([pred_df, pd.DataFrame(y_proba, columns=proba_cols)], axis=1)
    pred_df.to_csv(outdir/"audio_test_predictions_holdout.csv", index=False)
    save_cm_png(cm, list(le.classes_), outdir/"audio_confusion_matrix_holdout.png", "Audio: Confusion Matrix — Holdout")
    dump(pipe, outdir/"audio_rf_model.joblib"); dump(le, outdir/"audio_label_encoder.joblib")

def loso(X, y_enc, groups, le, outdir, top_k=30, rf_rank_est=300, rf_final_est=200):
    out_feats = outdir / "audio_loso_fold_top_features"; out_feats.mkdir(parents=True, exist_ok=True)
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []; rows = []
    labels_idx = list(range(len(le.classes_)))
    for i,(tr,te) in enumerate(logo.split(X, y_enc, groups=groups), start=1):
        Xt, yt = X.iloc[tr], y_enc[tr]; Xv, yv = X.iloc[te], y_enc[te]; g = groups.iloc[te].iloc[0]
        rank = rank_features_rf(Xt, yt, n_estimators=rf_rank_est)
        K = min(top_k, rank.shape[0]); top = rank.head(K)["feature"].tolist()
        pd.DataFrame({"selected_feature": top}).to_csv(out_feats/f"fold_{i:02d}_participant_{g}.csv", index=False)
        pipe = build_pipe(top, n_estimators=rf_final_est); pipe.fit(Xt[top], yt)
        yhat = pipe.predict(Xv[top]); proba = pipe.predict_proba(Xv[top])
        y_true_all.append(yv); y_pred_all.append(yhat)
        proba_cols = [f"proba_{c}" for c in le.classes_]
        dfp = pd.DataFrame({"participant": groups.iloc[te].values, "true_label": le.inverse_transform(yv), "pred_label": le.inverse_transform(yhat)})
        dfp = pd.concat([dfp, pd.DataFrame(proba, columns=proba_cols).reset_index(drop=True)], axis=1)
        rows.append(dfp)
    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels_idx, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels_idx, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    report = classification_report(y_true, y_pred, labels=labels_idx, target_names=le.classes_, zero_division=0)
    with open(outdir/"audio_metrics_loso.json", "w") as f:
        json.dump({"metrics": metrics, "classes": le.classes_.tolist(), "confusion_matrix": cm.tolist(), "report": report}, f, indent=2)
    pd.concat(rows, axis=0).reset_index(drop=True).to_csv(outdir/"audio_loso_predictions.csv", index=False)

def fit_final_all(X, y_enc, le, outdir, top_k=30, rf_rank_est=300, rf_final_est=200):
    rank = rank_features_rf(X, y_enc, n_estimators=rf_rank_est)
    K = min(top_k, rank.shape[0]); top = rank.head(K)["feature"].tolist()
    rank.to_csv(outdir/"audio_feature_importances_all.csv", index=False)
    pd.DataFrame({"selected_feature": top}).to_csv(outdir/"audio_selected_features.csv", index=False)
    pipe = build_pipe(top, n_estimators=rf_final_est); pipe.fit(X[top], y_enc)
    dump(pipe, outdir/"audio_rf_model.joblib"); dump(le, outdir/"audio_label_encoder.joblib")

def run_all():
    outdir = Path(OUTDIR); outdir.mkdir(parents=True, exist_ok=True)
    X, y_raw, groups, feat_cols = load_audio(CSV_PATH)
    print(f"[INFO] Audio loaded: {len(X)} samples, {len(feat_cols)} numeric features.")
    le = LabelEncoder(); y_enc = le.fit_transform(y_raw)
    print(f"[INFO] Classes: {list(le.classes_)}")
    if DO_HOLDOUT:
        print("[INFO] Holdout 80/20..."); holdout(X, y_enc, le, outdir, top_k=TOP_K, rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)
    print("[INFO] LOSO evaluation..."); loso(X, y_enc, groups, le, outdir, top_k=TOP_K, rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)
    print("[INFO] Final fit on ALL data..."); fit_final_all(X, y_enc, le, outdir, top_k=TOP_K, rf_rank_est=RF_RANK_EST, rf_final_est=RF_FINAL_EST)
    print(f"[DONE] Saved to: {outdir.resolve()}")

if __name__ == "__main__":
    run_all()
