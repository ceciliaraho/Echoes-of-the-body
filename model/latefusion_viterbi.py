# late_fusion_rf_logo_normalized.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# Config
# =============================
BIO_CSV   = "features_dataset.csv"
AUDIO_CSV = "audioFeatures_stats.csv"

# colonne da escludere (come da tue istruzioni)
DROP_BIO   = ['participant','label','time_center','start','end']
DROP_AUDIO = ['participant','label','label_id','abs_start_s','abs_end_s',
              'abs_time_center','rel_time_center','rel_end_s','rel_start_s']

# top-k feature per modality (mutual information)
K_BIO   = 30
K_AUDIO = 30

# attiva Viterbi smoothing dopo la fusione (True/False)
USE_VITERBI = True

# =============================
# Utils
# =============================
def make_rf(n_estimators=300, n_jobs=-1):
    """Random Forest con imputer median (lavora su feature già normalizzate)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=n_jobs
        ))
    ])

def per_participant_scale(df, feat_cols, participant_col='participant'):
    """z-score per partecipante: (x - mean_part) / std_part."""
    z = df.copy()
    for f in feat_cols:
        g = z.groupby(participant_col)[f]
        m = g.transform('mean')
        s = g.transform('std').replace(0.0, np.nan)
        z[f] = (z[f] - m) / s
    z[feat_cols] = z[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z

def mi_topk(X, y, k=40, random_state=42):
    """Mutual Information top-k sul TRAIN."""
    Xs = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(Xs, y, discrete_features=False, random_state=random_state)
    order = np.argsort(mi)[::-1]
    k_eff = min(k, X.shape[1])
    keep_idx = order[:k_eff]
    return keep_idx, mi

def align_proba(P, classes_seen, label_order):
    """
    Allinea una matrice di prob. (n, C_seen) alle classi in label_order (C_all).
    Gestisce classi non viste nel fold: colonne a 0 e riga uniformata se tutta zero.
    """
    idx = {c: i for i, c in enumerate(classes_seen)}
    n = P.shape[0]; C = len(label_order)
    out = np.zeros((n, C), dtype=float)
    for j, lab in enumerate(label_order):
        i = idx.get(lab, None)
        if i is not None:
            out[:, j] = P[:, i]
    row_sums = out.sum(axis=1, keepdims=True)   # (n,1)
    nonzero = (row_sums.squeeze() != 0)         # (n,)
    out[nonzero]  = out[nonzero] / row_sums[nonzero]
    out[~nonzero] = 1.0 / C
    return out

def learn_transition_matrix(train_keys_df, labels_sorted, add_k=1e-6):
    """
    Stima P(y_t | y_{t-1}) dalle sequenze di training (tutti i partecipanti TR).
    train_keys_df: colonne ['participant','time_center','label'] SOLO TRAIN.
    """
    idx = {c:i for i,c in enumerate(labels_sorted)}
    K = len(labels_sorted)
    A = np.full((K, K), add_k, dtype=float)
    for pid, g in train_keys_df.groupby('participant'):
        g = g.sort_values('time_center')
        prev = None
        for lab in g['label'].astype(str):
            if prev is not None and (prev in idx) and (lab in idx):
                A[idx[prev], idx[lab]] += 1.0
            prev = lab
    A = A / A.sum(axis=1, keepdims=True)
    return A

def viterbi_decode(emissions, trans, pi=None):
    """
    emissions: (T, C) con P(y_t = c | x_t) già allineate a label_order
    trans:     (C, C) con P(c_t | c_{t-1}); pi opzionale (uniforme se None)
    ritorna:   path (T,) indici classe max-prob
    """
    T, C = emissions.shape
    if pi is None:
        pi = np.full(C, 1.0/C)
    logE = np.log(emissions + 1e-12)
    logA = np.log(trans + 1e-12)
    logPi = np.log(pi + 1e-12)

    dp = np.full((T, C), -1e18)
    bp = np.zeros((T, C), dtype=int)
    dp[0] = logPi + logE[0]
    for t in range(1, T):
        prev_plus_trans = dp[t-1][:, None] + logA
        bp[t] = np.argmax(prev_plus_trans, axis=0)
        dp[t] = np.max(prev_plus_trans, axis=0) + logE[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1, path[t+1]]
    return path

# =============================
# 1) Carica e costruisci blocchi feature
# =============================
bio   = pd.read_csv(BIO_CSV)
audio = pd.read_csv(AUDIO_CSV)

# blocchi: salviamo i nomi PRIMA
bio_feat = bio.drop(columns=DROP_BIO, errors='ignore').copy()
bio_feat['participant'] = bio['participant']
bio_feat['time_center'] = bio['time_center']
bio_feat['label']       = bio['label']
bio_feature_names = bio_feat.columns.difference(['participant','time_center','label']).tolist()

audio_feat = audio.drop(columns=DROP_AUDIO, errors='ignore').copy()
audio_feat['participant'] = audio['participant']
audio_feat['time_center'] = audio['rel_time_center']  # sincronizzato
audio_feat['label']       = audio['label']
audio_feature_names = audio_feat.columns.difference(['participant','time_center','label']).tolist()

# forza numerico se arrivano come stringhe
for c in bio_feature_names:
    if bio_feat[c].dtype == 'object':
        bio_feat[c] = pd.to_numeric(bio_feat[c], errors='coerce')
for c in audio_feature_names:
    if audio_feat[c].dtype == 'object':
        audio_feat[c] = pd.to_numeric(audio_feat[c], errors='coerce')

# =============================
# 2) Normalizzazione z-score per partecipante (PRIMA del training)
# =============================
bio_feat_z   = per_participant_scale(bio_feat,   bio_feature_names,   participant_col='participant')
audio_feat_z = per_participant_scale(audio_feat, audio_feature_names, participant_col='participant')

# =============================
# 3) Allineamento chiavi
# =============================
keys = ['participant','time_center','label']
keys_aligned = (pd.merge(bio_feat_z[keys], audio_feat_z[keys], on=keys, how='inner')
                  .drop_duplicates().reset_index(drop=True))

# viste parallele normalizzate
bio_aligned   = keys_aligned.merge(bio_feat_z[keys + bio_feature_names],    on=keys, how='left')
audio_aligned = keys_aligned.merge(audio_feat_z[keys + audio_feature_names], on=keys, how='left')

# colonne numeriche finali
bio_cols   = [c for c in bio_feature_names   if np.issubdtype(bio_aligned[c].dtype,  np.number)]
audio_cols = [c for c in audio_feature_names if np.issubdtype(audio_aligned[c].dtype, np.number)]

print(f"#samples={len(keys_aligned)} | #bio_feat(z)={len(bio_cols)} | #audio_feat(z)={len(audio_cols)}")

participants = keys_aligned['participant'].astype(str).unique().tolist()
label_order  = sorted(keys_aligned['label'].astype(str).unique().tolist())

# =============================
# 4) LOGO + Late Fusion (su feature normalizzate) + (opzionale) Viterbi
# =============================
results = []
all_true, all_fused = [], []
all_viterbi = []

for p in participants:
    te_mask = (keys_aligned['participant'].astype(str) == p)
    tr_mask = ~te_mask

    # split (già normalizzato per partecipante)
    Xb_tr_raw = bio_aligned.loc[tr_mask, bio_cols].copy()
    Xa_tr_raw = audio_aligned.loc[tr_mask, audio_cols].copy()
    y_tr      = keys_aligned.loc[tr_mask, 'label'].astype(str).values

    Xb_te_raw = bio_aligned.loc[te_mask, bio_cols].copy()
    Xa_te_raw = audio_aligned.loc[te_mask, audio_cols].copy()
    y_te      = keys_aligned.loc[te_mask, 'label'].astype(str).values

    # MI per-modality sul TRAIN normalizzato
    kb = min(K_BIO,  Xb_tr_raw.shape[1]) if Xb_tr_raw.shape[1] > 0 else 0
    ka = min(K_AUDIO, Xa_tr_raw.shape[1]) if Xa_tr_raw.shape[1] > 0 else 0
    sel_bio   = Xb_tr_raw.columns[mi_topk(Xb_tr_raw.values, y_tr, k=kb)[0]].tolist()   if kb>0 else []
    sel_audio = Xa_tr_raw.columns[mi_topk(Xa_tr_raw.values, y_tr, k=ka)[0]].tolist()   if ka>0 else []

    # RF separati (su feature NORMALIZZATE e selezionate)
    rf_bio   = make_rf().fit(Xb_tr_raw[sel_bio]   if sel_bio   else Xb_tr_raw.iloc[:, :0], y_tr)
    rf_audio = make_rf().fit(Xa_tr_raw[sel_audio] if sel_audio else Xa_tr_raw.iloc[:, :0], y_tr)

    # proba in test
    Pb_raw = rf_bio.predict_proba(Xb_te_raw[sel_bio]   if sel_bio   else Xb_te_raw.iloc[:, :0])
    Pa_raw = rf_audio.predict_proba(Xa_te_raw[sel_audio] if sel_audio else Xa_te_raw.iloc[:, :0])

    Pb = align_proba(Pb_raw, rf_bio.named_steps['rf'].classes_,   label_order)
    Pa = align_proba(Pa_raw, rf_audio.named_steps['rf'].classes_, label_order)

    # Late fusion (media)
    Pfused = 0.5*Pb + 0.5*Pa
    y_pred_fused = np.array(label_order)[np.argmax(Pfused, axis=1)]

    # metriche per fold
    acc_bio   = (np.array(label_order)[np.argmax(Pb, axis=1)] == y_te).mean()
    acc_audio = (np.array(label_order)[np.argmax(Pa, axis=1)] == y_te).mean()
    acc_fused = (y_pred_fused == y_te).mean()

    row = {
        "participant": p, "n_test": int(len(y_te)),
        "acc_bio": float(acc_bio), "acc_audio": float(acc_audio), "acc_fusion": float(acc_fused),
        "k_bio": len(sel_bio), "k_audio": len(sel_audio)
    }

    # --- opzionale: Viterbi ---
    if USE_VITERBI:
        A = learn_transition_matrix(
            keys_aligned.loc[tr_mask, ['participant','time_center','label']].copy(),
            label_order, add_k=1e-6
        )
        # ordina la sequenza di test per tempo
        te_times = keys_aligned.loc[te_mask, 'time_center'].values
        order = np.argsort(te_times); inv = np.argsort(order)
        Pf_seq = Pfused[order]; y_te_seq = y_te[order]
        path = viterbi_decode(Pf_seq, A, pi=np.full(len(label_order), 1/len(label_order)))
        y_pred_vit = np.array(label_order)[path][inv]
        row["acc_fusion_viterbi"] = (y_pred_vit == y_te).mean()
        all_viterbi.extend(y_pred_vit.tolist())

    results.append(row)
    all_true.extend(y_te.tolist())
    all_fused.extend(y_pred_fused.tolist())

# =============================
# 5) Report finale
# =============================
res = pd.DataFrame(results).sort_values("participant")
print("\nAccuratezze per partecipante:")
print(res)

macro_cols = ['acc_bio','acc_audio','acc_fusion'] + (['acc_fusion_viterbi'] if USE_VITERBI else [])
print("\nMacro medie:")
print(res[macro_cols].mean())

print("\nClassification report — FUSION:")
print(classification_report(all_true, all_fused))

cm_fused = confusion_matrix(all_true, all_fused, labels=label_order)
print("\nConfusion matrix — FUSION:")
print(pd.DataFrame(cm_fused, index=[f"true_{l}" for l in label_order],
                             columns=[f"pred_{l}" for l in label_order]))

if USE_VITERBI:
    print("\nClassification report — FUSION + VITERBI:")
    print(classification_report(all_true, all_viterbi))
    cm_vit = confusion_matrix(all_true, all_viterbi, labels=label_order)
    print("\nConfusion matrix — FUSION + VITERBI:")
    print(pd.DataFrame(cm_vit, index=[f"true_{l}" for l in label_order],
                               columns=[f"pred_{l}" for l in label_order]))
