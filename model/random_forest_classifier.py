# rf_bio_optimized.py
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# ============== CONFIG ==============
CSV = "features_dataset.csv"     # <- le TUE feature NON normalizzate
N_SPLITS = 5
RANDOM_STATE = 42

# temporal context
USE_CONTEXT = True
LAGS = [1, 2]            # feature ritardate (solo passato)
ROLLS = [3, 5]           # rolling mean/std sul passato (size in finestre)

# imbalance
USE_OVERSAMPLE = True
# portiamo alcune classi rare a frazione della maggiore (usa i NOME classe qui)
TARGET_RATIOS = {
    "breath_retention": 0.8,
    "pranayama": 0.7,
}

# feature selection
USE_MI_SELECT = True
MI_KEEP_TOP = 60         # tieni le top feature per MI (a valle della pulizia)
USE_RF_FEATSEL = True
RF_FEATSEL_KEEP = 50     # (opzionale) ulteriore pruning con importanze RF

# tuning veloce RF
USE_TUNING = True
N_ITER_TUNE = 20

# colonne meta da escludere
EXCLUDE = {"label","participant","time_center","start","end"}
# ====================================


def select_numeric_features(df):
    return [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def add_temporal_context(df, feat_cols, lags=(1,2), rolls=(3,5)):
    """
    Versione veloce: crea tutte le colonne lag/rolling in un colpo solo
    e le concatena per evitare fragmentation warnings.
    """
    df = df.sort_values(["participant", "time_center"]).copy()

    new_cols = {}
    g = df.groupby("participant", sort=False)

    # LAG (solo passato)
    for L in lags:
        shifted = g[feat_cols].shift(L)
        shifted.columns = [f"{c}_lag{L}" for c in shifted.columns]
        for c in shifted.columns:
            new_cols[c] = shifted[c]

    # ROLLING mean/std sul passato (shift(1) per evitare leakage)
    for W in rolls:
        base = g[feat_cols].shift(1)
        rmean = base.rolling(W, min_periods=1).mean()
        rstd  = base.rolling(W, min_periods=1).std()
        rmean.columns = [f"{c}_rmean{W}" for c in rmean.columns]
        rstd.columns  = [f"{c}_rstd{W}"  for c in rstd.columns]
        for c in rmean.columns:
            new_cols[c] = rmean[c]
        for c in rstd.columns:
            new_cols[c] = rstd[c]

    ctx_df = pd.DataFrame(new_cols, index=df.index)
    # riempi buchi dei nuovi (inizio sequenze) con bfill/ffill
    ctx_df = ctx_df.bfill().ffill()

    # concat una volta sola
    out = pd.concat([df, ctx_df], axis=1)
    return out


def normalize_per_participant(df, feat_cols):
    """MinMax per participant: fit/transform separato su ciascun participant (niente leakage tra persone)."""
    df = df.copy()
    df[feat_cols] = df[feat_cols].astype("float32")   # evita warning dtype
    for pid, g in df.groupby("participant", sort=False):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(g[feat_cols].values)
        df.loc[g.index, feat_cols] = pd.DataFrame(scaled, index=g.index, columns=feat_cols)
    return df


def drop_low_variance_and_collinear(X_df, var_thresh=1e-8, corr_thresh=0.97):
    """Togli feature quasi-costanti e molto collineari (teniamo la prima)."""
    X = X_df.copy()
    keep = X.var() > var_thresh
    X = X.loc[:, keep]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    X = X.drop(columns=to_drop)

    return X


def mi_select(X_df, y, k=60):
    """Seleziona le top-k feature per mutual information."""
    X = X_df.copy()
    y = np.asarray(y).astype("int32")  # <<< target DISCRETO
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE, discrete_features=False)
    mi = np.nan_to_num(mi, nan=0.0)
    order = np.argsort(mi)[::-1]
    keep_idx = order[:min(k, X.shape[1])]
    return X.iloc[:, keep_idx], list(X.columns[keep_idx]), mi


def build_sampling_strategy(y_numeric, classes_names, target_ratios):
    """
    Crea sampling_strategy SOLO per le classi presenti nel TRAIN del fold.
    target_ratios usa i nomi delle classi.
    """
    y_ser = pd.Series(y_numeric)
    counts = y_ser.value_counts()
    if counts.empty:
        return None

    present_idx = set(counts.index.tolist())
    name2idx = {name: idx for idx, name in enumerate(classes_names)}
    maj = counts.max()

    strategy = counts.to_dict()  # base: mantieni le cardinalità attuali

    for cls_name, ratio in target_ratios.items():
        idx = name2idx.get(cls_name, None)
        if idx is None or idx not in present_idx:
            continue  # non forzare classi non presenti nel TRAIN
        cur = strategy.get(idx, 0)
        strategy[idx] = max(cur, int(round(ratio * maj)))

    return strategy


def oversample_train(X, y, sampling_strategy):
    """Applica ROS se possibile; altrimenti restituisce X,y senza crash."""
    if sampling_strategy is None:
        return (X.values if isinstance(X, pd.DataFrame) else X), y
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
        Xr, yr = ros.fit_resample(X, y)
        return Xr, yr
    except Exception as e:
        print(f"[warn] oversampling saltato (motivo: {e})")
        return (X.values if isinstance(X, pd.DataFrame) else X), y

def summarize_fold(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1w, f1m

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import numpy as np

def make_valid_group_splits(X, y, groups, n_splits=5, random_state=RANDOM_STATE):
    """
    Genera n_splits che rispettano:
      - split per gruppi (participant interi)
      - stratificazione per classe
      - VINCOLO: tutte le classi presenti nel TRAIN di ogni fold
    Se qualche split di SGKF non rispetta il vincolo (raro), si rigenera con seed diverso.
    """
    # safety: il numero di gruppi deve essere >= n_splits
    uniq_groups = np.unique(groups)
    if len(uniq_groups) < n_splits:
        n_splits = len(uniq_groups)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    tried = 0
    max_tries = 50  # basta e avanza

    rng = np.random.RandomState(random_state)

    while len(splits) < n_splits and tried < max_tries:
        tried += 1
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 10**9))
        for tr, te in sgkf.split(X, y, groups=groups):
            # vincolo forte: tutte le classi nel TRAIN
            if set(np.unique(y)) <= set(np.unique(y[tr])):
                splits.append((tr, te))
                if len(splits) == n_splits:
                    break

    if len(splits) < n_splits:
        print(f"[warn] trovati solo {len(splits)}/{n_splits} split che rispettano il vincolo 'tutte le classi nel TRAIN'.")
    return splits

def main():
    df = pd.read_csv(CSV)

    # ordina temporalmente
    if "time_center" in df.columns:
        df = df.sort_values(["participant","time_center"]).reset_index(drop=True)

    # encode label (y numerico, DISCRETO)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"]).astype("int32")
    y_enc_all = df["label_enc"].values
    class_names = list(le.classes_)

    # base feature set
    feat_cols = select_numeric_features(df)
    # cast a float PRIMA di creare context
    df[feat_cols] = df[feat_cols].astype("float32")

    # === (1) temporal context ===
    if USE_CONTEXT:
        df = add_temporal_context(df, feat_cols, lags=LAGS, rolls=ROLLS)
        feat_cols = select_numeric_features(df)  # aggiorna lista

    # === (2) normalizzazione per participant ===
    df = normalize_per_participant(df, feat_cols)

    # === X,y,groups ===
    X_all = df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y_all = df["label_enc"].astype("int32").values
    groups = df["participant"].values

    # === (3) pulizia feature globale (su tutto X per semplicità; alternativa: per-fold) ===
    X_all = drop_low_variance_and_collinear(X_all, var_thresh=1e-8, corr_thresh=0.97)
    feat_cols = list(X_all.columns)

    print(f"[INFO] features iniziali: {len(select_numeric_features(df))} | dopo pulizia: {X_all.shape[1]}")

    splits = make_valid_group_splits(X_all.values, y_all, groups, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    accs, f1ws, f1ms = [], [], []

    for fold, (tr, te) in enumerate(splits, start=1):
    
        X_tr, X_te = X_all.iloc[tr].copy(), X_all.iloc[te].copy()
        y_tr, y_te = y_all[tr], y_all[te]

        # === (4) feature selection (MI) sul TRAIN del fold ===
        if USE_MI_SELECT:
            X_tr, sel_cols, mi_scores = mi_select(X_tr, y_tr, k=MI_KEEP_TOP)
            X_te = X_te[sel_cols]
        else:
            sel_cols = X_tr.columns.tolist()

        # === (5) oversampling mirato SOLO sul train ===
        if USE_OVERSAMPLE:
            sampling_strategy = build_sampling_strategy(y_tr, class_names, TARGET_RATIOS)
            X_tr_os, y_tr_os = oversample_train(X_tr, y_tr, sampling_strategy)
        else:
            X_tr_os, y_tr_os = X_tr.values, y_tr

        # === (6) tuning veloce RF su train-fold ===
        if USE_TUNING:
            param_dist = {
                "n_estimators": [300, 400, 500, 600],
                "max_depth": [None, 16, 24, 32],
                "max_features": ["sqrt", 0.4, 0.6],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "bootstrap": [True],
                "class_weight": ["balanced"]
            }
            base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            rs = RandomizedSearchCV(
                base, param_dist, n_iter=N_ITER_TUNE, cv=3, n_jobs=-1, random_state=RANDOM_STATE,
                scoring="f1_weighted", verbose=0
            )
            rs.fit(X_tr_os, y_tr_os)
            clf = rs.best_estimator_
        else:
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=None, max_features="sqrt",
                min_samples_leaf=1, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )
            clf.fit(X_tr_os, y_tr_os)

        # (opz.) ulteriore pruning per importanza RF
        if USE_RF_FEATSEL:
            importances = clf.feature_importances_
            order = np.argsort(importances)[::-1][:min(RF_FEATSEL_KEEP, len(importances))]
            keep2 = [sel_cols[i] for i in order]

            clf2 = RandomForestClassifier(**clf.get_params())
            clf2.set_params(random_state=RANDOM_STATE+fold, n_jobs=-1)
            clf2.fit(X_tr_os[:, order], y_tr_os)
            y_pred = clf2.predict(X_te[keep2].values)
        else:
            y_pred = clf.predict(X_te.values)

        acc, f1w, f1m = summarize_fold(y_te, y_pred)
        accs.append(acc); f1ws.append(f1w); f1ms.append(f1m)

        print(f"\nFold {fold}  ACC={acc:.3f}  F1w={f1w:.3f}  F1m={f1m:.3f}")

    print("\n=== GroupKFold CV (per participant) — RF BIO OPT ===")
    print(f"Accuracy   : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1-weight  : {np.mean(f1ws):.3f} ± {np.std(f1ws):.3f}")
    print(f"F1-macro   : {np.mean(f1ms):.3f} ± {np.std(f1ms):.3f}")
    print("\nTip: alza/abbassa TARGET_RATIOS per spingere classi rare (es. breath_retention).")

if __name__ == "__main__":
    main()
