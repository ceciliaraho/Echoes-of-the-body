"""
inference_hub.py — late fusion of AUDIO/BIO predictions with EMA, temperature scaling, and gated online Viterbi.

Public API:
- init_models(...)
- handle_audio_features(row_dict)
- handle_bio_features(row_dict)
- close()

This module:
- loads/unwraps two models (audio, bio)
- ensures feature order & imputes missing values
- writes per-stream predictions to CSV (optional)
- time-aligns rows from the two streams and performs fused prediction (linear fusion → temperature → EMA → optional gated Viterbi)
- writes a merged CSV with audio/bio/fused/viterbi labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import threading
import csv
import math
import subprocess
import sys
import time
import numpy as np

# Optional: joblib for loading saved estimators
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None


# Parameters
LOCK = threading.Lock()

# Models and feature ordering
audio_model = None
bio_model   = None

audio_feat_order: List[str] = []   # column order for X_audio
bio_feat_order:   List[str] = []   # column order for X_bio
impute_value: float = 0.0

# Class labels (per stream)
audio_label_names: Optional[List[str]] = None
bio_label_names:   Optional[List[str]] = None

# CSV writers (per stream + merged)
audio_pred_fp = None
audio_pred_writer = None
audio_pred_headers: List[str] = []

bio_pred_fp = None
bio_pred_writer = None
bio_pred_headers: List[str] = []

merge_fp = None
merge_writer = None
merge_headers: List[str] = []

# Time alignment buffers (by integer-second bin)
BUF_AUDIO: Dict[int, dict] = {}
BUF_BIO:   Dict[int, dict] = {}
MERGE_TOL_S: float = 1.0   # max |delta_t| between A/B windows to fuse

# Live prints
PRINT_PRED: bool = False

# ===== Fused space & Viterbi/gating/smoothing =====
fused_label_names: Optional[List[str]] = None
fused_index: Dict[str, int] = {}
FUSED_K: Optional[int] = None

# Fusion weights
fuse_audio_w: float = 1.0
fuse_bio_w:   float = 1.0

# Transition matrix (log), online-Viterbi DP, priors
Alog_fused: Optional[np.ndarray] = None
delta_fused: Optional[np.ndarray] = None
fused_prior: Optional[np.ndarray] = None
fused_current: Optional[int] = None
fused_dwell: int = 0

# Smoothing + gating
FUSED_MIN_DWELL: int = 2           # min windows to dwell before allowing step
FUSED_NEXT_THRESH: float = 0.55    # p(next) threshold to unlock forward step
FUSED_EMA_ALPHA: float = 0.6       # EMA smoothing over fused probabilities
fused_ema: Optional[np.ndarray] = None
FUSED_TEMPERATURE: float = 0.9     # <1 sharpen, >1 soften

# Toggle online Viterbi on fused probabilities
USE_VITERBI_FUSED: bool = True

# Anti-spam for external notifier (optional)
LAST_SENT = None
LAST_TS   = 0.0
MIN_DWELL_S = 6.0  # don't notify more often than this (seconds)


# Utilities
def send_stage_via_cli(value):
    """Call an external script with the fused label/index (debounced) to send to Ableton session."""
    global LAST_SENT, LAST_TS
    now = time.time()
    if value == LAST_SENT and (now - LAST_TS) < MIN_DWELL_S:
        return
    try:
        subprocess.run([sys.executable, "stage.py", "--stage", str(value)], check=False)
        LAST_SENT, LAST_TS = value, now
    except Exception as e:
        print("[hub] error invoking stage.py:", e)


def open_writer(path: Optional[Union[Path, str]], headers: List[str]):
    if path is None:
        return None, None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fp = p.open("w", newline="", encoding="utf-8")
    wr = csv.DictWriter(fp, fieldnames=headers)
    wr.writeheader()
    fp.flush()
    return fp, wr


def close_writer(fp, wr):
    try:
        if fp:
            fp.flush()
            fp.close()
    except Exception:
        pass
    return None, None


def build_feature_vector(row: Dict[str, float], order: List[str], impute: float) -> np.ndarray:
    vals: List[float] = []
    for k in order:
        v = row.get(k, np.nan)
        if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
            v = impute
        vals.append(float(v))
    return np.asarray(vals, dtype=float).reshape(1, -1)


def predict_with_scores(est, X: np.ndarray) -> Tuple[Union[int, str], Optional[float], Optional[np.ndarray]]:
    """Return (pred_label_or_index, score, probs) with best-effort 'score'."""
    pred = est.predict(X)
    pred = pred[0] if hasattr(pred, "__len__") else pred
    score = None
    probs = None
    if hasattr(est, "predict_proba"):
        probs = est.predict_proba(X)
        probs = probs[0] if hasattr(probs, "__len__") else None
        if probs is not None:
            score = float(np.max(probs))
    if score is None and hasattr(est, "decision_function"):
        df = est.decision_function(X)
        if df is not None:
            score = float(np.max(df[0])) if np.ndim(df) == 2 else float(df[0])
    return pred, score, probs


def time_bin(t_center_s: float) -> int:
    """Map a center time to an integer-second bin for rough alignment."""
    return int(round(float(t_center_s)))


def unwrap_model_bundle(m):
    """
    Extract (estimator, class_names, selected_features) from your saved bundles.
    Accepts dict-like bundles with keys:
        'model'/'estimator'/'rf'/'clf', 'classes', 'selected_features'
    Or raw sklearn-like estimators (uses .classes_, .feature_names_in_ if present).
    """
    if isinstance(m, dict):
        est   = m.get("model") or m.get("estimator") or m.get("rf") or m.get("clf")
        names = m.get("classes", None)
        sel   = m.get("selected_features", None)
        return est, names, sel
    names = getattr(m, "classes_", None)
    sel   = getattr(m, "feature_names_in_", None)
    if sel is not None:
        sel = list(sel)
    return m, names, sel


def ensure_label_names(names, fallback_dim: Optional[int]) -> Optional[List[str]]:
    if names is None:
        return [str(i) for i in range(int(fallback_dim))] if fallback_dim else None
    return [str(x) for x in list(names)]


def label_to_name(idx, names: Optional[List[str]]):
    if isinstance(idx, str):
        return idx
    try:
        i = int(idx)
        if names and 0 <= i < len(names):
            return names[i]
        return str(i)
    except Exception:
        return str(idx)


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0.0:
        j = int(np.argmax(p))
        q = np.zeros_like(p)
        q[j] = 1.0
        return q
    return p / s


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        T = 1.0
    if abs(T - 1.0) < 1e-6:
        return p
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / T)
    return normalize(q)


def build_transition_matrix(names: List[str],
                            transition_order: List[str],
                            stay: float,
                            step: float,
                            extra_edges: Optional[Dict[str, Dict[str, float]]]) -> np.ndarray:
    """Row-stochastic transition matrix honoring a nominal order plus optional extra edges."""
    K = len(names)
    idx = {n: i for i, n in enumerate(names)}
    A = np.full((K, K), 1e-9, dtype=float)

    # stay
    for n in names:
        A[idx[n], idx[n]] = float(stay)

    # forward steps according to 'transition_order'
    for i in range(len(transition_order) - 1):
        a, b = transition_order[i], transition_order[i + 1]
        if a in idx and b in idx:
            A[idx[a], idx[b]] = max(A[idx[a], idx[b]], float(step))

    # extra edges
    if extra_edges:
        for src, dsts in extra_edges.items():
            if src not in idx:
                continue
            i = idx[src]
            for dst, w in dsts.items():
                if dst in idx:
                    A[i, idx[dst]] = max(A[i, idx[dst]], float(w))

    # row-normalize
    for i in range(K):
        A[i] = normalize(A[i])
    return A


def renorm_log_row(Alog_row: np.ndarray) -> np.ndarray:
    """Renormalize a single log-prob row after zeroing some entries."""
    row = np.exp(Alog_row - np.max(Alog_row))
    row = normalize(row)
    return np.log(row + 1e-12)


def viterbi_update_online(p_obs: np.ndarray) -> Tuple[int, float]:
    """
    Online Viterbi on fused probabilities with gating:
    - enforce minimum dwell in current state
    - block transition to immediate next state unless p(next) is high enough
    """
    global delta_fused, Alog_fused, fused_current, fused_dwell

    Alog = Alog_fused.copy()

    # Gate the forward step to the immediate next label in the sequence
    if fused_current is not None and FUSED_K and fused_current < (FUSED_K - 1):
        next_idx = fused_current + 1
        allow_step = (fused_dwell >= FUSED_MIN_DWELL) and (p_obs[next_idx] >= FUSED_NEXT_THRESH)
        if not allow_step:
            Alog[fused_current, next_idx] = math.log(1e-9)
            Alog[fused_current] = renorm_log_row(Alog[fused_current])

    log_obs = np.log(np.clip(p_obs, 1e-12, 1.0))

    if delta_fused is None:
        delta_fused = np.log(np.clip(fused_prior, 1e-12, 1.0)) + log_obs
    else:
        delta_fused = np.max(delta_fused[:, None] + Alog, axis=0) + log_obs

    j = int(np.argmax(delta_fused))
    fused_dwell = fused_dwell + 1 if (fused_current == j) else 1
    fused_current = j
    return j, float(delta_fused[j])


# API: init / close
def init_models(model_audio: Union[Path, str, object],
                model_bio:   Union[Path, str, object],
                audio_feature_order: Optional[List[str]] = None,
                bio_feature_order:   Optional[List[str]] = None,
                *,
                impute_value_param: float = 0.0,
                audio_pred_csv: Optional[Union[Path, str]] = None,
                bio_pred_csv:   Optional[Union[Path, str]] = None,
                merge_csv:      Optional[Union[Path, str]] = None,
                merge_tolerance_s: float = 1.0,
                print_predictions: bool = False,
                # fusion
                fuse_audio_weight: float = 0.7,
                fuse_bio_weight:   float = 0.8,
                # fused Viterbi
                use_viterbi_fused: bool = True,
                fused_order: Optional[List[str]] = None,
                fused_stay: float = 0.8,
                fused_step: float = 0.75,
                fused_extra_edges: Optional[Dict[str, Dict[str, float]]] = None,
                fused_start_label: Optional[str] = None,
                fused_start_strength: float = 1.0,
                # gating / smoothing
                fused_min_dwell: int = 5,
                fused_next_threshold: float = 0.7,
                fused_ema_alpha: float = 0.6,
                fused_temperature: float = 0.9) -> None:
    """
    Load/attach models, set feature order & IO, and configure fused Viterbi with gating/smoothing.
    """
    global audio_model, bio_model, audio_feat_order, bio_feat_order, impute_value, MERGE_TOL_S
    global audio_label_names, bio_label_names, PRINT_PRED
    global audio_pred_fp, audio_pred_writer, audio_pred_headers
    global bio_pred_fp, bio_pred_writer, bio_pred_headers
    global merge_fp, merge_writer, merge_headers
    global fused_label_names, fused_index, FUSED_K
    global Alog_fused, delta_fused, fused_prior, fused_current, fused_dwell
    global FUSED_MIN_DWELL, FUSED_NEXT_THRESH, FUSED_EMA_ALPHA, fused_ema, FUSED_TEMPERATURE
    global fuse_audio_w, fuse_bio_w, USE_VITERBI_FUSED

    with LOCK:
        def _load(x):
            if isinstance(x, (str, Path)):
                if joblib_load is None:
                    raise RuntimeError("joblib not available: pass an estimator object instead of a path.")
                return joblib_load(x)
            return x

        # Load / unwrap models
        a_raw = _load(model_audio)
        b_raw = _load(model_bio)
        a_est, a_names, a_sel = unwrap_model_bundle(a_raw)
        b_est, b_names, b_sel = unwrap_model_bundle(b_raw)
        if a_est is None or b_est is None:
            raise ValueError("Model bundle missing estimator (expected 'model'/'estimator'/'rf'/'clf').")

        audio_model = a_est
        bio_model   = b_est

        # Feature order (param → selected_features → feature_names_in_)
        audio_feat_order = list(audio_feature_order) if audio_feature_order else (list(a_sel) if a_sel else [])
        bio_feat_order   = list(bio_feature_order)   if bio_feature_order   else (list(b_sel) if b_sel else [])
        if not audio_feat_order:
            raise ValueError("Audio feature order not determined (pass audio_feature_order or use 'selected_features').")
        if not bio_feat_order:
            raise ValueError("Bio feature order not determined (pass bio_feature_order or use 'selected_features').")

        impute_value = float(impute_value_param)
        MERGE_TOL_S  = float(merge_tolerance_s)
        PRINT_PRED   = bool(print_predictions)

        # Class names per stream
        if a_names is None: a_names = getattr(audio_model, "classes_", None)
        if b_names is None: b_names = getattr(bio_model, "classes_", None)
        audio_label_names = ensure_label_names(a_names, None)
        bio_label_names   = ensure_label_names(b_names, None)

        # Per-stream writers (include prob_* if available)
        aud_prob_cols = [f"prob_{c}" for c in (audio_label_names or [])]
        bio_prob_cols = [f"prob_{c}" for c in (bio_label_names   or [])]
        audio_pred_headers = ["t_center_s", "prediction", "score"] + aud_prob_cols
        bio_pred_headers   = ["t_center_s", "prediction", "score"] + bio_prob_cols
        audio_pred_fp, audio_pred_writer = open_writer(audio_pred_csv, audio_pred_headers)
        bio_pred_fp,   bio_pred_writer   = open_writer(bio_pred_csv,   bio_pred_headers)

        # Merge writer (also includes fused + viterbi)
        merge_headers = ["t_center_s", "audio_pred", "audio_score", "bio_pred", "bio_score", "fused_pred", "fused_viterbi"]
        merge_fp, merge_writer = open_writer(merge_csv, merge_headers)

        # Fusion weights
        fuse_audio_w = float(fuse_audio_weight)
        fuse_bio_w   = float(fuse_bio_weight)

        # Fused label space (order matters for the nominal forward chain)
        if fused_order is None:
            base = audio_label_names or bio_label_names
            if base is None:
                raise ValueError("Cannot determine fused classes (no fused_order and models expose no classes_).")
            fused_label_names = list(base)
        else:
            fused_label_names = list(fused_order)

        fused_index = {n: i for i, n in enumerate(fused_label_names)}
        FUSED_K = len(fused_label_names)

        # Transition matrix for fused labels
        A = build_transition_matrix(fused_label_names,
                                    transition_order=fused_label_names,  # linear chain over the fused labels
                                    stay=fused_stay,
                                    step=fused_step,
                                    extra_edges=fused_extra_edges)
        Alog_fused = np.log(np.clip(A, 1e-12, 1.0))

        # Prior: optionally bias the starting label
        fused_prior = np.full(FUSED_K, 1.0 / FUSED_K, dtype=float)
        if fused_start_label and fused_start_label in fused_index:
            s = float(max(0.0, min(1.0, fused_start_strength)))
            fused_prior[:] = (1.0 - s) / max(1, FUSED_K - 1)
            fused_prior[fused_index[fused_start_label]] = s
        fused_prior = normalize(fused_prior)

        delta_fused = None
        fused_current = None
        fused_dwell = 0

        # Gating/smoothing
        FUSED_MIN_DWELL   = int(max(0, fused_min_dwell))
        FUSED_NEXT_THRESH = float(np.clip(fused_next_threshold, 0.0, 1.0))
        FUSED_EMA_ALPHA   = float(np.clip(fused_ema_alpha, 0.0, 0.99))
        fused_ema = None
        FUSED_TEMPERATURE = float(max(1e-3, fused_temperature))
        USE_VITERBI_FUSED = bool(use_viterbi_fused)

        print(f"[hub] audio feature order ({len(audio_feat_order)}): {audio_feat_order}")
        print(f"[hub] bio   feature order ({len(bio_feat_order)}):   {bio_feat_order}")
        print(f"[hub] fused labels: {fused_label_names}")
        print(f"[hub] merge tol: {MERGE_TOL_S}s | weights A/B: {fuse_audio_w}/{fuse_bio_w}")
        print(f"[hub] viterbi fused: min_dwell={FUSED_MIN_DWELL}, next_thr={FUSED_NEXT_THRESH}, ema={FUSED_EMA_ALPHA}, T={FUSED_TEMPERATURE}")


def close() -> None:
    """Close all writers and clear alignment buffers."""
    global audio_pred_fp, audio_pred_writer, bio_pred_fp, bio_pred_writer, merge_fp, merge_writer
    with LOCK:
        audio_pred_fp, audio_pred_writer = close_writer(audio_pred_fp, audio_pred_writer)
        bio_pred_fp,   bio_pred_writer  = close_writer(bio_pred_fp,   bio_pred_writer)
        merge_fp,      merge_writer     = close_writer(merge_fp,      merge_writer)
        BUF_AUDIO.clear()
        BUF_BIO.clear()



def handle_audio_features(row: Dict[str, float]) -> None:
    """Consume one AUDIO feature row, predict, log CSV, and try to fuse with a nearby BIO row."""
    global audio_model, audio_feat_order, impute_value
    global audio_pred_writer, audio_pred_fp, audio_label_names
    try:
        with LOCK:
            if audio_model is None:
                return
            t = float(row.get("t_center_s", np.nan))
            if not math.isfinite(t):
                return

            X = build_feature_vector(row, audio_feat_order, impute_value)
            pred_idx, score, probs = predict_with_scores(audio_model, X)
            pred_name = label_to_name(pred_idx, audio_label_names)

            # Per-stream CSV
            if audio_pred_writer is not None:
                out = {"t_center_s": t, "prediction": pred_name, "score": float(score) if score is not None else np.nan}
                if probs is not None and audio_label_names is not None:
                    for c, p in zip(audio_label_names, probs):
                        out[f"prob_{c}"] = float(p)
                safe = {k: out.get(k, np.nan) for k in audio_pred_headers}
                audio_pred_writer.writerow(safe)
                audio_pred_fp.flush()

            if PRINT_PRED:
                print(f"[AUDIO] t={t:6.1f}s → {pred_name}")

            # Save in buffer for fusion
            BUF_AUDIO[time_bin(t)] = {
                "t_center_s": t,
                "pred": pred_name,
                "score": float(score) if score is not None else np.nan,
                "probs": probs.copy() if probs is not None else None,
                "names": None if audio_label_names is None else list(audio_label_names),
            }

            try_merge_near(t)
    except Exception as e:
        print(f"[hub] handle_audio_features error: {e}")


def handle_bio_features(row: Dict[str, float]) -> None:
    """Consume one BIO feature row, predict, log CSV, and try to fuse with a nearby AUDIO row."""
    global bio_model, bio_feat_order, impute_value
    global bio_pred_writer, bio_pred_fp, bio_label_names
    try:
        with LOCK:
            if bio_model is None:
                return
            t = float(row.get("t_center_s", np.nan))
            if not math.isfinite(t):
                return

            X = build_feature_vector(row, bio_feat_order, impute_value)
            pred_idx, score, probs = predict_with_scores(bio_model, X)
            pred_name = label_to_name(pred_idx, bio_label_names)

            # Per-stream CSV
            if bio_pred_writer is not None:
                out = {"t_center_s": t, "prediction": pred_name, "score": float(score) if score is not None else np.nan}
                if probs is not None and bio_label_names is not None:
                    for c, p in zip(bio_label_names, probs):
                        out[f"prob_{c}"] = float(p)
                safe = {k: out.get(k, np.nan) for k in bio_pred_headers}
                bio_pred_writer.writerow(safe)
                bio_pred_fp.flush()

            if PRINT_PRED:
                print(f"[BIO  ] t={t:6.1f}s → {pred_name}")

            BUF_BIO[time_bin(t)] = {
                "t_center_s": t,
                "pred": pred_name,
                "score": float(score) if score is not None else np.nan,
                "probs": probs.copy() if probs is not None else None,
                "names": None if bio_label_names is None else list(bio_label_names),
            }

            try_merge_near(t)
    except Exception as e:
        print(f"[hub] handle_bio_features error: {e}")


# Late merge + fused logic
def map_probs_to_fused(names: Optional[List[str]], probs: Optional[np.ndarray]) -> np.ndarray:
    """Project a probability vector onto the fused label space (ordered by fused_label_names)."""
    p = np.zeros(FUSED_K, dtype=float)
    if probs is None or names is None:
        return p
    for n, val in zip(names, probs):
        if n in fused_index:
            p[fused_index[n]] = float(val)
    return p


def handle_fused(t: float, audio_entry: dict, bio_entry: dict):
    """Compute fused and (optionally) Viterbi-gated fused predictions; write merge CSV; optional external notify."""
    global fused_ema, delta_fused

    # 1) Project onto fused space
    pa = map_probs_to_fused(audio_entry.get("names"), audio_entry.get("probs"))
    pb = map_probs_to_fused(bio_entry.get("names"),   bio_entry.get("probs"))

    # Fallback: if no probs, use one-hot on predicted label
    if (pa.sum() <= 0.0) and isinstance(audio_entry.get("pred"), str):
        j = fused_index.get(audio_entry["pred"])
        if j is not None:
            pa[j] = 1.0
    if (pb.sum() <= 0.0) and isinstance(bio_entry.get("pred"), str):
        j = fused_index.get(bio_entry["pred"])
        if j is not None:
            pb[j] = 1.0

    # 2) Linear fusion
    p = fuse_audio_w * pa + fuse_bio_w * pb
    p = normalize(p)

    # 3) Temperature + EMA smoothing
    p = apply_temperature(p, FUSED_TEMPERATURE)
    if FUSED_EMA_ALPHA > 0.0:
        if fused_ema is None:
            fused_ema = p.copy()
        else:
            fused_ema = FUSED_EMA_ALPHA * fused_ema + (1.0 - FUSED_EMA_ALPHA) * p
        p = normalize(fused_ema)

    fused_pred = fused_label_names[int(np.argmax(p))]

    # 4) Online Viterbi (gated)
    viterbi_pred = fused_pred
    if USE_VITERBI_FUSED and Alog_fused is not None and fused_prior is not None:
        j, _ = viterbi_update_online(p)
        viterbi_pred = fused_label_names[j]

    if PRINT_PRED:
        print(f"[FUSED] t={t:6.1f}s → {fused_pred} | Viterbi={viterbi_pred}")
        send_stage_via_cli(viterbi_pred)

    # 5) Merge CSV
    if merge_writer is not None:
        out = {
            "t_center_s": t,
            "audio_pred": audio_entry.get("pred"),
            "audio_score": audio_entry.get("score"),
            "bio_pred": bio_entry.get("pred"),
            "bio_score": bio_entry.get("score"),
            "fused_pred": fused_pred,
            "fused_viterbi": viterbi_pred,
        }
        safe = {k: out.get(k, np.nan) for k in merge_headers}
        merge_writer.writerow(safe)
        merge_fp.flush()


def try_merge_near(t_center_s: float) -> None:
    """Try to find an AUDIO and BIO row whose centers are within MERGE_TOL_S seconds and fuse them."""
    if fused_label_names is None:
        return
    k = time_bin(t_center_s)

    # Look around k within ±MERGE_TOL_S seconds
    a = b = None
    k_a = k_b = None
    delta = int(max(1, round(MERGE_TOL_S)))

    for dk in range(-delta, delta + 1):
        key = k + dk
        if key in BUF_AUDIO and abs(BUF_AUDIO[key]["t_center_s"] - t_center_s) <= MERGE_TOL_S:
            a = BUF_AUDIO[key]; k_a = key; break

    for dk in range(-delta, delta + 1):
        key = k + dk
        if key in BUF_BIO and abs(BUF_BIO[key]["t_center_s"] - t_center_s) <= MERGE_TOL_S:
            b = BUF_BIO[key]; k_b = key; break

    if a is None or b is None:
        return

    # Use the average timestamp for the fused row
    t = 0.5 * (a["t_center_s"] + b["t_center_s"])
    handle_fused(t, a, b)

    # Cleanup consumed bins
    if k_a in BUF_AUDIO: del BUF_AUDIO[k_a]
    if k_b in BUF_BIO:   del BUF_BIO[k_b]
