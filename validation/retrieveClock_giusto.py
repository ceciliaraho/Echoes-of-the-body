from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys, csv
from joblib import load  # aggiungi
import pandas as pd      # aggiungi

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import sounddevice as sd
import soundfile as sf
import numpy as np

from realtime_audio_features import rt_start, rt_append, rt_stop
from realtime_bio_features   import bio_rt_start, bio_rt_append, bio_rt_stop

import inference_hub as hub


# Config
OSC_IP, OSC_PORT = "0.0.0.0", 6575

DEVICE_NAME = "CABLE Output"   # substring of the input device name
SAMPLERATE, CHANNELS, DTYPE = 44100, 2, "float32"

WAV_FILENAME = "rec_vbcable.wav"
FEATURES_DIR = Path("features");  FEATURES_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR     = Path("pred_logs"); PRED_DIR.mkdir(parents=True, exist_ok=True)

# Live inference
ENABLE_INFERENCE   = True
AUDIO_MODEL_PATH   = Path("models/audio_rf_model.joblib")
BIO_MODEL_PATH     = Path("models/bio_rf_model.joblib")

# ✅ nuovi path
AUDIO_LABEL_ENCODER = Path("models/audio_label_encoder.joblib")
BIO_LABEL_ENCODER   = Path("models/bio_label_encoder.joblib")
AUDIO_SELECTED_FEATS = Path("models/audio_selected_features.csv")
BIO_SELECTED_FEATS   = Path("models/selected_features.csv")  # (bio)

# ✅ usa NaN per lasciare che l'Imputer della pipeline faccia il suo lavoro
IMPUTE_VALUE       = float("nan")
MERGE_TOLERANCE_S  = 3.0   # alignment tolerance for fused

# State
recording = False
audio_stream = None
audio_chunks: list[np.ndarray] = []

audio_rt_active = False
bio_rt_active   = False

csv_file = None
csv_writer = None
bio_raw_path: Path | None = None

# de-dup for /bio 
last_bio_ts = None     # last seen timestamp (ms)
bio_acc_bf = 0.0       # sum for BF avg
bio_acc_hr = 0.0       # sum for HR avg
bio_acc_n  = 0         # count for that timestamp

#  first-sample guard + reset if time moves backwards 
FIRST_TS_GLITCH_MS = 1000.0   # first ts > 1 s → suspicious
BACKSTEP_RESET_MS  = 50.0     # if time decreases > 50 ms → reset
guard_has_first    = False
guard_first_ts     = 0.0
guard_first_bf_sum = 0.0
guard_first_hr_sum = 0.0
guard_first_n      = 0

# Utils
def find_device_index(name_substring: str):
    """Return first input device index whose name contains the given substring (case-insensitive)."""
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0 and name_substring.lower() in d["name"].lower():
                return i
    except Exception:
        pass
    return None

def audio_callback(indata, frames, time_info, status):
    """Collect raw audio and feed the RT audio pipeline while recording."""
    if not recording:
        return
    try:
        audio_chunks.append(indata.copy())
        rt_append(indata)
    except Exception:
        pass

def close_bio_csv():
    """Safely close the raw BIO CSV file."""
    global csv_file
    try:
        if csv_file and not csv_file.closed:
            csv_file.flush()
            csv_file.close()
    except Exception:
        pass

def stop_audio_stream():
    """Stop and close the audio input stream if active."""
    global audio_stream
    if audio_stream is not None:
        try: audio_stream.stop()
        except Exception: pass
        try: audio_stream.close()
        except Exception: pass
        audio_stream = None

def emit_bio_point(time_val_ms: float, bf: float, hr: float):
    """Write a single raw BIO row and forward the point to the RT pipeline."""
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        csv_writer.writerow([now, time_val_ms, bf, hr])
        csv_file.flush()
    except Exception:
        pass
    try:
        bio_rt_append(time_val_ms, bf, hr)
    except Exception:
        pass

def flush_pending_bio():
    """If we accumulated duplicates for the same timestamp, emit their average once."""
    global last_bio_ts, bio_acc_bf, bio_acc_hr, bio_acc_n
    if bio_acc_n > 0 and last_bio_ts is not None:
        bf_avg = bio_acc_bf / bio_acc_n
        hr_avg = bio_acc_hr / bio_acc_n
        if not (bf_avg == 0.0 and hr_avg == 0.0):  # ignore all-zero packets
            emit_bio_point(last_bio_ts, bf_avg, hr_avg)
    bio_acc_bf = 0.0
    bio_acc_hr = 0.0
    bio_acc_n  = 0
    last_bio_ts = None

def cleanup():
    """Finalize everything: stop audio, flush pending BIO, save WAV, close files, clear state."""
    global recording, audio_chunks
    try:
        stop_audio_stream()
        try: flush_pending_bio()
        except Exception: pass

        if audio_chunks:
            try:
                audio = np.vstack(audio_chunks)
                sf.write(WAV_FILENAME, audio, SAMPLERATE)
                print(f"Audio saved to '{WAV_FILENAME}'")
            except Exception:
                print("WAV save error (ignored).")
    finally:
        close_bio_csv()
        recording = False
        audio_chunks = []

# ==================== OSC HANDLERS ====================
def bio_handler(address, *args):
    """
    Handle /bio <time_ms> <BF> <HR>:
      - skip 0/0 packets
      - coalesce duplicates (average) for the same timestamp
      - guard suspicious first sample (very large ts)
      - reset accumulator if time goes backwards
    """
    global last_bio_ts, bio_acc_bf, bio_acc_hr, bio_acc_n
    global guard_has_first, guard_first_ts, guard_first_bf_sum, guard_first_hr_sum, guard_first_n

    if not recording or len(args) != 3:
        return

    try:
        time_val_ms, bf, hr = map(float, args)
    except Exception:
        return

    # Drop completely null packets (typical glitches)
    if bf == 0.0 and hr == 0.0:
        return

    # First-sample guard: if the very first ts is “huge”, wait for the second one
    if last_bio_ts is None and not guard_has_first:
        if time_val_ms > FIRST_TS_GLITCH_MS:
            guard_has_first   = True
            guard_first_ts    = time_val_ms
            guard_first_bf_sum = bf
            guard_first_hr_sum = hr
            guard_first_n      = 1
            return
        else:
            last_bio_ts = time_val_ms
            bio_acc_bf  = bf
            bio_acc_hr  = hr
            bio_acc_n   = 1
            return

    # Second sample while the first is in guard
    if guard_has_first:
        # If the new ts is notably smaller → discard the first (glitch) and start now
        if time_val_ms + BACKSTEP_RESET_MS < guard_first_ts:
            guard_has_first = False
            last_bio_ts = time_val_ms
            bio_acc_bf  = bf
            bio_acc_hr  = hr
            bio_acc_n   = 1
            return
        else:
            # First wasn’t a glitch → release it (average if duplicates)
            bf_avg = guard_first_bf_sum / max(1, guard_first_n)
            hr_avg = guard_first_hr_sum / max(1, guard_first_n)
            if not (bf_avg == 0.0 and hr_avg == 0.0):
                emit_bio_point(guard_first_ts, bf_avg, hr_avg)
            guard_has_first = False
            # This sample starts the new bucket
            last_bio_ts = time_val_ms
            bio_acc_bf  = bf
            bio_acc_hr  = hr
            bio_acc_n   = 1
            return

    # Reset accumulators if time goes backwards significantly
    if last_bio_ts is not None and time_val_ms + BACKSTEP_RESET_MS < last_bio_ts:
        flush_pending_bio()
        last_bio_ts = time_val_ms
        bio_acc_bf  = bf
        bio_acc_hr  = hr
        bio_acc_n   = 1
        return

    # Coalesce duplicates at the same timestamp
    if time_val_ms == last_bio_ts:
        bio_acc_bf += bf
        bio_acc_hr += hr
        bio_acc_n  += 1
        return

    # New timestamp → emit previous average and start a new bucket
    flush_pending_bio()
    last_bio_ts = time_val_ms
    bio_acc_bf  = bf
    bio_acc_hr  = hr
    bio_acc_n   = 1

def start_recording_handler(address, *args):
    global recording, audio_stream, audio_chunks, audio_rt_active, bio_rt_active
    global csv_file, csv_writer, bio_raw_path
    global last_bio_ts, bio_acc_bf, bio_acc_hr, bio_acc_n
    global guard_has_first, guard_first_ts, guard_first_bf_sum, guard_first_hr_sum, guard_first_n

    if recording:
        print("Already recording.")
        return

    print("START recording")
    recording = True

    # Find input device
    idx = find_device_index(DEVICE_NAME)
    if idx is None:
        print(f"Input device '{DEVICE_NAME}' not found.")
        recording = False
        return

    # Start audio stream
    audio_chunks = []
    try:
        audio_stream = sd.InputStream(
            device=idx, samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE,
            callback=audio_callback, blocksize=2048, latency='high'
        )
        audio_stream.start()
        print(f"Audio stream started on '{DEVICE_NAME}' (idx {idx}).")
    except Exception as e:
        recording = False
        print("Failed to start audio stream:", e)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw BIO CSV
    try:
        bio_raw_path = Path(f"bio_raw_{ts}.csv")
        csv_file = open(bio_raw_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["local_timestamp", "bio_time", "BF", "HR"])
        csv_file.flush()
        print(f"Raw BIO → {bio_raw_path}")
    except Exception as e:
        print("Cannot open raw BIO CSV:", e)

    # Reset de-dup + first-sample guard state
    last_bio_ts = None
    bio_acc_bf = bio_acc_hr = 0.0
    bio_acc_n  = 0
    guard_has_first = False
    guard_first_ts = 0.0
    guard_first_bf_sum = guard_first_hr_sum = 0.0
    guard_first_n = 0

    pipe_audio = load(AUDIO_MODEL_PATH)
    pipe_bio   = load(BIO_MODEL_PATH)
    le_audio   = load(AUDIO_LABEL_ENCODER)
    le_bio     = load(BIO_LABEL_ENCODER)

    aud_feats  = pd.read_csv(AUDIO_SELECTED_FEATS)["selected_feature"].tolist()
    bio_feats  = pd.read_csv(BIO_SELECTED_FEATS)["selected_feature"].tolist()

    # --- prepara i bundle per l'hub (modello + classi + ordine feature) ---
    audio_bundle = {"model": pipe_audio,
                    "classes": le_audio.classes_.tolist(),
                    "selected_features": aud_feats}
    bio_bundle   = {"model": pipe_bio,
                    "classes": le_bio.classes_.tolist(),
                    "selected_features": bio_feats}
    # Init models 
    enable_cb = False
    if ENABLE_INFERENCE and hub is not None and AUDIO_MODEL_PATH.exists() and BIO_MODEL_PATH.exists():
        try:
            hub.init_models(
                model_audio=audio_bundle,
                model_bio=bio_bundle,
                impute_value_param=IMPUTE_VALUE,
                audio_pred_csv=PRED_DIR / f"pred_audio_{ts}.csv",
                bio_pred_csv=PRED_DIR / f"pred_bio_{ts}.csv",
                merge_csv=PRED_DIR / f"pred_merge_{ts}.csv",
                merge_tolerance_s=MERGE_TOLERANCE_S,
                print_predictions=True,
                fuse_audio_weight=0.7,
                fuse_bio_weight=0.8,

                # usa le STESSE classi del bio come ordine "fuso" (coerente con training)
                fused_order=["pranayama","chanting","viparita_swasa","breath_retention","meditation"],
                
                use_viterbi_fused=True,
                fused_stay=0.50,
                fused_step=0.60,
                fused_extra_edges={"viparita_swasa": {"breath_retention": 0.90}, "viparita_swasa": {"meditation": 0.60  }},
                fused_start_label="pranayama",
                fused_start_strength=0.8,
                # gating/smoothing
                fused_min_dwell=1,
                fused_next_threshold=0.5,
                fused_ema_alpha=0.5,
                fused_temperature=1.0
            )
            print("Models loaded. Live inference ON.")
            enable_cb = True
        except Exception as e:
            print("Model init failed:", e)
    else:
        if ENABLE_INFERENCE:
            print("Live inference OFF (hub or models missing).")

    # Start realtime pipelines (+ optional callbacks into hub)
    out_csv_audio = FEATURES_DIR / f"audio_features_rt_{ts}.csv"
    out_csv_bio   = FEATURES_DIR / f"bio_features_rt_{ts}.csv"

    try:
        rt_start(
            output_csv=out_csv_audio,
            input_sr=SAMPLERATE,
            streaming=True,
            on_window=hub.handle_audio_features if enable_cb else None
        )
        audio_rt_active = True
        print(f"Audio RT → {out_csv_audio}")
    except Exception as e:
        audio_rt_active = False
        print("Audio RT start error:", e)

    try:
        bio_rt_start(
            output_csv=out_csv_bio,
            fs=120, win_sec=10.0, hop_sec=10.0, long_sec=40.0,
            streaming=True,
            on_window=hub.handle_bio_features if enable_cb else None
        )
        bio_rt_active = True
        print(f"Bio RT → {out_csv_bio}")
    except Exception as e:
        bio_rt_active = False
        print("Bio RT start error:", e)

def stop_recording_handler(address, *args):
    global recording, audio_rt_active, bio_rt_active
    if not recording:
        print("Not recording.")
        return

    print(">>> STOP recording")
    recording = False

    # Flush pending /bio before stopping RT
    try: flush_pending_bio()
    except Exception: pass

    # Close RT
    try:
        if audio_rt_active:
            feat_csv = rt_stop()
            print(f"AUDIO features CSV: {feat_csv}")
    except Exception as e:
        print("Audio RT stop error:", e)
    finally:
        audio_rt_active = False

    try:
        if bio_rt_active:
            bio_csv = bio_rt_stop()
            print(f"BIO features CSV: {bio_csv}")
    except Exception as e:
        print("Bio RT stop error:", e)
    finally:
        bio_rt_active = False

    # Close hub
    try:
        if hub is not None:
            hub.close()
    except Exception:
        pass

    cleanup()

def default_handler(address, *args):
    
    pass

# Main
if __name__ == "__main__":
    try:
        disp = Dispatcher()
        disp.set_default_handler(default_handler)
        disp.map("/bio", bio_handler)
        disp.map("/startRecording", start_recording_handler)
        disp.map("/stopRecording", stop_recording_handler)

        server = BlockingOSCUDPServer((OSC_IP, OSC_PORT), disp)
        print(f"Listening on {OSC_IP}:{OSC_PORT} … (/bio, /startRecording, /stopRecording)")
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nCtrl+C: shutting down…")
        try:
            try: flush_pending_bio()
            except Exception: pass
            if audio_rt_active:
                feat_csv = rt_stop()
                print(f"AUDIO features CSV: {feat_csv}")
        except Exception:
            pass
        try:
            if bio_rt_active:
                bio_csv = bio_rt_stop()
                print(f"BIO features CSV: {bio_csv}")
        except Exception:
            pass
        try:
            if hub is not None:
                hub.close()
        except Exception:
            pass
        cleanup()

    except Exception as e:
        print("Unexpected error:", e)
        try:
            try: flush_pending_bio()
            except Exception: pass
            if audio_rt_active:
                feat_csv = rt_stop()
                print(f"AUDIO features CSV: {feat_csv}")
        except Exception:
            pass
        try:
            if bio_rt_active:
                bio_csv = bio_rt_stop()
                print(f"BIO features CSV: {bio_csv}")
        except Exception:
            pass
        try:
            if hub is not None:
                hub.close()
        except Exception:
            pass
        cleanup()
        sys.exit(1)
