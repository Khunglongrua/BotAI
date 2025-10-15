# -*- coding: utf-8 -*-
"""
F3_train_residual_mi38_single_input_light.py

Train LSTM 1-input cho dữ liệu F3 (MI38 + growth + 3 target residual).
Nguồn dữ liệu: /home/dzu/Desktop/BOT_2025/F3_RESIDUAL/{STEM}_feat_mi{XX}_APPLIED.parquet
  - X: toàn bộ cột numeric trừ 3 cột Y cuối (Y_5, Y_10, Y_15)
  - Y: 3 cột cuối là residual: [ln(C_{t+k}) - ln(C_t)] - g_raw_now

Hỗ trợ:
  * Đọc nhiều file của 1 symbol (theo INTERVAL) và nối lại theo thời gian (dedup theo open_time).
  * tf.data: cache -> (train: shuffle) -> batch -> prefetch (NO repeat)
  * Validation không shuffle để ổn định val_loss
  * Warmup–Cosine LR schedule (bs=1024): lr0=1.4e-3, warmup 15%, lr_min=0.1*lr0
  * EarlyStopping: min_delta=3e-4, patience=10, restore_best_weights
  * In ra MASE / Theil’s U / MAE (scaled & original)
  * Ghi JSON những symbol có MASE target1 < 0.75

Tham khảo cấu trúc từ N3_train_single_input_light.py và điều chỉnh cho MI38.
"""

import os, math, time, json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib

# ===== GPU & Precision =====
import tensorflow as tf
try:
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
except Exception:
    pass
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import RobustScaler

mixed_precision.set_global_policy("mixed_float16")

# ===== Paths & const =====
from pathlib import Path

# Lấy thư mục gốc của project (chính là thư mục chứa file code hiện tại)
BASE_DIR = Path(__file__).resolve().parent

# Thư mục chứa dữ liệu parquet MI38 (nằm cùng cấp với code, hoặc bạn có thể đổi cho phù hợp)
DATA_DIR = BASE_DIR / "DATA_BINANCE_FEAT"

# Thư mục output kết quả LSTM
OUT_DIR = BASE_DIR / "LSTM_F3_mi38_64"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL     = "15m"
WINDOW_SIZE  = 64
WINDOW_STEP  = 1

TARGET_NAMES = ["Y_5", "Y_10", "Y_15"]
TARGET_COLS  = 3

# ==== Batch & Train ====
BATCH_SIZE   = 1024
EPOCHS       = 80
PATIENCE     = 10
DROPOUT      = 0.10

# ==== Warmup–Cosine cho BS=1024 ====
LR_INIT        = 1.4e-3
MIN_LR_RATIO   = 0.10
WARMUP_FRAC    = 0.15
WEIGHT_DECAY   = 1e-5
CLIPNORM       = 5.0

# ===== JSON “đạt chuẩn” theo MASE target1 =====
MASE_T1_THRESHOLD = 0.75
SELECTED_JSON_PATH = OUT_DIR / f"selected_symbols_t1_lt_{MASE_T1_THRESHOLD:.2f}.json"


# ====== WarmupCosine schedule ======
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr0: float, total_steps: int, warmup_steps: int, min_lr: float):
        super().__init__()
        self.lr0 = float(lr0)
        self.total_steps = int(max(1, total_steps))
        self.warmup_steps = int(max(0, warmup_steps))
        self.min_lr = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr0 = tf.cast(self.lr0, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warm  = tf.cast(self.warmup_steps, tf.float32)
        lr_warm = lr0 * (step + 1.0) / tf.maximum(1.0, warm)  # warmup tuyến tính
        step_after  = tf.maximum(0.0, step - warm)
        total_after = tf.maximum(1.0, total - warm)
        cosine = 0.5 * (1.0 + tf.cos(math.pi * (step_after / total_after)))
        lr_decay = min_lr + (lr0 - min_lr) * cosine
        return tf.where(step < warm, lr_warm, lr_decay)
    def get_config(self):
        return {"lr0": self.lr0, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr}

class EpochLogger(Callback):
    def __init__(self, total_epochs: int): super().__init__(); self.total_epochs = int(total_epochs)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr_fn = self.model.optimizer.learning_rate
        try:
            step = tf.keras.backend.get_value(self.model.optimizer.iterations)
            lr_val = float(tf.keras.backend.get_value(lr_fn(step))) if callable(lr_fn) else float(tf.keras.backend.get_value(lr_fn))
        except Exception:
            try: lr_val = float(lr_fn.numpy())
            except Exception: lr_val = float(lr_fn)
        loss = logs.get("loss", float("nan")); vloss = logs.get("val_loss", float("nan"))
        print(f"Epoch {epoch+1}/{self.total_epochs} - loss: {loss:.6f} - val_loss: {vloss:.6f} - lr: {lr_val:.6f}")


# ===== Helpers =====
def split_train_val_test(N: int, train=0.60, val=0.20) -> Tuple[int, int]:
    return int(train*N), int((train+val)*N)

def create_windows_X(X_scaled: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    N = len(X_scaled)
    if N <= window: return np.empty((0, window, X_scaled.shape[1]), dtype=np.float32)
    idx = range(0, N - window, step)
    return np.asarray([X_scaled[i:i+window, :] for i in idx], dtype=np.float32)

def create_windows_y_level(Y_scaled: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    N = len(Y_scaled)
    if N <= window: return np.empty((0, Y_scaled.shape[1]), dtype=np.float32)
    idx = range(0, N - window, step)
    return np.asarray([Y_scaled[i+window, :] for i in idx], dtype=np.float32)

def sanitize_array(arr, fill=0.0):
    arr = np.asarray(arr); bad = ~np.isfinite(arr)
    if bad.any(): arr = arr.copy(); arr[bad] = fill
    return arr

def _load_and_concat_symbol_files(symbol: str) -> pd.DataFrame:
    """
    Đọc TẤT CẢ file {SYMBOL}_{INTERVAL}_feat_mi*_APPLIED.parquet, nối theo thời gian và dedup theo open_time.
    Nếu không có file theo INTERVAL, fallback: {SYMBOL}_*.parquet (vẫn lọc *_APPLIED).
    """
    pats = [
        f"{symbol}_{INTERVAL}_feat_mi*_APPLIED.parquet",
        f"{symbol}_*_feat_mi*_APPLIED.parquet",
    ]
    paths = []
    for pat in pats:
        paths.extend(sorted(DATA_DIR.glob(pat)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy parquet MI38 cho {symbol} trong {DATA_DIR}")

    dfs = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            if "open_time" not in d.columns:
                continue
            dfs.append(d)
        except Exception as e:
            print(f"⚠️ Bỏ qua {p.name}: {e}")

    if not dfs:
        raise FileNotFoundError(f"{symbol}: đọc được 0 file hợp lệ.")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    # sort & dedup theo open_time
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    # ffill nhẹ nếu còn NaN
    if df.isna().any().any():
        df = df.ffill()
    return df

def load_parquet_symbol(symbol: str) -> pd.DataFrame:
    return _load_and_concat_symbol_files(symbol)

def extract_XY(df: pd.DataFrame, target_names: Optional[List[str]] = None, target_cols: int = TARGET_COLS):
    df_num = df.select_dtypes(include=["number"]).copy()
    # bỏ các cột thời gian nếu còn sót
    for c in list(df_num.columns):
        lc = str(c).lower()
        if ("time" in lc) or ("timestamp" in lc) or ("date" in lc) or (lc in {"open_time","close_time"}):
            df_num.drop(columns=c, inplace=True, errors="ignore")
    if target_names and all(t in df_num.columns for t in target_names):
        y_df = df_num[target_names]; x_df = df_num.drop(columns=target_names)
    else:
        y_df = df_num.iloc[:, -target_cols:]; x_df = df_num.iloc[:, :-target_cols]
    return x_df.values.astype(np.float32), y_df.values.astype(np.float32)


def build_model(input_shape, n_targets: int, dropout=DROPOUT):
    x_in = Input(shape=input_shape, dtype=tf.float32, name="X_seq")
    t, f = input_shape
    proj = ((f + 7)//8)*8
    x = x_in if proj == f else TimeDistributed(Dense(proj, use_bias=False, dtype=tf.float32), name="project_to_8")(x_in)
    x = LSTM(464, return_sequences=True,  name="lstm_464")(x); x = Dropout(dropout)(x)
    x = LSTM(256, return_sequences=True,  name="lstm_256")(x); x = Dropout(dropout)(x)
    x = LSTM(128, return_sequences=False, name="lstm_128")(x); x = Dropout(dropout)(x)
    y_hat = Dense(n_targets, dtype=tf.float32, name="y_hat")(x)
    return Model(x_in, y_hat, name="lstm_single_input_light")


def build_optimizer(n_train_samples: int, epochs: int, lr0: float=LR_INIT,
                    min_lr_ratio: float=MIN_LR_RATIO, warmup_frac: float=WARMUP_FRAC):
    steps_per_epoch = max(1, math.ceil(n_train_samples / float(BATCH_SIZE)))
    total_steps  = max(1, steps_per_epoch * int(epochs))
    warmup_steps = int(round(total_steps * float(warmup_frac)))
    min_lr = float(lr0) * float(min_lr_ratio)
    sched = WarmupCosine(lr0=float(lr0), total_steps=int(total_steps),
                         warmup_steps=int(warmup_steps), min_lr=float(min_lr))
    opt = AdamW(learning_rate=sched, weight_decay=float(WEIGHT_DECAY), clipnorm=float(CLIPNORM))
    return opt


# ===== Metrics (MASE/TheilU/MAE) =====
def _mase_theil_per_target(y_true_vec, y_pred_vec, y_train_vec_for_mase=None, eps=1e-8):
    y_true_vec = np.asarray(y_true_vec, dtype=float)
    y_pred_vec = np.asarray(y_pred_vec, dtype=float)
    if y_train_vec_for_mase is not None and len(y_train_vec_for_mase) > 1:
        denom = np.mean(np.abs(np.diff(y_train_vec_for_mase)))
    else:
        denom = np.mean(np.abs(np.diff(y_true_vec))) if len(y_true_vec) > 1 else np.nan
    mae = np.mean(np.abs(y_true_vec - y_pred_vec))
    mase = mae / (denom + eps) if not np.isnan(denom) else np.nan
    num = np.sum((y_true_vec - y_pred_vec) ** 2)
    den = np.sum(np.diff(y_true_vec) ** 2) + eps
    theil_u = np.sqrt(num / den)
    return mase, theil_u, mae

def mase_and_theil_matrix(y_true, y_pred, y_train_for_mase=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_targets = y_true.shape[1]
    mase, theil, mae = [], [], []
    for i in range(n_targets):
        tr = None if y_train_for_mase is None else np.asarray(y_train_for_mase, dtype=float)[:, i]
        m, t, a = _mase_theil_per_target(y_true[:, i], y_pred[:, i], tr)
        mase.append(m); theil.append(t); mae.append(a)
    return np.array(mase), np.array(theil), np.array(mae)

def print_metric_tables(symbol, mase_s, theil_s, mae_s, mase_o, theil_o, mae_o):
    def fmt(x): 
        return f"{x:.6f}" if (x is not None and np.isfinite(x)) else "nan"
    print("\n=== METRICS ===")
    print(f"Symbol: {symbol}")
    print("[Scaled]")
    print(f"  MASE : {fmt(mase_s[0])} | {fmt(mase_s[1])} | {fmt(mase_s[2])}")
    print(f"  Theil: {fmt(theil_s[0])} | {fmt(theil_s[1])} | {fmt(theil_s[2])}")
    print(f"  MAE  : {fmt(mae_s[0])}  | {fmt(mae_s[1])}  | {fmt(mae_s[2])}")
    print("[Original]")
    if (mase_o is None) or (theil_o is None) or (mae_o is None):
        print("  (inverse failed)")
    else:
        print(f"  MASE : {fmt(mase_o[0])} | {fmt(mase_o[1])} | {fmt(mase_o[2])}")
        print(f"  Theil: {fmt(theil_o[0])} | {fmt(theil_o[1])} | {fmt(theil_o[2])}")
        print(f"  MAE  : {fmt(mae_o[0])}  | {fmt(mae_o[1])}  | {fmt(mae_o[2])}")


# ===== Append JSON nếu MASE target1 đạt chuẩn =====
def append_selected_symbol_if_qualified(symbol, mase_s, mase_o,
                                        threshold=MASE_T1_THRESHOLD,
                                        json_path=SELECTED_JSON_PATH):
    if (mase_o is not None) and (len(mase_o) > 0) and np.isfinite(mase_o[0]):
        m_eff = float(mase_o[0])
    else:
        m_eff = float(mase_s[0]) if (mase_s is not None and len(mase_s) > 0) else np.inf

    if not np.isfinite(m_eff):
        print(f"[WARN] {symbol}: MASE t1 không khả dụng → bỏ qua append.")
        return

    if m_eff < float(threshold):
        payload = {"threshold": float(threshold),
                   "metric": "mase target1",
                   "updated_at": int(time.time()),
                   "symbols": [],
                   "metrics": {}}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        payload.update({k: loaded.get(k, payload.get(k)) for k in payload.keys()})
                        if "metrics" not in payload or not isinstance(payload["metrics"], dict):
                            payload["metrics"] = {}
            except Exception:
                pass

        syms = list(dict.fromkeys(list(payload.get("symbols", [])) + [symbol]))
        payload["symbols"] = syms
        metrics = payload.get("metrics", {})
        metrics[symbol] = metrics.get(symbol, {})
        metrics[symbol]["mase_t1"] = float(m_eff)
        payload["metrics"] = metrics
        payload["threshold"] = float(threshold)
        payload["updated_at"] = int(time.time())

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✅ {symbol}: t1 MASE={m_eff:.4f} < {threshold} → appended to {json_path.name}")
    else:
        print(f"ℹ️ {symbol}: t1 MASE={m_eff:.4f} ≥ {threshold} → không append.")


# ===== Train một symbol =====
def train_symbol(symbol: str):
    print(f"\n====== Train {symbol} ({INTERVAL}) ws={WINDOW_SIZE} bs={BATCH_SIZE} ======")
    df = load_parquet_symbol(symbol)
    X_raw, y_raw = extract_XY(df, target_names=TARGET_NAMES, target_cols=TARGET_COLS)

    N = len(X_raw)
    if N <= WINDOW_SIZE + 1:
        print(f"⚠️ {symbol}: quá ngắn (N={N}) → bỏ qua."); return

    tr_end, val_end = split_train_val_test(N, train=0.60, val=0.20)
    X_tr_raw, y_tr_raw = X_raw[:tr_end],          y_raw[:tr_end]
    X_val_raw, y_val_raw = X_raw[tr_end:val_end], y_raw[tr_end:val_end]
    X_te_raw,  y_te_raw  = X_raw[val_end:],       y_raw[val_end:]

    # RobustScaler fit trên TRAIN
    sx = RobustScaler().fit(X_tr_raw)
    sy = RobustScaler().fit(y_tr_raw)
    joblib.dump(sx, OUT_DIR / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_robust_X.pkl")
    joblib.dump(sy, OUT_DIR / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_robust_Y_t{TARGET_COLS}.pkl")

    # Transform + sanitize
    X_tr = sanitize_array(sx.transform(X_tr_raw)); y_tr_s = sanitize_array(sy.transform(y_tr_raw))
    X_val = sanitize_array(sx.transform(X_val_raw)); y_val_s = sanitize_array(sy.transform(y_val_raw))
    X_te  = sanitize_array(sx.transform(X_te_raw));  y_te_s  = sanitize_array(sy.transform(y_te_raw))

    # Window hoá
    X_tr_w = create_windows_X(X_tr,  WINDOW_SIZE, WINDOW_STEP)
    X_val_w = create_windows_X(X_val, WINDOW_SIZE, WINDOW_STEP)
    X_te_w  = create_windows_X(X_te,  WINDOW_SIZE, WINDOW_STEP)
    y_tr_w  = create_windows_y_level(y_tr_s, WINDOW_SIZE, WINDOW_STEP)
    y_val_w = create_windows_y_level(y_val_s, WINDOW_SIZE, WINDOW_STEP)
    y_te_w  = create_windows_y_level(y_te_s,  WINDOW_SIZE, WINDOW_STEP)

    # tf.data
    ds_train = make_train_ds(X_tr_w, y_tr_w)
    ds_val   = make_val_ds(X_val_w, y_val_w)

    # Model & optimizer
    model = build_model((WINDOW_SIZE, X_tr_w.shape[2]), n_targets=TARGET_COLS, dropout=DROPOUT)
    opt   = build_optimizer(n_train_samples=len(X_tr_w), epochs=EPOCHS, lr0=LR_INIT,
                            min_lr_ratio=MIN_LR_RATIO, warmup_frac=WARMUP_FRAC)
    try:
        model.compile(optimizer=opt, loss="mae", jit_compile=False)
    except Exception:
        model.compile(optimizer=opt, loss="mae")

    # Callbacks
    ckpt_path = OUT_DIR / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_best.keras"
    callbacks = [
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", mode="min",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", mode="min",
                      min_delta=3e-4, patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        EpochLogger(EPOCHS)
    ]

    # Fit
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=0)

    # ===== Evaluate & Metrics =====
    y_pred_te_s = model.predict(X_te_w, verbose=0, batch_size=BATCH_SIZE)
    y_true_te_s = y_te_w

    # Mẫu train cho mẫu số MASE (scaled)
    y_train_for_mase_s = y_tr_s[WINDOW_SIZE:][:len(y_pred_te_s)]

    # Scaled metrics
    mase_s, theil_s, mae_s = mase_and_theil_matrix(
        y_true_te_s, y_pred_te_s, y_train_for_mase=y_train_for_mase_s
    )

    # Original metrics (inverse transform)
    mase_o = theil_o = mae_o = None
    try:
        y_pred_te = sy.inverse_transform(y_pred_te_s)
        y_true_te = sy.inverse_transform(y_true_te_s)
        y_train_for_mase = sy.inverse_transform(y_train_for_mase_s)
        mase_o, theil_o, mae_o = mase_and_theil_matrix(
            y_true_te, y_pred_te, y_train_for_mase=y_train_for_mase
        )
    except Exception as e:
        print(f"[WARN] Inverse metrics failed: {e}")

    # In
    print_metric_tables(symbol, mase_s, theil_s, mae_s, mase_o, theil_o, mae_o)

    # Append JSON nếu đạt chuẩn theo target1
    append_selected_symbol_if_qualified(symbol, mase_s, mase_o)


def load_symbols():
    p = Path("symbols.txt")
    if p.exists():
        try:
            return [s.strip().upper() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            print(f"❌ Lỗi đọc symbols.txt: {e}")
    print("⚠️ Không thấy symbols.txt (mỗi dòng 1 mã, ví dụ BTCUSDT)."); return []


def make_train_ds(X, Y):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    return (
        ds.cache()
          .shuffle(min(len(X), 5000), seed=42, reshuffle_each_iteration=True)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE)
    )

def make_val_ds(X, Y):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    # KHÔNG shuffle validation để ổn định val_loss
    return ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    for sym in load_symbols():
        try:
            train_symbol(sym)
        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện {sym}: {e}")
    print(f"✅ Done. JSON symbols đạt chuẩn (nếu có): {SELECTED_JSON_PATH}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
