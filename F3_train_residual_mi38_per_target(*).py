# -*- coding: utf-8 -*-
"""
F3_train_residual_mi38_per_target.py (per-target training, optimized for RTX 4090)

- Train LSTM single-input cho residual MI38 theo TỪNG TARGET riêng.
- Cố định: TARGET = Y_10, WINDOW_SIZE = 64, INTERVAL = "15m".
- Lưu mô hình/metrics theo thư mục: TARGET_3Y/Y_10/
- Ghi metrics (MASE/TheilU/MAE) scaled & original + append CSV.
- Tự động cập nhật JSON 'selected_symbols_mase_lt_0.75.json' trong TARGET_3Y/Y_10/
  chỉ chứa những symbol có MASE < 0.75 (ưu tiên original; nếu inverse fail thì dùng scaled).
- Loại sạch mọi cột Y_* khỏi X để tránh leakage.

Yêu cầu:
- symbols.txt (mỗi dòng 1 symbol, ví dụ BTCUSDT)
- Bộ parquet MI38 *_APPLIED.parquet trong DATA_DIR

Có thể đổi đường dẫn qua env (không bắt buộc):
  DATA_DIR=/path/to/data  OUT_ROOT=/path/to/TARGET_3Y  python F3_train_residual_mi38_per_target.py
"""

import os, math, time, json, tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

# ===== GPU & Precision =====
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import RobustScaler

# ---- GPU setup
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
mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)  # XLA JIT
except Exception:
    pass

# ===== Paths & const =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "DATA_BINANCE_FEAT"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", BASE_DIR / "TARGET_3Y"))

# ---- Cố định cấu hình lõi (không đọc env)
INTERVAL     = "15m"
TARGET_NAME  = "Y_5"
WINDOW_SIZE  = 176
WINDOW_STEP  = 1

# ==== Batch & Train (optimized for RTX 4090) ====
BATCH_SIZE   = 128*8
EPOCHS       = 80
PATIENCE     = 22          # rộng hơn để vượt warm-up
DROPOUT      = 0.15

# ==== Warmup–Cosine schedule ====
LR_INIT      = 1.6e-3      # tuned cho bs=1024
MIN_LR_RATIO = 0.10        # min_lr = 0.1 * LR_INIT
WARMUP_FRAC  = 0.08        # warm-up ngắn hơn
WEIGHT_DECAY = 1e-5
CLIPNORM     = 5.0

# ---- MASE threshold & JSON output
MASE_THRESHOLD      = 0.8
SELECTED_JSON_NAME  = "selected_symbols_mase_lt_0.75.json"

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
        lr_warm = lr0 * (step + 1.0) / tf.maximum(1.0, warm)
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
    # tìm các parquet *_APPLIED.parquet theo interval
    pats = [
        f"{symbol}_{INTERVAL}_feat_mi*_APPLIED.parquet",
        f"{symbol}_*_feat_mi*_APPLIED.parquet",
    ]
    paths = []
    for pat in pats: paths.extend(sorted(DATA_DIR.glob(pat)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy parquet MI38 cho {symbol} trong {DATA_DIR}")

    dfs = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            if "open_time" not in d.columns: continue
            dfs.append(d)
        except Exception as e:
            print(f"⚠️ Bỏ qua {p.name}: {e}")

    if not dfs: raise FileNotFoundError(f"{symbol}: đọc được 0 file hợp lệ.")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    if df.isna().any().any(): df = df.ffill()
    return df

def load_parquet_symbol(symbol: str) -> pd.DataFrame:
    return _load_and_concat_symbol_files(symbol)

def extract_XY_per_target(df: pd.DataFrame, target_name: str):
    """
    X: toàn bộ numeric HỢP LỆ (loại sạch Y_* và cột thời gian).
    y: chỉ 1 cột target (shape [N, 1]).
    """
    df_num = df.select_dtypes(include=["number"]).copy()
    # drop cột thời gian
    for c in list(df_num.columns):
        lc = str(c).lower()
        if ("time" in lc) or ("timestamp" in lc) or ("date" in lc) or (lc in {"open_time","close_time"}):
            df_num.drop(columns=c, inplace=True, errors="ignore")

    if target_name not in df_num.columns:
        raise KeyError(f"Không thấy cột target '{target_name}' trong dữ liệu.")

    # LOẠI TẤT CẢ Y_* khỏi X
    y_cols = [c for c in df_num.columns if str(c).upper().startswith("Y_")]
    y_df = df_num[[target_name]].copy()
    x_df = df_num.drop(columns=y_cols, errors="ignore")

    return x_df.values.astype(np.float32), y_df.values.astype(np.float32)

def build_model(input_shape, dropout=DROPOUT):
    """
    Model output 1 chiều (regression 1 target)
    """
    x_in = Input(shape=input_shape, dtype=tf.float32, name="X_seq")
    t, f = input_shape
    proj = ((f + 7)//8)*8
    x = x_in if proj == f else TimeDistributed(Dense(proj, use_bias=False, dtype=tf.float32), name="project_to_8")(x_in)
    x = LSTM(512, return_sequences=True,  name="lstm_512")(x); x = Dropout(dropout)(x)
    x = LSTM(256, return_sequences=True,  name="lstm_256")(x); x = Dropout(dropout)(x)
    x = LSTM(128, return_sequences=False, name="lstm_128")(x); x = Dropout(dropout)(x)

    y_hat = Dense(1, dtype=tf.float32, name="y_hat")(x)
    return Model(x_in, y_hat, name="lstm_single_input_1target")

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

# ===== Metrics (MASE/TheilU/MAE) 1 target =====
def mase_theil_mae_single(y_true_vec, y_pred_vec, y_train_vec_for_mase=None, eps=1e-8):
    y_true_vec = np.asarray(y_true_vec, dtype=float).reshape(-1)
    y_pred_vec = np.asarray(y_pred_vec, dtype=float).reshape(-1)
    if y_train_vec_for_mase is not None and len(y_train_vec_for_mase) > 1:
        denom = np.mean(np.abs(np.diff(y_train_vec_for_mase.reshape(-1))))
    else:
        denom = np.mean(np.abs(np.diff(y_true_vec))) if len(y_true_vec) > 1 else np.nan
    mae = np.mean(np.abs(y_true_vec - y_pred_vec))
    mase = mae / (denom + eps) if not np.isnan(denom) else np.nan
    num = np.sum((y_true_vec - y_pred_vec) ** 2)
    den = np.sum(np.diff(y_true_vec) ** 2) + eps
    theil_u = np.sqrt(num / den)
    return float(mase), float(theil_u), float(mae)

def print_metric_single(symbol, target_name, mase_s, theil_s, mae_s, mase_o, theil_o, mae_o):
    def fmt(x): 
        return f"{x:.6f}" if (x is not None and np.isfinite(x)) else "nan"
    print("\n=== METRICS (single target) ===")
    print(f"Symbol: {symbol} | Target: {target_name}")
    print("[Scaled]")
    print(f"  MASE : {fmt(mase_s)}")
    print(f"  Theil: {fmt(theil_s)}")
    print(f"  MAE  : {fmt(mae_s)}")
    print("[Original]")
    if any(v is None or not np.isfinite(v) for v in [mase_o, theil_o, mae_o]):
        print("  (inverse failed)")
    else:
        print(f"  MASE : {fmt(mase_o)}")
        print(f"  Theil: {fmt(theil_o)}")
        print(f"  MAE  : {fmt(mae_o)}")

# ===== TF Datasets =====
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
    return ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ===== JSON helpers =====
def _atomic_write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def update_selected_symbols_json(out_dir: Path, target_name: str, interval: str,
                                 window_size: int, symbol: str,
                                 mase: float, theil: float, mae: float):
    path = out_dir / SELECTED_JSON_NAME
    data = {"target": target_name, "interval": interval, "window_size": window_size,
            "threshold": MASE_THRESHOLD, "symbols": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "symbols" not in data or not isinstance(data["symbols"], dict):
                data["symbols"] = {}
        except Exception:
            data = {"target": target_name, "interval": interval,
                    "window_size": window_size, "threshold": MASE_THRESHOLD, "symbols": {}}
    data["target"] = target_name
    data["interval"] = interval
    data["window_size"] = window_size
    data["threshold"] = MASE_THRESHOLD
    data["symbols"][symbol] = {
        "mase": float(mase),
        "theil": float(theil),
        "mae": float(mae)
    }
    _atomic_write_json(data, path)

def remove_symbol_from_selected_json(out_dir: Path, symbol: str):
    path = out_dir / SELECTED_JSON_NAME
    if not path.exists(): return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("symbols"), dict) and symbol in data["symbols"]:
            del data["symbols"][symbol]
            _atomic_write_json(data, path)
    except Exception:
        pass

# ===== Train 1 symbol cho 1 target =====
def train_symbol_for_target(symbol: str, target_name: str):
    # Thư mục ra theo target: TARGET_3Y/<target_name>/
    out_dir = OUT_ROOT / target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = target_name.replace("_", "").lower()  # y5, y10, y15
    print(f"\n====== Train {symbol} ({INTERVAL}) ws={WINDOW_SIZE} bs={BATCH_SIZE} | target={target_name} ======")

    df = load_parquet_symbol(symbol)
    X_raw, y_raw = extract_XY_per_target(df, target_name=target_name)   # y_raw shape [N,1]

    N = len(X_raw)
    if N <= WINDOW_SIZE + 1:
        print(f"⚠️ {symbol}:{target_name}: quá ngắn (N={N}) → bỏ qua."); return

    tr_end, val_end = split_train_val_test(N, train=0.60, val=0.20)
    X_tr_raw, y_tr_raw = X_raw[:tr_end],          y_raw[:tr_end]
    X_val_raw, y_val_raw = X_raw[tr_end:val_end], y_raw[tr_end:val_end]
    X_te_raw,  y_te_raw  = X_raw[val_end:],       y_raw[val_end:]

    # RobustScaler fit trên TRAIN
    sx = RobustScaler().fit(X_tr_raw)
    sy = RobustScaler().fit(y_tr_raw)  # 1 chiều
    joblib.dump(sx, out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{suffix}_robust_X.pkl")
    joblib.dump(sy, out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{suffix}_robust_Y.pkl")

    # Transform + sanitize
    X_tr = sanitize_array(sx.transform(X_tr_raw)); y_tr_s = sanitize_array(sy.transform(y_tr_raw))
    X_val = sanitize_array(sx.transform(X_val_raw)); y_val_s = sanitize_array(sy.transform(y_val_raw))
    X_te  = sanitize_array(sx.transform(X_te_raw));  y_te_s  = sanitize_array(sy.transform(y_te_raw))

    # Window hoá
    X_tr_w = create_windows_X(X_tr,  WINDOW_SIZE, WINDOW_STEP)
    X_val_w = create_windows_X(X_val, WINDOW_SIZE, WINDOW_STEP)
    X_te_w  = create_windows_X(X_te,  WINDOW_SIZE, WINDOW_STEP)
    y_tr_w  = create_windows_y_level(y_tr_s, WINDOW_SIZE, WINDOW_STEP)  # shape [M,1]
    y_val_w = create_windows_y_level(y_val_s, WINDOW_SIZE, WINDOW_STEP)
    y_te_w  = create_windows_y_level(y_te_s,  WINDOW_SIZE, WINDOW_STEP)

    # tf.data
    ds_train = make_train_ds(X_tr_w, y_tr_w)
    ds_val   = make_val_ds(X_val_w, y_val_w)

    # Model & optimizer
    model = build_model((WINDOW_SIZE, X_tr_w.shape[2]), dropout=DROPOUT)
    opt   = build_optimizer(n_train_samples=len(X_tr_w), epochs=EPOCHS, lr0=LR_INIT,
                            min_lr_ratio=MIN_LR_RATIO, warmup_frac=WARMUP_FRAC)
    try:
        model.compile(optimizer=opt, loss=Huber(delta=0.002))
    except Exception:
        model.compile(optimizer=opt, loss="mae")

    # Callbacks (nếu Keras hỗ trợ start_from_epoch, dùng; nếu không -> tăng patience đã set)
    ckpt_path = out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{suffix}_best.keras"
    callbacks = [
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", mode="min",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", mode="min",
                      min_delta=1e-5, patience=PATIENCE,
                      restore_best_weights=False, start_from_epoch=int(EPOCHS * WARMUP_FRAC) + 5, verbose=1),
        EpochLogger(EPOCHS)
    ]

    # Fit
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=0)

    # ===== Evaluate & Metrics =====
    y_pred_te_s = model.predict(X_te_w, verbose=0, batch_size=BATCH_SIZE)  # shape [M,1]
    y_true_te_s = y_te_w

    # Mẫu train cho mẫu số MASE (scaled)
    y_train_for_mase_s = y_tr_s[WINDOW_SIZE:][:len(y_pred_te_s)]

    # Scaled metrics (single)
    mase_s, theil_s, mae_s = mase_theil_mae_single(
        y_true_te_s, y_pred_te_s, y_train_vec_for_mase=y_train_for_mase_s
    )

    # Original metrics (inverse transform)
    mase_o = theil_o = mae_o = None
    try:
        y_pred_te = sy.inverse_transform(y_pred_te_s)
        y_true_te = sy.inverse_transform(y_true_te_s)
        y_train_for_mase = sy.inverse_transform(y_train_for_mase_s)
        mase_o, theil_o, mae_o = mase_theil_mae_single(
            y_true_te, y_pred_te, y_train_vec_for_mase=y_train_for_mase
        )
    except Exception as e:
        print(f"[WARN] Inverse metrics failed ({target_name}): {e}")

    # In metrics
    print_metric_single(symbol, target_name, mase_s, theil_s, mae_s, mase_o, theil_o, mae_o)

    # Lưu metrics CSV (append) trong thư mục target
    try:
        import csv
        metrics_csv = out_dir / f"metrics_{target_name}.csv"
        header = ["timestamp","symbol","interval","window_size","target",
                  "mase_scaled","theil_scaled","mae_scaled",
                  "mase_orig","theil_orig","mae_orig"]
        row = [int(time.time()), symbol, INTERVAL, WINDOW_SIZE, target_name,
               mase_s, theil_s, mae_s, mase_o, theil_o, mae_o]
        write_header = not metrics_csv.exists()
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print(f"[WARN] Ghi metrics CSV lỗi: {e}")

    # ===== Cập nhật JSON symbol đạt chuẩn trong TARGET_3Y/<target> =====
    mase_ref  = mase_o if (mase_o is not None and np.isfinite(mase_o)) else mase_s
    theil_ref = theil_o if (theil_o is not None and np.isfinite(theil_o)) else theil_s
    mae_ref   = mae_o if (mae_o is not None and np.isfinite(mae_o)) else mae_s

    if np.isfinite(mase_ref) and mase_ref < MASE_THRESHOLD:
        update_selected_symbols_json(out_dir, target_name, INTERVAL, WINDOW_SIZE,
                                     symbol, mase_ref, theil_ref, mae_ref)
        print(f"✅ {symbol} đạt chuẩn MASE < {MASE_THRESHOLD:.2f} -> cập nhật {SELECTED_JSON_NAME}")
    else:
        remove_symbol_from_selected_json(out_dir, symbol)
        print(f"⛔ {symbol} KHÔNG đạt chuẩn MASE < {MASE_THRESHOLD:.2f} (mase={mase_ref:.6f}).")

def load_symbols():
    p = Path("symbols.txt")
    if p.exists():
        try:
            return [s.strip().upper() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            print(f"❌ Lỗi đọc symbols.txt: {e}")
    print("⚠️ Không thấy symbols.txt (mỗi dòng 1 mã, ví dụ BTCUSDT)."); return []

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output root: {OUT_ROOT}")
    print(f"🔒 Target cố định: {TARGET_NAME} | INTERVAL={INTERVAL} | WINDOW_SIZE={WINDOW_SIZE} | BS={BATCH_SIZE}")

    for sym in load_symbols():
        try:
            train_symbol_for_target(sym, TARGET_NAME)
        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện {sym} / {TARGET_NAME}: {e}")

    print("✅ Done.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
