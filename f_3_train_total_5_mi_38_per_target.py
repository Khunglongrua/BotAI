# -*- coding: utf-8 -*-
"""
F3_train_total5_mi38_per_target.py  (per-target classification, clean version)

- Input: parquet từ F3_apply_selected_features_total5_mi38.py (label 5 lớp 0..4).
- Khi tạo cửa sổ trượt để huấn luyện: chỉ dùng nhãn {0,2,4} và map {0,2,4}->{0,1,2}.
- Loss: Cross-Entropy + ordinal penalty (λ=0.2).
- Kiến trúc: LSTM gọn.
- Không dùng focal, không class_weight, không flags phụ.

Chạy:
    python f_3_train_total_5_mi_38_per_target.py
"""

import os, math, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

# ===== GPU setup (tối giản) =====
try:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass
mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

# ===== Paths & basic params =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "DATA_BINANCE_TOTAL5"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", BASE_DIR / "TARGET_3Y"))

INTERVAL     = os.environ.get("INTERVAL", "15m")
TARGET_NAME  = os.environ.get("TARGET", "Y_10")   # Y_5 / Y_10 / Y_15
WINDOW_SIZE  = int(os.environ.get("WINDOW_SIZE", 64))
WINDOW_STEP  = int(os.environ.get("WINDOW_STEP", 1))

# Khai thác tối đa 4090 như bạn muốn: 128 * 10 (có thể đổi qua ENV)
BATCH_X      = int(os.environ.get("BATCH_X", 128))
BATCH_MULT   = int(os.environ.get("BATCH_MULT", 10))
BATCH_SIZE   = BATCH_X * BATCH_MULT

EPOCHS       = int(os.environ.get("EPOCHS", 80))
PATIENCE     = int(os.environ.get("PATIENCE", 22))
DROPOUT      = float(os.environ.get("DROPOUT", 0.25))
LR_INIT      = float(os.environ.get("LR_INIT", 1.6e-3))
MIN_LR_RATIO = 0.10
WARMUP_FRAC  = 0.10
WEIGHT_DECAY = 1e-5
CLIPNORM     = 5.0

# ===== Warmup–Cosine schedule =====
@tf.keras.utils.register_keras_serializable(package="custom")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr0: float, total_steps: int, warmup_steps: int, min_lr: float):
        super().__init__()
        self.lr0 = float(lr0)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr0 = tf.cast(self.lr0, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)
        total = tf.cast(tf.maximum(1, self.total_steps), tf.float32)
        warm  = tf.cast(tf.maximum(0, self.warmup_steps), tf.float32)
        # linear warmup
        lr_warm = lr0 * (step + 1.0) / tf.maximum(1.0, warm)
        # cosine decay after warmup
        step_after  = tf.maximum(0.0, step - warm)
        total_after = tf.maximum(1.0, total - warm)
        cosine = 0.5 * (1.0 + tf.cos(math.pi * (step_after / total_after)))
        lr_decay = min_lr + (lr0 - min_lr) * cosine
        return tf.where(step < warm, lr_warm, lr_decay)

    def get_config(self):
        return {
            "lr0": float(self.lr0),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": float(self.min_lr),
        }

class EpochLogger(Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = int(total_epochs)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr_fn = self.model.optimizer.learning_rate
        try:
            step = tf.keras.backend.get_value(self.model.optimizer.iterations)
            lr_val = float(tf.keras.backend.get_value(lr_fn(step))) if callable(lr_fn) else float(tf.keras.backend.get_value(lr_fn))
        except Exception:
            try:
                lr_val = float(lr_fn.numpy())
            except Exception:
                lr_val = float(lr_fn)
        loss = logs.get("loss", float("nan"))
        vloss = logs.get("val_loss", float("nan"))
        acc  = logs.get("sparse_categorical_accuracy", float("nan"))
        vacc = logs.get("val_sparse_categorical_accuracy", float("nan"))
        print(f"Epoch {epoch+1}/{self.total_epochs} - loss: {loss:.6f} - acc: {acc:.4f} - val_loss: {vloss:.6f} - val_acc: {vacc:.4f} - lr: {lr_val:.6f}")

def split_train_val_test(N: int, train=0.60, val=0.20) -> Tuple[int, int]:
    """Trả về chỉ số kết thúc train và kết thúc val (mở đầu test)."""
    return int(train * N), int((train + val) * N)

def create_windows_X(X_scaled: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    N = len(X_scaled)
    if N <= window:
        return np.empty((0, window, X_scaled.shape[1]), dtype=np.float32)
    idx = range(0, N - window, step)
    return np.asarray([X_scaled[i:i + window, :] for i in idx], dtype=np.float32)

def create_windows_y_class(y_idx: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Lấy nhãn tại cuối cửa sổ."""
    yy = y_idx.reshape(-1)
    N = len(yy)
    if N <= window:
        return np.empty((0,), dtype=np.int32)
    idx = range(0, N - window, step)
    return np.asarray([yy[i + (window - 1)] for i in idx], dtype=np.int32)

def sanitize_array(arr, fill=0.0):
    arr = np.asarray(arr)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        arr[bad] = fill
    return arr

# ==== Giữ 0-2-4 và map → 0-1-2 (cố định) ====
LBL_KEEP = np.array([0, 2, 4], dtype=np.int32)
LBL_MAP  = {0: 0, 2: 1, 4: 2}

def filter_and_map_extremes(X_w: np.ndarray, y_w: np.ndarray):
    mask = np.isin(y_w, LBL_KEEP)
    X_f = X_w[mask]
    y_f = y_w[mask]
    if len(y_f) == 0:
        return X_f, y_f
    y_m = np.vectorize(LBL_MAP.get, otypes=[np.int32])(y_f).astype(np.int32)
    return X_f, y_m

# ===== Data IO =====
def _load_and_concat_symbol_files(symbol: str) -> pd.DataFrame:
    pats = [
        f"{symbol}_{INTERVAL}_feat_mi*_APPLIED.parquet",
        f"{symbol}_*_feat_mi*_APPLIED.parquet",
    ]
    paths = []
    for pat in pats:
        paths.extend(sorted(DATA_DIR.glob(pat)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy parquet TOTAL5 cho {symbol} trong {DATA_DIR}")

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
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ===== Feature/Label extraction =====
def extract_XY_per_target(df: pd.DataFrame, target_name: str):
    df_num = df.select_dtypes(include=["number"]).copy()
    for c in list(df_num.columns):
        lc = str(c).lower()
        if ("time" in lc) or ("timestamp" in lc) or ("date" in lc) or (lc in {"open_time", "close_time"}):
            df_num.drop(columns=c, inplace=True, errors="ignore")

    if target_name not in df_num.columns:
        raise KeyError(f"Không thấy cột target '{target_name}' trong dữ liệu.")

    y_cols = [c for c in df_num.columns if str(c).upper().startswith("Y_")]
    y_df = df_num[[target_name]].copy()
    x_df = df_num.drop(columns=y_cols, errors="ignore")

    y_vals = y_df.values.reshape(-1).astype(np.int32)
    return x_df.values.astype(np.float32), y_vals

# ===== Model (LSTM gọn) =====
def build_model(input_shape, n_classes=3, dropout=DROPOUT):
    x_in = Input(shape=input_shape, dtype=tf.float32, name="X_seq")
    t, f = input_shape
    proj = ((f + 7) // 8) * 8
    x = x_in if proj == f else TimeDistributed(Dense(proj, use_bias=False, dtype=tf.float32), name="project_to_8")(x_in)
    x = LSTM(512, return_sequences=True, name="lstm_512")(x); x = Dropout(dropout)(x)
    x = LSTM(256, return_sequences=True, name="lstm_256")(x); x = Dropout(dropout)(x)
    x = LSTM(128, return_sequences=False, name="lstm_128")(x); x = Dropout(dropout)(x)
    logits = Dense(n_classes, dtype=tf.float32, name="logits")(x)
    return Model(x_in, logits, name="lstm_cls3")

# ===== Optimizer =====
def build_optimizer(n_train_samples: int, epochs: int, lr0: float = LR_INIT,
                    min_lr_ratio: float = MIN_LR_RATIO, warmup_frac: float = WARMUP_FRAC):
    steps_per_epoch = max(1, math.ceil(n_train_samples / float(BATCH_SIZE)))
    total_steps  = max(1, steps_per_epoch * int(epochs))
    warmup_steps = int(round(total_steps * float(warmup_frac)))
    min_lr = float(lr0) * float(min_lr_ratio)
    sched = WarmupCosine(lr0=float(lr0), total_steps=int(total_steps),
                         warmup_steps=int(warmup_steps), min_lr=float(min_lr))
    return AdamW(learning_rate=sched, weight_decay=float(WEIGHT_DECAY), clipnorm=float(CLIPNORM))

# ===== TF Datasets =====
def make_train_ds(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.cache().shuffle(min(len(X), 8000), seed=42, reshuffle_each_iteration=True)\
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_val_ds(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ===== Loss: CE + Ordinal penalty =====
LAMBDA_ORDINAL = 0.2
def ce_with_ordinal_penalty(n_classes: int, lam: float):
    """CE + penalty theo khoảng cách |j - y| kỳ vọng dưới p."""
    sce = SparseCategoricalCrossentropy(from_logits=True)
    cls_idx = tf.range(n_classes, dtype=tf.float32)[None, :]
    def loss(y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        ce = sce(y_true, logits)
        p = tf.nn.softmax(logits)
        yt = tf.cast(y_true, tf.float32)[:, None]
        dist = tf.abs(cls_idx - yt)
        exp_dist = tf.reduce_mean(tf.reduce_sum(dist * p, axis=1))
        return ce + lam * exp_dist
    return loss

# ===== Train 1 symbol / 1 target =====
def train_symbol_for_target(symbol: str, target_name: str):
    out_dir = OUT_ROOT / target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n====== Train {symbol} ({INTERVAL}) ws={WINDOW_SIZE} bs={BATCH_SIZE} | target={target_name} ======")

    df = _load_and_concat_symbol_files(symbol)
    X_raw, y_idx = extract_XY_per_target(df, target_name=target_name)

    N = len(X_raw)
    if N <= WINDOW_SIZE + 1:
        print(f"⚠️ {symbol}:{target_name}: quá ngắn (N={N}) → bỏ qua.")
        return

    tr_end, val_end = split_train_val_test(N, train=0.60, val=0.20)
    X_tr_raw, y_tr = X_raw[:tr_end],          y_idx[:tr_end]
    X_val_raw, y_val = X_raw[tr_end:val_end], y_idx[tr_end:val_end]
    X_te_raw,  y_te  = X_raw[val_end:],       y_idx[val_end:]

    sx = RobustScaler().fit(X_tr_raw)
    joblib.dump(sx, out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{target_name.lower()}_robust_X.pkl")

    X_tr = sanitize_array(sx.transform(X_tr_raw))
    X_val = sanitize_array(sx.transform(X_val_raw))
    X_te  = sanitize_array(sx.transform(X_te_raw))

    X_tr_w = create_windows_X(X_tr,  WINDOW_SIZE, WINDOW_STEP)
    X_val_w = create_windows_X(X_val, WINDOW_SIZE, WINDOW_STEP)
    X_te_w  = create_windows_X(X_te,  WINDOW_SIZE, WINDOW_STEP)
    y_tr_w  = create_windows_y_class(y_tr, WINDOW_SIZE, WINDOW_STEP)
    y_val_w = create_windows_y_class(y_val, WINDOW_SIZE, WINDOW_STEP)
    y_te_w  = create_windows_y_class(y_te,  WINDOW_SIZE, WINDOW_STEP)

    # Filter extremes và map → 0/1/2
    X_tr_w, y_tr_w = filter_and_map_extremes(X_tr_w, y_tr_w)
    X_val_w, y_val_w = filter_and_map_extremes(X_val_w, y_val_w)
    X_te_w,  y_te_w  = filter_and_map_extremes(X_te_w,  y_te_w)

    if len(y_tr_w) == 0 or len(y_val_w) == 0 or len(y_te_w) == 0:
        print(f"⛔ {symbol}:{target_name}: không đủ mẫu sau khi lọc extremes. Bỏ qua.")
        return

    # In phân bố nhãn
    for name, y_arr in {"train": y_tr_w, "val": y_val_w, "test": y_te_w}.items():
        u, c = np.unique(y_arr, return_counts=True)
        print(f"[{name}] label counts:", dict(zip(u.tolist(), c.tolist())))

    ds_train = make_train_ds(X_tr_w, y_tr_w)
    ds_val   = make_val_ds(X_val_w, y_val_w)

    model = build_model((WINDOW_SIZE, X_tr_w.shape[2]), n_classes=3, dropout=DROPOUT)
    opt   = build_optimizer(n_train_samples=len(X_tr_w), epochs=EPOCHS, lr0=LR_INIT,
                            min_lr_ratio=MIN_LR_RATIO, warmup_frac=WARMUP_FRAC)

    loss_fn = ce_with_ordinal_penalty(3, LAMBDA_ORDINAL)
    model.compile(optimizer=opt, loss=loss_fn,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ckpt_path = out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{target_name.lower()}_best.keras"
    callbacks = [
        ModelCheckpoint(filepath=str(ckpt_path),
                        monitor="val_sparse_categorical_accuracy",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-5, patience=PATIENCE,
                      restore_best_weights=False,
                      start_from_epoch=int(EPOCHS * WARMUP_FRAC) + 5,  # đúng yêu cầu của bạn
                      verbose=1),
        EpochLogger(EPOCHS),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=0)

    # ===== Evaluate =====
    logits = model.predict(X_te_w, verbose=0, batch_size=BATCH_SIZE)
    preds  = np.argmax(logits, axis=1).astype(int)

    acc  = accuracy_score(y_te_w, preds)
    bacc = balanced_accuracy_score(y_te_w, preds)
    macro_f1 = f1_score(y_te_w, preds, average='macro')
    per_class_f1 = f1_score(y_te_w, preds, average=None, labels=[0,1,2])
    cm = confusion_matrix(y_te_w, preds, labels=[0,1,2])

    print("\n=== CLASSIFICATION METRICS ===")
    print(f"Symbol: {symbol} | Target: {target_name}")
    print(f"  Acc  : {acc:.4f} | BAcc: {bacc:.4f} | MacroF1: {macro_f1:.4f}")
    print(f"  F1 per class [0..2]: {np.round(per_class_f1,4).tolist()}")

    # Ghi metrics CSV
    try:
        import csv
        metrics_csv = out_dir / f"metrics_{target_name}.csv"
        header = ["timestamp","symbol","interval","window_size","target",
                  "accuracy","balanced_accuracy","macro_f1","f1_c0","f1_c1","f1_c2"]
        row = [int(time.time()), symbol, INTERVAL, WINDOW_SIZE, target_name,
               acc, bacc, macro_f1, *per_class_f1.tolist()]
        write_header = not metrics_csv.exists()
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print(f"[WARN] Ghi metrics CSV lỗi: {e}")

    # Confusion matrix CSV
    try:
        cm_path = out_dir / f"{symbol}_{INTERVAL}_ws{WINDOW_SIZE}_{target_name.lower()}_cm.csv"
        pd.DataFrame(cm, index=["t0","t1","t2"], columns=["p0","p1","p2"]).to_csv(cm_path)
    except Exception as e:
        print(f"[WARN] Ghi confusion matrix lỗi: {e}")

# ===== Symbols loader =====
def load_symbols():
    p = Path("symbols.txt")
    if p.exists():
        try:
            return [s.strip().upper() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            print(f"❌ Lỗi đọc symbols.txt: {e}")
    print("⚠️ Không thấy symbols.txt (mỗi dòng 1 mã, ví dụ BTCUSDT).")
    return []

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output root: {OUT_ROOT}")
    print(f"🔒 Target: {TARGET_NAME} | INTERVAL={INTERVAL} | WINDOW_SIZE={WINDOW_SIZE} | BS={BATCH_SIZE}")

    for sym in load_symbols():
        try:
            train_symbol_for_target(sym, TARGET_NAME)
        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện {sym} / {TARGET_NAME}: {e}")

    print("✅ Done.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
