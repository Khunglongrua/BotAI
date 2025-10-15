# -*- coding: utf-8 -*-
"""
HPO Keras/LSTM cho phân loại 5 lớp Y_k (k ∈ {5,10,15}) trên bộ FEAT30 (30 đặc trưng).
- KHÔNG bin lại; dùng trực tiếp nhãn 0..4: Y_5 | Y_10 | Y_15
- Giữ flow giống bản cls3 cũ: Optuna + ASHA + cosine LR + early stopping + pruning.

Yêu cầu: pip install tensorflow==2.15.0 optuna pandas numpy scikit-learn pyarrow
"""

import os, math, json, random, argparse, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from tensorflow.keras.optimizers import AdamW
except Exception:
    from tensorflow.keras.optimizers import Adam as AdamW  # fallback

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import TFKerasPruningCallback

# ===== Setup =====
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "DATA_BINANCE_FEAT30"
DEFAULT_OUT  = BASE_DIR / "CLS5_HPO_LSTM"

mixed_precision.set_global_policy("mixed_float16")
tf.get_logger().setLevel("ERROR")
SEED = 123
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

for g in tf.config.experimental.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
try: tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception: pass

WARMUP_FRAC  = 0.10
MIN_LR_RATIO = 0.10

# ===== File pickers (naming cho FEAT30) =====
def _pick_parquets(data_folder: Path, symbol: str, interval: str):
    """
    FEAT30 output từ F3: <SYMBOL>_15m_feat_top<xx>_15m_3y.parquet
    """
    patterns = [
        f"{symbol}_{interval}_feat_top*_{interval}_3y.parquet",
        f"{symbol}_{interval}_feat_top*.parquet",  # nới lỏng để tương thích biến thể
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(data_folder.glob(pat)))
    return [p for p in files if p.exists()]

def _load_concat(data_folder: Path, symbol: str, interval: str) -> pd.DataFrame:
    paths = _pick_parquets(data_folder, symbol, interval)
    if not paths:
        raise FileNotFoundError(f"Không thấy parquet của {symbol} trong {data_folder}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            if "open_time" not in d.columns:
                continue
            dfs.append(d)
        except Exception as e:
            logging.warning(f"Bỏ qua {p.name}: {e}")
    if not dfs:
        raise FileNotFoundError(f"{symbol}: đọc được 0 file hợp lệ.")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True)
    return df

# ===== Chuẩn bị X, y =====
def extract_Xy(df: pd.DataFrame, target_name: str):
    """
    - X: giữ numeric, bỏ 'open_time', mọi cột bắt đầu 'Y_' và mọi *_log/*_pct
    - y: cột target_name (0..4, int)
    """
    num = df.select_dtypes(include=["number"]).copy()
    # bỏ thời gian nếu convert thành số ở đâu đó
    for c in list(num.columns):
        lc = str(c).lower()
        if ("time" in lc) or ("timestamp" in lc) or (lc in {"open_time", "close_time"}):
            num.drop(columns=c, inplace=True, errors="ignore")

    if target_name not in df.columns:
        raise KeyError(f"Thiếu cột target '{target_name}' (Y_5 | Y_10 | Y_15).")

    # loại mọi target khác + *_log/*_pct khỏi X
    drop_cols = [c for c in num.columns if c.upper().startswith("Y_") or c.endswith("_log") or c.endswith("_pct")]
    X = num.drop(columns=drop_cols, errors="ignore").astype(np.float32).values
    y = df[target_name].astype(np.int32).values
    return X, y

def time_split_index(n_rows: int, ratios=(0.6, 0.2, 0.2)):
    tr_end = int(n_rows * ratios[0])
    va_end = int(n_rows * (ratios[0] + ratios[1]))
    return tr_end, va_end

def create_windows_X(X_scaled: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    N = len(X_scaled)
    if N <= window: return np.empty((0, window, X_scaled.shape[1]), dtype=np.float32)
    idx = range(0, N - window + 1, step)
    return np.asarray([X_scaled[i:i + window, :] for i in idx], dtype=np.float32)

def create_windows_y(y_idx: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    yy = y_idx.reshape(-1); N = len(yy)
    if N <= window: return np.empty((0,), dtype=np.int32)
    idx = range(0, N - window + 1, step)
    return np.asarray([yy[i + (window - 1)] for i in idx], dtype=np.int32)

# ===== Model & Loss =====
def make_cosine_warmup_lrfn(base_lr: float, total_epochs: int, warmup_frac: float, min_lr_ratio: float):
    warmup_epochs = int(max(0, round(total_epochs * warmup_frac)))
    min_lr = float(base_lr) * float(min_lr_ratio)
    def lrfn(epoch, _cur):
        if epoch < warmup_epochs:
            return float(base_lr) * (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return float(min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t)))
    return lrfn

def ce_with_ordinal_penalty(n_classes: int, lam: float):
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cls_idx = tf.range(n_classes, dtype=tf.float32)[None, :]  # shape (1,C)
    def loss(y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        ce = sce(y_true, logits)
        p = tf.nn.softmax(logits)
        yt = tf.cast(y_true, tf.float32)[:, None]              # (B,1)
        dist = tf.abs(cls_idx - yt)                            # (B,C)
        ord_pen = tf.reduce_mean(tf.reduce_sum(dist * p, axis=1))
        return ce + float(lam) * ord_pen
    return loss

def build_model(input_shape, n_classes=5,
                lstm1=512, lstm2=256, lstm3=128,
                dropout=0.15, project_to_mult8=True):
    x_in = Input(shape=input_shape, dtype=tf.float32, name="X_seq")
    t, f = input_shape
    if project_to_mult8:
        proj = ((f + 7) // 8) * 8
        x = x_in if proj == f else TimeDistributed(Dense(proj, use_bias=False, dtype=tf.float32), name="proj8")(x_in)
    else:
        x = x_in
    x = LSTM(int(lstm1), return_sequences=True, name="lstm1")(x); x = Dropout(float(dropout))(x)
    x = LSTM(int(lstm2), return_sequences=True, name="lstm2")(x); x = Dropout(float(dropout))(x)
    x = LSTM(int(lstm3), return_sequences=False, name="lstm3")(x); x = Dropout(float(dropout))(x)
    logits = Dense(n_classes, dtype=tf.float32, name="logits")(x)
    return tf.keras.Model(x_in, logits, name="lstm_cls5_hpo")

# ===== Optuna Objective =====
def make_objective(args, epochs: int):
    data_folder = Path(args.data_folder)
    symbol      = args.symbol
    interval    = args.interval
    target_name = args.target

    def objective(trial: optuna.Trial):
        try:
            # --- Search space ---
            window_size = trial.suggest_int("window_size", 64, 256, step=16)
            batch_size  = trial.suggest_int("batch_size", 256, 1024, step=64)
            batch_size  = max(32, (int(batch_size) // 32) * 32)

            lstm1 = trial.suggest_int("lstm1", 256, 768, step=64)
            lstm2 = trial.suggest_int("lstm2", 128, 512, step=64)
            lstm3 = trial.suggest_int("lstm3",  64, 256, step=32)
            dropout = trial.suggest_float("dropout", 0.05, 0.40)
            lam_ord = trial.suggest_float("lambda_ordinal", 0.0, 0.40)

            base_lr = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)

            # --- Load & split ---
            df = _load_concat(data_folder, symbol, interval)
            X_raw, y_idx = extract_Xy(df, target_name)

            N = len(X_raw)
            if N <= window_size + 10:
                raise optuna.TrialPruned(f"N quá nhỏ (N={N}) cho window={window_size}")

            tr_end_idx, val_end_idx = time_split_index(N, ratios=(0.60, 0.20, 0.20))
            X_tr_raw, y_tr = X_raw[:tr_end_idx],            y_idx[:tr_end_idx]
            X_val_raw, y_val = X_raw[tr_end_idx:val_end_idx], y_idx[tr_end_idx:val_end_idx]

            scaler = RobustScaler().fit(X_tr_raw)
            X_tr = scaler.transform(X_tr_raw).astype(np.float32)
            X_val = scaler.transform(X_val_raw).astype(np.float32)

            X_tr_w  = create_windows_X(X_tr,  window_size, step=1)
            y_tr_w  = create_windows_y(y_tr,  window_size, step=1)
            X_val_w = create_windows_X(X_val, window_size, step=1)
            y_val_w = create_windows_y(y_val, window_size, step=1)

            # Class weights (5 lớp)
            classes = np.arange(5)
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_w)
            cw_map = {int(c): float(w) for c, w in zip(classes, cw)}

            # Datasets
            train_ds = tf.data.Dataset.from_tensor_slices((X_tr_w, y_tr_w))\
                        .cache().shuffle(min(len(X_tr_w), 8000), seed=SEED, reshuffle_each_iteration=True)\
                        .batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)
            valid_ds = tf.data.Dataset.from_tensor_slices((X_val_w, y_val_w))\
                        .cache().batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)

            # Model
            model = build_model((window_size, X_tr_w.shape[2]),
                                n_classes=5, lstm1=lstm1, lstm2=lstm2, lstm3=lstm3,
                                dropout=dropout, project_to_mult8=True)

            loss_fn = ce_with_ordinal_penalty(5, lam_ord)
            opt = AdamW(learning_rate=float(base_lr), weight_decay=float(weight_decay))
            model.compile(optimizer=opt, loss=loss_fn,
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

            cb = [
                EarlyStopping(monitor="val_loss", mode="min", patience=10,
                              restore_best_weights=True,
                              start_from_epoch=int(epochs * WARMUP_FRAC) + 5, verbose=0),
                TFKerasPruningCallback(trial, monitor="val_loss"),
                LearningRateScheduler(make_cosine_warmup_lrfn(base_lr, epochs, WARMUP_FRAC, MIN_LR_RATIO), verbose=0),
            ]

            model.fit(train_ds, validation_data=valid_ds, epochs=epochs, verbose=0, callbacks=cb)

            # VAL metrics
            val_logits = model.predict(X_val_w, verbose=0, batch_size=batch_size)
            val_pred   = np.argmax(val_logits, axis=1).astype(int)
            bacc = balanced_accuracy_score(y_val_w, val_pred)
            macro_f1 = f1_score(y_val_w, val_pred, average="macro")
            acc = accuracy_score(y_val_w, val_pred)

            trial.set_user_attr("val_bacc", float(bacc))
            trial.set_user_attr("val_macro_f1", float(macro_f1))
            trial.set_user_attr("val_acc", float(acc))
            trial.set_user_attr("feat_dim", int(X_tr_w.shape[2]))
            trial.set_user_attr("train_windows", int(len(X_tr_w)))
            trial.set_user_attr("val_windows", int(len(X_val_w)))

            return float(bacc)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logging.exception(f"Trial {trial.number} ERROR: {e}")
            return 0.0
        finally:
            tf.keras.backend.clear_session()
    return objective

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--target", type=str, default="Y_10", help="Y_5 | Y_10 | Y_15")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--study_name", type=str, default="cls5_bacc_max")
    parser.add_argument("--storage", type=str, default=None, help='VD "sqlite:///optuna.db" để lưu study')
    args = parser.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=10)
    pruner  = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)

    study = optuna.create_study(
        study_name=args.study_name, direction="maximize",
        sampler=sampler, pruner=pruner,
        storage=args.storage, load_if_exists=bool(args.storage)
    )

    objective = make_objective(args, epochs=args.epochs)
    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)

    best = study.best_trial
    best_params = best.params.copy()
    for k in ["batch_size", "window_size", "lstm1", "lstm2", "lstm3"]:
        if k in best_params: best_params[k] = int(best_params[k])

    out = {
        "symbol": args.symbol,
        "interval": args.interval,
        "target": args.target,
        "best_params": best_params,
        "score": float(best.value),
        "score_metric": "val_bacc",
        "cosine_schedule": {"warmup_frac": WARMUP_FRAC, "min_lr_ratio": MIN_LR_RATIO, "epochs": args.epochs},
        "study_name": args.study_name,
        "trials": args.trials,
        "notes": "Data in DATA_BINANCE_FEAT30, pattern *_feat_top*_15m_3y.parquet",
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"{args.symbol}_{args.interval}_{args.target}_best_params_cls5_asha_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Best val_bacc: {best.value:.6f}")
    print(f"Saved best params: {out_path}")
    print(json.dumps(best_params, indent=2))

if __name__ == "__main__":
    main()
