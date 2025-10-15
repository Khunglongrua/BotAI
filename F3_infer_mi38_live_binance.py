# -*- coding: utf-8 -*-
"""
F3_infer_mi38_live_binance.py (Simplified)
---------------------------------
Dự báo *live* trên Binance cho mô hình residual-logs (Y_5, Y_10, Y_15)
• BỎ HẾT chức năng "cắm cờ" (flags/CLI/ENV).
• KHÔNG dùng argparse, KHÔNG đọc ENV.
• Chỉ dùng CÁC FILE CỤ THỂ đặt CÙNG THƯ MỤC với script:
   - symbols:        ./symbols.txt  (mỗi dòng một symbol, ví dụ BTCUSDT)
   - feature list:   ./selected_features_mi38.json  (key: "features": [...])
   - discord webhook (tuỳ chọn): ./discord.txt (một dòng URL). Nếu thiếu file -> không gửi.
   - artifacts (model & scaler): ./LSTM_F3_mi38_{WS}/{SYMBOL}_{INTERVAL}_ws{WS}_*.{keras,pkl}
• Các tham số cấu hình cố định nằm trong mục HẰNG SỐ bên dưới.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os, time, json

import numpy as np
import pandas as pd
import requests
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model

# ============= GPU tweaks (an toàn) =============
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

# ============= Hằng số & Đường dẫn =============
BASE_DIR = Path(__file__).resolve().parent
BINANCE_BASE = "https://api.binance.com"

# Cấu hình CỐ ĐỊNH (chỉnh tại đây nếu cần)
INTERVAL = "15m"   # chỉ dùng một interval
WS       = 64       # window size khớp khi train
BARS     = 3000     # số nến kéo về để tạo cửa sổ
THR      = 0.30     # ngưỡng |y5|% để ra tín hiệu
BODY_TH  = 0.0      # lọc thân nến tối thiểu (%); đặt 0 nếu không dùng
COMPOSE  = "exp"    # "exp" (chuẩn) hoặc "add" (xấp xỉ cộng phần trăm)
BATCH_SZ = 256
EMA_SPAN = 20

# Đường dẫn FILE CỤ THỂ (cùng cấp)
SYMBOLS_TXT   = BASE_DIR / "symbols.txt"
FEATURES_JSON = BASE_DIR / "selected_features_mi38.json"
DISCORD_TXT   = BASE_DIR / "discord.txt"   # nếu không tồn tại -> không gửi

# OUT_DIR chứa model + scaler đã train (đồng bộ với pipeline F3_mi38)
OUT_DIR = BASE_DIR / f"LSTM_F3_mi38_{WS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUT_DIR / "infer_logs_live"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============= Utilities =============

def _print(*args, **kwargs):
    print(*args, flush=True, **kwargs)

# ---------- Discord ----------

def _read_text_if_exists(p: Path) -> Optional[str]:
    try:
        if p.exists():
            s = p.read_text(encoding="utf-8").strip()
            return s or None
    except Exception:
        pass
    return None

def resolve_webhook() -> Optional[str]:
    """Chỉ đọc ./discord.txt. Không dùng ENV/CLI."""
    if DISCORD_TXT.exists():
        s = _read_text_if_exists(DISCORD_TXT)
        return s
    return None

def post_discord(webhook: str, payload: Dict) -> None:
    try:
        r = requests.post(webhook, json=payload, timeout=10)
        _print(f"Discord {r.status_code}")
    except Exception as e:
        _print(f"Discord error: {e}")

# ---------- Binance ----------

def fetch_binance_klines(symbol: str, interval: str = "15m",
                         limit: int = 1500, max_bars: int = 3000) -> pd.DataFrame:
    """Kéo lịch sử nến lùi dần theo endTime để ghép đủ ~max_bars (tối đa ~3000)."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    frames = []
    end_ts = int(time.time() * 1000)
    got = 0
    while got < max_bars:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, max_bars - got),
            "endTime": end_ts,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            break
        df = pd.DataFrame(arr, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df = df[["open_time","open","high","low","close","volume"]].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)
        got += len(df)
        # set end_ts = ms của nến đầu tiên - 1ms
        first_ms = int(df["open_time"].iloc[0].timestamp() * 1000)
        end_ts = first_ms - 1
        if len(df) <= 1:
            break
    if not frames:
        raise RuntimeError("Không lấy được dữ liệu từ Binance")
    full = pd.concat(frames, ignore_index=True).dropna()
    return full.sort_values("open_time").reset_index(drop=True)

# ---------- Feature engineering (TA + growth logs) ----------
import ta

def add_all_ta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ta.add_all_ta_features(out, open="open", high="high", low="low",
                                 close="close", volume="volume", fillna=False)
    return out.replace([np.inf, -np.inf], np.nan)


def add_growth_logs(df: pd.DataFrame, close_col: str = "close", ema_span: int = EMA_SPAN) -> pd.DataFrame:
    d = df.copy()
    # g_raw_now (log-diff), đồng thời tạo thêm alias *_log để tương thích các bộ tên cũ
    g_raw_log = np.log(d[close_col]) - np.log(d[close_col].shift(1))
    d["g_raw_now"] = g_raw_log
    d["g_raw_now_log"] = g_raw_log
    # g_ema_now (log-diff trên EMA)
    ema_col = f"ema{ema_span}"
    d[ema_col] = d[close_col].ewm(span=ema_span, adjust=False).mean()
    g_ema_log = np.log(d[ema_col]) - np.log(d[ema_col].shift(1))
    d[f"g_ema_now_{ema_span}"] = g_ema_log
    d[f"g_ema_now_log_{ema_span}"] = g_ema_log
    return d

# ---------- Artifacts & chuẩn hoá ----------
N_TARGETS = 3
TARGET_NAMES = ["Y_5", "Y_10", "Y_15"]


def load_artifacts(symbol: str) -> Tuple[Path, Path, Path]:
    model_path = OUT_DIR / f"{symbol}_{INTERVAL}_ws{WS}_best.keras"
    sx_path    = OUT_DIR / f"{symbol}_{INTERVAL}_ws{WS}_robust_X.pkl"
    sy_path    = OUT_DIR / f"{symbol}_{INTERVAL}_ws{WS}_robust_Y_t{N_TARGETS}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Thiếu model: {model_path.name}")
    if not sx_path.exists():
        raise FileNotFoundError(f"Thiếu scaler X: {sx_path.name}")
    if not sy_path.exists():
        raise FileNotFoundError(f"Thiếu scaler Y: {sy_path.name}")
    return model_path, sx_path, sy_path


def load_feature_list() -> List[str]:
    """Đọc JSON cố định: FEATURES_JSON với key "features"."""
    if not FEATURES_JSON.exists():
        raise FileNotFoundError(f"Không thấy feature JSON: {FEATURES_JSON}")
    obj = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    feats = obj.get("features") or obj.get("cols") or obj.get("columns")
    if not isinstance(feats, list) or not feats:
        raise ValueError("selected_features_mi38.json không hợp lệ, thiếu mảng 'features'.")
    return [str(x) for x in feats]


def expected_feature_count(scaler) -> Optional[int]:
    # sklearn >=1.0 có n_features_in_
    n = getattr(scaler, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)
    # fallback cho RobustScaler: dùng độ dài scale_ / center_
    for attr in ("scale_", "center_", "center", "median_"):
        v = getattr(scaler, attr, None)
        if v is not None:
            try:
                return int(len(v))
            except Exception:
                pass
    return None


def complete_feature_list(feat_order: List[str], df: pd.DataFrame, expected_n: Optional[int]) -> List[str]:
    """Nếu scaler đòi 40 features mà JSON chỉ có 38 → tự bổ sung 2 cột tăng trưởng.
    Thứ tự bổ sung: g_raw_now, g_ema_now_{EMA_SPAN}, rồi tới biến *_log tương ứng.
    """
    if expected_n is None:
        return feat_order
    if len(feat_order) == expected_n:
        return feat_order
    if len(feat_order) > expected_n:
        raise ValueError(f"Danh sách features ({len(feat_order)}) > số cột scaler yêu cầu ({expected_n}). Kiểm tra JSON!")

    cands = [
        "g_raw_now",
        f"g_ema_now_{EMA_SPAN}",
        "g_raw_now_log",
        f"g_ema_now_log_{EMA_SPAN}",
    ]
    addables = [c for c in cands if (c in df.columns) and (c not in feat_order)]

    out = list(feat_order)
    for c in addables:
        if len(out) >= expected_n:
            break
        out.append(c)

    if len(out) != expected_n:
        raise ValueError(
            f"Không thể tự bổ sung để khớp scaler: cần {expected_n} cột, hiện có {len(out)}. "
            f"Hãy cung cấp JSON đúng 40 cột hoặc kiểm tra artifacts.")

    _print(f"ℹ️ Auto-append features → {addables[:max(0, expected_n-len(feat_order))]}")
    return out


def make_windows(X: np.ndarray, ws: int) -> np.ndarray:
    if len(X) < ws:
        return np.empty((0, ws, X.shape[1]), dtype=np.float32)
    return np.asarray([X[i:i+ws] for i in range(len(X) - ws + 1)], dtype=np.float32)

# ---------- Ra quyết định ----------

def decide_signal(y5: float, y10: float, y15: float, thr: float) -> str:
    if (y5 >= thr) and (y15 > y10 > y5):
        return "LONG"
    if (y5 <= -thr) and (y15 < y10 < y5):
        return "SHORT"
    return "HOLD"

# ---------- Symbols loader ----------

def _dedup_keep_order(lst: List[str]) -> List[str]:
    seen, out = set(), []
    for x in lst:
        x = str(x).strip().upper()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def load_symbol_list() -> List[str]:
    if SYMBOLS_TXT.exists():
        try:
            return _dedup_keep_order(SYMBOLS_TXT.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass
    return []

# ============= Inference core =============

def infer_live_once(symbol: str) -> Dict:
    t0 = time.time()

    # artifacts
    model_path, sx_path, sy_path = load_artifacts(symbol)
    model = load_model(model_path, compile=False)
    sx = joblib.load(sx_path)
    sy = joblib.load(sy_path)

    # data → TA → growth logs → chọn đúng cột theo feature_json
    raw = fetch_binance_klines(symbol, interval=INTERVAL, max_bars=BARS)
    df = add_all_ta(raw)
    df = add_growth_logs(df, ema_span=EMA_SPAN)

    # feature list từ JSON (tự bổ sung nếu scaler đòi nhiều hơn)
    feat_order = load_feature_list()
    expected_n = expected_feature_count(sx)
    feat_order = complete_feature_list(feat_order, df, expected_n)

    missing = [f for f in feat_order if f not in df.columns]
    if missing:
        raise KeyError(f"{symbol}: thiếu cột đặc trưng: {missing[:6]}{'...' if len(missing) > 6 else ''}")


    X_df = df[feat_order].replace([np.inf, -np.inf], np.nan)
    if X_df.isna().values.any():
        X_df = X_df.ffill().bfill()

    X_scaled = sx.transform(X_df.values.astype("float32"))
    X_win = make_windows(X_scaled, WS)
    if X_win.shape[0] == 0:
        raise RuntimeError(f"{symbol}: không đủ bars để tạo cửa sổ ws={WS}.")

    y_resid_log_scaled = model.predict(X_win, verbose=0, batch_size=BATCH_SZ)
    if y_resid_log_scaled.ndim != 2 or y_resid_log_scaled.shape[1] < 3:
        raise RuntimeError(f"{symbol}: output model không hợp lệ: {y_resid_log_scaled.shape}")

    y_resid_log_hat = sy.inverse_transform(y_resid_log_scaled).astype(float)  # (T,3)

    # Hợp nhất growth_now_log đúng index cửa sổ
    g_log = df["g_raw_now_log"].values.astype(float)
    g_log_for_windows = g_log[WS-1:]  # mỗi cửa sổ kết thúc tại vị trí này

    if COMPOSE == "add":
        y_resid_pct = (np.exp(y_resid_log_hat) - 1.0) * 100.0
        g_pct = (np.exp(g_log_for_windows).reshape(-1,1) - 1.0) * 100.0
        y_total_pct_all = y_resid_pct + g_pct
        y_total_log_all = np.log(1.0 + y_total_pct_all/100.0)
    else:  # "exp"
        y_total_log_all = y_resid_log_hat + g_log_for_windows.reshape(-1,1)
        y_total_pct_all = (np.exp(y_total_log_all) - 1.0) * 100.0

    # lấy phần tử cuối
    y_total_pct_last = y_total_pct_all[-1]
    y5, y10, y15 = map(float, y_total_pct_last)

    signal = decide_signal(y5, y10, y15, THR)
    reason = "rules"
    if BODY_TH > 0 and {"open","close"}.issubset(raw.columns):
        o, c = float(raw["open"].iloc[-1]), float(raw["close"].iloc[-1])
        if np.isfinite(o) and np.isfinite(c) and c != 0:
            body_pct = abs(c - o) / c * 100.0
            if body_pct < BODY_TH:
                signal = "HOLD"; reason = f"body%<{BODY_TH}"

    # in thông tin
    def fmt_pct3(arr: np.ndarray) -> str:
        a = [f"{float(x):+.3f}%" for x in arr]
        return " | ".join(a)

    _print("==========================")
    _print(f"Symbol  : {symbol}")
    _print(f"Interval: {INTERVAL} | ws={WS} | bars={BARS}")
    _print(f"Compose : {COMPOSE} | thr={THR:.2f}% | body_th={BODY_TH:.2f}%")
    _print(f"TOTAL % (y5|y10|y15): {fmt_pct3(y_total_pct_last)}")
    _print(f"→ SIGNAL: {signal} ({reason}) | {time.time()-t0:.3f}s")

    # log file
    now = time.strftime("%Y%m%d-%H%M%S")
    logline = {
        "ts": now, "symbol": symbol, "interval": INTERVAL, "ws": WS,
        "compose": COMPOSE, "thr": float(THR), "body_th": float(BODY_TH),
        "y_total_pct": [float(x) for x in y_total_pct_last],
        "signal": signal, "reason": reason,
    }
    with (LOG_DIR / f"infer_live_{symbol}_{INTERVAL}_ws{WS}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(logline, ensure_ascii=False) + "")

    # discord (nếu có file discord.txt)
    webhook = resolve_webhook()
    if webhook and signal in {"LONG","SHORT"}:
        price = float(raw["close"].iloc[-1]) if "close" in raw.columns else None
        emb = {
            "title": f"{signal} | {symbol}",
            "description": (
                f"Interval: **{INTERVAL}** | ws=**{WS}**"
                f"Compose: **{COMPOSE}** (live)"
            ),
            "color": 5814783 if signal == "LONG" else 15548997,
            "fields": [
                {"name": "y_total % (5|10|15)", "value": fmt_pct3(y_total_pct_last), "inline": False},
                {"name": "price", "value": f"{price:.4f}" if price is not None else "-", "inline": True},
                {"name": "thr | reason", "value": f"{THR:.2f}% | {reason}", "inline": True},
            ],
            "footer": {"text": f"F3_mi38 live • ws={WS}"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        post_discord(webhook, {"embeds": [emb]})

    return logline

# ============= Runner đơn giản =============

def main():
    syms = load_symbol_list()
    if not syms:
        _print("⚠️ Không có symbol trong ./symbols.txt")
        return

    _print(f"Symbols: {', '.join(syms)}")
    _print(f"Artifacts dir: {OUT_DIR}")
    _print(f"Discord: {'ON' if resolve_webhook() else 'OFF'}")

    for s in syms:
        try:
            infer_live_once(s)
        except Exception as e:
            _print(f"❌ {s}: {e}")

    _print("✅ Done.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
