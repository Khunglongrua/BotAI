# -*- coding: utf-8 -*-
"""
F3_infer_mi38_live_3y_any.py — Inference residual & tổng hợp % + Notification Discord/Telegram.

- Đọc symbols.txt (MASTER)
- Chỉ predict nếu symbol có trong TARGET_3Y/<target>/selected_symbols_mase_lt_0.75.json
- Kiểm tra đủ artifacts *_best.keras + *_robust_X.pkl
- Tự sinh 2 đặc trưng tăng trưởng close: g_raw_now_log, g_ema_now_log (+ alias)
- Khớp số cột X với scaler; log rõ ràng

Kết quả:
- y_resid_log  : phần dư (log-return) model trả ra
- y_total_pct  : % cuối cùng sau khi cộng g_raw_now_log và quy đổi *100

Ra quyết định:
- DECISION_SPACE = "PCT" hoặc "RESID_LOG"

Bổ sung bộ lọc:
- 3 EMA: LONG (ema9 > ema27 > ema72) | SHORT (ema9 < ema27 < ema72) — tính theo INTERVAL giao dịch
- body_strong đa khung: tính |close - open|/open * 100 với ngưỡng theo hướng LONG/SHORT
  * BODY_STRONG_TFS = ["15m", "30m", "1h"]
  * BODY_STRONG_REQUIRE_MODE = "ANY" hoặc "ALL"

Thông báo:
- SEND_DISCORD: on/off (webhook từ discord4.txt)
- SEND_TELEGRAM: on/off (token/chat_id từ ENV hoặc file telegram_token.txt / telegram_chat_id.txt)
  * Telegram dùng parse_mode="HTML" để hiển thị đẹp (bold, dấu đầu dòng, emoji).
"""

import os, time, json, requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import ta
from datetime import datetime, timezone

# ============= CẤU HÌNH =============
INTERVAL   = "15m"
WS         = 176
MAX_BARS   = 3000
BATCH_SZ   = 256

# Chọn miền so ngưỡng: "PCT" (khuyến nghị) hoặc "RESID_LOG"
DECISION_SPACE = "PCT"
THRESH_PCT     = 3      # ±3.00% (nếu DECISION_SPACE="PCT")
THRESH_RESID   = 0.30   # ±0.30 (log) (nếu DECISION_SPACE="RESID_LOG")

# Bật/tắt từng target
USE_Y5  = True
USE_Y10 = False
USE_Y15 = False

# ==== BỘ LỌC BỔ SUNG ====
REQUIRE_3EMA         = False
EMA_SPANS            = (9, 27, 72)   # ema9, ema27, ema72

REQUIRE_BODY_STRONG  = True
BODY_STRONG_PCT      = 0.5           # 0.3% theo yêu cầu ban đầu (tùy chỉnh)
BODY_STRONG_TFS      = ["15m"]
BODY_STRONG_REQUIRE_MODE = "ANY"

# ==== NOTIFY TOGGLES ====
SEND_DISCORD  = True
SEND_TELEGRAM = True

BASE_DIR    = Path(__file__).resolve().parent
SYMBOLS_TXT = BASE_DIR / "symbols.txt"
FEATURES_JS = BASE_DIR / "selected_features_mi38.json"
TARGET_ROOT = BASE_DIR / "TARGET_3Y"

# Discord webhook lưu trong file
DISCORD_TXT = BASE_DIR / "discord4.txt"

# Telegram creds: ENV ưu tiên, fallback file
TELEGRAM_TOKEN_TXT  = BASE_DIR / "telegram_token.txt"
TELEGRAM_CHATID_TXT = BASE_DIR / "telegram_chat_id.txt"

BINANCE_BASE = "https://api.binance.com"

# ============= GPU SAFE =============
try:
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

# ============= DATA =============
def fetch_klines(symbol: str, interval: str = INTERVAL, limit=1500, max_bars=MAX_BARS) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    frames = []
    end_ts = int(time.time() * 1000)
    got = 0
    while got < max_bars:
        params = {"symbol": symbol.upper(), "interval": interval,
                  "limit": min(limit, max_bars - got), "endTime": end_ts}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        arr = r.json()
        if not arr: break
        df = pd.DataFrame(arr, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ]) [["open_time","open","high","low","close","volume"]]
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.dropna())
        got += len(df)
        first_ms = int(df["open_time"].iloc[0].timestamp() * 1000)
        end_ts = first_ms - 1
        if len(df) <= 1: break
    if not frames:
        raise RuntimeError(f"Không lấy được dữ liệu từ Binance ({symbol},{interval}).")
    full = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    return full

def add_all_ta(df: pd.DataFrame) -> pd.DataFrame:
    out = ta.add_all_ta_features(df.copy(), open="open", high="high", low="low",
                                 close="close", volume="volume", fillna=False)
    return out.replace([np.inf, -np.inf], np.nan)

def add_growth_logs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g_raw_log = np.log(d["close"]) - np.log(d["close"].shift(1))
    d["g_raw_now_log"] = g_raw_log
    d["g_raw_now"]     = g_raw_log  # alias

    ema20 = d["close"].ewm(span=20, adjust=False).mean()
    g_ema_log = np.log(ema20) - np.log(ema20.shift(1))
    d["g_ema_now_log"]    = g_ema_log
    d["g_ema_now"]        = g_ema_log
    d["g_ema_now_20"]     = g_ema_log
    d["g_ema_now_log_20"] = g_ema_log
    return d

def read_feature_list() -> List[str]:
    obj = json.loads(FEATURES_JS.read_text(encoding="utf-8"))
    feats = obj.get("features") or obj.get("cols") or obj.get("columns")
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"File features không hợp lệ: {FEATURES_JS}")
    return [str(x) for x in feats]

def make_last_window(X_scaled: np.ndarray, ws: int) -> np.ndarray:
    if len(X_scaled) < ws:
        raise RuntimeError(f"Không đủ bars để tạo cửa sổ ws={ws}.")
    return X_scaled[-ws:].reshape(1, ws, X_scaled.shape[1]).astype("float32")

# ============= ARTIFACTS =============
def _first_existing(base: Path, patterns: list) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits: return hits[0]
    return None

def artifact_paths(symbol: str, target: str) -> Tuple[Path, Path]:
    td = TARGET_ROOT / target
    t_std = target.lower()               # 'y_5'
    t_cmp = t_std.replace("_","")        # 'y5'
    model_pat = [
        f"{symbol}_{INTERVAL}_ws{WS}_{t_std}_best.keras",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_cmp}_best.keras",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_std}*.keras",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_cmp}*.keras",
    ]
    scaler_pat = [
        f"{symbol}_{INTERVAL}_ws{WS}_{t_std}_robust_X.pkl",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_cmp}_robust_X.pkl",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_std}_robust_x.pkl",
        f"{symbol}_{INTERVAL}_ws{WS}_{t_cmp}_robust_x.pkl",
    ]
    m = _first_existing(td, model_pat)
    s = _first_existing(td, scaler_pat)
    if m is None or s is None:
        raise FileNotFoundError(f"Không thấy artifact cho {symbol}/{target} tại {td}")
    print(f"🔗 {symbol}/{target}: model={m.name} | scaler={s.name}")
    return m, s

def read_selected_symbols(target: str) -> Set[str]:
    p = TARGET_ROOT / target / "selected_symbols_mase_lt_0.75.json"
    if not p.exists():
        print(f"ℹ️  {target}: thiếu file selected.")
        return set()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  {target}: lỗi đọc JSON ({e})")
        return set()

    syms: List[str] = []
    if isinstance(obj, dict) and isinstance(obj.get("symbols"), dict):
        syms = list(obj["symbols"].keys())
    elif isinstance(obj, dict):
        meta = {"target","interval","window_size","threshold"}
        syms = [k for k in obj.keys() if k not in meta]
    elif isinstance(obj, list):
        syms = obj
    return set(s.strip().upper() for s in syms if str(s).strip())

def load_symbols_txt() -> List[str]:
    if not SYMBOLS_TXT.exists(): return []
    return [s.strip().upper() for s in SYMBOLS_TXT.read_text(encoding="utf-8").splitlines() if s.strip()]

# ============= DISCORD =============
def read_webhook() -> Optional[str]:
    if not DISCORD_TXT.exists(): return None
    return DISCORD_TXT.read_text(encoding="utf-8").strip() or None

def post_discord(webhook: str, payload: Dict) -> None:
    try:
        payload.setdefault("username", "predict_kst")
        payload.setdefault("allowed_mentions", {"parse": []})
        r = requests.post(webhook, json=payload, timeout=10)
        print(f"Discord {r.status_code}")
    except Exception as e:
        print(f"Discord error: {e}")

def discord_embed(symbol: str, direction: str, preds: Dict[str, float], price: float) -> Dict:
    is_long = (direction == "LONG")
    color   = 0x00C853 if is_long else 0xD32F2F
    dot     = "🟢" if is_long else "🔴"

    y5_pct   = preds.get("Y_5_%", preds.get("Y5%", 0.0))
    y5_resid = preds.get("Y_5_resid", preds.get("Y5_resid", 0.0))
    y10_pct  = preds.get("Y_10_%", None)
    y15_pct  = preds.get("Y_15_%", None)

    f2 = lambda x: f"{x:.2f}"
    f6 = lambda x: f"{x:.6f}"

    title = f"{dot}  **{direction} | {symbol} ({INTERVAL})**"
    desc_lines = []
    desc_lines.append(f"**Y5**: {f2(y5_pct)}%   •   **Y₅_resid**: {f2(y5_resid)}")
    extras = []
    if y10_pct is not None: extras.append(f"**Y10**: {f2(y10_pct)}%")
    if y15_pct is not None: extras.append(f"**Y15**: {f2(y15_pct)}%")
    if extras: desc_lines.append("   •   ".join(extras))
    desc_lines.append(f"**Price**: {f6(price)}")
    description = "\n".join(desc_lines)

    ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    footer_text = f"ws={WS} • LSTM_F3_mi38"
    fields = [
        {"name": "Interval", "value": f"**{INTERVAL}**", "inline": True},
        {"name": "ws",       "value": f"**{WS}**",       "inline": True},
        {"name": "Time",     "value": time.strftime("%H:%M", time.localtime()), "inline": True},
    ]
    return {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": footer_text},
        "timestamp": ts_iso,
    }

# ============= TELEGRAM (HTML) =============
def _read_telegram_creds() -> Tuple[Optional[str], Optional[str]]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token and TELEGRAM_TOKEN_TXT.exists():
        token = TELEGRAM_TOKEN_TXT.read_text(encoding="utf-8").strip() or None
    if not chat_id and TELEGRAM_CHATID_TXT.exists():
        chat_id = TELEGRAM_CHATID_TXT.read_text(encoding="utf-8").strip() or None
    return token, chat_id

def _html_escape(s: str) -> str:
    # Escape nhẹ cho HTML parse_mode
    return (s.replace("&","&amp;")
             .replace("<","&lt;")
             .replace(">","&gt;"))

def telegram_message_html(symbol: str, direction: str, preds: Dict[str, float], price: float) -> str:
    """
    Format tin nhắn Telegram đẹp bằng HTML (bold + emoji).
    """
    dot = "🟢" if direction == "LONG" else "🔴"
    f2 = lambda x: f"{x:.2f}"
    f6 = lambda x: f"{x:.6f}"

    y5_pct   = preds.get("Y_5_%", 0.0)
    y5_resid = preds.get("Y_5_resid", 0.0)
    y10_pct  = preds.get("Y_10_%", None)
    y15_pct  = preds.get("Y_15_%", None)

    # Escape tên symbol/phòng hờ có ký tự đặc biệt
    sym_safe = _html_escape(symbol)

    lines = []
    lines.append(f"{dot} <b>{direction}</b> | <b>{sym_safe}</b> ({_html_escape(INTERVAL)})")
    lines.append(f"• <b>Y5</b>: {f2(y5_pct)}%   • <b>Y5_resid</b>: {f2(y5_resid)}")
    extra = []
    if y10_pct is not None: extra.append(f"<b>Y10</b>: {f2(y10_pct)}%")
    if y15_pct is not None: extra.append(f"<b>Y15</b>: {f2(y15_pct)}%")
    if extra: lines.append("• " + "   •   ".join(extra))
    lines.append(f"• <b>Price</b>: {f6(price)}")
    lines.append(f"• ws={WS} | {time.strftime('%H:%M', time.localtime())}")
    return "\n".join(lines)

def post_telegram(text: str) -> None:
    token, chat_id = _read_telegram_creds()
    if not token or not chat_id:
        print("Telegram disabled/missing creds.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, data=data, timeout=10)
        print(f"Telegram {r.status_code}")
    except Exception as e:
        print(f"Telegram error: {e}")

# ============= FEATURE ALIGNMENT =============
def ensure_growth_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "g_raw_now_log" not in d.columns or "g_raw_now" not in d.columns:
        g_raw_log = np.log(d["close"]) - np.log(d["close"].shift(1))
        d["g_raw_now_log"] = g_raw_log
        d["g_raw_now"]     = g_raw_log
    need_cols = {"g_ema_now_log","g_ema_now","g_ema_now_20","g_ema_now_log_20"}
    if not need_cols.issubset(set(d.columns)):
        ema20 = d["close"].ewm(span=20, adjust=False).mean()
        g_ema_log = np.log(ema20) - np.log(ema20.shift(1))
        for c in need_cols:
            d[c] = g_ema_log
    return d

def reconcile_features_with_scaler(feats_json: List[str], df: pd.DataFrame, scaler) -> List[str]:
    df = ensure_growth_feature_columns(df)

    feat_in = getattr(scaler, "feature_names_in_", None)
    if feat_in is not None and len(feat_in) > 0:
        alias = {
            "g_raw_now": "g_raw_now_log",
            "g_ema_now": "g_ema_now_log",
            "g_ema_now_20": "g_ema_now_log",
            "g_ema_now_log_20": "g_ema_now_log",
        }
        ordered, missing = [], []
        for name in list(feat_in):
            col = alias.get(name, name)
            if col in df.columns: ordered.append(col)
            else: missing.append(name)
        if missing:
            raise KeyError(f"Thiếu cột theo scaler.feature_names_in_: {missing}")
        print(f"⚙️  Dùng thứ tự scaler.feature_names_in_ ({len(ordered)} cột).")
        return ordered

    expected = getattr(scaler, "n_features_in_", None)
    if expected is None: return feats_json

    if len(feats_json) != expected:
        new_feats = feats_json[:]
        for c in ["g_raw_now_log","g_ema_now_log","g_raw_now","g_ema_now_log_20"]:
            if len(new_feats) >= expected: break
            if c not in new_feats: new_feats.append(c)
        print(f"⚙️  JSON features = {len(feats_json)} | Scaler expects = {expected} "
              f"-> dùng {len(new_feats)} cột (auto-append: {[c for c in new_feats if c not in feats_json]})")
        return new_feats
    return feats_json

# ============= PREDICT =============
def predict_target(symbol: str, target: str, feats_json: List[str]) -> Tuple[float, float, float]:
    model_path, sx_path = artifact_paths(symbol, target)
    model = load_model(model_path, compile=False)
    sx    = joblib.load(sx_path)

    raw = fetch_klines(symbol, interval=INTERVAL)
    df  = add_all_ta(raw)
    df  = add_growth_logs(df)
    feats = reconcile_features_with_scaler(feats_json, df, sx)

    X_df = df[feats].ffill().bfill().replace([np.inf,-np.inf],0)
    print(f"✅ X_df shape = {X_df.shape}, scaler expects {sx.n_features_in_}")
    assert X_df.shape[1] == sx.n_features_in_, "Mismatch số cột X so với scaler!"

    X_scaled = sx.transform(X_df.values.astype("float32"))
    X_last = make_last_window(X_scaled, WS)
    yhat = model.predict(X_last, verbose=0, batch_size=BATCH_SZ)
    y_resid_log = float(np.ravel(yhat)[0])

    g_raw_now_log = float(df["g_raw_now_log"].iloc[-1])
    y_total_log = y_resid_log + g_raw_now_log
    y_total_pct = (np.exp(y_total_log) - 1) * 100.0

    last_price = float(df["close"].iloc[-1])
    return y_resid_log, y_total_pct, last_price

def decide_direction(value: float, threshold: float) -> Optional[str]:
    if value >  threshold: return "LONG"
    if value < -threshold: return "SHORT"
    return None

# ============= BỘ LỌC 3EMA & BODY (đa khung) =============
def get_ema_flags(symbol: str) -> Dict[str, float]:
    """
    EMA filter luôn tính theo INTERVAL giao dịch (ví dụ: 15m).
    """
    try:
        df = fetch_klines(symbol, interval=INTERVAL, limit=1000, max_bars=240)  # đủ cho ema72
        close = df["close"]
        ema9  = close.ewm(span=EMA_SPANS[0], adjust=False).mean().iloc[-1]
        ema27 = close.ewm(span=EMA_SPANS[1], adjust=False).mean().iloc[-1]
        ema72 = close.ewm(span=EMA_SPANS[2], adjust=False).mean().iloc[-1]
        return {
            "ema9": float(ema9),
            "ema27": float(ema27),
            "ema72": float(ema72),
            "ema_up":  bool(ema9 > ema27 > ema72),
            "ema_down":bool(ema9 < ema27 < ema72),
        }
    except Exception as e:
        print(f"⚠️  Lỗi tính 3EMA cho {symbol}: {e}")
        return {"ema9": np.nan, "ema27": np.nan, "ema72": np.nan, "ema_up": False, "ema_down": False}

def get_body_pcts(symbol: str, tfs: List[str]) -> Dict[str, float]:
    """
    Trả về dict {tf: body_pct} cho mỗi khung trong tfs.
    body_pct = (close - open)/open * 100 của cây nến mới nhất ở khung đó.
    """
    out: Dict[str, float] = {}
    for tf in tfs:
        try:
            df = fetch_klines(symbol, interval=tf, limit=200, max_bars=200)
            body_pct = (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1] * 100.0
            out[tf] = float(body_pct)
        except Exception as e:
            print(f"⚠️  Lỗi tính body_pct {symbol}@{tf}: {e}")
            out[tf] = np.nan
    return out

def body_pass(direction: str, body_map: Dict[str, float], threshold_pct: float, mode: str = "ANY") -> Tuple[bool, str]:
    """
    Kiểm tra body_strong theo nhiều khung:
      - direction: "LONG" hoặc "SHORT"
      - body_map: {tf: body_pct}
      - threshold_pct: ví dụ 0.3
      - mode: "ANY" (>= 1 tf đạt) hoặc "ALL" (mọi tf đều đạt)
    """
    if not body_map:
        return False, "Không tính được body_pct ở bất kỳ khung nào"
    checks = []
    for tf, bp in body_map.items():
        if np.isnan(bp):
            checks.append(False)
            continue
        if direction == "LONG":
            checks.append(bp >= threshold_pct)
        else:
            checks.append(bp <= -threshold_pct)
    if mode.upper() == "ALL":
        ok = all(checks) if checks else False
    else:
        ok = any(checks) if checks else False

    if ok:
        passed = [tf for (tf, bp) in body_map.items()
                  if not np.isnan(bp) and ((bp >= threshold_pct) if direction=="LONG" else (bp <= -threshold_pct))]
        reason = f"OK body({mode}): " + ", ".join([f"{tf}={body_map[tf]:.3f}%" for tf in passed])
    else:
        reason = f"Loại body({mode}): " + ", ".join([f"{tf}={bp:.3f}%" for tf, bp in body_map.items() if not np.isnan(bp)])
    return ok, reason

def pass_filters_for(direction: str, ema_info: Dict[str, float], body_tfs: List[str]) -> Tuple[bool, str]:
    """
    Trả về (ok, reason_if_fail) cho toàn bộ bộ lọc.
    - 3EMA: theo INTERVAL
    - body_strong: theo danh sách BODY_STRONG_TFS và chế độ BODY_STRONG_REQUIRE_MODE
    """
    # 3 EMA
    if REQUIRE_3EMA:
        if direction == "LONG" and not ema_info["ema_up"]:
            return False, (f"Loại: 3EMA không up (ema9={ema_info['ema9']:.4f} ≤/≥ "
                           f"ema27={ema_info['ema27']:.4f} ≤/≥ ema72={ema_info['ema72']:.4f})")
        if direction == "SHORT" and not ema_info["ema_down"]:
            return False, (f"Loại: 3EMA không down (ema9={ema_info['ema9']:.4f} ≥/≤ "
                           f"ema27={ema_info['ema27']:.4f} ≥/≤ ema72={ema_info['ema72']:.4f})")
    # Body strong đa khung
    if REQUIRE_BODY_STRONG:
        body_map = get_body_pcts(symbol, body_tfs)  # <- cần symbol; truyền bằng closure trong main
        ok, why = body_pass(direction, body_map, BODY_STRONG_PCT, BODY_STRONG_REQUIRE_MODE)
        if not ok:
            return False, why
        return True, why  # why ở đây là thông điệp OK chi tiết
    return True, ""

# ============= MAIN =============
def main():
    print(f"USE_Y5={USE_Y5} | USE_Y10={USE_Y10} | USE_Y15={USE_Y15} | "
          f"DECISION_SPACE={DECISION_SPACE} | THRESH={'±'+str(THRESH_PCT)+'%' if DECISION_SPACE=='PCT' else '±'+str(THRESH_RESID)}")
    print(f"[FILTERS] 3EMA={REQUIRE_3EMA} (spans={EMA_SPANS}) | "
          f"body_strong={REQUIRE_BODY_STRONG} (≥{BODY_STRONG_PCT}%, TFS={BODY_STRONG_TFS}, MODE={BODY_STRONG_REQUIRE_MODE})")
    print(f"[NOTIFY] SEND_DISCORD={SEND_DISCORD} | SEND_TELEGRAM={SEND_TELEGRAM}")

    enabled = [t for t,flag in [("Y_5",USE_Y5),("Y_10",USE_Y10),("Y_15",USE_Y15)] if flag]
    if not enabled:
        print("⚠️ Không có target nào được bật."); return

    syms_main = load_symbols_txt()
    if not syms_main:
        print("⚠️ Không thấy symbols.txt"); return

    feats_json = read_feature_list()
    webhook = read_webhook() if SEND_DISCORD else None
    tg_token, tg_chat = _read_telegram_creds() if SEND_TELEGRAM else (None, None)

    selected: Dict[str, Set[str]] = {}
    for t in enabled:
        selected[t] = read_selected_symbols(t)
        inter = set(syms_main) & selected[t]
        print(f"🔎 {t}: selected={len(selected[t])} | symbols.txt={len(syms_main)} | giao={len(inter)}")

    # ------------ CHÚ Ý: pass_filters_for cần symbol để tính body đa khung ------------
    def run_filters(direction: str, sym: str) -> Tuple[bool, str]:
        ema_info = get_ema_flags(sym)  # theo INTERVAL giao dịch
        # Truyền 'symbol' vào scope cho pass_filters_for
        global symbol
        symbol = sym
        try:
            return pass_filters_for(direction, ema_info, BODY_STRONG_TFS)
        finally:
            symbol = None

    for sym in syms_main:
        try:
            preds_show: Dict[str, float] = {}
            base_direction: Optional[str] = None
            last_price_for_embed: float = 0.0

            for t in enabled:
                if sym not in selected[t]:
                    print(f"⏭  Bỏ {sym} ({t} not selected)")
                    base_direction = None
                    break

                y_resid_log, y_total_pct, last_price = predict_target(sym, t, feats_json)
                last_price_for_embed = last_price
                print(f"[{sym}] {t}: resid_log={y_resid_log:.4f} | total%={y_total_pct:.2f}")

                if DECISION_SPACE == "PCT":
                    dir_t = decide_direction(y_total_pct, THRESH_PCT)
                    preds_show[f"{t}_%"] = y_total_pct
                else:
                    dir_t = decide_direction(y_resid_log, THRESH_RESID)
                    preds_show[f"{t}_resid"] = y_resid_log

                preds_show[f"{t}_resid"] = y_resid_log
                preds_show[f"{t}_%"]     = y_total_pct

                print(f"→ {t} direction: {dir_t or 'NEUTRAL'}")

                if base_direction is None:
                    if dir_t is None:
                        base_direction = None
                        break
                    base_direction = dir_t
                else:
                    if dir_t != base_direction:
                        base_direction = None
                        break

            # ---- Áp dụng bộ lọc bổ sung sau khi đã có base_direction hợp lệ
            if base_direction:
                ok, why = run_filters(base_direction, sym)
                if not ok:
                    print(f"🚫 {sym} {base_direction} bị loại bởi bộ lọc: {why}")
                    base_direction = None  # hủy tín hiệu
                else:
                    print(f"✅ {sym} {base_direction} qua lọc: {why}")

            if base_direction:
                print(f"✅ SIGNAL {base_direction} | {sym} | {preds_show}")

                # Discord
                if SEND_DISCORD and webhook:
                    embed = discord_embed(sym, base_direction, preds_show, last_price_for_embed)
                    post_discord(webhook, {"embeds":[embed]})
                elif SEND_DISCORD and not webhook:
                    print("Discord enabled nhưng thiếu webhook (discord4.txt).")

                # Telegram (HTML)
                if SEND_TELEGRAM:
                    if tg_token and tg_chat:
                        text = telegram_message_html(sym, base_direction, preds_show, last_price_for_embed)
                        post_telegram(text)
                    else:
                        print("Telegram enabled nhưng thiếu TOKEN/CHAT_ID (ENV hoặc file).")

        except FileNotFoundError as e:
            print(f"❌ {sym}: {e}")
        except Exception as e:
            print(f"❌ {sym}: {e}")

    print("Done.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    main()
