# -*- coding: utf-8 -*-
"""
F3_apply_selected_features_residual_mi38.py  (multiprocessing + interval filter)

- Đọc JSON 38 đặc trưng ở **cùng thư mục**: selected_features_mi38_no_visual_ichimoku.json
- Chỉ xử lý các file .parquet có đúng interval (mặc định 15m) trong thư mục DATA_BINANCE
- Áp 38 feature (đã loại 2 visual Ichimoku) + 2 growth (g_raw_now, g_ema_now_{EMA_SPAN})
- Tạo 3 target residual trên g_raw: Y_5, Y_10, Y_15
- Lưu ra thư mục DATA_BINANCE_FEAT
- Tận dụng CPU bằng ProcessPoolExecutor

Yêu cầu:
    pip install ta pandas pyarrow
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, json
import numpy as np
import pandas as pd

# ========= CONFIG =========
BASE_DIR       = Path(__file__).parent
INPUT_DIR      = BASE_DIR / "DATA_BINANCE"        # dữ liệu thô
OUTPUT_DIR     = BASE_DIR / "DATA_BINANCE_FEAT"   # dữ liệu sau xử lý
FEATURE_JSON   = BASE_DIR / "selected_features_mi38_no_visual_ichimoku.json"

INTERVAL       = "15m"          # chỉ xử lý các file có hậu tố _15m.parquet
EMA_SPAN       = 10
HORIZONS       = (5, 10, 15)
WARMUP_BARS    = 250
REQ_COLS       = {"open_time", "open", "high", "low", "close", "volume"}

# Tối ưu CPU
NUM_WORKERS    = int(os.getenv("NUM_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
SKIP_IF_EXISTS = True  # bỏ qua nếu đã có file output tương ứng

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= Utils =========
def _load_feature_list(json_path: Path) -> list[str]:
    if not json_path.exists():
        raise FileNotFoundError(f"Không thấy JSON: {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    feats = meta.get("features", meta if isinstance(meta, list) else [])
    feats = [str(x) for x in feats]
    # đảm bảo loại 2 visual nếu còn sót
    remove_set = {"trend_visual_ichimoku_a", "trend_visual_ichimoku_b"}
    seen, dedup = set(), []
    for f in feats:
        if f not in remove_set and f not in seen:
            seen.add(f); dedup.append(f)
    return dedup

def _add_all_features(df_src: pd.DataFrame) -> pd.DataFrame:
    # Import trong worker để tránh vấn đề khi fork
    import ta
    df_feat = df_src.copy()
    df_feat = ta.add_all_ta_features(
        df_feat, open="open", high="high", low="low", close="close", volume="volume", fillna=False
    )
    return df_feat

def _make_growth_features(df: pd.DataFrame, close_col="close", ema_span=EMA_SPAN) -> pd.DataFrame:
    df = df.copy()
    df["g_raw_now"] = np.log(df[close_col]) - np.log(df[close_col].shift(1))
    ema_col = f"ema{ema_span}"
    df[ema_col] = df[close_col].ewm(span=ema_span, adjust=False).mean()
    df[f"g_ema_now_{ema_span}"] = np.log(df[ema_col]) - np.log(df[ema_col].shift(1))
    return df

def _make_targets_resid_graw(df: pd.DataFrame, close_col="close", horizons=HORIZONS) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k in horizons:
        fwd = np.log(df[close_col].shift(-k)) - np.log(df[close_col])
        out[f"Y_{k}"] = (fwd - df["g_raw_now"]).astype(np.float32)
    return out

def _out_path_for(input_path: Path, kept_feat_count: int) -> Path:
    stem = input_path.stem  # ví dụ: BTCUSDT_15m
    # tránh lặp INTERVAL nếu stem đã có _15m
    suffix_interval = "" if stem.endswith(f"_{INTERVAL}") else f"_{INTERVAL}"
    out_name = f"{stem}{suffix_interval}_feat_mi{kept_feat_count}_{INTERVAL}_APPLIED.parquet"
    return OUTPUT_DIR / out_name

def _process_one(parquet_path_str: str, feature_list: list[str]) -> tuple[str, str]:
    p = Path(parquet_path_str)
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        missing = list(REQ_COLS - set(df.columns))
        if missing:
            return (p.name, f"❌ Thiếu cột bắt buộc {missing}")

        df = df.sort_values("open_time").reset_index(drop=True)

        # 1) full indicators
        df_feat = _add_all_features(df)

        # 2) warmup cắt đầu dãy
        cut = min(WARMUP_BARS, max(len(df_feat) - 1, 0))
        if cut > 0:
            df_feat = df_feat.iloc[cut:].reset_index(drop=True)

        # 3) growth features
        df_feat = _make_growth_features(df_feat, close_col="close", ema_span=EMA_SPAN)

        # 4) clean
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feat.ffill(inplace=True)

        # 5) targets
        targets = _make_targets_resid_graw(df_feat, close_col="close", horizons=HORIZONS)

        # 6) cắt đuôi theo horizon lớn nhất
        max_h = max(HORIZONS)
        valid_len = len(df_feat) - (max_h - 1)
        if valid_len <= 0:
            return (p.name, f"❌ Dữ liệu quá ngắn cho horizons={HORIZONS} với warmup={WARMUP_BARS}")
        df_feat  = df_feat.iloc[:valid_len].reset_index(drop=True)
        targets  = targets.iloc[:valid_len].reset_index(drop=True)

        # 7) ép bool -> int8
        for c in df_feat.columns:
            if pd.api.types.is_bool_dtype(df_feat[c]):
                df_feat[c] = df_feat[c].astype(np.int8)

        # 8) chọn 38 feature theo JSON + chèn 2 growth
        avail = [c for c in feature_list if c in df_feat.columns]
        avail_num = [c for c in avail if pd.api.types.is_numeric_dtype(df_feat[c])]
        growth_feats = ["g_raw_now", f"g_ema_now_{EMA_SPAN}"]
        feature_cols = avail_num + [g for g in growth_feats if g in df_feat.columns and g not in avail_num]

        # 9) downcast
        for c in feature_cols:
            if pd.api.types.is_float_dtype(df_feat[c]):
                df_feat[c] = df_feat[c].astype(np.float32)
            elif pd.api.types.is_integer_dtype(df_feat[c]) and df_feat[c].dtype != np.int8:
                df_feat[c] = df_feat[c].astype(np.int8)

        # 10) kết quả + lưu
        first_cols  = ["open_time"]
        target_cols = [f"Y_{k}" for k in HORIZONS]
        df_out = pd.concat([df_feat[first_cols], df_feat[feature_cols], targets[target_cols]], axis=1)

        out_path = _out_path_for(p, len(feature_cols))
        if SKIP_IF_EXISTS and out_path.exists():
            return (p.name, f"⏭  Bỏ qua (đã có) -> {out_path.name}")

        df_out.to_parquet(out_path, index=False)
        return (p.name, f"✅ rows={len(df_out)}, kept_features={len(feature_cols)} | saved={out_path.name}")
    except Exception as e:
        return (p.name, f"❌ Error: {e}")

def _match_interval(filename: str, interval: str) -> bool:
    # downloader đặt tên kiểu: SYMBOL_INTERVAL.parquet (ví dụ: BTCUSDT_15m.parquet)
    return filename.endswith(f"_{interval}.parquet")

def main():
    # 1) đọc danh sách feature (FIX: truyền đúng json_path để tránh lỗi positional argument)
    selected_features = _load_feature_list(FEATURE_JSON)

    # 2) lọc file theo interval
    all_files = sorted(INPUT_DIR.glob("*.parquet"))
    files = [p for p in all_files if _match_interval(p.name, INTERVAL)]

    if not files:
        print(f"⚠️ Không thấy file interval={INTERVAL} trong {INPUT_DIR.resolve()}")
        for p in all_files[:10]:
            print("  ->", p.name)
        return

    print(f"→ Xử lý {len(files)} file với {NUM_WORKERS} tiến trình | interval={INTERVAL}")
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        fut2name = {ex.submit(_process_one, str(p), selected_features): p.name for p in files}
        for fut in as_completed(fut2name):
            name, msg = fut.result()
            print(f"{name}: {msg}")
            results.append((name, msg))

    ok = sum(1 for _, m in results if m.startswith("✅") or m.startswith("⏭"))
    print(f"✔ Hoàn tất: {ok}/{len(results)} file (bao gồm đã bỏ qua nếu tồn tại). Kết quả: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
