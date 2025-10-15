# -*- coding: utf-8 -*-
"""
F3_apply_selected_features_total5_mi38.py  (multiprocessing + interval filter)

Cập nhật: **Target rời rạc 5 lớp theo TOTAL log‑return** (không dùng residual).
- Đọc JSON 38 đặc trưng ở **cùng thư mục**: selected_features_mi38_no_visual_ichimoku.json
- Chỉ xử lý các file .parquet có đúng interval (mặc định 15m) trong thư mục DATA_BINANCE
- Áp 38 feature (đã loại 2 visual Ichimoku) + 2 growth (g_raw_now, g_ema_now_{EMA_SPAN})
- Tạo 3 target rời rạc 5 lớp trên tổng log-return: Y_5, Y_10, Y_15 (int8: 0..4)
- Lưu ra thư mục DATA_BINANCE_FEAT
- Tận dụng CPU bằng ProcessPoolExecutor
- Lưu kèm file .bins.json cho mỗi output để kiểm tra ranh giới & mid

Tùy chọn BINs:
- Nếu tồn tại file bins_config_total5.json ở cùng thư mục, script sẽ dùng ranh giới cố định trong file này
  cho toàn bộ symbol (khuyến nghị: file này nên được tạo từ dữ liệu TRAIN tập trung).
- Nếu KHÔNG có file bins_config_total5.json, script sẽ tự tính quantile 20/40/60/80 theo từng file.

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
OUTPUT_DIR     = BASE_DIR / "DATA_BINANCE_TOTAL5"   # dữ liệu sau xử lý
FEATURE_JSON   = BASE_DIR / "selected_features_mi38_no_visual_ichimoku.json"

INTERVAL       = "15m"          # chỉ xử lý các file có hậu tố _15m.parquet
EMA_SPAN       = 10
HORIZONS       = (5, 10, 15)
WARMUP_BARS    = 250
REQ_COLS       = {"open_time", "open", "high", "low", "close", "volume"}

# Tối ưu CPU
NUM_WORKERS    = int(os.getenv("NUM_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
SKIP_IF_EXISTS = True  # bỏ qua nếu đã có file output tương ứng

# Bins cấu hình tùy chọn
BINS_JSON      = BASE_DIR / "bins_config_total5.json"  # nếu có sẽ dùng cố định cho mọi file
USE_FIXED_BINS = BINS_JSON.exists()

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


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá NaN/Inf để tránh lỗi khi cắt bins & ép kiểu.
    - Thay inf -> NaN
    - FFill rồi BFill để lấp cả đầu/đuôi
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


def _make_growth_features(df: pd.DataFrame, close_col="close", ema_span=EMA_SPAN) -> pd.DataFrame:
    df = df.copy()
    df["g_raw_now"] = np.log(df[close_col]) - np.log(df[close_col].shift(1))
    ema_col = f"ema{ema_span}"
    df[ema_col] = df[close_col].ewm(span=ema_span, adjust=False).mean()
    df[f"g_ema_now_{ema_span}"] = np.log(df[ema_col]) - np.log(df[ema_col].shift(1))
    return df


def _calc_forward_log_return(df: pd.DataFrame, close_col: str, k: int) -> pd.Series:
    """fwd_k = log(close[t+k]) - log(close[t])"""
    fwd = np.log(df[close_col].shift(-k)) - np.log(df[close_col])
    return fwd.astype(np.float32)


def _ensure_strictly_increasing(arr: np.ndarray) -> np.ndarray:
    """Đảm bảo ranh giới tăng nghiêm ngặt để pd.cut không lỗi/NaN bất thường."""
    e = arr.astype(np.float64, copy=True)
    for i in range(1, len(e)):
        if not (e[i] > e[i-1]):
            # đẩy nhẹ ranh giới hiện tại lên một bước epsilon
            e[i] = np.nextafter(e[i-1], np.inf)
    return e.astype(np.float32)


def _compute_edges_and_mids_from_series(fwd: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Tạo ranh giới 5 bins theo quantile 20/40/60/80 và mid (trung điểm) cho mỗi bin.
       Hai đầu ±inf sẽ được cắt cụt tại p5/p95 để mid không quá cực đoan.
    """
    s = fwd.dropna()
    if len(s) == 0:
        # fallback an toàn nếu toàn NaN: dùng biên mặc định cực nhỏ
        edges = np.array([-np.inf, -1e-6, -1e-9, 1e-9, 1e-6, np.inf], dtype=np.float32)
        mids  = np.array([-5e-7, -5e-10, 0.0, 5e-10, 5e-7], dtype=np.float32)
        return edges, mids

    q = s.quantile([0.2, 0.4, 0.6, 0.8]).values.astype(np.float32)
    edges = np.array([-np.inf, q[0], q[1], q[2], q[3], np.inf], dtype=np.float32)
    edges = _ensure_strictly_increasing(edges)

    # mid
    p5  = float(s.quantile(0.05))
    p95 = float(s.quantile(0.95))
    mids = []
    for i in range(5):
        lo, hi = edges[i], edges[i+1]
        if np.isneginf(lo): lo = p5
        if np.isposinf(hi): hi = p95
        mids.append(float((lo + hi) / 2.0))
    return edges, np.array(mids, dtype=np.float32)


def _make_targets_total_classes(
    df: pd.DataFrame,
    close_col: str = "close",
    horizons = HORIZONS,
    fixed_edges: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, list[float]]]:
    """
    Trả về:
      - targets: DataFrame cột Y_5, Y_10, Y_15 (pandas 'Int8' cho phép NA trong bước trung gian)
      - edges:   ranh giới (6 giá trị) cho mỗi Y_k
      - mids:    giá trị đại diện (5 giá trị) cho mỗi Y_k (trung điểm)
    """
    out = pd.DataFrame(index=df.index)
    edges_dict, mids_dict = {}, {}

    for k in horizons:
        key = f"Y_{k}"
        fwd = _calc_forward_log_return(df, close_col, k)

        if fixed_edges and key in fixed_edges:
            e = np.array(fixed_edges[key], dtype=np.float32)
            e = _ensure_strictly_increasing(e)
            s = fwd.dropna()
            p5  = float(s.quantile(0.05)) if len(s) else -1e-6
            p95 = float(s.quantile(0.95)) if len(s) else  1e-6
            mids = []
            for i in range(5):
                lo, hi = e[i], e[i+1]
                if np.isneginf(lo): lo = p5
                if np.isposinf(hi): hi = p95
                mids.append(float((lo + hi) / 2.0))
            m = np.array(mids, dtype=np.float32)
        else:
            e, m = _compute_edges_and_mids_from_series(fwd)

        # labels=False -> trả về chỉ số bin 0..4; cho phép NA
        cats = pd.cut(fwd, bins=e, labels=False, include_lowest=True, right=True)
        out[key] = cats.astype('Int8')  # pandas nullable Int8, chưa ép np.int8 ngay để tránh lỗi nếu còn NA
        edges_dict[key] = e.tolist()
        mids_dict[key]  = m.astype(float).tolist()

    return out, edges_dict, mids_dict


def _out_path_for(input_path: Path, kept_feat_count: int) -> Path:
    stem = input_path.stem  # ví dụ: BTCUSDT_15m
    # tránh lặp INTERVAL nếu stem đã có _15m
    suffix_interval = "" if stem.endswith(f"_{INTERVAL}") else f"_{INTERVAL}"
    out_name = f"{stem}{suffix_interval}_feat_mi{kept_feat_count}_{INTERVAL}_APPLIED.parquet"
    return OUTPUT_DIR / out_name


def _process_one(parquet_path_str: str, feature_list: list[str], fixed_edges: dict | None) -> tuple[str, str]:
    p = Path(parquet_path_str)
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        missing = list(REQ_COLS - set(df.columns))
        if missing:
            return (p.name, f"❌ Thiếu cột bắt buộc {missing}")

        # loại bỏ hàng thiếu close trước khi tính toán
        df = df.sort_values("open_time").reset_index(drop=True)
        df = df[df["close"].notna()].reset_index(drop=True)

        # 1) full indicators
        df_feat = _add_all_features(df)

        # 2) warmup cắt đầu dãy
        cut = min(WARMUP_BARS, max(len(df_feat) - 1, 0))
        if cut > 0:
            df_feat = df_feat.iloc[cut:].reset_index(drop=True)

        # 3) growth features
        df_feat = _make_growth_features(df_feat, close_col="close", ema_span=EMA_SPAN)

        # 4) clean kỹ: inf->nan, ffill + bfill để KHÔNG còn NaN đầu/đuôi
        df_feat = _clean_numeric(df_feat)

        # 5) targets (TOTAL classes 5 mức, KHÔNG dùng residual)
        targets, edges, mids = _make_targets_total_classes(
            df_feat, close_col="close", horizons=HORIZONS, fixed_edges=fixed_edges
        )

        # 6) cắt đuôi theo horizon lớn nhất + loại các hàng còn NA label (nếu có)
        max_h = max(HORIZONS)
        valid_len = len(df_feat) - (max_h - 1)
        if valid_len <= 0:
            return (p.name, f"❌ Dữ liệu quá ngắn cho horizons={HORIZONS} với warmup={WARMUP_BARS}")
        df_feat  = df_feat.iloc[:valid_len].reset_index(drop=True)
        targets  = targets.iloc[:valid_len].reset_index(drop=True)

        # loại hàng có NA ở bất kỳ target nào rồi mới ép np.int8
        mask_ok = targets.notna().all(axis=1)
        if not mask_ok.all():
            df_feat = df_feat[mask_ok].reset_index(drop=True)
            targets = targets[mask_ok].reset_index(drop=True)
        for c in [f"Y_{k}" for k in HORIZONS]:
            targets[c] = targets[c].astype(np.int8)

        # 7) ép bool -> int8 cho features
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

        # Lưu bins cạnh file để kiểm tra nhanh
        bins_sidecar = out_path.with_suffix(".bins.json")
        side = {"horizons": list(HORIZONS), "bins": edges, "bin_mids": mids, "mode": "total_class5"}
        bins_sidecar.write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")

        return (p.name, f"✅ rows={len(df_out)}, kept_features={len(feature_cols)} | saved={out_path.name}")
    except Exception as e:
        return (p.name, f"❌ Error: {e}")


def _match_interval(filename: str, interval: str) -> bool:
    # downloader đặt tên kiểu: SYMBOL_INTERVAL.parquet (ví dụ: BTCUSDT_15m.parquet)
    return filename.endswith(f"_{interval}.parquet")


def _load_fixed_edges() -> dict | None:
    if not USE_FIXED_BINS:
        return None
    try:
        cfg = json.loads(BINS_JSON.read_text(encoding="utf-8"))
        # kỳ vọng cấu trúc: {"horizons": [5,10,15], "bins": {"Y_5": [...], ...}, "bin_mids": {...} (optional)}
        bins = cfg.get("bins", None)
        if not isinstance(bins, dict):
            print("⚠️ bins_config_total5.json không hợp lệ: thiếu 'bins' -> fallback tự tính theo file")
            return None
        return bins
    except Exception as e:
        print(f"⚠️ Không thể đọc {BINS_JSON.name}: {e} -> fallback tự tính theo file")
        return None


def main():
    # 1) đọc danh sách feature
    selected_features = _load_feature_list(FEATURE_JSON)

    # 2) lọc file theo interval
    all_files = sorted(INPUT_DIR.glob("*.parquet"))
    files = [p for p in all_files if _match_interval(p.name, INTERVAL)]

    if not files:
        print(f"⚠️ Không thấy file interval={INTERVAL} trong {INPUT_DIR.resolve()}")
        for p in all_files[:10]:
            print("  ->", p.name)
        return

    fixed_edges = _load_fixed_edges()

    print(f"→ Xử lý {len(files)} file với {NUM_WORKERS} tiến trình | interval={INTERVAL} | bins_fixed={fixed_edges is not None}")
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        fut2name = {ex.submit(_process_one, str(p), selected_features, fixed_edges): p.name for p in files}
        for fut in as_completed(fut2name):
            name, msg = fut.result()
            print(f"{name}: {msg}")
            results.append((name, msg))

    ok = sum(1 for _, m in results if m.startswith("✅") or m.startswith("⏭"))
    print(f"✔ Hoàn tất: {ok}/{len(results)} file (bao gồm đã bỏ qua nếu tồn tại). Kết quả: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
