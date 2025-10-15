# -*- coding: utf-8 -*-
"""
F3_apply_selected_features_top30_3y.py (UPDATED)

Quy trình target đúng yêu cầu:
1) Dịch chuyển tương lai k nến
2) Y_k_log = ln(close[t+k]) - ln(close[t])
3) Dùng qcut chia 5 khúc -> nhãn 0..4 cho mỗi k: Y_5, Y_10, Y_15

Giữ nguyên:
- Tính full TA rồi chỉ chọn đúng 30 đặc trưng từ JSON
- Warmup, làm sạch, downcast
- Cắt đuôi theo horizon lớn nhất để tránh lookahead
- Đa luồng theo file
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, json
import numpy as np
import pandas as pd

# ========= CONFIG =========
BASE_DIR          = Path(__file__).parent
INPUT_DIR         = BASE_DIR / "DATA_BINANCE"          # thư mục parquet thô (SYMBOL_15m.parquet)
OUTPUT_DIR        = BASE_DIR / "DATA_BINANCE_FEAT30"   # thư mục output
FEATURE_JSON      = BASE_DIR / "top30_features_btc_y10.json"  # JSON chứa đúng 30 tên đặc trưng
INTERVAL          = "15m"
HORIZONS          = (5, 10, 15)
WARMUP_BARS       = 250
REQ_COLS          = {"open_time", "open", "high", "low", "close", "volume"}

# Binning cấu hình
N_BINS            = 5
BIN_LABEL_COLS    = True   # xuất các cột nhãn 5 lớp: Y_5, Y_10, Y_15 (0..4, int8)
KEEP_LOG_TARGETS  = True   # giữ lại Y_5_log, Y_10_log, Y_15_log để tiện kiểm tra/so sánh

# Tối ưu đa luồng
NUM_WORKERS       = int(os.getenv("NUM_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
SKIP_IF_EXISTS    = True  # tồn tại file thì bỏ qua
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ========= Utils =========
def _match_interval(filename: str, interval: str) -> bool:
    return filename.endswith(f"_{interval}.parquet")


def _add_all_features(df_src: pd.DataFrame) -> pd.DataFrame:
    import ta  # import trong worker
    df_feat = df_src.copy()
    df_feat = ta.add_all_ta_features(
        df_feat, open="open", high="high", low="low", close="close", volume="volume", fillna=False
    )
    return df_feat


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


def _calc_forward_log_return(close: pd.Series, k: int) -> pd.Series:
    """Y_k_log = ln(close[t+k]) - ln(close[t])"""
    return (np.log(close.shift(-k)) - np.log(close)).astype(np.float32)


def _qcut_5bins_safe(y: pd.Series, q: int = 5) -> pd.Series:
    """
    Binning 5 khúc robust:
    - Dùng qcut(labels=False, duplicates='drop')
    - Nếu số biên trùng lặp (không đủ 5 khúc), fallback sang cut theo
      các quantile duy nhất còn lại.
    """
    y = y.astype("float32")
    try:
        bins = pd.qcut(y, q, labels=False, duplicates="drop")
        if bins.notna().sum() == 0:
            raise ValueError("qcut produced all-NaN")
        # Nếu qcut rớt xuống < q bins (do duplicates='drop'), ta chuẩn hoá về 0..(nbins-1)
        nb = int(bins.max()) + 1 if bins.notna().any() else 0
        if nb < q:
            # tính lại edges thủ công từ quantile duy nhất
            qs = np.unique(y.dropna().quantile(np.linspace(0, 1, q + 1)).values)
            if len(qs) <= 2:
                # dữ liệu hầu như phẳng -> tất cả về 2 nhóm, hoặc 1 nhóm
                bins2 = pd.cut(y, bins=len(qs)-1, labels=False, include_lowest=True)
            else:
                bins2 = pd.cut(y, bins=qs, labels=False, include_lowest=True, duplicates="drop")
            return bins2.astype("float32")
        return bins.astype("float32")
    except Exception:
        # Fallback: dùng rank để ép đủ phân vị (giảm rủi ro dữ liệu phẳng)
        r = y.rank(method="average", pct=True)
        bins = pd.qcut(r, q, labels=False, duplicates="drop")
        return bins.astype("float32")


def _load_top30(json_path: Path) -> list[str]:
    if not json_path.exists():
        raise FileNotFoundError(f"Không thấy JSON: {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    feats = meta.get("top_features") if isinstance(meta, dict) else meta
    feats = [str(x) for x in feats]
    remove_set = {"trend_visual_ichimoku_a", "trend_visual_ichimoku_b"}
    out, seen = [], set()
    for f in feats:
        if f not in remove_set and f not in seen:
            out.append(f); seen.add(f)
    return out


def _out_path_for(input_path: Path, kept_feat_count: int) -> Path:
    stem = input_path.stem  # e.g., BTCUSDT_15m
    suffix_interval = "" if stem.endswith(f"_{INTERVAL}") else f"_{INTERVAL}"
    out_name = f"{stem}{suffix_interval}_feat_top{kept_feat_count}_{INTERVAL}_3y.parquet"
    return OUTPUT_DIR / out_name


def _process_one(parquet_path_str: str, top30: list[str]) -> tuple[str, str]:
    p = Path(parquet_path_str)
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        missing = list(REQ_COLS - set(df.columns))
        if missing:
            return (p.name, f"❌ Thiếu cột bắt buộc {missing}")

        # Sắp xếp & loại close NaN
        df = df.sort_values("open_time").reset_index(drop=True)
        df = df[df["close"].notna()].reset_index(drop=True)

        # 1) Tính toàn bộ TA
        df_feat = _add_all_features(df)

        # 2) Warmup
        cut = min(WARMUP_BARS, max(len(df_feat) - 1, 0))
        if cut > 0:
            df_feat = df_feat.iloc[cut:].reset_index(drop=True)

        # 3) Làm sạch số
        df_feat = _clean_numeric(df_feat)

        # 4) Tạo targets Y_k_log và nhãn 5 lớp Y_k
        tgts = {}
        for k in HORIZONS:
            y_log = _calc_forward_log_return(df_feat["close"], k)  # ln(future) - ln(now)
            tgts[f"Y_{k}_log"] = y_log

        df_y = pd.DataFrame(tgts)

        # 5) Cắt đuôi theo horizon lớn nhất để tránh nhìn trước tương lai
        max_h = max(HORIZONS)
        valid_len = len(df_feat) - (max_h - 1)
        if valid_len <= 0:
            return (p.name, f"❌ Dữ liệu quá ngắn cho horizons={HORIZONS} với warmup={WARMUP_BARS}")
        df_feat = df_feat.iloc[:valid_len].reset_index(drop=True)
        df_y    = df_y.iloc[:valid_len].reset_index(drop=True)

        # 6) Binning 5 khúc bằng qcut cho từng Y_k_log -> nhãn 0..4
        if BIN_LABEL_COLS:
            for k in HORIZONS:
                bins = _qcut_5bins_safe(df_y[f"Y_{k}_log"], q=N_BINS)
                # convert về int8; NaN -> -1 (nếu có), rồi loại bỏ hàng NaN
                labs = bins.astype("float32")
                # Một số fallback có thể trả về float với NaN; ép kiểu
                labs = labs.where(~labs.isna(), other=-1).astype("int8")
                df_y[f"Y_{k}"] = labs

        # 7) Loại hàng nào còn NA ở targets (hoặc nhãn = -1)
        keep_mask = df_y.notna().all(axis=1)
        if BIN_LABEL_COLS:
            for k in HORIZONS:
                keep_mask &= df_y[f"Y_{k}"] >= 0
        if not keep_mask.all():
            df_feat = df_feat[keep_mask].reset_index(drop=True)
            df_y    = df_y[keep_mask].reset_index(drop=True)

        # 8) Chỉ giữ đúng 30 đặc trưng (nếu thiếu thì bỏ qua cột đó) & downcast
        avail = [c for c in top30 if c in df_feat.columns]
        feature_cols = [c for c in avail if pd.api.types.is_numeric_dtype(df_feat[c])]
        for c in feature_cols:
            if pd.api.types.is_float_dtype(df_feat[c]):
                df_feat[c] = df_feat[c].astype(np.float32)
            elif pd.api.types.is_integer_dtype(df_feat[c]) and df_feat[c].dtype != np.int8:
                df_feat[c] = df_feat[c].astype(np.int8)

        # Downcast targets
        for c in df_y.columns:
            if c.endswith("_log"):
                df_y[c] = df_y[c].astype(np.float32)
            else:
                # nhãn Y_5/Y_10/Y_15
                df_y[c] = df_y[c].astype(np.int8)

        # 9) Lưu
        #   - Luôn có: open_time + 30 features + Y_k_log
        #   - Nếu BIN_LABEL_COLS=True: thêm Y_k (0..4)
        out_cols = ["open_time"] + feature_cols + [f"Y_{k}_log" for k in HORIZONS]
        if BIN_LABEL_COLS:
            out_cols += [f"Y_{k}" for k in HORIZONS]

        df_out = pd.concat([df_feat[["open_time"] + feature_cols], df_y], axis=1)[out_cols]

        out_path = _out_path_for(p, len(feature_cols))
        if SKIP_IF_EXISTS and out_path.exists():
            return (p.name, f"⏭  Bỏ qua (đã có) -> {out_path.name}")

        df_out.to_parquet(out_path, index=False)
        return (p.name, f"✅ rows={len(df_out)}, kept_features={len(feature_cols)} | saved={out_path.name}")

    except Exception as e:
        return (p.name, f"❌ Error: {e}")


def main():
    # Đọc top-30 từ JSON
    top30 = _load_top30(FEATURE_JSON)

    # Lọc file theo interval
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
        fut2name = {ex.submit(_process_one, str(p), top30): p.name for p in files}
        for fut in as_completed(fut2name):
            name, msg = fut.result()
            print(f"{name}: {msg}")
            results.append((name, msg))

    ok = sum(1 for _, m in results if m.startswith("✅") or m.startswith("⏭"))
    print(f"✔ Hoàn tất: {ok}/{len(results)} file. Kết quả tại: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
