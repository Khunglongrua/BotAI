# -*- coding: utf-8 -*-
"""
select_top30_features_btc_y10_ta_dropna.py
------------------------------------------
- Đọc OHLCV BTCUSDT từ ./DATA_BINANCE/BTCUSDT_<interval>.parquet
- Sinh TA features bằng ta.add_all_ta_features + giữ nguyên OHLCV
- TRƯỚC KHI GỘP TARGET:
    + Thay inf→NaN
    + Bỏ cột all-NaN
    + Bỏ cột có tỷ lệ NaN > max_nan_col_ratio (mặc định 0.20 = 20%)
    + Bỏ cột zero-variance (nunique ≤ 1)
- Tạo nhãn Y5/Y10/Y15 = forward log-return, chia 5 đoạn (0..4)
- Gộp X (đã lọc cột) với Y, rồi mới drop theo hàng (NaN/Inf)
- RF + TimeSeriesSplit (n_splits tự co) → chọn TOP-30 đặc trưng cho Y10
"""

import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
SEED = 123

# ---------- TA features ----------
def add_all_ta_features(df_src: pd.DataFrame) -> pd.DataFrame:
    import ta  # import trong hàm để an toàn
    df_feat = ta.add_all_ta_features(
        df_src.copy(),
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=False
    )
    return df_feat

# ---------- Helpers ----------
def forward_logret(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon)) - np.log(close)

def discretize_5bins_force(y_cont: pd.Series) -> pd.Series:
    """Chia 5 đoạn {0..4} theo quantile 20/40/60/80; đảm bảo biên tăng nghiêm ngặt; trả về Int64 (nullable)."""
    s = y_cont.replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = s.notna()
    if mask.sum() == 0:
        return out
    non = s[mask].to_numpy()
    q20, q40, q60, q80 = np.quantile(non, [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf, float(q20), float(q40), float(q60), float(q80), np.inf]
    eps = np.finfo(float).eps
    for i in range(1, len(edges)):
        if not (edges[i] > edges[i-1]):
            edges[i] = edges[i-1] + eps
    bins = pd.cut(s[mask], bins=edges, labels=[0,1,2,3,4], include_lowest=True)
    out.loc[mask] = bins.astype(int).astype("Int64")
    return out

def rf_importance_timeseries(X: np.ndarray, y: np.ndarray, n_splits=5, seed=SEED):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    imps = []
    for i, (tr, va) in enumerate(tscv.split(X)):
        clf = RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed + i
        )
        clf.fit(X[tr], y[tr])
        imps.append(clf.feature_importances_)
    imps = np.vstack(imps)
    return imps.mean(axis=0), imps.std(axis=0)

def counts_jsonable(s: pd.Series) -> dict:
    # khóa và giá trị đều ép về kiểu native của Python
    return {int(k): int(v) for k, v in s.value_counts().sort_index().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", type=str, default="./DATA_BINANCE")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="15m")
    ap.add_argument("--output_dir", type=str, default="./FEATURE_SELECTION_BTC_Y10")
    ap.add_argument("--h5", type=int, default=5)
    ap.add_argument("--h10", type=int, default=10)
    ap.add_argument("--h15", type=int, default=15)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--max_nan_col_ratio", type=float, default=0.20, help="bỏ cột có NaN ratio > ngưỡng này (vd 0.2)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pq = Path(args.data_folder) / f"{args.symbol}_{args.interval}.parquet"
    if not pq.exists():
        raise SystemExit(f"Không thấy file: {pq}")

    logging.info(f"Đọc dữ liệu: {pq.name}")
    df_raw = pd.read_parquet(pq)
    df_raw = df_raw.rename(columns={c: c.lower() for c in df_raw.columns})
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df_raw.columns:
            raise SystemExit(f"{pq.name} thiếu cột {c}")

    # 1) Sinh TA
    logging.info("Sinh toàn bộ TA features…")
    df_feat = add_all_ta_features(df_raw[["open","high","low","close","volume"]])

    # 2) Gộp OHLCV + TA
    X_df = pd.concat(
        [df_raw[["open","high","low","close","volume"]],
         df_feat.drop(columns=["open","high","low","close","volume"], errors="ignore")],
        axis=1
    )

    # 3) LỌC CỘT trước khi gộp target (đúng yêu cầu)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    before_cols = X_df.shape[1]

    # 3.1 all-NaN
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        X_df = X_df.drop(columns=all_nan_cols)
        logging.info(f"Bỏ {len(all_nan_cols)} cột all-NaN")

    # 3.2 NaN ratio > threshold (mặc định 20%)
    nan_ratio = X_df.isna().mean()
    high_nan_cols = nan_ratio[nan_ratio > args.max_nan_col_ratio].index.tolist()
    if high_nan_cols:
        X_df = X_df.drop(columns=high_nan_cols)
        logging.info(f"Bỏ {len(high_nan_cols)} cột NaN ratio > {args.max_nan_col_ratio:.2f}")

    # 3.3 zero-variance
    nunique = X_df.nunique(dropna=True)
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        X_df = X_df.drop(columns=zero_var_cols)
        logging.info(f"Bỏ {len(zero_var_cols)} cột zero-variance")

    logging.info(f"Sau lọc cột: {before_cols} → {X_df.shape[1]} đặc trưng")

    # 4) Tạo nhãn
    logging.info("Tạo nhãn Y5/Y10/Y15 (5 lớp)…")
    y5  = discretize_5bins_force(forward_logret(df_raw["close"], args.h5)).rename("Y5")
    y10 = discretize_5bins_force(forward_logret(df_raw["close"], args.h10)).rename("Y10")
    y15 = discretize_5bins_force(forward_logret(df_raw["close"], args.h15)).rename("Y15")

    # 5) Gộp X (đã lọc cột) với Y, rồi MỚI drop theo hàng
    data = pd.concat([X_df, y5, y10, y15], axis=1)
    before_rows = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    after_rows = len(data)
    logging.info(f"Đã loại {before_rows - after_rows} dòng có NaN/Inf (còn lại {after_rows})")

    if after_rows < 200:
        raise SystemExit(
            "Sau khi lọc còn quá ít mẫu (<200). Hãy tăng --max_nan_col_ratio (vd 0.4/0.6/0.8) "
            "hoặc dùng interval dài hơn (30m/1h)."
        )

    # 6) Chuẩn bị dữ liệu cho Y10
    feature_names = [c for c in data.columns if c not in ["Y5","Y10","Y15"]]
    X = data[feature_names].to_numpy(dtype=np.float32)
    y = data["Y10"].astype(int).to_numpy()

    # 7) Chọn n_splits hợp lý theo số mẫu (mỗi fold ~200 mẫu trở lên)
    n_splits = min(5, max(2, after_rows // 200))
    logging.info(f"TimeSeriesSplit n_splits = {n_splits}")

    imp_mean, imp_std = rf_importance_timeseries(X, y, n_splits=n_splits, seed=SEED)
    imp_tbl = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": imp_mean,
        "importance_std": imp_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    top_k = int(args.top_k)
    top_features = imp_tbl.head(top_k)["feature"].tolist()

    # 8) Lưu output
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_importance_btc_y10.csv").write_text(imp_tbl.to_csv(index=False))
    with open(out_dir / "top30_features_btc_y10.json", "w", encoding="utf-8") as f:
        json.dump({"symbol": args.symbol, "interval": args.interval,
                   "target": "Y10_5classes", "top_k": top_k,
                   "top_features": top_features}, f, indent=2, ensure_ascii=False)


    with open(out_dir / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "Y5_counts":  counts_jsonable(data["Y5"]),
            "Y10_counts": counts_jsonable(data["Y10"]),
            "Y15_counts": counts_jsonable(data["Y15"]),
        }, f, indent=2, ensure_ascii=False)


    logging.info("Đã lưu CSV/JSON kết quả")
    print("TOP-30 (preview):", top_features[:30])

if __name__ == "__main__":
    main()
