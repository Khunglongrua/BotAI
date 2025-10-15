# get_data_binance_async.py
# Fast, robust Binance kline downloader (async) with .env support
# Added:
#   1) Ensure OUT_DIR exists at start
#   2) Per-task error handling (won't stop the whole batch)
#   3) Validate symbols via exchange info to skip invalid ones
#
# Requirements:
#   pip install python-binance pandas pyarrow tqdm python-dotenv
#
# Usage:
#   1) Put your symbols (one per line) in symbols.txt
#   2) Create .env with BINANCE_API_KEY, BINANCE_API_SECRET
#   3) python get_data_binance_async.py

import os
import asyncio
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException
from tqdm.asyncio import tqdm_asyncio

# ================== Load config from .env ==================
# Load .env (or fallback to "env") that sits next to this script, regardless of CWD
_ENV_CANDIDATES = [Path(__file__).with_name(".env"), Path(__file__).with_name("env")]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        break
else:
    # Also try default load (might be set by shell)
    load_dotenv(override=True)

SYMBOLS_FILE = os.getenv("SYMBOLS_FILE", "symbols.txt")
INTERVALS = [s.strip() for s in os.getenv("INTERVALS", "15m,30m,1h").split(",") if s.strip()]
YEARS_BACK = int(os.getenv("YEARS_BACK", "8"))
OUT_DIR = Path(os.getenv("OUT_DIR", "DRAW"))
WRITE_PARQUET = os.getenv("WRITE_PARQUET", "true").lower() == "true"
WRITE_EXCEL = os.getenv("WRITE_EXCEL", "false").lower() == "true"
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
SKIP_INTERVALS = set(s.strip() for s in os.getenv("SKIP_INTERVALS", "5m").split(",") if s.strip())

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ===========================================================

BINANCE_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]

KEEP_COLS = ["open_time","open","high","low","close","volume"]

def _start_ms(years_back: int) -> int:
    start = dt.datetime.utcnow() - dt.timedelta(days=365*years_back)
    return int(start.timestamp() * 1000)

def _cast_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # downcast numeric for memory and speed
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.sort_values("open_time").reset_index(drop=True)

async def _fetch_one(
    client: AsyncClient, symbol: str, interval: str, start_time_ms: int
) -> pd.DataFrame:
    """Fetch all klines for (symbol, interval) from start_time_ms to now."""
    rows = []
    cur = start_time_ms
    while True:
        klines = await client.get_klines(
            symbol=symbol, interval=interval, limit=1000, startTime=cur
        )
        if not klines:
            break
        rows.extend(klines)
        cur = klines[-1][0] + 1  # avoid duplicate last kline
        if len(klines) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.DataFrame(rows, columns=BINANCE_COLS)[KEEP_COLS]
    return _cast_and_sort(df)

async def _valid_symbols(client: AsyncClient, symbols: List[str]) -> List[str]:
    """Return only valid spot symbols; print and drop invalid ones."""
    info = await client.get_exchange_info()
    valid_set = {s['symbol'] for s in info['symbols']}
    cleaned = [s.strip().upper() for s in symbols if s.strip()]
    bad = [s for s in cleaned if s not in valid_set]
    if bad:
        print(f"⚠️  Bỏ qua symbol không hợp lệ (spot): {bad}")
    return [s for s in cleaned if s in valid_set]

async def fetch_all(
    api_key: str, api_secret: str, symbols: List[str], intervals: List[str]
) -> List[Tuple[str, str, pd.DataFrame, Optional[BaseException]]]:
    start_ms = _start_ms(YEARS_BACK)

    # Create client explicitly (compatibility across python-binance versions)
    client = await AsyncClient.create(api_key, api_secret)
    try:
        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        valid_syms = await _valid_symbols(client, symbols)
        if not valid_syms:
            print("⚠️  Không còn symbol hợp lệ sau khi lọc.")
            return []

        async def guarded_fetch(sym: str, itv: str):
            async with sem:
                try:
                    df = await _fetch_one(client, sym, itv, start_ms)
                    return sym, itv, df, None
                except (BinanceAPIException, BinanceRequestException, Exception) as e:
                    return sym, itv, pd.DataFrame(columns=KEEP_COLS), e

        tasks = [
            guarded_fetch(s, i) for s in valid_syms for i in intervals if i not in SKIP_INTERVALS
        ]

        results = []
        for item in await tqdm_asyncio.gather(*tasks, desc="Fetching Data", ncols=100):
            results.append(item)
        return results
    finally:
        # Close aiohttp session cleanly
        await client.close_connection()

def _load_symbols(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Symbols file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _save_one(symbol: str, interval: str, df: pd.DataFrame):
    if df is None or df.empty:
        print(f"ℹ️  {symbol} {interval}: không có dữ liệu để lưu.")
        return
    pq_path = OUT_DIR / f"{symbol}_{interval}.parquet"
    if WRITE_PARQUET:
        df.to_parquet(pq_path, index=False)
        print(f"✔ Saved {pq_path}")
    if WRITE_EXCEL:
        xlsx_path = OUT_DIR / f"{symbol}_{interval}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"✔ Saved {xlsx_path}")


async def main():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET in environment or .env")

    # 1) Ensure OUT_DIR exists at start
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    symbols = _load_symbols(SYMBOLS_FILE)

    # 2) Fetch all with per-task error handling + 3) symbol validation inside
    results = await fetch_all(BINANCE_API_KEY, BINANCE_API_SECRET, symbols, INTERVALS)

    # Save & log
    for sym, itv, df, err in results:
        if err:
            print(f"⚠️  {sym} {itv}: {err}")
        _save_one(sym, itv, df)

if __name__ == "__main__":
    asyncio.run(main())
