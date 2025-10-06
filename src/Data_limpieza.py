#%%
# Importaciones
from __future__ import annotations 
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import re

#%%
# Función para la carga y limpieza del data set
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _find_header_row(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            cols = [c.strip().strip('"').lower() for c in line.split(",")]
            s = set(cols)
            if {"date", "open", "high", "low", "close"}.issubset(s):
                return i
            if {"unix", "open", "high", "low", "close"}.issubset(s):
                return i
    return 0 

def _ms_to_datetime(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df.get("Date"), errors="coerce", utc=False)
    if d.isna().all() and "unix" in df.columns:
        d = pd.to_datetime(pd.to_numeric(df["unix"], errors="coerce"), unit="s", utc=False)
    return d


def cargar_y_limpiar(path_csv: str) -> pd.DataFrame:
    hdr = _find_header_row(path_csv)
    df = pd.read_csv(
        path_csv,
        header=hdr,
        sep=",",
        encoding="utf-8",
        low_memory=False,
        on_bad_lines="skip"
    )

    # 3) Normaliza nombres y mapea a tu estándar
    raw_cols = list(df.columns)
    df.columns = [_norm(c) for c in df.columns]

    # Mapeo de encabezados
    mapping = {
        "date": "Date",
        "open_time": "Date",

        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",

        "volume": "Volume BTC",
        "volume btc": "Volume BTC",
        "volume_btc": "Volume BTC",
        "volume_BTC": "Volume BTC",
    }

    df = df.rename(columns={c: mapping[c] for c in df.columns if c in mapping})
    df["Date"] = _ms_to_datetime(df)
    for col in ["Open", "High", "Low", "Close", "Volume BTC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    required = ["Date", "Open", "High", "Low", "Close", "Volume BTC"]
    
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. "
            f"Cabeceras originales: {raw_cols}"
        )

    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[required + [c for c in df.columns if c not in required]]

    return df

#%%
def split_train_test_val(df: pd.DataFrame, split=(0.6, 0.2, 0.2)) -> Dict[str, pd.DataFrame]:
    """
    Split de forma cronológica 60/20/20
    """
    a, b, c = split
    assert abs((a+b+c) - 1.0) < 1e-9
    n = len(df)
    n_tr = int(n * a)
    n_te = int(n * b)
    train = df.iloc[:n_tr].copy()
    test  = df.iloc[n_tr:n_tr+n_te].copy()
    val   = df.iloc[n_tr+n_te:].copy()
    return {"train": train, "test": test, "val": val}

#%%
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def bollinger_mid_band(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return lower, ma, upper


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range clásico.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    return tr.ewm(alpha=1/window, adjust=False).mean()

# %%
