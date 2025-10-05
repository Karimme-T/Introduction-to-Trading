#%%
# Importaciones
from __future__ import annotations 
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

#%%
# Función para la carga y limpieza del data set

def cargar_y_limpiar(path_csv: str) -> pd.DataFrame:
    """
    Lee el CSV, normaliza columnas, parsea fechas y asegura que las columnas sean numéricas
    """
    df = pd.read_csv(path_csv)
    columnas = ["Date", "Open", "High", "Low", "Close", "Volume BTC"]
    colmap = {c: c.strip() for c in df.columns}
    df = df.rename(columns=colmap)
    missing = [c for c in columnas if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df[columnas].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.set_index("Date", inplace=True)
    df.index.name = "Date"

    for col in ["Open", "High", "Low", "Close", "Volume BTC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    df["High"] = df[['High', 'Open', 'Close']].max(axis=1)
    df["Low"]  = df[['Low', 'Open', 'Close']].min(axis=1)
    df = df[df["Low"] <= df["High"]]
    return df[["Open","High","Low","Close","Volume BTC"]]

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
