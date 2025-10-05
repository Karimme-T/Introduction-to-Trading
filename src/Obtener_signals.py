#%%
# Importaciones
import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


#%%
# Obtener señales

@dataclass
class StrategyParams:
    rsi_window: int = 14
    rsi_buy: int = 30        
    rsi_sell: int = 70      

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_window: int = 20
    bb_nstd: float = 2.0

    confirm_bars: int = 1    

    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0

    n_shares: float = 0.05 

def get_signals(data: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
    - rsi_sig: +1 (bull), -1 (bear), 0 (neutral)
    - macd_sig
    - bb_sig
    - signal_raw: +1, -1, 0 según mayoría (2/3)
    - signal: versión con confirmación por p.confirm_bars (requiere p.e. 1..2 barras consecutivas)
    - ATR: para SL/TP
    """
    df = data.copy()

    # Indicadores
    df["RSI"] = rsi(df["Close"], p.rsi_window)
    macd_line, macd_signal = macd(df["Close"], p.macd_fast, p.macd_slow, p.macd_signal)
    df["MACD"], df["MACD_SIGNAL"] = macd_line, macd_signal
    bb_low, bb_mid, bb_up = bollinger_bands(df["Close"], p.bb_window, p.bb_nstd)
    df["BB_LOW"], df["BB_MID"], df["BB_UP"] = bb_low, bb_mid, bb_up
    df["ATR"] = atr(df, window=14)

    # Señales individuales
    df["rsi_sig"]  = np.where(df["RSI"] < p.rsi_buy, +1,
                       np.where(df["RSI"] > p.rsi_sell, -1, 0))
    df["macd_sig"] = np.where(df["MACD"] > df["MACD_SIGNAL"], +1,
                       np.where(df["MACD"] < df["MACD_SIGNAL"], -1, 0))
    # Bollinger: tocar banda inferior = bull, superior = bear; cerca de media = neutro
    touch_low  = df["Close"] <= df["BB_LOW"]
    touch_up   = df["Close"] >= df["BB_UP"]
    df["bb_sig"] = np.where(touch_low, +1, np.where(touch_up, -1, 0))

    # Mayoría simple (2/3)
    votes = df[["rsi_sig","macd_sig","bb_sig"]].sum(axis=1)
    df["signal_raw"] = np.where(votes >= 2, +1, np.where(votes <= -2, -1, 0))

    # Confirmación: requiere p.confirm_bars velas consecutivas iguales
    if p.confirm_bars > 1:
        s = df["signal_raw"].copy()
        # acumulador por rachas
        streak = (s != s.shift(1)).cumsum()
        count  = s.groupby(streak).cumcount() + 1
        conf   = (count >= p.confirm_bars).astype(int)
        df["signal"] = np.where(conf.eq(1), s, 0)
    else:
        df["signal"] = df["signal_raw"]

    df = df.dropna(subset=["RSI","MACD","MACD_SIGNAL","BB_LOW","BB_MID","BB_UP","ATR"])

    return df
# %%
