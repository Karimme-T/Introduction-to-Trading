#%%
# Importaciones
from __future__ import annotations 
import numpy as np
import pandas as pd
from .Backtest import BacktestResult, Trade
from .config import horas_anuales
from typing import List, TYPE_CHECKING
from typing import Dict, Tuple, List

#%%

def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())

def calmar_ratio(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[0] == 0:
        return -np.inf
    
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365*24*3600)
    cagr = (1 + total_return)**(1/years) - 1 if years > 0 and (1 + total_return) >= 0 else total_return
    mdd = max_drawdown(equity)
    if mdd >= 0: 
        return -np.inf
    return float(cagr / abs(mdd))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    returns: retornos por barra (horaria). Se anualiza con ANNUALIZATION_HOURS.
    """
    if returns.empty: return 0.0
    std_dev = returns.std(ddof=0)
    if std_dev == 0:
        return 0.0
    
    mean_excess = returns.mean() - rf/horas_anuales
    return float(mean_excess / (std_dev + 1e-12) * np.sqrt(horas_anuales))

def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.empty: return 0.0

    downside = returns.clip(upper=0.0)
    denom = downside.std(ddof=0)
    if denom == 0:
        return 0.0
    mean_excess = returns.mean() - rf/horas_anuales
    return float(mean_excess / (denom + 1e-12) * np.sqrt(horas_anuales))

def win_rate(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for tr in trades if tr.pnl is not None and tr.pnl > 0)
    return wins / len(trades)

def compute_metrics(result):
    eq = result["equity"] if isinstance(result, dict) else result
    eq = pd.Series(eq)
    if eq.empty or len(eq) < 2:
        return {
            "Total Return": 0.0, "CAGR": 0.0, "MDD": 0.0, "CALMAR": -np.inf, "SHARPE": 0.0, 
            "SORTINO": 0.0, "WINRATE": 0.0, "NUMTRADES": len(result.trades)
        }
    
    idx = eq.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx_dt = pd.to_datetime(idx, units="s", errors="coerce")
        if idx_dt.isna().all():
            idx_dt = pd.to_datetime(idx, units="ms", errors="coerce")
        if not idx_dt.isna().all():
            eq = eq.copy()
            eq.index = idx_dt
            idx = eq.index
        else:
            vals = np.asarray(idx, dtype="float64")
            d = np.median(np.diff(vals)) if len(vals) > 1 else 1.0
            if d > 1e6:
                seconds_total = (vals[-1] - vals[0]) / 1e3
            else:
                seconds_total = (vals[-1] - vals[0])

            years_cagr = seconds_total / (365 * 24 * 3600)
    if isinstance(idx, pd.DatetimeIndex):
        delta = (idx[-1] - idx[0])
        years_cagr = delta.total_seconds() / (365 * 24 * 3600)
    years_cagr = max(float(years_cagr), 1e-9)

    stats = {
        "Total Return": float(eq.iloc[-1]/eq.iloc[0] - 1),
        "CAGR": float(cagr_val),
        "MDD": max_drawdown(eq),
        "Calmar": calmar_ratio(eq),
        "Sharpe": sharpe_ratio(ret),
        "Sortino": sortino_ratio(ret),
        "WinRate": win_rate(result.trades),
        "NumTrades": len(result.trades)
    }
    return stats
# %%
