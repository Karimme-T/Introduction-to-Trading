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
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25*24*3600)
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

def compute_metrics(result: BacktestResult) -> Dict[str, float]:
    eq = result.equity_curve
    if eq.empty or len(eq) < 2:
        return {
            "Total Return": 0.0, "CAGR": 0.0, "MDD": 0.0, "CALMAR": -np.inf, "SHARPE": 0.0, 
            "SORTINO": 0.0, "WINRATE": 0.0, "NUMTRADES": len(result.trades)
        }
    
    ret = eq.pct_change().fillna(0.0) 
    years_cagr = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25*24*3600)
    if years_cagr > 0 and (1 + (eq.iloc[-1]/eq.iloc[0] -1)) >= 0:
        cagr_val = (1 + (eq.iloc[-1]/eq.iloc[0]-1)) ** (1/years_cagr) -1
    else:
        cagr_val = 0.0
    
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
