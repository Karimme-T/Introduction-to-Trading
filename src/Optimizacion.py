#%%

# Importaciones

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import optuna
from .Obtener_signals import StrategyParams, get_signals
from .Backtest import backtest
from .Ratios import compute_metrics
from .config import semilla
#%%

# Optimización OPTUNA

def suggest_params(trial) -> StrategyParams:
    """
    Espacio de búsqueda razonable (ajústalo con cuidado para evitar overfitting).
    """
    p = StrategyParams(
        rsi_window = trial.suggest_int("rsi_window", 8, 24),
        rsi_buy = trial.suggest_int("rsi_buy", 20, 40),
        rsi_sell = trial.suggest_int("rsi_sell", 60, 80),
        macd_fast = trial.suggest_int("macd_fast", 8, 15),
        macd_slow = trial.suggest_int("macd_slow", 20, 35),
        macd_signal = trial.suggest_int("macd_signal", 6, 12),
        bb_window = trial.suggest_int("bb_window", 18, 24),
        bb_nstd = trial.suggest_float("bb_nstd", 1.5, 2.5, step=0.1),
        confirm_bars = trial.suggest_int("confirm_bars", 1, 2),
        sl_atr_mult = trial.suggest_float("sl_atr_mult", 0.8, 2.5, step=0.1),
        tp_atr_mult = trial.suggest_float("tp_atr_mult", 1.5, 5.0, step=0.25),
        n_shares = trial.suggest_float("n_shares", 0.01, 0.2, step=0.01)
    )

    # coherencia MACD (slow > fast):
    if p.macd_slow <= p.macd_fast:
        pass
    return p

def objective_factory(train_df: pd.DataFrame, test_df: pd.DataFrame, fee_rate: float, start_equity: float):
    """
    Maximizamos Calmar en TEST para seleccionar hiperparámetros (evita sobreajuste al train).
    """
    def objective(trial):
        p = suggest_params(trial)
        sig_te = get_signals(train_df, p)
        res_te = backtest(sig_te, p, fee_rate=fee_rate, start_equity=start_equity)

        stats  = compute_metrics(res_te)
        score = stats.get("CALMAR", -np.inf)
        if not np.isfinite(score):
            score = -1e6
        return score
    return objective

def optimize_params(train_df: pd.DataFrame, test_df: pd.DataFrame, n_trials: int = 50, study_name: str = "calmar_opt", tasa: float = None, capital_inicial: float = None):
    if optuna is None:
        raise RuntimeError("Optuna no está instalado. Instala con: pip install optuna")
    sampler = optuna.samplers.TPESampler(seed=semilla)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    fee = tasa if tasa is not None else config.tasa
    start_eq = capital_inicial if capital_inicial is not None else config.capital_inicial

    study.optimize(objective_factory(train_df, test_df, fee, start_eq), n_trials=n_trials, show_progress_bar=False)
    return study
# %%
