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
from .config import semilla, tasa
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
    Maximiza Calmar en TEST/VALIDACIÓN para seleccionar hiperparámetros.
    """
    def objective(trial):
        import numpy as np
        import pandas as pd

        try:
            p = suggest_params(trial)
            sig_te = get_signals(test_df, p)

            if isinstance(sig_te, pd.Series):
                te_input = test_df.copy()
                te_input["signal"] = sig_te
            elif isinstance(sig_te, pd.DataFrame):
                if "signal" in sig_te.columns:
                    te_input = test_df.join(sig_te[["signal"]], how="left")
                else:
                    col_sig = next((c for c in sig_te.columns if c.lower() == "signal"), None)
                    if col_sig:
                        te_input = test_df.join(sig_te[[col_sig]].rename(columns={col_sig: "signal"}), how="left")
                    else:
                        te_input = test_df.join(sig_te, how="left")
            else:
                raise TypeError("get_signals debe devolver pd.Series o pd.DataFrame")

            if "Close" not in te_input.columns:
                if "close" in te_input.columns:
                    te_input = te_input.rename(columns={"close": "Close"})
                else:
                    raise KeyError("Falta columna 'Close' para backtest")
            if "signal" not in te_input.columns:
                raise KeyError("Falta columna 'signal' para backtest")

            res_te = backtest(te_input, p, fee_rate=fee_rate, start_equity=start_equity)

            stats = compute_metrics(res_te) or {}
            score = (stats.get("CALMAR")
                     or stats.get("Calmar")
                     or stats.get("calmar"))
            if score is None or not np.isfinite(float(score)):
                score = -1e6

            return float(score)

        except Exception as e:
            try:
                trial.set_user_attr("error", str(e))
            except Exception:
                pass
            return -1e9

    return objective


def optimize_params(train_df: pd.DataFrame, test_df: pd.DataFrame, n_trials: int = 50, study_name: str = "calmar_opt", tasa: float = None, capital_inicial: float = None):
    if optuna is None:
        raise RuntimeError("Optuna no está instalado. Instala con: pip install optuna")
    sampler = optuna.samplers.TPESampler(seed=semilla)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    fee = tasa if tasa is not None else tasa
    start_eq = capital_inicial if capital_inicial is not None else capital_inicial

    study.optimize(objective_factory(train_df, test_df, fee, start_eq), n_trials=n_trials, show_progress_bar=False)
    return study
# %%
