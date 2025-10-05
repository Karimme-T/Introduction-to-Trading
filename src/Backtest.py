#%%
# Importaciones
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import optuna
from .Obtener_signals import StrategyParams

#%%
# Backtest

# Parámetros

data_folder = "data"
data = "Binance_BTCUSDT_1h.csv"   
tasa = 0.00125                   
capital_inicial = 100_000_000.00               
semilla = 2111544
rng = np.random.default_rng(semilla)

horas_anuales = 24 * 365 

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str      
    entry_price: float
    exit_price: Optional[float]
    units: float
    fee_entry: float
    fee_exit: float
    pnl: Optional[float]

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    stats: Dict[str, float]

def backtest(signals_df: pd.DataFrame, p: StrategyParams, fee_rate: float = tasa,
             start_equity: float = capital_inicial) -> BacktestResult:
    """
    Backtest OHLC (sin apalancamiento). Ejecuta a close de la barra donde aparece la señal confirmada.
    SL/TP fijos por múltiplos de ATR desde la entrada. 1 posición a la vez (simple).
    """
    df = signals_df.copy()
    close = df["Close"]
    atr_ = df["ATR"]

    equity = start_equity
    equity_curve = []
    trades: List[Trade] = []

    position = 0      # +1 long, -1 short, 0 flat
    entry_px = None
    units = 0.0
    sl_px = None
    tp_px = None
    fee_entry = 0.0

    for t, row in df.iterrows():
        px = row["Close"]

        # 1) Gestionar salida por SL/TP si en posición
        if position != 0:
            hit_sl = hit_tp = False
            if position == +1:
                hit_sl = (row["Low"] <= sl_px) if sl_px is not None else False
                hit_tp = (row["High"] >= tp_px) if tp_px is not None else False
            else:  # corto
                hit_sl = (row["High"] >= sl_px) if sl_px is not None else False
                hit_tp = (row["Low"]  <= tp_px) if tp_px is not None else False

            exit_reason = None
            if hit_sl and hit_tp:
                # prioridad intrabar: SL primero (conservador)
                exit_reason = "SL"
            elif hit_sl:
                exit_reason = "SL"
            elif hit_tp:
                exit_reason = "TP"

            if exit_reason is not None:
                exit_px = sl_px if exit_reason=="SL" else tp_px
                fee_out = abs(exit_px * units) * fee_rate
                pnl = (position * (exit_px - entry_px) * units) - fee_entry - fee_out
                equity += pnl
                trades.append(Trade(entry_time=entry_time, exit_time=t, side="long" if position==1 else "short",
                                    entry_price=float(entry_px), exit_price=float(exit_px), units=float(units),
                                    fee_entry=float(fee_entry), fee_exit=float(fee_out), pnl=float(pnl)))
                position, entry_px, units, sl_px, tp_px = 0, None, 0.0, None, None
                fee_entry = 0.0

        # 2) Si estamos flat, procesar entrada por señal confirmada
        sig = row["signal"]
        if position == 0 and sig != 0:
            # Entramos a cierre de barra actual
            side = +1 if sig > 0 else -1
            entry_px = px
            units = p.n_units  # fracción de BTC (permite “partes”)
            fee_entry = abs(entry_px * units) * fee_rate
            equity -= fee_entry  # fee reduce equity instantáneamente
            position = side
            entry_time = t

            # Niveles SL/TP a partir de ATR
            atr_val = atr_.loc[t]
            if atr_val is None or np.isnan(atr_val):
                sl_px = tp_px = None
            else:
                if position == +1:
                    sl_px = entry_px - p.sl_atr_mult * atr_val
                    tp_px = entry_px + p.tp_atr_mult * atr_val
                else:
                    sl_px = entry_px + p.sl_atr_mult * atr_val
                    tp_px = entry_px - p.tp_atr_mult * atr_val

        # Registrar equity mark-to-market (si hay posición, aún no realizamos PnL salvo fees)
        equity_curve.append((t, equity))

    # Cerrar posición al final a precio de cierre
    if position != 0 and entry_px is not None:
        last_t = df.index[-1]
        exit_px = df["Close"].iloc[-1]
        fee_out = abs(exit_px * units) * fee_rate
        pnl = (position * (exit_px - entry_px) * units) - fee_entry - fee_out
        equity += pnl
        trades.append(Trade(entry_time=entry_time, exit_time=last_t, side="long" if position==1 else "short",
                            entry_price=float(entry_px), exit_price=float(exit_px), units=float(units),
                            fee_entry=float(fee_entry), fee_exit=float(fee_out), pnl=float(pnl)))
        equity_curve[-1] = (last_t, equity)

    equity_curve = pd.Series(
        data=[v for _, v in equity_curve],
        index=[t for t, _ in equity_curve],
        name="Equity"
    )
    stats = {}  # se llena en metrics()
    return BacktestResult(equity_curve=equity_curve, trades=trades, stats=stats)
# %%
