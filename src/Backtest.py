#%%
# Importaciones
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .Obtener_signals import StrategyParams

#%%
# Backtest

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

def backtest(signals_df: pd.DataFrame, p: StrategyParams, fee_rate: float,
             start_equity: float) -> BacktestResult:
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

    position = 0 
    entry_px = None
    units = 0.0
    sl_px = None
    tp_px = None
    fee_entry = 0.0
    entry_time = None

    for t, row in df.iterrows():
        px = row["Close"]

        # 1) Gestionar salida por SL/TP si en posición
        if position != 0:
            hit_sl = hit_tp = False
            if position == +1:
                hit_sl = (sl_px is not None) and (row['Low'] <= sl_px)
                hit_tp = (tp_px is not None) and (row['High'] >= tp_px)
            else:  # corto
                hit_sl = (sl_px is not None) and (row['High'] >= sl_px)
                hit_tp = (tp_px is not None) and (row['Low'] <= tp_px)

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

                if exit_px is None:
                    exit_px = px
                if units == 0:
                    pnl = 0.0
                    fee_out = 0.0
                else:
                    fee_out = abs(exit_px * units) * fee_rate
                    pnl = (position * (exit_px - entry_px) * units) - fee_entry - fee_out

                equity += pnl
                trades.append(Trade(entry_time=entry_time, exit_time=t, side="long" if position==1 else "short",
                                    entry_price=float(entry_px), exit_price=float(exit_px), units=float(units),
                                    fee_entry=float(fee_entry), fee_exit=float(fee_out), pnl=float(pnl)))
                position, entry_px, units, sl_px, tp_px = 0, None, 0.0, None, None
                fee_entry = 0.0
                entry_time = None

        # 2) Si estamos flat, procesar entrada por señal confirmada
        sig = row["signal"]
        if position == 0 and sig != 0:
            # Entramos a cierre de barra actual
            side = +1 if sig > 0 else -1
            entry_px = px
            units = p.n_shares 
            if units <= 0:
                print(f"Advertencia: Unidades de entrada calculas como {units} en {t}. No se abrirá posición")
                continue

            fee_entry = abs(entry_px * units) * fee_rate
            equity -= fee_entry  
            position = side
            entry_time = t

            # Niveles SL/TP a partir de ATR
            atr_val = atr_.loc[t]
            if np.isnan(atr_val):
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
        if units == 0:
            pnl = 0.0
            fee_out = 0.0
        else:
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
    stats = {}
    return BacktestResult(equity_curve=equity_curve, trades=trades, stats=stats)
# %%
