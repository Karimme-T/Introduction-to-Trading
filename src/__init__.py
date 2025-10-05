# src/__init__.py
from .Data_limpieza import cargar_y_limpiar, split_train_test_val
from .Data_limpieza import rsi, ema, macd, bollinger_bands, atr
from .Obtener_signals import StrategyParams, get_signals
from .Backtest import backtest, Trade, BacktestResult # Añadir Trade y BacktestResult si quieres exportarlos
from .Ratios import compute_metrics
from .Optimizacion import optimize_params, suggest_params # Asumo que también querrás exportar suggest_params
from .Tablas_graficas import plot_equity, returns_tables
from .config import tasa, capital_inicial, semilla, horas_anuales

__all__ = [
    "cargar_y_limpiar", "split_train_test_val",
    "rsi", "ema", "macd", "bollinger_bands", "atr",
    "StrategyParams", "get_signals",
    "backtest", "Trade", "BacktestResult", 
    "compute_metrics", "optimize_params", "suggest_params", 
    "plot_equity", "returns_tables",
    "tasa", "capital_inicial", "semilla", "horas_anuales"
]
