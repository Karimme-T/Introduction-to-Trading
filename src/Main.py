#%%
# Importaciones
from .Data_limpieza import cargar_y_limpiar, split_train_test_val
from .Data_limpieza import rsi, ema, macd, bollinger_bands, atr
from .Obtener_signals import StrategyParams, get_signals
from .Backtest import backtest
from . Ratios import compute_metrics
from .Optimizacion import optimize_params
from .Tablas_graficas import plot_equity, returns_tables

from pathlib import Path
import os, json
import numpy as np
from .config import tasa, capital_inicial, semilla, horas_anuales

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "Data" 
data_file = "Binance_BTCUSDT_1h.csv"
csv_path = DATA_DIR / data_file


# Optuna y helpers de optimización 
try:
    import optuna
    from Optimizacion import optimize_params, suggest_params
except Exception:
    optuna = None
    optimize_params = None
    suggest_params = None

#%%

# Parámetros  

rng = np.random.default_rng(semilla)

def main():
    path = ("CWD:", Path.cwd())
    print("CSV esperado:", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró: {csv_path}")
    
    df = cargar_y_limpiar(str(csv_path))

    splits = split_train_test_val(df, split=(0.60, 0.20, 0.20))
    tr, te, va = splits["train"], splits["test"], splits["val"]

    if optuna is not None and optimize_params is not None and suggest_params is not None:
        study = optimize_params(tr, te, n_trials=50)
        best_p = suggest_params(study.best_trial)  
        print("Mejores hiperparámetros:", study.best_trial.params)
    else:
        best_p = StrategyParams()

    # Enfoque walk-forward simple:
    #  1) Ajuste/selección con train+test 
    #  2) Evaluación FINAL en val:
    sig_val = get_signals(va, best_p)
    res_val = backtest(sig_val, best_p, fee_rate=tasa, start_equity=capital_inicial)
    res_val.stats = compute_metrics(res_val)

    print(json.dumps(res_val.stats, indent=2))

    # Gráficos y tablas
    outdir = ROOT / "report_out"
    outdir.mkdir(exist_ok=True)

    plot_equity(res_val, title="Equity en VALIDATION (final)")


    tables = returns_tables(res_val)
    for nombre, tb in tables.items():
        tb.to_csv(outdir / f"{nombre}.csv")
        
    print("\nRetornos mensuales (compuestos):")
    print(tables["Mensual_compuesto"].tail(12))



# %%
