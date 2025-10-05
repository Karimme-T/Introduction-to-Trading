#%%
# Importaciones
from __future__ import annotations 
from .Backtest import BacktestResult
import matplotlib.pyplot as plt
import pandas as pd

#%%

# Tablas y gráficos

def plot_equity(result: BacktestResult, title="Curva de Portafolio"):
    eq = result.equity_curve
    if eq.empty:
        print(f"Advertencia: No hay datos en la curva de capital para el gráfico '{title}'.")
        return
    
    plt.figure(figsize=(10,4))
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def returns_tables(result: BacktestResult) -> Dict[str, pd.DataFrame]:
    """
    Retornos por barra → agregados a mensual, trimestral, anual.
    """
    eq = result.equity_curve
    if eq.empty or len(eq) < 2:
        return {
            "Mensual_compuesto": pd.DataFrame(columns=["Return"]),
            "Trimestral_compuesto": pd.DataFrame(columns=["Return"]),
            "Anual_compuesto": pd.DataFrame(columns=["Return"])
        }
    
    rets = eq.pct_change().fillna(0.0)
    df = rets.to_frame("ret")

    monthly_c   = df.resample("M").apply(lambda s: (1+s).prod()-1)
    quarterly_c = df.resample("Q").apply(lambda s: (1+s).prod()-1)
    annual_c    = df.resample("Y").apply(lambda s: (1+s).prod()-1)

    return {
        "Mensual_compuesto": monthly_c.rename(columns={"ret":"Return"}),
        "Trimestral_compuesto": quarterly_c.rename(columns={"ret":"Return"}),
        "Anual_compuesto": annual_c.rename(columns={"ret":"Return"})
    }
# %%
