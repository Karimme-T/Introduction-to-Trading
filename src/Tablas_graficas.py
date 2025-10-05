#%%
# Importaciones
from __future__ import annotations 
from .Backtest import BacktestResult
import matplotlib.pyplot as plt

#%%

# Tablas y gráficos

def plot_equity(result: BacktestResult, title="Curva de Portafolio"):
    eq = result.equity_curve
    plt.figure(figsize=(10,4))
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

def returns_tables(result: BacktestResult) -> Dict[str, pd.DataFrame]:
    """
    Retornos por barra → agregados a mensual, trimestral, anual.
    """
    eq = result.equity_curve
    rets = eq.pct_change().fillna(0.0)
    df = rets.to_frame("ret")
    monthly   = (1+df.resample("M").sum()).rename(columns={"ret":"ret_sum"}) - 1.0  # aproximación simple
    quarterly = (1+df.resample("Q").sum()).rename(columns={"ret_sum":"ret_q"} ) - 1.0
    annual    = (1+df.resample("Y").sum()).rename(columns={"ret_sum":"ret_y"} ) - 1.0

    # Mejor: usar composición exacta: (1+r1)*(1+r2)*...-1
    monthly_c   = df.resample("M").apply(lambda s: (1+s).prod()-1)
    quarterly_c = df.resample("Q").apply(lambda s: (1+s).prod()-1)
    annual_c    = df.resample("Y").apply(lambda s: (1+s).prod()-1)

    return {
        "Mensual_compuesto": monthly_c.rename(columns={"ret":"Return"}),
        "Trimestral_compuesto": quarterly_c.rename(columns={"ret":"Return"}),
        "Anual_compuesto": annual_c.rename(columns={"ret":"Return"})
    }
# %%
