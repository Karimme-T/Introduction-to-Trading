# Introduction-to-Trading


Este proyecto implementa un **sistema de backtesting y optimización** para estrategias de trading algorítmico, utilizando datos históricos de **Binance BTC/USDT (1 hora)**.  
El objetivo es evaluar, comparar y optimizar estrategias técnicas combinando indicadores clásicos (RSI, MACD, Bandas de Bollinger, ATR) con criterios de gestión de riesgo (Stop Loss, Take Profit y tamaño de posición).

---

El sistema sigue una arquitectura modular en Python con los siguientes componentes principales:

| **Data_limpieza.py** | Carga, limpieza y normalización de datos históricos. Incluye funciones de RSI, MACD, EMA, ATR y Bandas de Bollinger. |
| **Obtener_signals.py** | Generación de señales de compra/venta basadas en múltiples indicadores técnicos. |
| **Backtest.py** | Simulación de operaciones *long* y *short* con SL/TP, comisiones y capital inicial configurable. |
| **Ratios.py** | Cálculo de métricas de rendimiento (Calmar Ratio, retorno total, drawdown máximo). |
| **Optimizacion.py** | Optimización de hiperparámetros con Optuna para maximizar el Calmar Ratio. |
| **Tablas_graficas.py** | Generación de gráficas de equity y tablas de retornos mensuales. |
| **Main.py** | Orquestador principal: ejecuta la limpieza, particiona los datos (train/test/validation), entrena, optimiza y evalúa la estrategia. |

---

## Flujo general del sistema

1. **Carga de datos:**  
   Se leen los precios históricos desde `Data/Binance_BTCUSDT_1h.csv`.

2. **Preprocesamiento:**  
   - Se renombran columnas y se ordenan las fechas.  
   - Se calculan indicadores técnicos.

3. **Generación de señales:**  
   `get_signals()` evalúa la confluencia de RSI, MACD y Bandas de Bollinger para producir una señal:
   - `+1`: compra (BUY)  
   - `-1`: venta (SELL)  
   - `0`: sin señal

4. **Backtesting:**  
   Se simulan operaciones con Stop Loss (SL) y Take Profit (TP), aplicando comisiones y gestión de efectivo.

5. **Optimización (Optuna):**  
   Se ajustan los parámetros de los indicadores y niveles de SL/TP para maximizar el **Calmar Ratio**.

6. **Visualización y métricas:**  
   Se grafican los resultados de la curva de equity y se exportan los retornos mensuales y métricas clave.

---

## Ejemplo de salida

### Curva de Equity

### Métricas principales

| **CALMAR** | Relación entre CAGR y drawdown máximo |
| **TOTAL_RETURN_%** | Retorno total del portafolio |
| **MAX_DRAWDOWN_%** | Caída máxima desde el pico |
| **START_EQ / END_EQ** | Capital inicial / final |

---

## ⚙️ Requisitos

- Python ≥ 3.9  
- Librerías:
  ```bash
  pip install numpy pandas matplotlib optuna
