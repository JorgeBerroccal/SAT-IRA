"""
Modelos de comparación (líneas base) y métricas de evaluación.
    - Ingenuo estacional: la misma semana del año anterior.
    - Persistencia: el último valor observado se repite.
    - SARIMA(1,0,1)(0,1,1)52 sobre log(casos+1).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    m = y > 0
    return float(100 * np.mean(np.abs((y[m] - p[m]) / y[m])))


def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE_%": mape(y, p)}


def ingenuo_estacional(casos: np.ndarray, origenes, horizonte=4, periodo=52) -> np.ndarray:
    return np.array([[casos[t + h - periodo] for h in range(1, horizonte + 1)] for t in origenes], float)


def persistencia(casos: np.ndarray, origenes, horizonte=4) -> np.ndarray:
    return np.array([[casos[t]] * horizonte for t in origenes], float)


def sarima(casos: np.ndarray, idx_inicio_ajuste: int, idx_fin_ajuste: int, origenes,
           horizonte=4, orden=(1, 0, 1), estacional=(0, 1, 1, 52)) -> np.ndarray:
    """Ajusta SARIMA con casos[idx_inicio:idx_fin] y pronostica desde cada origen
    reutilizando los parámetros estimados (sin re-estimar)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    z = np.log1p(casos.astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = SARIMAX(z[idx_inicio_ajuste:idx_fin_ajuste], order=orden, seasonal_order=estacional,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        salida = []
        for t in origenes:
            r = ajuste.apply(z[idx_inicio_ajuste: t + 1])
            salida.append(np.expm1(r.forecast(horizonte)))
    return np.clip(np.array(salida), 0, None), ajuste
