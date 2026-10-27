"""
Componente 2 — Modelo LSTM
==========================
Aprende del historial y pronostica los casos de las próximas 4 semanas
(horizonte de 2 a 4 semanas, más la semana 1 como referencia).

Aprendizaje supervisado de regresión con ventanas deslizantes:
    entrada X  = últimas VENTANA semanas, 4 variables por semana:
                 [nivel relativo a la última semana, casos (log + Min-Max), sen(semana), cos(semana)]
    etiqueta y = cambio de los casos (en escala log normalizada) en las HORIZONTE semanas
                 siguientes respecto a la última semana observada.

Predecir el *cambio* en vez del valor absoluto hace al modelo robusto a los cambios de
nivel de la serie (p. ej. la caída de casos después de la pandemia de COVID-19).

Modelos entrenados:
    - Departamento de Ayacucho: ensamble de 5 LSTM (distinta semilla); el pronóstico es el promedio.
    - Provincias: una LSTM "global" compartida por las 11 provincias (cada serie normalizada aparte).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VENTANA = 16       # semanas de historia que mira el modelo
HORIZONTE = 4      # semanas que pronostica
PERIODO = 52.18    # semanas por año (para la codificación estacional)


# ---------------------------------------------------------------------------
# Escalado (Min-Max sobre log1p, ajustado solo con datos de entrenamiento)
# ---------------------------------------------------------------------------
class Escalador:
    def __init__(self, minimo: float = 0.0, maximo: float = 1.0):
        self.minimo, self.maximo = minimo, maximo

    def ajustar(self, casos: np.ndarray) -> "Escalador":
        z = np.log1p(casos)
        self.minimo, self.maximo = float(z.min()), float(z.max())
        return self

    def transformar(self, casos):
        return (np.log1p(casos) - self.minimo) / (self.maximo - self.minimo)

    def invertir(self, z):
        return np.expm1(np.asarray(z) * (self.maximo - self.minimo) + self.minimo).clip(min=0)

    def a_dict(self):
        return {"minimo": self.minimo, "maximo": self.maximo}


def caracteristicas(serie: pd.DataFrame, esc: Escalador) -> np.ndarray:
    """Matriz (n_semanas, 3): casos escalados, seno y coseno de la semana del año."""
    ang = 2 * np.pi * serie["semana"].values / PERIODO
    cols = [esc.transformar(serie["casos"].values.astype(float)), np.sin(ang), np.cos(ang)]
    if "tmin" in serie.columns:               # temperatura mínima SENAMHI (opcional)
        t = serie["tmin"].values.astype(float)
        cols.append((t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t)))
    return np.column_stack(cols).astype("float32")


def entrada(bloque: np.ndarray) -> np.ndarray:
    """Agrega la columna 'nivel relativo' (casos de cada semana menos los de la última)."""
    return np.column_stack([bloque[:, 0] - bloque[-1, 0], bloque]).astype("float32")


def ventanas(X: np.ndarray, ventana=VENTANA, horizonte=HORIZONTE):
    """Ventanas deslizantes sobre la matriz de características X.

    Devuelve X_v (n, ventana, f+1), y_v (n, horizonte) = cambio respecto a la última semana,
    e idx = posición de la última semana observada (origen del pronóstico).
    """
    xs, ys, idx = [], [], []
    for t in range(ventana - 1, len(X) - horizonte):
        xs.append(entrada(X[t - ventana + 1: t + 1]))
        ys.append(X[t + 1: t + 1 + horizonte, 0] - X[t, 0])
        idx.append(t)
    return np.array(xs), np.array(ys), np.array(idx)


# ---------------------------------------------------------------------------
# Red neuronal
# ---------------------------------------------------------------------------
def construir_modelo(n_caracteristicas: int, ventana=VENTANA, horizonte=HORIZONTE,
                     unidades=64, dropout=0.2, lr=1e-3):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    modelo = keras.Sequential([
        layers.Input(shape=(ventana, n_caracteristicas)),
        layers.LSTM(unidades, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(unidades // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizonte),
    ], name="lstm_ira_ayacucho")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return modelo


# ---------------------------------------------------------------------------
# Pronóstico
# ---------------------------------------------------------------------------
def pronosticar(modelos, serie: pd.DataFrame, esc: Escalador, origenes=None,
                ventana=VENTANA) -> np.ndarray:
    """Pronostica HORIZONTE semanas desde cada origen (índice de la última semana observada).

    `modelos` puede ser un modelo o una lista (ensamble: se promedian las salidas).
    Si `origenes` es None se pronostica solo desde la última semana de la serie.
    Devuelve una matriz (n_origenes, HORIZONTE) en número de casos.
    """
    if not isinstance(modelos, (list, tuple)):
        modelos = [modelos]
    X = caracteristicas(serie, esc)
    if origenes is None:
        origenes = [len(serie) - 1]
    origenes = np.asarray(origenes)
    lotes = np.array([entrada(X[t - ventana + 1: t + 1]) for t in origenes])
    delta = np.mean([np.asarray(m(lotes, training=False)) for m in modelos], axis=0)
    return esc.invertir(X[origenes, 0][:, None] + delta)


def guardar_escaladores(escaladores: dict, ruta: Path):
    ruta.write_text(json.dumps({k: v.a_dict() for k, v in escaladores.items()}, indent=2,
                               ensure_ascii=False), encoding="utf-8")


def cargar_escaladores(ruta: Path) -> dict:
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {k: Escalador(**v) for k, v in d.items()}


def cargar_modelos(carpeta: Path) -> dict:
    """Carga los modelos entrenados: {'departamento': [5 modelos], 'provincias': modelo}."""
    from tensorflow import keras
    carpeta = Path(carpeta)
    depto = [keras.models.load_model(p) for p in sorted(carpeta.glob("lstm_departamento_*.keras"))]
    prov = keras.models.load_model(carpeta / "lstm_provincias.keras")
    return {"departamento": depto, "provincias": prov}
