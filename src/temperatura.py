"""
Temperatura mínima del SENAMHI
==============================
Lee el archivo de texto que entrega la descarga de datos del SENAMHI
(columnas: año, mes, día, precipitación, temperatura máxima, temperatura mínima;
-99.9 = dato faltante) y lo agrega por semana epidemiológica.

Estación usada: PUQUIO (archivo qc00000736.txt), provincia de Lucanas, Ayacucho.
Datos disponibles: 24/11/1962 a 30/06/2014.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tratamiento import semana_epi_de_fecha

FALTANTE = -99.9
MIN_DIAS_SEMANA = 4      # una semana es válida si tiene al menos 4 días con dato


def leer_senamhi(ruta: str | Path) -> pd.DataFrame:
    """Datos diarios con fecha y las tres variables; -99.9 pasa a NaN."""
    d = pd.read_csv(ruta, sep=r"\s+", header=None,
                    names=["anio", "mes", "dia", "precipitacion", "tmax", "tmin"])
    d = d.replace(FALTANTE, np.nan)
    d["fecha"] = pd.to_datetime(dict(year=d.anio, month=d.mes, day=d.dia))
    # control de calidad básico
    d.loc[d["tmin"] > d["tmax"], ["tmin", "tmax"]] = np.nan
    return d[["fecha", "precipitacion", "tmax", "tmin"]]


def tmin_semanal(diario: pd.DataFrame, desde: str = "1999-12-26") -> pd.DataFrame:
    """Temperatura mínima promedio por semana epidemiológica.

    Las semanas con menos de MIN_DIAS_SEMANA días válidos quedan vacías y luego se
    interpolan solo si el hueco es de 1 o 2 semanas; los huecos mayores quedan vacíos.
    """
    d = diario[diario["fecha"] >= desde].copy()
    ew = [semana_epi_de_fecha(f) for f in d["fecha"]]
    d["anio"] = [a for a, _ in ew]
    d["semana"] = [s for _, s in ew]
    w = d.groupby(["anio", "semana"]).agg(tmin=("tmin", "mean"), dias_validos=("tmin", "count"),
                                          dias=("tmin", "size")).reset_index()
    w.loc[w["dias_validos"] < MIN_DIAS_SEMANA, "tmin"] = np.nan
    w = w[w["dias"] == 7].reset_index(drop=True)          # solo semanas completas en el calendario
    w["tmin_interpolada"] = w["tmin"].isna()
    w["tmin"] = w["tmin"].interpolate(limit=2, limit_area="inside")
    w["tmin_interpolada"] &= w["tmin"].notna()
    return w[["anio", "semana", "tmin", "dias_validos", "tmin_interpolada"]]
