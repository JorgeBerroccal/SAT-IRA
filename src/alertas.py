"""
Componente 3 — Módulo de alertas
================================
Compara el pronóstico con lo "normal" para esa época del año (canal endémico)
y lo convierte en un semáforo:  🟢 normal · 🟡 alerta · 🔴 emergencia.

Canal endémico (Bortman, 1999): para cada semana epidemiológica se toman los
casos de esa misma semana en los últimos N años y se calculan los cuartiles.
    Q1  -> límite de la zona de éxito
    Q2  -> (mediana) límite de la zona de seguridad
    Q3  -> límite de la zona de alarma
    > Q3 -> zona de epidemia

Correspondencia con el semáforo del sistema:
    pronóstico <= Q2        -> 🟢 NORMAL      (zonas de éxito y seguridad)
    Q2 < pronóstico <= Q3   -> 🟡 ALERTA      (zona de alarma)
    pronóstico > Q3         -> 🔴 EMERGENCIA  (zona de epidemia)

Los años 2020 y 2021 se excluyen del canal porque la pandemia de COVID-19
alteró la notificación de IRA (caída artificial de casos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANIOS_EXCLUIDOS = (2020, 2021)
N_ANIOS = 7

NIVELES = {
    0: {"nivel": "NORMAL", "emoji": "🟢", "color": "#2e9e5b",
        "mensaje": "Casos dentro de lo esperado para la época."},
    1: {"nivel": "ALERTA", "emoji": "🟡", "color": "#e0a100",
        "mensaje": "Casos por encima de la mediana histórica: reforzar vigilancia y stock."},
    2: {"nivel": "EMERGENCIA", "emoji": "🔴", "color": "#d63b3b",
        "mensaje": "Casos por encima del tercer cuartil: activar plan de respuesta."},
}


def anios_referencia(anio: int, disponibles, n: int = N_ANIOS, excluir=ANIOS_EXCLUIDOS) -> list[int]:
    """Los `n` años anteriores a `anio` con datos, sin contar los años excluidos."""
    candidatos = sorted([a for a in disponibles if a < anio and a not in excluir], reverse=True)
    return sorted(candidatos[:n])


def canal_endemico(serie: pd.DataFrame, anio: int, n: int = N_ANIOS,
                   excluir=ANIOS_EXCLUIDOS, suavizar: bool = True) -> pd.DataFrame:
    """Canal endémico (Q1, Q2, Q3 por semana) para el año `anio`.

    `serie` debe tener columnas: anio, semana, casos.
    Si `suavizar`, cada semana usa también las semanas vecinas (±1) de los años de
    referencia, lo que estabiliza el canal en provincias con pocos casos.
    """
    refs = anios_referencia(anio, serie["anio"].unique(), n, excluir)
    base = serie[serie["anio"].isin(refs)][["anio", "semana", "casos"]].copy()
    filas = []
    for s in range(1, 54):
        if s == 53:                       # pocos años tienen semana 53: se apoya en la 52
            vecinas = [52, 53]
        elif suavizar:
            vecinas = [52 if s == 1 else s - 1, s, 1 if s == 52 else s + 1]
        else:
            vecinas = [s]
        v = base[base["semana"].isin(vecinas)]["casos"].values
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        filas.append({"semana": s, "q1": q1, "q2": q2, "q3": q3})
    canal = pd.DataFrame(filas)
    canal["anio"] = anio
    canal.attrs["anios_referencia"] = refs
    return canal


def nivel_alerta(valor: float, q2: float, q3: float) -> int:
    if valor > q3:
        return 2
    if valor > q2:
        return 1
    return 0


def evaluar_pronostico(pronostico: pd.DataFrame, canal: pd.DataFrame,
                       ultimo_observado: float | None = None) -> dict:
    """Asigna el semáforo a un pronóstico de 2-4 semanas.

    pronostico: DataFrame con columnas anio, semana, pred (una fila por semana futura).
    canal: DataFrame del canal endémico con columnas anio, semana, q1, q2, q3
           (puede incluir dos años si el horizonte cruza fin de año).
    Devuelve el nivel global (el peor de las semanas pronosticadas), el detalle por
    semana y la tendencia respecto a la última semana observada.
    """
    det = pronostico.merge(canal, on=["anio", "semana"], how="left")
    det["nivel_id"] = [nivel_alerta(p, a, b) for p, a, b in zip(det["pred"], det["q2"], det["q3"])]
    det["nivel"] = det["nivel_id"].map(lambda i: NIVELES[i]["nivel"])
    det["emoji"] = det["nivel_id"].map(lambda i: NIVELES[i]["emoji"])
    det["exceso_vs_mediana_%"] = 100 * (det["pred"] - det["q2"]) / det["q2"].replace(0, np.nan)

    peor = int(det["nivel_id"].max())
    tendencia, cambio = "estable", 0.0
    if ultimo_observado and ultimo_observado > 0:
        cambio = 100 * (det["pred"].mean() - ultimo_observado) / ultimo_observado
        tendencia = "en aumento" if cambio > 10 else ("en descenso" if cambio < -10 else "estable")
    return {
        **NIVELES[peor],
        "nivel_id": peor,
        "detalle": det,
        "tendencia": tendencia,
        "cambio_%": cambio,
    }
