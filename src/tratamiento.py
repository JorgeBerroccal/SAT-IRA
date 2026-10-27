"""
Componente 1 — Tratamiento de datos
===================================
Lee el CSV de datos abiertos de vigilancia de IRA del MINSA (CDC Perú),
lo limpia y arma la serie semanal del departamento de Ayacucho
(y de cada una de sus 11 provincias).

El archivo original no trae encabezados. Las columnas son (diccionario MINSA):
    departamento; provincia; distrito; ano; semana; sub_reg_nt; ubigeo;
    ira_no_neumonia; neumonias_men5; neumonias_60mas;
    hospitalizados_men5; hospitalizados_60mas;
    defunciones_men5; defunciones_60mas
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNAS = [
    "departamento", "provincia", "distrito", "anio", "semana", "sub_reg_nt", "ubigeo",
    "ira_no_neumonia", "neumonias_men5", "neumonias_60mas",
    "hospitalizados_men5", "hospitalizados_60mas",
    "defunciones_men5", "defunciones_60mas",
]
NUMERICAS = COLUMNAS[7:]
DEPARTAMENTO = "AYACUCHO"


# ---------------------------------------------------------------------------
# Semana epidemiológica -> fecha
# ---------------------------------------------------------------------------
def inicio_semana_epi(anio: int, semana: int) -> dt.date:
    """Fecha (domingo) en que empieza la semana epidemiológica `semana` del año `anio`.

    La semana 1 es la primera semana domingo–sábado que tiene al menos 4 días en enero.
    """
    ene1 = dt.date(anio, 1, 1)
    dias_desde_domingo = (ene1.weekday() + 1) % 7      # lunes=0 ... domingo=6  -> domingo=0
    domingo = ene1 - dt.timedelta(days=dias_desde_domingo)
    if dias_desde_domingo > 3:                         # 1 de enero cae jueves, viernes o sábado
        domingo += dt.timedelta(days=7)
    return domingo + dt.timedelta(weeks=semana - 1)


def semana_epi_de_fecha(fecha) -> tuple[int, int]:
    """(año, semana) epidemiológicos de una fecha cualquiera."""
    d = pd.Timestamp(fecha).date()
    for anio in (d.year + 1, d.year, d.year - 1):
        inicio = inicio_semana_epi(anio, 1)
        if d >= inicio:
            return anio, (d - inicio).days // 7 + 1
    raise ValueError(fecha)


def semanas_del_anio(anio: int) -> int:
    """52 o 53 semanas epidemiológicas."""
    return 53 if inicio_semana_epi(anio, 53) < inicio_semana_epi(anio + 1, 1) else 52


# ---------------------------------------------------------------------------
# Lectura y limpieza
# ---------------------------------------------------------------------------
def leer_csv_minsa(ruta: str | Path, departamento: str = DEPARTAMENTO) -> pd.DataFrame:
    """Lee el CSV crudo (sin encabezado, separado por ';') y deja solo el departamento pedido."""
    df = pd.read_csv(
        ruta, sep=";", header=None, names=COLUMNAS, encoding="utf-8-sig",
        dtype={"sub_reg_nt": str, "ubigeo": str},
    )
    for c in ["departamento", "provincia", "distrito"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df = df[df["departamento"] == departamento.upper()].copy()

    # Limpieza
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df = df[(df["semana"] >= 1) & (df["semana"] <= 53)]
    df = df.drop_duplicates()
    return df


def construir_serie(df: pd.DataFrame, provincia: str | None = None) -> pd.DataFrame:
    """Agrega por semana epidemiológica y devuelve una serie continua (sin semanas faltantes).

    casos = IRA no neumónica + neumonías (<5 años y 60+ años)
    """
    if provincia:
        df = df[df["provincia"] == provincia.upper()]
    g = df.groupby(["anio", "semana"], as_index=False)[NUMERICAS].sum()

    # Calendario completo de semanas epidemiológicas para no dejar huecos
    anios = range(int(g["anio"].min()), int(g["anio"].max()) + 1)
    cal = pd.DataFrame(
        [(a, s) for a in anios for s in range(1, semanas_del_anio(a) + 1)],
        columns=["anio", "semana"],
    )
    serie = cal.merge(g, on=["anio", "semana"], how="left")
    faltantes = serie[NUMERICAS[0]].isna().sum()
    serie[NUMERICAS] = serie[NUMERICAS].interpolate(limit_direction="both").round().astype(int)

    serie["neumonias"] = serie["neumonias_men5"] + serie["neumonias_60mas"]
    serie["casos"] = serie["ira_no_neumonia"] + serie["neumonias"]
    serie["fecha"] = pd.to_datetime([inicio_semana_epi(a, s) for a, s in zip(serie.anio, serie.semana)])
    serie["semana_txt"] = serie["anio"].astype(str) + "-S" + serie["semana"].astype(str).str.zfill(2)
    serie.attrs["semanas_interpoladas"] = int(faltantes)
    return serie[["fecha", "anio", "semana", "semana_txt", "casos", "ira_no_neumonia", "neumonias",
                  "neumonias_men5", "neumonias_60mas", "hospitalizados_men5", "hospitalizados_60mas",
                  "defunciones_men5", "defunciones_60mas"]]


def construir_todas(ruta_csv: str | Path, carpeta_salida: str | Path) -> pd.DataFrame:
    """Arma la serie del departamento y de cada provincia y la guarda en un solo CSV largo."""
    df = leer_csv_minsa(ruta_csv)
    partes = []
    s = construir_serie(df)
    s.insert(0, "ambito", "AYACUCHO (departamento)")
    partes.append(s)
    for prov in sorted(df["provincia"].unique()):
        s = construir_serie(df, prov)
        s.insert(0, "ambito", prov)
        partes.append(s)
    largo = pd.concat(partes, ignore_index=True)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    largo.to_csv(carpeta_salida / "series_semanales_ayacucho.csv", index=False)
    return largo


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_abiertos_vigilancia_iras_2000_2024.csv"
    out = construir_todas(ruta, Path(__file__).resolve().parents[1] / "data")
    print(out.groupby("ambito")["casos"].agg(["count", "sum", "mean", "max"]))
