"""
Exporta los resultados del modelo a una versión web estática del tablero
(un solo archivo HTML que se abre en cualquier navegador, sin Python).

    python exportar_web.py     ->  web/tablero_estatico.html

Usa: data/series_semanales_ayacucho.csv, resultados/*.csv, modelos/info_modelo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from alertas import canal_endemico, nivel_alerta  # noqa: E402
from tratamiento import semana_epi_de_fecha  # noqa: E402

DEPTO = "AYACUCHO (departamento)"
ANIO_INICIO_WEB = 2016          # semanas disponibles en el modo simulación


def main():
    largo = pd.read_csv(RAIZ / "data" / "series_semanales_ayacucho.csv", parse_dates=["fecha"])
    pron = pd.read_csv(RAIZ / "resultados" / "pronosticos_historicos.csv")
    met = pd.read_csv(RAIZ / "resultados" / "metricas_prueba.csv", dtype={"horizonte": str})
    info = json.loads((RAIZ / "modelos" / "info_modelo.json").read_text(encoding="utf-8"))
    ambitos = [DEPTO] + sorted(a for a in largo["ambito"].unique() if a != DEPTO)

    base = largo[largo["ambito"] == DEPTO].reset_index(drop=True)
    i0 = int(np.argmax(base["anio"] >= ANIO_INICIO_WEB - 2))   # 2 años previos para el gráfico
    cal = base.iloc[i0:][["fecha", "anio", "semana"]].copy()
    ultima = base["fecha"].iloc[-1]
    fut = [semana_epi_de_fecha(ultima + pd.Timedelta(weeks=k)) for k in range(1, 5)]
    cal_f = cal["fecha"].dt.strftime("%Y-%m-%d").tolist() + \
        [(ultima + pd.Timedelta(weeks=k)).strftime("%Y-%m-%d") for k in range(1, 5)]
    cal_a = cal["anio"].tolist() + [a for a, _ in fut]
    cal_s = cal["semana"].tolist() + [s for _, s in fut]
    n_obs = len(cal)
    o0 = int(np.argmax(cal["anio"].values >= ANIO_INICIO_WEB))

    datos = {
        "meta": {"ultima": base["semana_txt"].iloc[-1], "n_obs": n_obs, "o0": o0,
                 "ventana": info["ventana"], "division": info["division"]},
        "cal": {"f": cal_f, "a": cal_a, "s": cal_s},
        "ambitos": ambitos, "series": {}, "pron": {}, "canal": {}, "bandas": info["bandas_log_80"],
        "metricas": {}, "eval": {},
    }
    for amb in ambitos:
        s = largo[largo["ambito"] == amb].reset_index(drop=True)
        datos["series"][amb] = s["casos"].iloc[i0:].astype(int).tolist()

        p = pron[pron["ambito"] == amb]
        p = p[p["origen_anio"] >= ANIO_INICIO_WEB].sort_values(["origen_anio", "origen_semana", "h"])
        datos["pron"][amb] = np.round(p["pred"].values.reshape(-1, 4), 1).tolist()

        can = {}
        for anio in range(ANIO_INICIO_WEB - 2, 2026):
            c = canal_endemico(s, anio)
            can[str(anio)] = np.round(c[["q1", "q2", "q3"]].values, 1).tolist()
        datos["canal"][amb] = can

        m = met[met["ambito"] == amb]
        datos["metricas"][amb] = {
            mod: {"MAE": round(float(g.loc[g.horizonte == "1-4", "MAE"].iloc[0]), 1),
                  "RMSE": round(float(g.loc[g.horizonte == "1-4", "RMSE"].iloc[0]), 1),
                  "MAPE": round(float(g.loc[g.horizonte == "1-4", "MAPE_%"].iloc[0]), 1),
                  "porH": [round(float(g.loc[g.horizonte == str(h), "MAE"].iloc[0]), 1) for h in range(1, 5)]}
            for mod, g in m.groupby("modelo")}

        # ¿Acierta el semáforo? (prueba 2023-2024, 2 semanas de anticipación)
        ev = pron[(pron["ambito"] == amb) & (pron["h"] == 2) & (pron["anio"].isin([2023, 2024]))]
        cc = pd.concat([canal_endemico(s, a).assign(anio=a) for a in (2023, 2024)])
        ev = ev.merge(cc, on=["anio", "semana"]).merge(s[["anio", "semana", "casos"]], on=["anio", "semana"])
        npred = np.array([nivel_alerta(a, b, c) for a, b, c in zip(ev["pred"], ev["q2"], ev["q3"])])
        nreal = np.array([nivel_alerta(a, b, c) for a, b, c in zip(ev["casos"], ev["q2"], ev["q3"])])
        datos["eval"][amb] = {
            "acierto": round(100 * float((npred == nreal).mean()), 0),
            "sens": round(100 * float(((npred > 0) & (nreal > 0)).sum() / max((nreal > 0).sum(), 1)), 0),
            "n": int(len(ev)),
        }

    plantilla = (RAIZ / "web" / "plantilla.html").read_text(encoding="utf-8")
    salida = plantilla.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    completo = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                '<style>body{margin:0}</style>\n</head>\n<body>\n' + salida + '\n</body>\n</html>\n')
    (RAIZ / "web" / "tablero_estatico.html").write_text(completo, encoding="utf-8")
    print(f"web/tablero_estatico.html generado ({len(completo) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
