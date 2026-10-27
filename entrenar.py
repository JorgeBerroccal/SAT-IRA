"""
Pipeline completo de entrenamiento y evaluación.

    python entrenar.py --csv datos_abiertos_vigilancia_iras_2000_2024.csv

1. Tratamiento de datos  -> data/series_semanales_ayacucho.csv
2. Entrena las LSTM      -> modelos/*.keras + modelos/escaladores.json
       · Departamento: ensamble de 5 LSTM (semillas 1..5)
       · Provincias:   1 LSTM global compartida por las 11 provincias
3. Evalúa vs. líneas base (ingenuo estacional, persistencia, SARIMA) -> resultados/
4. Genera los pronósticos de todo el historial (modo simulación del tablero)
   y el pronóstico de las próximas 4 semanas.

División temporal (sin mezclar, respetando el orden del tiempo):
    entrenamiento : 2000 – feb 2020
    (excluido)    : mar 2020 – dic 2021   (pandemia COVID-19: notificación anómala)
    validación    : 2022                  (parada temprana e intervalos de pronóstico)
    prueba        : 2023 – 2024           (métricas finales)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from tratamiento import construir_todas, semana_epi_de_fecha  # noqa: E402
from modelo_lstm import (VENTANA, HORIZONTE, Escalador, caracteristicas, ventanas,  # noqa: E402
                         construir_modelo, guardar_escaladores, pronosticar)
import baselines as bl  # noqa: E402

DEPTO = "AYACUCHO (departamento)"
FIN_ENTRENAMIENTO = pd.Timestamp("2020-03-01")
FIN_PANDEMIA = pd.Timestamp("2021-12-31")
ANIO_VALIDACION = 2022
ANIOS_PRUEBA = (2023, 2024)
SEMILLAS_DEPTO = (1, 2, 3, 4, 5)


def dividir(fechas: pd.Series, idx: np.ndarray) -> np.ndarray:
    """Asigna cada ventana (por su origen t) a train / val / test / fuera."""
    f = fechas.values
    split = []
    for t in idx:
        f_in0, f_out0, f_out1 = f[t - VENTANA + 1], f[t + 1], f[t + HORIZONTE]
        if f_out1 < np.datetime64(FIN_ENTRENAMIENTO):
            split.append("train")
        elif f_in0 > np.datetime64(FIN_PANDEMIA) and pd.Timestamp(f_out1).year == ANIO_VALIDACION:
            split.append("val")
        elif pd.Timestamp(f_out0).year in ANIOS_PRUEBA and pd.Timestamp(f_out1).year in ANIOS_PRUEBA:
            split.append("test")
        else:
            split.append("fuera")
    return np.array(split)


def entrenar_red(datos, semilla, epocas, etiqueta):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(semilla)
    modelo = construir_modelo(datos["train"][0].shape[-1])
    t0 = time.time()
    hist = modelo.fit(
        *datos["train"], validation_data=datos["val"], epochs=epocas, batch_size=64, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)],
    )
    h = pd.DataFrame(hist.history)
    print(f"      {etiqueta}: {len(h)} épocas, {time.time() - t0:.0f} s, "
          f"mejor val_loss={h['val_loss'].min():.5f}")
    return modelo, h


def main(ruta_csv: str | None, epocas: int = 300):
    (RAIZ / "modelos").mkdir(exist_ok=True)
    (RAIZ / "resultados").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1. datos
    ruta_series = RAIZ / "data" / "series_semanales_ayacucho.csv"
    if ruta_csv:
        print("[1/4] Tratamiento de datos ...")
        largo = construir_todas(ruta_csv, RAIZ / "data")
    else:
        print("[1/4] Usando serie ya procesada:", ruta_series)
        largo = pd.read_csv(ruta_series, parse_dates=["fecha"])
    largo["fecha"] = pd.to_datetime(largo["fecha"])
    series = {a: largo[largo["ambito"] == a].reset_index(drop=True) for a in largo["ambito"].unique()}

    # ------------------------------------------------------------------ 2. ventanas
    escaladores, prep = {}, {}
    for a, s in series.items():
        esc = Escalador().ajustar(s.loc[s["fecha"] < FIN_ENTRENAMIENTO, "casos"].values)
        escaladores[a] = esc
        Xv, yv, idx = ventanas(caracteristicas(s, esc))
        prep[a] = (Xv, yv, idx, dividir(s["fecha"], idx))

    def juntar(ambitos):
        return {k: (np.concatenate([prep[a][0][prep[a][3] == k] for a in ambitos]),
                    np.concatenate([prep[a][1][prep[a][3] == k] for a in ambitos]))
                for k in ("train", "val", "test")}

    provincias = [a for a in series if a != DEPTO]
    datos_depto, datos_prov = juntar([DEPTO]), juntar(provincias)
    for nombre, d in (("departamento", datos_depto), ("provincias", datos_prov)):
        print(f"      {nombre:12s} " + "  ".join(f"{k}={v[0].shape[0]}" for k, v in d.items()))

    # ------------------------------------------------------------------ 3. entrenamiento
    print("[2/4] Entrenando LSTM ...")
    for p in (RAIZ / "modelos").glob("*.keras"):
        p.unlink()
    modelos_depto, historiales = [], []
    for sem in SEMILLAS_DEPTO:
        m, h = entrenar_red(datos_depto, sem, epocas, f"departamento semilla {sem}")
        m.save(RAIZ / "modelos" / f"lstm_departamento_s{sem}.keras")
        modelos_depto.append(m)
        historiales.append(h.assign(modelo=f"departamento_s{sem}"))
    modelo_prov, h = entrenar_red(datos_prov, 42, epocas, "provincias (global)")
    modelo_prov.save(RAIZ / "modelos" / "lstm_provincias.keras")
    historiales.append(h.assign(modelo="provincias"))
    pd.concat([h.rename_axis("epoca").reset_index() for h in historiales]).to_csv(
        RAIZ / "resultados" / "historial_entrenamiento.csv", index=False)
    guardar_escaladores(escaladores, RAIZ / "modelos" / "escaladores.json")
    modelos_depto[0].summary(print_fn=lambda l: print("      " + l))

    def modelo_de(a):
        return modelos_depto if a == DEPTO else modelo_prov

    # ------------------------------------------------------------------ 4. evaluación
    print("[3/4] Evaluando en prueba (2023-2024) ...")
    filas, bandas, prueba_depto = [], {}, None
    for a, s in series.items():
        esc, (_, _, idx, sp) = escaladores[a], prep[a]
        casos = s["casos"].values.astype(float)

        # Intervalo empírico de 80 % con los residuos (log) del año de validación
        org_val = idx[sp == "val"]
        p_val = pronosticar(modelo_de(a), s, esc, org_val)
        y_val = np.array([casos[t + 1:t + 1 + HORIZONTE] for t in org_val])
        r = np.log1p(y_val) - np.log1p(p_val)
        bandas[a] = {"inf": np.quantile(r, 0.10, axis=0).round(4).tolist(),
                     "sup": np.quantile(r, 0.90, axis=0).round(4).tolist()}

        org = idx[sp == "test"]
        y = np.array([casos[t + 1:t + 1 + HORIZONTE] for t in org])
        preds = {
            "LSTM": pronosticar(modelo_de(a), s, esc, org),
            "Ingenuo estacional": bl.ingenuo_estacional(casos, org),
            "Persistencia": bl.persistencia(casos, org),
        }
        if a == DEPTO:
            i0 = int(np.argmax(s["fecha"] >= pd.Timestamp("2012-01-01")))
            i1 = int(np.argmax(s["fecha"] >= FIN_ENTRENAMIENTO))
            t0 = time.time()
            preds["SARIMA"], _ = bl.sarima(casos, i0, i1, org)
            print(f"      SARIMA(1,0,1)(0,1,1)52 listo en {time.time() - t0:.0f} s")
            prueba_depto = (s.iloc[org], y, preds)
        for nombre, p in preds.items():
            for h in range(HORIZONTE):
                filas.append({"ambito": a, "modelo": nombre, "horizonte": str(h + 1),
                              **bl.metricas(y[:, h], p[:, h])})
            filas.append({"ambito": a, "modelo": nombre, "horizonte": "1-4",
                          **bl.metricas(y.ravel(), p.ravel())})
    met = pd.DataFrame(filas)
    met.to_csv(RAIZ / "resultados" / "metricas_prueba.csv", index=False)

    tabla = met[met["ambito"] == DEPTO].pivot_table(index="modelo", columns="horizonte",
                                                    values=["MAE", "RMSE", "MAPE_%"], aggfunc="first")
    print(tabla.round(1).to_string())
    resumen_prov = met[(met["ambito"] != DEPTO) & (met["horizonte"] == "1-4")].pivot_table(
        index="ambito", columns="modelo", values="MAPE_%")
    print("\nMAPE (%) por provincia, horizontes 1-4:\n" + resumen_prov.round(1).to_string())

    base, y, preds = prueba_depto
    filas = []
    for i, (_, r0) in enumerate(base.iterrows()):
        for h in range(HORIZONTE):
            fila = {"origen": r0["semana_txt"], "h": h + 1, "real": y[i, h]}
            fila.update({m: round(float(p[i, h]), 1) for m, p in preds.items()})
            filas.append(fila)
    pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "pronosticos_prueba_departamento.csv", index=False)

    # ------------------------------------------------------------------ 5. historial completo
    print("[4/4] Pronósticos para el tablero ...")
    filas = []
    for a, s in series.items():
        org = np.arange(VENTANA - 1, len(s))
        p = pronosticar(modelo_de(a), s, escaladores[a], org)
        ultima = pd.Timestamp(s["fecha"].iloc[-1])
        for i, t in enumerate(org):
            for h in range(HORIZONTE):
                j = t + h + 1
                if j < len(s):
                    af, sf = int(s.at[j, "anio"]), int(s.at[j, "semana"])
                else:  # semanas futuras, más allá del último dato
                    af, sf = semana_epi_de_fecha(ultima + pd.Timedelta(weeks=j - len(s) + 1))
                filas.append({"ambito": a, "origen_anio": int(s.at[t, "anio"]),
                              "origen_semana": int(s.at[t, "semana"]), "h": h + 1,
                              "anio": af, "semana": sf, "pred": round(float(p[i, h]), 1)})
    pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "pronosticos_historicos.csv", index=False)

    info = {
        "ventana": VENTANA, "horizonte": HORIZONTE,
        "parametros_por_red": int(modelos_depto[0].count_params()),
        "division": {"entrenamiento": "2000-S01 a 2020-S09",
                     "excluido": "2020-S10 a 2021-S52 (COVID-19)",
                     "validacion": str(ANIO_VALIDACION), "prueba": "2023-2024"},
        "n_ventanas": {"departamento": {k: int(v[0].shape[0]) for k, v in datos_depto.items()},
                       "provincias": {k: int(v[0].shape[0]) for k, v in datos_prov.items()}},
        "bandas_log_80": bandas,
        "ultima_semana": series[DEPTO]["semana_txt"].iloc[-1],
    }
    (RAIZ / "modelos" / "info_modelo.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print("Listo. Modelos en modelos/, métricas y pronósticos en resultados/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV crudo del MINSA (si se omite, usa data/series_semanales_ayacucho.csv)")
    ap.add_argument("--epocas", type=int, default=300)
    a = ap.parse_args()
    main(a.csv, a.epocas)
