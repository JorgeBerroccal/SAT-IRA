"""
Hipótesis H3 — ¿Agregar la temperatura mínima reduce el error del LSTM?

    python experimento_temperatura.py

Como la estación de Puquio solo tiene datos hasta junio de 2014, la comparación se hace
dentro de 2000-2013, con la misma arquitectura y la misma división para los dos modelos:
    entrenamiento : objetivos hasta 2010-S52
    validación    : 2011 (parada temprana)
    prueba        : 2012-2013

Modelos (ensamble de 5 semillas cada uno, una red por ámbito):
    LSTM sin temperatura : casos (log, Min-Max), nivel relativo, seno y coseno de la semana
    LSTM con temperatura : lo mismo + temperatura mínima semanal de Puquio (Min-Max)
Referencias: persistencia e ingenuo estacional.
Comparación de errores: prueba de Diebold-Mariano con la corrección de Harvey et al. (1997).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from modelo_lstm import VENTANA, HORIZONTE, Escalador, caracteristicas, entrada, construir_modelo  # noqa: E402
import baselines as bl  # noqa: E402

AMBITOS = ["AYACUCHO (departamento)", "LUCANAS"]
FIN_TRAIN = pd.Timestamp("2011-01-02")      # inicio de 2011-S01
ANIO_VAL, ANIOS_TEST = 2011, (2012, 2013)
SEMILLAS = (1, 2, 3, 4, 5)
OUT = RAIZ / "resultados" / "temperatura"


def preparar(ambito):
    largo = pd.read_csv(RAIZ / "data" / "series_semanales_ayacucho.csv", parse_dates=["fecha"])
    tmin = pd.read_csv(OUT / "tmin_semanal_puquio.csv")
    s = largo[largo["ambito"] == ambito].merge(tmin[["anio", "semana", "tmin"]], on=["anio", "semana"], how="left")
    s = s[(s["anio"] >= 2000) & (s["anio"] <= 2013)].reset_index(drop=True)
    assert s["tmin"].notna().all(), "faltan semanas de temperatura"
    entreno = s["fecha"] < FIN_TRAIN
    esc = Escalador().ajustar(s.loc[entreno, "casos"].values)
    base = caracteristicas(s.drop(columns=["tmin"]), esc)                 # z, sen, cos
    tmn, tmx = s.loc[entreno, "tmin"].min(), s.loc[entreno, "tmin"].max()  # escala solo con entrenamiento
    t = ((s["tmin"] - tmn) / (tmx - tmn)).values.astype("float32")[:, None]
    X = {"sin temperatura": base, "con temperatura": np.hstack([base, t])}
    idx = np.arange(VENTANA - 1, len(s) - HORIZONTE)
    f = s["fecha"].values
    sp = []
    for o in idx:
        fin = pd.Timestamp(f[o + HORIZONTE]); ini = pd.Timestamp(f[o + 1])
        if fin < FIN_TRAIN:
            sp.append("train")
        elif ini.year == ANIO_VAL and fin.year == ANIO_VAL:
            sp.append("val")
        elif ini.year in ANIOS_TEST and fin.year in ANIOS_TEST:
            sp.append("test")
        else:
            sp.append("fuera")
    return s, esc, X, idx, np.array(sp)


def ventanas_de(Xm, idx):
    xs = np.array([entrada(Xm[t - VENTANA + 1:t + 1]) for t in idx])
    ys = np.array([Xm[t + 1:t + 1 + HORIZONTE, 0] - Xm[t, 0] for t in idx])
    return xs, ys


def diebold_mariano(e1, e2, h):
    """DM con pérdida absoluta y corrección de Harvey, Leybourne y Newbold (1997).
    d = |e1| - |e2|; d < 0 favorece al modelo 1. Devuelve (estadístico, p bilateral)."""
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)


def main(epocas=300):
    import tensorflow as tf
    from tensorflow import keras
    OUT.mkdir(parents=True, exist_ok=True)
    filas, filas_semilla, dm_filas, preds_guardar = [], [], [], []
    for amb in AMBITOS:
        s, esc, X, idx, sp = preparar(amb)
        casos = s["casos"].values.astype(float)
        org_test = idx[sp == "test"]
        y = np.array([casos[t + 1:t + 1 + HORIZONTE] for t in org_test])
        preds = {}
        for variante, Xm in X.items():
            xtr, ytr = ventanas_de(Xm, idx[sp == "train"])
            xva, yva = ventanas_de(Xm, idx[sp == "val"])
            xte, _ = ventanas_de(Xm, org_test)
            deltas = []
            for sem in SEMILLAS:
                tf.keras.utils.set_random_seed(sem)
                m = construir_modelo(xtr.shape[-1])
                t0 = time.time()
                h = m.fit(xtr, ytr, validation_data=(xva, yva), epochs=epocas, batch_size=64, verbose=0,
                          callbacks=[keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                                     keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)])
                dlt = np.asarray(m(xte, training=False))
                deltas.append(dlt)
                p_sem = esc.invertir(Xm[org_test, 0][:, None] + dlt)
                filas_semilla.append({"ambito": amb, "modelo": f"LSTM {variante}", "semilla": sem,
                                      "epocas": len(h.history["loss"]), "MAE": bl.mae(y.ravel(), p_sem.ravel())})
                print(f"{amb} | {variante} | semilla {sem}: {len(h.history['loss'])} épocas, "
                      f"{time.time() - t0:.0f} s, MAE={filas_semilla[-1]['MAE']:.1f}", flush=True)
            preds[f"LSTM {variante}"] = esc.invertir(Xm[org_test, 0][:, None] + np.mean(deltas, axis=0))
        preds["Persistencia"] = bl.persistencia(casos, org_test)
        preds["Ingenuo estacional"] = bl.ingenuo_estacional(casos, org_test)
        for mod, p in preds.items():
            for hz in range(HORIZONTE):
                filas.append({"ambito": amb, "modelo": mod, "horizonte": str(hz + 1), **bl.metricas(y[:, hz], p[:, hz])})
            filas.append({"ambito": amb, "modelo": mod, "horizonte": "1-4", **bl.metricas(y.ravel(), p.ravel())})
        for hz in range(HORIZONTE):
            e_con = y[:, hz] - preds["LSTM con temperatura"][:, hz]
            e_sin = y[:, hz] - preds["LSTM sin temperatura"][:, hz]
            est, p = diebold_mariano(e_con, e_sin, hz + 1)
            dm_filas.append({"ambito": amb, "horizonte": hz + 1, "n": len(e_con),
                             "MAE_con": float(np.mean(np.abs(e_con))), "MAE_sin": float(np.mean(np.abs(e_sin))),
                             "DM_HLN": est, "p": p})
        for i, t in enumerate(org_test):
            for hz in range(HORIZONTE):
                preds_guardar.append({"ambito": amb, "origen": s.at[t, "semana_txt"], "h": hz + 1,
                                      "fecha_objetivo": s.at[t + hz + 1, "fecha"].date().isoformat(),
                                      "real": y[i, hz], **{m: round(float(p[i, hz]), 1) for m, p in preds.items()}})
        print(f"{amb}: ventanas train={int((sp == 'train').sum())} val={int((sp == 'val').sum())} "
              f"test={int((sp == 'test').sum())}")
    met = pd.DataFrame(filas)
    met.to_csv(OUT / "metricas_h3.csv", index=False)
    pd.DataFrame(filas_semilla).to_csv(OUT / "metricas_h3_por_semilla.csv", index=False)
    dm = pd.DataFrame(dm_filas)
    dm.to_csv(OUT / "diebold_mariano_h3.csv", index=False)
    pd.DataFrame(preds_guardar).to_csv(OUT / "pronosticos_h3.csv", index=False)
    for amb in AMBITOS:
        print("\n", amb)
        print(met[met["ambito"] == amb].pivot_table(index="modelo", columns="horizonte",
                                                   values=["MAE", "MAPE_%"], aggfunc="first").round(1).to_string())
    print(pd.DataFrame(filas_semilla).groupby(["ambito", "modelo"])["MAE"].agg(["mean", "std", "min", "max"]).round(1))
    print(dm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
