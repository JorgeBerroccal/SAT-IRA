"""
Objetivo específico b) — Relación entre la temperatura mínima (SENAMHI, Puquio)
y los casos semanales de IRA y neumonía, 2000-2013.

    python analisis_temperatura.py ruta/qc00000736.txt

Salidas en resultados/temperatura/:
    tmin_semanal_puquio.csv, correlaciones.csv, resumen.json y figuras.

Se calculan dos correlaciones:
    1. Directa: casos vs. temperatura mínima (Pearson y Spearman), rezagos de 0 a 8 semanas.
    2. Sin la estacionalidad: anomalías de log(casos+1) y de temperatura respecto al
       promedio de la misma semana epidemiológica en 2000-2013.
Las semanas consecutivas no son independientes, por eso los intervalos de confianza
se estiman con bootstrap circular por bloques (bloques de 52 semanas, 2000 réplicas).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from temperatura import leer_senamhi, tmin_semanal  # noqa: E402

AMBITOS = ["AYACUCHO (departamento)", "LUCANAS"]
ANIO_INI, ANIO_FIN = 2000, 2013
REZAGOS = range(0, 9)
BLOQUE, REPLICAS, SEMILLA = 52, 2000, 42
OUT = RAIZ / "resultados" / "temperatura"


def bootstrap_r(x, y, bloque=BLOQUE, replicas=REPLICAS, semilla=SEMILLA):
    """IC 95 % de Pearson con bootstrap circular por bloques."""
    rng = np.random.default_rng(semilla)
    n = len(x)
    nb = int(np.ceil(n / bloque))
    rs = np.empty(replicas)
    for k in range(replicas):
        inicios = rng.integers(0, n, nb)
        idx = (inicios[:, None] + np.arange(bloque)[None, :]).ravel()[:n] % n
        rs[k] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.percentile(rs, [2.5, 97.5])


def serie_con_temperatura(ambito: str, tmin: pd.DataFrame) -> pd.DataFrame:
    largo = pd.read_csv(RAIZ / "data" / "series_semanales_ayacucho.csv", parse_dates=["fecha"])
    s = largo[largo["ambito"] == ambito][["fecha", "anio", "semana", "casos"]]
    s = s.merge(tmin[["anio", "semana", "tmin"]], on=["anio", "semana"], how="left")
    return s.reset_index(drop=True)


def main(ruta_txt: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 200, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    OUT.mkdir(parents=True, exist_ok=True)

    diario = leer_senamhi(ruta_txt)
    w = tmin_semanal(diario)
    w.to_csv(OUT / "tmin_semanal_puquio.csv", index=False)
    periodo = w[(w["anio"] >= ANIO_INI) & (w["anio"] <= ANIO_FIN)]
    d_per = diario[(diario["fecha"].dt.year >= ANIO_INI) & (diario["fecha"].dt.year <= ANIO_FIN)]
    resumen = {
        "estacion": "PUQUIO (SENAMHI, qc00000736)",
        "periodo_diario_archivo": f"{diario['fecha'].min():%d/%m/%Y} - {diario['fecha'].max():%d/%m/%Y}",
        "dias_2000_2013": int(len(d_per)),
        "dias_sin_tmin_2000_2013": int(d_per["tmin"].isna().sum()),
        "semanas_2000_2013": int(len(periodo)),
        "semanas_interpoladas": int(periodo["tmin_interpolada"].sum()),
        "semanas_sin_dato": int(periodo["tmin"].isna().sum()),
        "tmin_semanal_media": round(float(periodo["tmin"].mean()), 2),
        "tmin_semanal_min": round(float(periodo["tmin"].min()), 2),
        "tmin_semanal_max": round(float(periodo["tmin"].max()), 2),
    }
    clim_t = periodo.groupby("semana")["tmin"].mean()
    resumen["semana_mas_fria_promedio"] = int(clim_t.idxmin())
    resumen["tmin_semana_mas_fria"] = round(float(clim_t.min()), 2)

    filas = []
    series = {}
    for amb in AMBITOS:
        s = serie_con_temperatura(amb, w)
        s = s[(s["anio"] >= ANIO_INI) & (s["anio"] <= ANIO_FIN)].reset_index(drop=True)
        s["lc"] = np.log1p(s["casos"])
        s["tmin_an"] = s["tmin"] - s.groupby("semana")["tmin"].transform("mean")
        s["lc_an"] = s["lc"] - s.groupby("semana")["lc"].transform("mean")
        series[amb] = s
        clim_c = s.groupby("semana")["casos"].mean()
        resumen[f"semana_mas_casos_promedio_{amb}"] = int(clim_c.idxmax())
        for lag in REZAGOS:
            t = s["tmin"].shift(lag)
            ta = s["tmin_an"].shift(lag)
            m = t.notna() & s["casos"].notna()
            ma = ta.notna() & s["lc_an"].notna()
            r, p = pearsonr(t[m], s["casos"][m])
            rho, prho = spearmanr(t[m], s["casos"][m])
            ra, pa = pearsonr(ta[ma], s["lc_an"][ma])
            ci = bootstrap_r(t[m].values, s["casos"][m].values)
            cia = bootstrap_r(ta[ma].values, s["lc_an"][ma].values)
            filas.append({"ambito": amb, "rezago_semanas": lag, "n": int(m.sum()),
                          "pearson": r, "pearson_ic95_inf": ci[0], "pearson_ic95_sup": ci[1],
                          "p_pearson": p, "spearman": rho, "p_spearman": prho,
                          "pearson_anomalias": ra, "anom_ic95_inf": cia[0], "anom_ic95_sup": cia[1],
                          "p_anomalias": pa})
    cor = pd.DataFrame(filas)
    cor.to_csv(OUT / "correlaciones.csv", index=False)
    (OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(cor.round(3).to_string(index=False))

    # ---------------- Figura A: series semanales
    s = series[AMBITOS[0]]
    fig, ax1 = plt.subplots(figsize=(8, 3.3))
    ax1.plot(s["fecha"], s["casos"], color="#1b2a3c", lw=0.9, label="Casos (departamento)")
    ax1.set_ylabel("Casos por semana")
    ax2 = ax1.twinx()
    ax2.plot(s["fecha"], s["tmin"], color="#2a7fb8", lw=0.8, alpha=.85, label="Temperatura mínima (Puquio)")
    ax2.set_ylabel("Temperatura mínima (°C)")
    ax2.invert_yaxis()
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_t1_series_2000_2013.png"); plt.close(fig)

    # ---------------- Figura B: comportamiento promedio por semana del año
    fig, ax1 = plt.subplots(figsize=(8, 3.3))
    c = s.groupby("semana")["casos"].mean()
    t = s.groupby("semana")["tmin"].mean()
    ax1.plot(c.index, c.values, color="#1b2a3c", lw=1.6, label="Casos promedio (departamento)")
    ax1.set_xlabel("Semana epidemiológica"); ax1.set_ylabel("Casos por semana")
    ax2 = ax1.twinx()
    ax2.plot(t.index, t.values, color="#2a7fb8", lw=1.6, label="Temperatura mínima promedio (Puquio)")
    ax2.set_ylabel("Temperatura mínima (°C)"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax1.set_xlim(1, 52)
    fig.tight_layout(); fig.savefig(OUT / "fig_t2_promedio_semanal.png"); plt.close(fig)

    # ---------------- Figura C: correlación por rezago
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, col, inf, sup, tit in [(axs[0], "pearson", "pearson_ic95_inf", "pearson_ic95_sup", "Correlación directa"),
                                   (axs[1], "pearson_anomalias", "anom_ic95_inf", "anom_ic95_sup", "Sin estacionalidad (anomalías)")]:
        for amb, color in zip(AMBITOS, ["#1b2a3c", "#e08e2b"]):
            g = cor[cor["ambito"] == amb]
            ax.errorbar(g["rezago_semanas"] + (0.1 if amb == "LUCANAS" else -0.1), g[col],
                        yerr=[g[col] - g[inf], g[sup] - g[col]], fmt="o", ms=4, capsize=2, color=color,
                        label="Departamento" if amb == AMBITOS[0] else "Lucanas")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Rezago de la temperatura (semanas)"); ax.set_title(tit, fontsize=9)
    axs[0].set_ylabel("Coeficiente de Pearson (IC 95 %)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_t3_correlacion_rezagos.png"); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "data" / "qc00000736.txt"))
