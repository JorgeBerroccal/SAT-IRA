"""
Genera las figuras para el informe final (resultados/figuras/*.png).

    python generar_figuras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from alertas import canal_endemico  # noqa: E402

DEPTO = "AYACUCHO (departamento)"
OUT = RAIZ / "resultados" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

largo = pd.read_csv(RAIZ / "data" / "series_semanales_ayacucho.csv", parse_dates=["fecha"])
s = largo[largo["ambito"] == DEPTO].reset_index(drop=True)

# Figura 1 — serie histórica
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(s["fecha"], s["casos"], color="#1b2a3c", lw=0.9)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color="gray", alpha=.15, label="COVID-19 (excluido)")
ax.axvspan(pd.Timestamp("2023-01-01"), s["fecha"].max(), color="#4338ca", alpha=.08, label="Prueba (2023-2024)")
ax.set_ylabel("Casos de IRA por semana")
ax.set_title("Casos semanales de IRA y neumonía — departamento de Ayacucho, 2000-2024")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig1_serie_historica.png"); plt.close(fig)

# Figura 2 — real vs pronóstico a 2 semanas (prueba)
pr = pd.read_csv(RAIZ / "resultados" / "pronosticos_prueba_departamento.csv")
d = pr[pr["h"] == 2].copy()
d["fecha"] = [s.loc[s["semana_txt"] == o].index[0] + 2 for o in d["origen"]]
d["fecha"] = s.loc[d["fecha"], "fecha"].values
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(d["fecha"], d["real"], color="#1b2a3c", lw=1.6, label="Real")
ax.plot(d["fecha"], d["LSTM"], color="#4338ca", lw=1.4, label="LSTM")
ax.plot(d["fecha"], d["SARIMA"], color="#2a9d8f", lw=1, ls="--", label="SARIMA")
ax.set_ylabel("Casos por semana")
ax.set_title("Pronóstico con 2 semanas de anticipación — periodo de prueba")
ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(OUT / "fig2_real_vs_pronostico_h2.png"); plt.close(fig)

# Figura 3 — MAE por horizonte
met = pd.read_csv(RAIZ / "resultados" / "metricas_prueba.csv", dtype={"horizonte": str})
m = met[(met["ambito"] == DEPTO) & (met["horizonte"] != "1-4")]
piv = m.pivot(index="horizonte", columns="modelo", values="MAE")[["LSTM", "SARIMA", "Persistencia", "Ingenuo estacional"]]
ax = piv.plot.bar(figsize=(6, 3), color=["#4338ca", "#2a9d8f", "#e08e2b", "#98a2b0"], rot=0, width=.8)
ax.set_xlabel("Semanas de anticipación"); ax.set_ylabel("MAE (casos)")
ax.set_title("Error absoluto medio por horizonte — prueba 2023-2024")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig3_mae_por_horizonte.png"); plt.close()

# Figura 4 — canal endémico 2024 con casos reales
c = canal_endemico(s, 2024)
r = s[s["anio"] == 2024]
fig, ax = plt.subplots(figsize=(8, 3))
ax.fill_between(c["semana"], 0, c["q1"], color="#1f8a55", alpha=.10, label="Éxito")
ax.fill_between(c["semana"], c["q1"], c["q2"], color="#1f8a55", alpha=.25, label="Seguridad")
ax.fill_between(c["semana"], c["q2"], c["q3"], color="#c98f00", alpha=.30, label="Alarma")
ax.fill_between(c["semana"], c["q3"], c["q3"] * 1.35, color="#c93232", alpha=.15, label="Epidemia")
ax.plot(r["semana"], r["casos"], color="#1b2a3c", lw=1.6, label="Casos 2024")
ax.set_xlim(1, 52); ax.set_xlabel("Semana epidemiológica"); ax.set_ylabel("Casos")
ax.set_title(f"Canal endémico 2024 (años de referencia: {', '.join(map(str, c.attrs['anios_referencia']))})")
ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig4_canal_endemico_2024.png"); plt.close(fig)

# Figura 5 — curvas de entrenamiento
h = pd.read_csv(RAIZ / "resultados" / "historial_entrenamiento.csv")
fig, ax = plt.subplots(figsize=(6, 3))
for mod, g in h.groupby("modelo"):
    if mod == "departamento_s1" or mod == "provincias":
        ax.plot(g["epoca"], g["loss"], label=f"entrenamiento · {mod}")
        ax.plot(g["epoca"], g["val_loss"], ls="--", label=f"validación · {mod}")
ax.set_yscale("log"); ax.set_xlabel("Época"); ax.set_ylabel("MSE")
ax.set_title("Curvas de aprendizaje"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig5_curvas_entrenamiento.png"); plt.close(fig)
print("Figuras en", OUT)
