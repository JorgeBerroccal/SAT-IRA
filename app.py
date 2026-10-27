"""
Componente 4 — Tablero web (Streamlit)
=====================================
    streamlit run app.py

Muestra el gráfico de casos, el pronóstico LSTM de las próximas 4 semanas,
el canal endémico y el semáforo de alerta para el personal de la DIRESA Ayacucho.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from alertas import NIVELES, canal_endemico, evaluar_pronostico  # noqa: E402
from modelo_lstm import HORIZONTE, VENTANA  # noqa: E402
from tratamiento import construir_serie, leer_csv_minsa, semana_epi_de_fecha  # noqa: E402

DEPTO = "AYACUCHO (departamento)"
COLOR_REAL = "#1f3b57"
COLOR_PRED = "#7b3fe4"
ZONAS = [("Éxito", "rgba(46,158,91,0.10)"), ("Seguridad", "rgba(46,158,91,0.22)"),
         ("Alarma", "rgba(224,161,0,0.22)"), ("Epidemia", "rgba(214,59,59,0.11)")]

st.set_page_config(page_title="Alerta temprana IRA · Ayacucho", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.semaforo {display:flex; gap:14px; align-items:center; background:#20262e; padding:14px 18px;
           border-radius:40px; width:fit-content; margin:auto;}
.luz {width:54px; height:54px; border-radius:50%; opacity:.18;}
.luz.on {opacity:1; box-shadow:0 0 22px 6px currentColor;}
.tarjeta {border-radius:14px; padding:18px 22px; color:white;}
.tarjeta h2 {margin:0; color:white; font-size:2rem; letter-spacing:.04em;}
.tarjeta p {margin:.3rem 0 0 0; font-size:1rem; opacity:.95;}
.prov {border-radius:10px; padding:10px 12px; margin-bottom:10px; border-left:8px solid;
       background:rgba(127,127,127,.08);}
.prov b {font-size:.95rem;}
.prov small {opacity:.8;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de datos y modelos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_series_base() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data" / "series_semanales_ayacucho.csv", parse_dates=["fecha"])


@st.cache_data
def cargar_resultados():
    pron = pd.read_csv(RAIZ / "resultados" / "pronosticos_historicos.csv")
    met = pd.read_csv(RAIZ / "resultados" / "metricas_prueba.csv", dtype={"horizonte": str})
    prueba = pd.read_csv(RAIZ / "resultados" / "pronosticos_prueba_departamento.csv")
    hist = pd.read_csv(RAIZ / "resultados" / "historial_entrenamiento.csv")
    info = json.loads((RAIZ / "modelos" / "info_modelo.json").read_text(encoding="utf-8"))
    return pron, met, prueba, hist, info


@st.cache_resource(show_spinner="Cargando modelos LSTM ...")
def cargar_motor():
    """Intenta cargar TensorFlow y los modelos. Si no hay TF, el tablero usa los
    pronósticos precalculados por entrenar.py (mismos resultados)."""
    try:
        from modelo_lstm import cargar_escaladores, cargar_modelos
        return cargar_modelos(RAIZ / "modelos"), cargar_escaladores(RAIZ / "modelos" / "escaladores.json")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


@st.cache_data(show_spinner="Procesando el CSV del MINSA ...")
def procesar_csv_subido(contenido: bytes) -> pd.DataFrame:
    df = leer_csv_minsa(io.BytesIO(contenido))
    partes = [construir_serie(df).assign(ambito=DEPTO)]
    for prov in sorted(df["provincia"].unique()):
        partes.append(construir_serie(df, prov).assign(ambito=prov))
    return pd.concat(partes, ignore_index=True)


@st.cache_data
def canal(ambito: str, anio: int, _serie: pd.DataFrame, n_anios: int, filas: int = 0) -> pd.DataFrame:
    """Canal endémico en caché (la clave incluye el n.º de filas por si se suben datos nuevos)."""
    return canal_endemico(_serie, anio, n=n_anios)


series = cargar_series_base()
pron_csv, metricas, prueba, historial, info = cargar_resultados()
modelos, escaladores = cargar_motor()
motor_vivo = modelos is not None

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫁 Alerta temprana IRA")
    st.caption("DIRESA Ayacucho · Prototipo IS-484")

    subido = None
    with st.expander("📤 Actualizar con nuevo CSV del MINSA"):
        archivo = st.file_uploader("CSV de vigilancia de IRA (formato datos abiertos)", type=["csv"])
        if archivo is not None:
            if motor_vivo:
                subido = procesar_csv_subido(archivo.getvalue())
                st.success(f"Datos cargados hasta {subido['semana_txt'].iloc[-1]}")
            else:
                st.warning("Para pronosticar con datos nuevos se necesita TensorFlow instalado.")
    if subido is not None:
        series = subido

    ambitos = [DEPTO] + sorted(a for a in series["ambito"].unique() if a != DEPTO)
    ambito = st.selectbox("Ámbito", ambitos, index=0)
    serie = series[series["ambito"] == ambito].reset_index(drop=True)

    modo = st.radio("Modo", ["Pronóstico actual", "Simulación histórica"],
                    help="En simulación se elige una semana del pasado: el sistema solo usa los datos "
                         "hasta esa semana y luego se compara con lo que realmente ocurrió.")
    etiquetas = serie["semana_txt"].tolist()
    if modo == "Pronóstico actual":
        idx_origen = len(serie) - 1
    else:
        candidatas = etiquetas[VENTANA - 1: len(etiquetas) - 1]
        defecto = candidatas.index("2023-S13") if "2023-S13" in candidatas else len(candidatas) - 1
        sel = st.select_slider("Última semana conocida", options=candidatas, value=candidatas[defecto])
        idx_origen = etiquetas.index(sel)

    st.divider()
    semanas_hist = st.slider("Semanas de historia en el gráfico", 26, 260, 78, step=26)
    n_anios = st.slider("Años de referencia del canal endémico", 5, 7, 7)
    st.divider()
    if motor_vivo:
        st.success(f"Motor LSTM activo · {len(modelos['departamento'])} redes (depto.) + 1 red provincial",
                   icon="🧠")
    else:
        st.info("Motor LSTM no disponible (sin TensorFlow): usando pronósticos precalculados.", icon="💾")
    st.caption(f"Datos: MINSA – CDC Perú, vigilancia de IRA. Último dato: **{etiquetas[-1]}**")


# ---------------------------------------------------------------------------
# Lógica de pronóstico + alerta
# ---------------------------------------------------------------------------
def fechas_futuras(s: pd.DataFrame, t: int) -> pd.DataFrame:
    filas = []
    for h in range(1, HORIZONTE + 1):
        j = t + h
        if j < len(s):
            filas.append((s.at[j, "fecha"], int(s.at[j, "anio"]), int(s.at[j, "semana"])))
        else:
            f = s["fecha"].iloc[-1] + pd.Timedelta(weeks=j - len(s) + 1)
            a, w = semana_epi_de_fecha(f)
            filas.append((f, a, w))
    return pd.DataFrame(filas, columns=["fecha", "anio", "semana"])


def pronostico(amb: str, s: pd.DataFrame, t: int) -> pd.DataFrame:
    fut = fechas_futuras(s, t)
    if motor_vivo:
        from modelo_lstm import pronosticar
        m = modelos["departamento"] if amb == DEPTO else modelos["provincias"]
        fut["pred"] = pronosticar(m, s, escaladores[amb], [t])[0]
    else:
        a0, s0 = int(s.at[t, "anio"]), int(s.at[t, "semana"])
        p = pron_csv[(pron_csv.ambito == amb) & (pron_csv.origen_anio == a0) & (pron_csv.origen_semana == s0)]
        fut["pred"] = p.sort_values("h")["pred"].values
    b = info["bandas_log_80"][amb]
    fut["inf"] = np.expm1(np.log1p(fut["pred"]) + np.array(b["inf"])).clip(0)
    fut["sup"] = np.expm1(np.log1p(fut["pred"]) + np.array(b["sup"]))
    fut["h"] = range(1, HORIZONTE + 1)
    return fut


def alerta_para(amb: str, s: pd.DataFrame, t: int):
    fut = pronostico(amb, s, t)
    can = pd.concat([canal(amb, a, s, n_anios, len(s)) for a in sorted(fut["anio"].unique())])
    return fut, evaluar_pronostico(fut[["anio", "semana", "pred"]], can, float(s.at[t, "casos"]))


fut, res = alerta_para(ambito, serie, idx_origen)
ultima = serie.iloc[idx_origen]

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(f"# Sistema de alerta temprana de IRA — {ambito.title()}")
sub = (f"Pronóstico LSTM de las próximas {HORIZONTE} semanas desde la semana epidemiológica "
       f"**{ultima['semana_txt']}** ({ultima['fecha']:%d/%m/%Y}).")
if modo == "Simulación histórica":
    sub += " 🔁 *Modo simulación: se compara con lo que realmente ocurrió.*"
st.markdown(sub)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🚦 Alerta", "🗺️ Provincias", "📈 Desempeño del modelo", "🗂️ Datos", "ℹ️ Cómo funciona"])

# ---------------------------------------------------------------------------
# TAB 1 — Alerta
# ---------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        luces = "".join(
            f'<div class="luz {"on" if i == res["nivel_id"] else ""}" '
            f'style="background:{NIVELES[i]["color"]}; color:{NIVELES[i]["color"]}"></div>'
            for i in (0, 1, 2))
        st.markdown(f'<div class="semaforo">{luces}</div>', unsafe_allow_html=True)
        st.markdown("")
        flecha = {"en aumento": "↗", "en descenso": "↘", "estable": "→"}[res["tendencia"]]
        st.markdown(
            f'<div class="tarjeta" style="background:{res["color"]}">'
            f'<h2>{res["emoji"]} {res["nivel"]}</h2><p>{res["mensaje"]}</p>'
            f'<p><b>Tendencia:</b> {flecha} {res["tendencia"]} ({res["cambio_%"]:+.0f} % '
            f'vs. última semana)</p></div>', unsafe_allow_html=True)
        st.markdown("")
        st.metric(f"Casos en {ultima['semana_txt']}", f"{int(ultima['casos']):,}".replace(",", " "))

    with c2:
        cols = st.columns(HORIZONTE)
        det = res["detalle"]
        for i, col in enumerate(cols):
            r = det.iloc[i]
            delta = r["exceso_vs_mediana_%"]
            col.metric(
                f"{r['emoji']} S+{i + 1} · {int(r['anio'])}-S{int(r['semana']):02d}",
                f"{r['pred']:,.0f}".replace(",", " "),
                f"{delta:+.0f} % vs. mediana" if pd.notna(delta) else None,
                delta_color="inverse")
            col.caption(f"IC 80 %: {fut['inf'].iloc[i]:,.0f} – {fut['sup'].iloc[i]:,.0f}".replace(",", " "))

        # ---- Gráfico principal
        ini = max(0, idx_origen - semanas_hist + 1)
        fin_real = min(len(serie), idx_origen + 1 + (HORIZONTE + 4 if modo == "Simulación histórica" else 0))
        hist_v = serie.iloc[ini: idx_origen + 1]
        fechas_eje = pd.concat([serie.iloc[ini:fin_real][["fecha", "anio", "semana"]],
                                fut[["fecha", "anio", "semana"]]]).drop_duplicates("fecha")
        can_all = pd.concat([canal(ambito, a, serie, n_anios, len(serie)) for a in sorted(fechas_eje["anio"].unique())])
        z = fechas_eje.merge(can_all, on=["anio", "semana"], how="left").sort_values("fecha")
        techo = max(z["q3"].max() * 1.25, hist_v["casos"].max() * 1.1, fut["sup"].max() * 1.05)

        fig = go.Figure()
        capas = [z["q1"], z["q2"], z["q3"], pd.Series(techo, index=z.index)]
        base = pd.Series(0.0, index=z.index)
        for (nombre, color), sup in zip(ZONAS, capas):
            fig.add_trace(go.Scatter(x=z["fecha"], y=base, mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=z["fecha"], y=sup, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=color, name=f"Zona de {nombre.lower()}", hoverinfo="skip"))
            base = sup
        fig.add_trace(go.Scatter(x=z["fecha"], y=z["q2"], mode="lines", name="Mediana histórica",
                                 line=dict(color="rgba(80,80,80,.55)", dash="dot", width=1)))
        fig.add_trace(go.Scatter(x=hist_v["fecha"], y=hist_v["casos"], mode="lines+markers",
                                 name="Casos notificados", line=dict(color=COLOR_REAL, width=2.4),
                                 marker=dict(size=4), customdata=hist_v["semana_txt"],
                                 hovertemplate="%{customdata}: %{y:,.0f} casos<extra></extra>"))
        if modo == "Simulación histórica":
            desp = serie.iloc[idx_origen + 1: fin_real]
            fig.add_trace(go.Scatter(x=desp["fecha"], y=desp["casos"], mode="lines+markers",
                                     name="Lo que realmente ocurrió",
                                     line=dict(color=COLOR_REAL, width=1.6, dash="dash"),
                                     marker=dict(size=7, symbol="circle-open"),
                                     customdata=desp["semana_txt"],
                                     hovertemplate="%{customdata}: %{y:,.0f} casos reales<extra></extra>"))
        xf = [ultima["fecha"]] + fut["fecha"].tolist()
        fig.add_trace(go.Scatter(x=xf + xf[::-1],
                                 y=[ultima["casos"]] + fut["sup"].tolist() + fut["inf"].tolist()[::-1] + [ultima["casos"]],
                                 fill="toself", fillcolor="rgba(123,63,228,.16)", line=dict(width=0), mode="lines",
                                 name="Intervalo 80 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xf, y=[ultima["casos"]] + fut["pred"].tolist(), mode="lines+markers",
                                 name="Pronóstico LSTM", line=dict(color=COLOR_PRED, width=3),
                                 marker=dict(size=[0] + [11] * HORIZONTE,
                                             color=["rgba(0,0,0,0)"] + [NIVELES[i]["color"] for i in det["nivel_id"]],
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f} casos pronosticados<extra></extra>"))
        fig.add_vline(x=ultima["fecha"], line=dict(color="gray", dash="dash", width=1))
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          yaxis=dict(title="Casos de IRA por semana", range=[0, techo]),
                          legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                          title=dict(text="Casos semanales, canal endémico y pronóstico", font=dict(size=15)))
        st.plotly_chart(fig, width="stretch")

    # ---- Tabla detalle
    tabla = det[["anio", "semana", "pred", "q1", "q2", "q3", "emoji", "nivel"]].copy()
    tabla.insert(2, "fecha inicio", fut["fecha"].dt.strftime("%d/%m/%Y").values)
    tabla["IC 80 %"] = [f"{a:,.0f} – {b:,.0f}" for a, b in zip(fut["inf"], fut["sup"])]
    if modo == "Simulación histórica":
        reales = [serie.at[idx_origen + h, "casos"] if idx_origen + h < len(serie) else np.nan
                  for h in range(1, HORIZONTE + 1)]
        tabla["casos reales"] = reales
        tabla["error %"] = 100 * (tabla["pred"] - tabla["casos reales"]) / tabla["casos reales"]
    tabla = tabla.rename(columns={"anio": "año", "pred": "pronóstico", "q1": "Q1 (éxito)",
                                  "q2": "Q2 mediana", "q3": "Q3 (alarma)", "emoji": ""})
    st.dataframe(tabla.round(0), hide_index=True, width="stretch")
    refs = canal(ambito, int(det['anio'].iloc[0]), serie, n_anios, len(serie)).attrs.get("anios_referencia", [])
    st.caption("Canal endémico por cuartiles (Bortman, 1999) con los años "
               f"{', '.join(map(str, refs))}. Se excluyen 2020-2021 (COVID-19). "
               "🟢 ≤ mediana · 🟡 entre mediana y Q3 · 🔴 > Q3.")

# ---------------------------------------------------------------------------
# TAB 2 — Provincias
# ---------------------------------------------------------------------------
with tab2:
    st.markdown(f"#### Semáforo por provincia desde {ultima['semana_txt']}")
    filas = []
    for amb in ambitos[1:]:
        s = series[series["ambito"] == amb].reset_index(drop=True)
        t = int(np.where(s["semana_txt"] == ultima["semana_txt"])[0][0])
        f, r = alerta_para(amb, s, t)
        filas.append({"provincia": amb, "nivel_id": r["nivel_id"], "nivel": r["nivel"], "emoji": r["emoji"],
                      "color": r["color"], "casos_ult": int(s.at[t, "casos"]),
                      "pred_media": f["pred"].mean(), "max_exceso": r["detalle"]["exceso_vs_mediana_%"].max(),
                      "tendencia": r["tendencia"], "cambio": r["cambio_%"]})
    pv = pd.DataFrame(filas).sort_values(["nivel_id", "max_exceso"], ascending=False)
    k1, k2, k3 = st.columns(3)
    for col, i in zip((k1, k2, k3), (2, 1, 0)):
        col.metric(f"{NIVELES[i]['emoji']} {NIVELES[i]['nivel'].title()}", int((pv["nivel_id"] == i).sum()))
    cc = st.columns(4)
    for i, r in enumerate(pv.itertuples()):
        cc[i % 4].markdown(
            f'<div class="prov" style="border-color:{r.color}"><b>{r.emoji} {r.provincia.title()}</b><br>'
            f'<small>{r.nivel} · última semana: {r.casos_ult} casos<br>'
            f'pronóstico medio: {r.pred_media:.0f} ({r.cambio:+.0f} %, {r.tendencia})</small></div>',
            unsafe_allow_html=True)
    figp = go.Figure(go.Bar(x=pv["max_exceso"], y=pv["provincia"].str.title(), orientation="h",
                            marker_color=pv["color"],
                            hovertemplate="%{y}: %{x:+.0f} % sobre la mediana<extra></extra>"))
    figp.add_vline(x=0, line=dict(color="gray", width=1))
    figp.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                       title=dict(text="Mayor exceso pronosticado sobre la mediana histórica (%)", font=dict(size=14)))
    st.plotly_chart(figp, width="stretch")
    st.caption("En provincias con pocos casos semanales, variaciones pequeñas cambian el nivel: "
               "interpretar el semáforo junto con el número absoluto de casos.")

# ---------------------------------------------------------------------------
# TAB 3 — Desempeño del modelo
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("#### Evaluación en el periodo de prueba (2023 – 2024), datos que el modelo nunca vio")
    m = metricas[metricas["ambito"] == ambito]
    orden = ["LSTM", "SARIMA", "Persistencia", "Ingenuo estacional"]
    glob_ = m[m["horizonte"] == "1-4"].set_index("modelo").reindex([o for o in orden if o in m["modelo"].unique()])
    st.dataframe(glob_[["MAE", "RMSE", "MAPE_%"]].round(1).rename(columns={"MAPE_%": "MAPE (%)"}),
                 width="stretch")
    a1, a2 = st.columns(2)
    with a1:
        fm = go.Figure()
        paleta = {"LSTM": COLOR_PRED, "SARIMA": "#2a9d8f", "Persistencia": "#e9a03b",
                  "Ingenuo estacional": "#9aa5b1"}
        for mod in glob_.index:
            d = m[(m["modelo"] == mod) & (m["horizonte"] != "1-4")]
            fm.add_trace(go.Bar(x=[f"{h} sem." for h in d["horizonte"]], y=d["MAE"], name=mod,
                                marker_color=paleta[mod]))
        fm.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=40, b=10),
                         title=dict(text="MAE por horizonte (casos por semana)", font=dict(size=14)),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fm, width="stretch")
    with a2:
        if ambito == DEPTO:
            hsel = st.radio("Horizonte", [1, 2, 3, 4], index=1, horizontal=True,
                            format_func=lambda h: f"{h} semana{'s' if h > 1 else ''} antes")
            d = prueba[prueba["h"] == hsel].copy()
            d["obj"] = [serie.loc[serie["semana_txt"] == o].index[0] + hsel for o in d["origen"]]
            d["fecha"] = serie.loc[d["obj"], "fecha"].values
            fp = go.Figure()
            fp.add_trace(go.Scatter(x=d["fecha"], y=d["real"], name="Real", line=dict(color=COLOR_REAL, width=2.4)))
            fp.add_trace(go.Scatter(x=d["fecha"], y=d["LSTM"], name="LSTM", line=dict(color=COLOR_PRED, width=2)))
            fp.add_trace(go.Scatter(x=d["fecha"], y=d["SARIMA"], name="SARIMA",
                                    line=dict(color="#2a9d8f", width=1.4, dash="dot")))
            fp.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                             title=dict(text=f"Real vs. pronosticado con {hsel} semana(s) de anticipación",
                                        font=dict(size=14)), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fp, width="stretch")
        else:
            st.info("La comparación con SARIMA y la curva real vs. pronóstico se muestran para el departamento.")
    # ---- ¿El semáforo acierta? (periodo de prueba)
    st.markdown("#### ¿El semáforo acierta? Nivel pronosticado vs. nivel real (2023 – 2024)")
    hs = st.radio("Anticipación", [1, 2, 3, 4], index=1, horizontal=True, key="h_sem",
                  format_func=lambda h: f"{h} semana{'s' if h > 1 else ''}")
    ev = pron_csv[(pron_csv["ambito"] == ambito) & (pron_csv["h"] == hs) & (pron_csv["anio"].isin([2023, 2024]))]
    can_ev = pd.concat([canal(ambito, a, serie, n_anios, len(serie)) for a in (2023, 2024)])
    ev = ev.merge(can_ev, on=["anio", "semana"]).merge(serie[["anio", "semana", "casos"]], on=["anio", "semana"])
    from alertas import nivel_alerta
    ev["pronosticado"] = [nivel_alerta(a, b, c) for a, b, c in zip(ev["pred"], ev["q2"], ev["q3"])]
    ev["real"] = [nivel_alerta(a, b, c) for a, b, c in zip(ev["casos"], ev["q2"], ev["q3"])]
    nombres = {i: f"{NIVELES[i]['emoji']} {NIVELES[i]['nivel'].title()}" for i in NIVELES}
    mc = pd.crosstab(ev["real"].map(nombres), ev["pronosticado"].map(nombres),
                     rownames=["Real ↓"], colnames=["Pronosticado →"]).reindex(
        index=list(nombres.values()), columns=list(nombres.values()), fill_value=0)
    e1, e2 = st.columns([1.6, 1])
    e1.dataframe(mc, width="stretch")
    acierto = (ev["real"] == ev["pronosticado"]).mean() * 100
    reales_alerta = ev["real"] > 0
    sens = ((ev["pronosticado"] > 0) & reales_alerta).sum() / max(reales_alerta.sum(), 1) * 100
    e2.metric("Semanas con el nivel correcto", f"{acierto:.0f} %")
    e2.metric("Semanas de alerta/emergencia detectadas", f"{sens:.0f} %",
              help="De las semanas que realmente estuvieron en alerta o emergencia, "
                   "porcentaje en que el sistema también emitió alerta o emergencia.")

    with st.expander("Curvas de entrenamiento y configuración"):
        hh = historial.copy()
        fh = go.Figure()
        for mod, d in hh.groupby("modelo"):
            fh.add_trace(go.Scatter(x=d["epoca"], y=d["val_loss"], name=f"val · {mod}", mode="lines"))
        fh.update_layout(height=300, yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10),
                         title=dict(text="Pérdida de validación (MSE) por época", font=dict(size=13)))
        st.plotly_chart(fh, width="stretch")
        st.json({k: info[k] for k in ("ventana", "horizonte", "parametros_por_red", "division", "n_ventanas")})

# ---------------------------------------------------------------------------
# TAB 4 — Datos
# ---------------------------------------------------------------------------
with tab4:
    st.markdown(f"#### Serie semanal 2000 – 2024 · {ambito.title()}")
    fd = go.Figure()
    fd.add_trace(go.Scatter(x=serie["fecha"], y=serie["ira_no_neumonia"], name="IRA no neumónica",
                            stackgroup="a", line=dict(color="#6c8ebf")))
    fd.add_trace(go.Scatter(x=serie["fecha"], y=serie["neumonias"], name="Neumonías",
                            stackgroup="a", line=dict(color="#d6604d")))
    fd.add_vrect(x0="2020-03-01", x1="2021-12-31", fillcolor="gray", opacity=.12, line_width=0,
                 annotation_text="COVID-19 (excluido)", annotation_position="top left")
    fd.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                     legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fd, width="stretch")
    anual = serie.groupby("anio")[["casos", "ira_no_neumonia", "neumonias", "hospitalizados_men5",
                                   "defunciones_men5", "defunciones_60mas"]].sum()
    st.dataframe(anual, width="stretch")
    st.download_button("⬇️ Descargar serie semanal (CSV)", serie.to_csv(index=False).encode("utf-8"),
                       file_name=f"serie_{ambito.split(' ')[0].lower()}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# TAB 5 — Cómo funciona
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("""
#### Flujo del sistema
```
Datos MINSA ──▶ Tratamiento ──▶ Modelo LSTM ──▶ Pronóstico ──▶ Semáforo ──▶ Tablero web
(semanales)     (limpieza)      (aprende)       (1-4 semanas)  (🟢 🟡 🔴)    (DIRESA)
```
| # | Componente | Qué hace | Archivo |
|---|---|---|---|
| 1 | Tratamiento de datos | Filtra Ayacucho, suma IRA no neumónica + neumonías por semana epidemiológica, rellena huecos y arma 12 series (departamento + 11 provincias) | `src/tratamiento.py` |
| 2 | Modelo LSTM | Mira las últimas **16 semanas** (casos + semana del año) y pronostica las **4 siguientes**. Departamento: ensamble de 5 redes; provincias: 1 red global | `src/modelo_lstm.py`, `entrenar.py` |
| 3 | Módulo de alertas | Compara el pronóstico con el **canal endémico** (cuartiles de 7 años, sin 2020-21) | `src/alertas.py` |
| 4 | Tablero web | Este tablero | `app.py` |

#### Arquitectura de la red
`Entrada (16 × 4) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(4)`

- **Variables de entrada por semana:** casos en escala log normalizada (Min-Max), nivel relativo a la última semana, seno y coseno de la semana del año (estacionalidad de heladas).
- **Salida:** cambio esperado de los casos en las 4 semanas siguientes → se convierte a número de casos.
- **Entrenamiento:** Adam (lr = 0.001), pérdida MSE, *early stopping* (paciencia 30) y reducción de la tasa de aprendizaje.
- **División temporal:** entrenamiento 2000 – feb 2020 · excluido mar 2020 – 2021 (pandemia) · validación 2022 · prueba 2023 – 2024.

#### Reglas del semáforo
| Pronóstico | Zona del canal | Nivel |
|---|---|---|
| ≤ mediana (Q2) | éxito / seguridad | 🟢 NORMAL |
| entre Q2 y Q3 | alarma | 🟡 ALERTA |
| > Q3 | epidemia | 🔴 EMERGENCIA |

El nivel global es el peor de las 4 semanas pronosticadas. La **tendencia** compara el promedio pronosticado con la última semana (±10 %).
""")
