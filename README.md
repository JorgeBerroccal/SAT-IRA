# Sistema de alerta temprana de IRA — Ayacucho (prototipo)

Curso **Inteligencia Artificial I (IS-484)** · UNSCH
Pronóstico de casos semanales de infecciones respiratorias agudas (IRA) y neumonía con redes **LSTM**,
a partir de los datos abiertos de vigilancia epidemiológica del MINSA (CDC Perú), 2000 – 2024.

```
Datos MINSA ──▶ Tratamiento ──▶ Modelo LSTM ──▶ Pronóstico ──▶ Semáforo ──▶ Tablero web
(semanales)     (limpieza)      (aprende)       (1-4 semanas)  (🟢 🟡 🔴)    (DIRESA / puestos de salud)
```

| # | Componente | Archivo | Herramienta |
|---|---|---|---|
| 1 | Tratamiento de datos | `src/tratamiento.py` | pandas |
| 2 | Modelo LSTM (+ líneas base) | `src/modelo_lstm.py`, `src/baselines.py`, `entrenar.py` | TensorFlow / Keras, statsmodels |
| 3 | Módulo de alertas | `src/alertas.py` | Python |
| 4 | Tablero web | `app.py` (+ versión estática `web/tablero_estatico.html`) | Streamlit, Plotly |

---

## 1. Cómo ejecutarlo (para la exposición)

Requiere **Python 3.10 – 3.12**.

```bash
cd sistema_alerta_ira
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Se abre en el navegador (http://localhost:8501). En Windows también puedes hacer doble clic en `iniciar.bat`.

**Los modelos ya vienen entrenados** (carpeta `modelos/`), así que no hace falta volver a entrenar para presentar.
Si TensorFlow no se instala en tu PC, el tablero igual funciona: usa los pronósticos precalculados en
`resultados/pronosticos_historicos.csv` (son los mismos números que da el modelo).

**Plan B sin instalar nada:** abre `web/tablero_estatico.html` con doble clic en cualquier navegador.

### Guion sugerido para la demo
1. **Pestaña Alerta, modo "Pronóstico actual":** pronóstico de 2025-S01 a S04 desde el último dato (2024-S52) → 🟢 NORMAL, tendencia en descenso.
2. **Modo "Simulación histórica", semana 2023-S13:** el brote de marzo-abril de 2023. El sistema da 🔴 EMERGENCIA y la línea discontinua muestra que efectivamente los casos siguieron por encima del canal.
3. Mover a **2023-S10** (antes del brote): el modelo todavía dice 🟢 → sirve para hablar de limitaciones (no anticipa brotes súbitos sin señal previa; ver Rojas-Paucar et al., 2025).
4. **Pestaña Provincias:** semáforo de las 11 provincias en la misma semana.
5. **Pestaña Desempeño:** LSTM vs. SARIMA, persistencia e ingenuo estacional; matriz de aciertos del semáforo.

---

## 2. Reentrenar desde el CSV del MINSA

```bash
python entrenar.py --csv ruta/datos_abiertos_vigilancia_iras_2000_2024.csv   # ~4 min en CPU
python generar_figuras.py      # figuras para el informe  -> resultados/figuras/
python exportar_web.py         # versión estática         -> web/tablero_estatico.html
```
En Google Colab: sube la carpeta, `!pip install -r requirements.txt` y ejecuta las mismas líneas con `!`.

El CSV crudo (135 MB) no viene incluido; la serie ya procesada sí (`data/series_semanales_ayacucho.csv`).
Desde el tablero también se puede subir un CSV nuevo del MINSA (barra lateral → *Actualizar con nuevo CSV*) y el modelo pronostica con esos datos.

---

## 3. Decisiones de diseño

**Datos.** El CSV no trae encabezados; las columnas son: departamento; provincia; distrito; año; semana; sub_reg_nt; ubigeo;
IRA no neumónica; neumonías <5; neumonías 60+; hospitalizados <5; hospitalizados 60+; defunciones <5; defunciones 60+.
Variable objetivo: **casos = IRA no neumónica + neumonías**, sumados por semana epidemiológica. Se arman 12 series
(departamento + 11 provincias), 1 304 semanas cada una. Las fechas se calculan con la regla de semana epidemiológica
(domingo a sábado; la semana 1 es la primera con ≥ 4 días en enero).

**Pandemia.** En 2020 – 2021 la notificación de IRA cayó de forma artificial (de ~1 300 a ~200 casos/semana).
Esos años se **excluyen** del entrenamiento y del canal endémico.

**División temporal** (sin mezclar): entrenamiento 2000 – feb 2020 · validación 2022 · prueba 2023 – 2024.

**Modelo.** Ventana de 16 semanas → 4 semanas de pronóstico.
`Entrada(16×4) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(4)` · Adam (lr 0.001) · MSE · early stopping.
Variables por semana: casos en log + Min-Max, nivel relativo a la última semana, seno y coseno de la semana del año.
La red predice el **cambio** respecto a la última semana (no el valor absoluto); así se adapta a cambios de nivel
como la caída de casos post-pandemia. Departamento: ensamble de 5 redes (semillas 1-5). Provincias: 1 red compartida.
Intervalo de 80 % a partir de los errores del año de validación.

**Alertas.** Canal endémico por cuartiles (Bortman, 1999) de los 7 años previos, sin 2020-2021, suavizado con ±1 semana.
🟢 pronóstico ≤ mediana · 🟡 entre mediana y Q3 · 🔴 > Q3. Nivel global = el peor de las 4 semanas.

**Temperatura SENAMHI.** No está incluida porque esos datos no venían en los archivos. El código ya está preparado:
si la serie trae una columna `tmin`, `caracteristicas()` la agrega como variable de entrada (hay que reentrenar).

---

## 4. Resultados (prueba 2023 – 2024, departamento de Ayacucho)

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **LSTM** | **114.1** | **153.9** | **11.6 %** |
| SARIMA(1,0,1)(0,1,1)₅₂ | 119.5 | 163.8 | 11.8 % |
| Persistencia | 132.6 | 178.2 | 13.7 % |
| Ingenuo estacional | 154.8 | 201.1 | 16.0 % |

MAE por semanas de anticipación:

| Modelo | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| **LSTM** | **91.6** | **110.5** | **124.3** | **130.0** |
| SARIMA | 92.9 | 113.8 | 130.4 | 141.0 |
| Persistencia | 95.2 | 122.4 | 147.9 | 164.8 |
| Ingenuo estacional | 156.1 | 156.0 | 154.5 | 152.4 |

Semáforo con 2 semanas de anticipación: nivel correcto en el 76 % de las semanas; anticipó el 61 % de las semanas
que realmente estuvieron en alerta o emergencia. En las 11 provincias la LSTM también tiene el menor MAPE
frente a persistencia e ingenuo estacional (`resultados/metricas_prueba.csv`).

Limitación: la ventaja de la LSTM sobre SARIMA es moderada (≈ 5 % en MAE) y crece con el horizonte; ningún modelo
anticipa el inicio de un brote súbito sin señal previa en los datos (p. ej. marzo 2023).

---

## 5. Estructura

```
sistema_alerta_ira/
├── app.py                      tablero Streamlit
├── entrenar.py                 pipeline: datos → entrenamiento → evaluación → pronósticos
├── generar_figuras.py          figuras del informe
├── exportar_web.py             genera web/tablero_estatico.html
├── src/  tratamiento.py · modelo_lstm.py · alertas.py · baselines.py
├── data/series_semanales_ayacucho.csv
├── modelos/  lstm_departamento_s1..s5.keras · lstm_provincias.keras · escaladores.json · info_modelo.json
├── resultados/  metricas_prueba.csv · pronosticos_*.csv · historial_entrenamiento.csv · figuras/
└── web/  tablero_estatico.html · plantilla.html
```
