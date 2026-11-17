"""Demo: La voz del usuario · análisis de sentimiento de reseñas de apps bancarias.

Ejecutar desde la raíz del proyecto:
    streamlit run app/app.py
Requiere haber corrido el notebook (crea models/ y data/processed/).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "models"
DATA = RAIZ / "data" / "processed"
ETIQUETAS = ["negativa", "neutra", "positiva"]
COLORES = {"negativa": "🔴", "neutra": "🟡", "positiva": "🟢"}

st.set_page_config(page_title="La voz del usuario", page_icon="💬", layout="wide")


@st.cache_resource
def cargar_beto():
    ruta = MODELOS / "beto_sentimiento"
    if not ruta.exists():
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta).eval()

    def predecir(textos):
        enc = tok(list(textos), truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(modelo(**enc).logits, -1).numpy()
    return predecir


@st.cache_resource
def cargar_pysentimiento():
    try:
        from pysentimiento import create_analyzer
    except ImportError:
        return None
    analizador = create_analyzer(task="sentiment", lang="es")

    def predecir(textos):
        salida = analizador.predict(list(textos))
        return np.array([[o.probas["NEG"], o.probas["NEU"], o.probas["POS"]] for o in salida])
    return predecir


@st.cache_resource
def cargar_base():
    ruta = MODELOS / "linea_base_tfidf_lr.joblib"
    if not ruta.exists():
        return None
    import joblib
    pipe = joblib.load(ruta)
    return lambda textos: pipe.predict_proba(list(textos))


@st.cache_data
def cargar_csv(nombre, **kw):
    ruta = DATA / nombre
    return pd.read_csv(ruta, **kw) if ruta.exists() else None


st.title("💬 La voz del usuario")
st.caption("Sentimiento y quejas frecuentes en reseñas de Google Play de apps bancarias colombianas · "
           "Proyecto Integrador IA Avanzada (Ruta A, reto A1)")

modelos = {k: v for k, v in {"pysentimiento (mejor F1-macro)": cargar_pysentimiento(),
                             "BETO (ajuste fino)": cargar_beto(),
                             "TF-IDF + RegLog (línea base)": cargar_base()}.items() if v is not None}
if not modelos:
    st.error("No se encontraron modelos en `models/`. Ejecuta primero `notebooks/pipeline_resenas.ipynb`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Analizar una reseña", "Tendencias por app", "¿De qué se quejan?"])

with tab1:
    nombre_modelo = st.radio("Modelo", list(modelos), horizontal=True)
    texto = st.text_area("Escribe o pega una reseña", height=120,
                         value="La app se cae cada vez que intento hacer una transferencia, pésimo servicio")
    if st.button("Analizar", type="primary") and texto.strip():
        p = modelos[nombre_modelo]([texto])[0]
        etq = ETIQUETAS[int(np.argmax(p))]
        st.subheader(f"{COLORES[etq]} {etq.capitalize()} ({p.max():.0%} de confianza)")
        st.bar_chart(pd.DataFrame({"probabilidad": p}, index=ETIQUETAS))
        if len(modelos) > 1:
            st.markdown("**Comparación entre modelos**")
            st.dataframe(pd.DataFrame({n: f(np.array([texto]))[0] for n, f in modelos.items()},
                                      index=ETIQUETAS).style.format("{:.1%}"))
    st.info("El modelo puede equivocarse con sarcasmo, reseñas mixtas o textos muy cortos. "
            "Es una herramienta de apoyo, no reemplaza la lectura humana.")

with tab2:
    tend = cargar_csv("tendencia_negativas.csv")
    if tend is None:
        st.warning("Falta `data/processed/tendencia_negativas.csv` (sale del notebook).")
    else:
        apps = st.multiselect("Apps", sorted(tend["app"].unique()), default=sorted(tend["app"].unique()))
        pivote = tend[tend["app"].isin(apps)].pivot(index="mes", columns="app", values="pct_negativas")
        st.markdown("**% de reseñas negativas por mes**")
        st.line_chart(pivote)
    clas = cargar_csv("resenas_clasificadas.csv")
    if clas is not None:
        st.markdown("**Distribución de sentimiento por app**")
        st.dataframe(pd.crosstab(clas["app"], clas["sentimiento"], normalize="index")
                     .style.format("{:.1%}"))

with tab3:
    temas = cargar_csv("temas_queja_por_app.csv", index_col=0)
    if temas is not None:
        st.markdown("**% de reseñas negativas que mencionan cada tema**")
        st.bar_chart(temas, stack=False, horizontal=True)
        st.dataframe(temas.style.format("{:.1f} %"))
    kw = cargar_csv("palabras_clave_negativas.csv")
    if kw is None or kw.empty:
        st.warning("Falta `data/processed/palabras_clave_negativas.csv` (sale del notebook).")
    else:
        c1, c2 = st.columns(2)
        app = c1.selectbox("App", sorted(kw["app"].unique()))
        meses = sorted(kw.loc[kw["app"] == app, "mes"].unique())
        mes = c2.selectbox("Mes", meses, index=len(meses) - 1)
        sel = kw[(kw["app"] == app) & (kw["mes"] == mes)].sort_values("puntaje_yake")
        st.markdown(f"**Palabras clave de las reseñas negativas** ({int(sel['n_negativas'].iloc[0])} reseñas). "
                    "En YAKE, menor puntaje = más relevante.")
        st.dataframe(sel[["palabra_clave", "puntaje_yake"]].reset_index(drop=True), use_container_width=True)
        if clas is not None:
            st.markdown("**Ejemplos de reseñas negativas de ese mes** (anonimizadas)")
            ej = clas[(clas["app"] == app) & (clas["mes"] == mes) & (clas["sentimiento"] == "negativa")]
            st.dataframe(ej[["texto", "estrellas"]].head(10), use_container_width=True)
