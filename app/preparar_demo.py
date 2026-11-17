"""Prepara localmente lo que necesita la demo, sin bajar nada de Colab.

Usa los archivos de data/processed/ que genera el notebook (train, val y resenas_clasificadas) y:
  1. Reentrena la línea base TF-IDF + regresión logística (misma configuración y semilla que el notebook).
  2. Calcula la tendencia mensual de reseñas negativas (solo meses con >= 30 reseñas).
  3. Calcula los temas de queja por app y las palabras clave YAKE de las reseñas negativas.

Ejecutar desde la raíz del proyecto:
    python app/preparar_demo.py
El modelo BETO no se reconstruye (pesa 440 MB); si lo quieres en la demo, copia models/beto_sentimiento/ desde Drive.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
MODELOS.mkdir(exist_ok=True)
SEMILLA = 42
ETIQUETAS = ["negativa", "neutra", "positiva"]
ETQ2ID = {e: i for i, e in enumerate(ETIQUETAS)}

# 1. Línea base
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
y_train = train["etiqueta_debil"].map(ETQ2ID).values
y_val = val["etiqueta_debil"].map(ETQ2ID).values
mejor_f1, base = -1, None
for C in [0.1, 0.3, 1, 3, 10]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEMILLA)),
    ])
    pipe.fit(train["texto"], y_train)
    f1 = f1_score(y_val, pipe.predict(val["texto"]), average="macro")
    if f1 > mejor_f1:
        mejor_f1, base, mejor_C = f1, pipe, C
joblib.dump(base, MODELOS / "linea_base_tfidf_lr.joblib")
print(f"Línea base reentrenada: C={mejor_C}, F1-macro val={mejor_f1:.3f}")

# 2. Tendencia mensual
df = pd.read_csv(DATA / "resenas_clasificadas.csv")
tendencia = (df.assign(neg=df["sentimiento"].eq("negativa"))
               .groupby(["app", "mes"])["neg"].agg(pct_negativas="mean", n_resenas="size").reset_index())
tendencia["pct_negativas"] *= 100
tendencia = tendencia[tendencia["n_resenas"] >= 30].sort_values("mes")
tendencia.to_csv(DATA / "tendencia_negativas.csv", index=False)
print("Tendencia:", len(tendencia), "meses-app")

# 3. Temas y palabras clave (mismas reglas que la sección 13 del notebook)
GENERICAS = {"app", "apps", "aplicación", "aplicacion", "aplicaciones", "nequi", "daviplata", "davivienda",
             "bancolombia", "banco", "deja", "dejar", "deja nada", "mala", "malo", "pésima", "pesima", "pésimo",
             "pesimo", "horrible", "terrible", "porquería", "servicio", "nada", "siempre", "ahora", "ser", "hacer",
             "puedo", "puede", "sirve", "funciona", "bien", "mal", "mejor", "peor", "gracias", "favor"}
TEMAS = {
    "Acceso / no deja entrar": r"\b(?:entrar|ingresar|ingreso|acceder|abrir|abre|inicio de sesi[oó]n|bloque)",
    "Actualización": r"actualiz",
    "Compatibilidad del celular": r"compatib|android|huawei|celular nuevo|cambiar de celular",
    "Caídas / lentitud": r"\b(?:ca[ií]d|lent|demora|se cierra|congela|error|falla)",
    "Clave / verificación": r"clave|contraseña|c[oó]digo|verificaci|huella|reconocimiento",
    "Transferencias / pagos": r"transfer|pag[oa]|retir|cajero|dinero|plata",
    "Atención al cliente": r"atenci[oó]n|asesor|soporte|servicio al cliente|l[ií]nea|chat|respuesta",
    "Cobros / comisiones": r"cobr|cuota de manejo|comisi|4 ?x ?1000|cuatro por mil",
}
neg = df[df["sentimiento"] == "negativa"].copy()
for tema, patron in TEMAS.items():
    neg[tema] = neg["texto"].str.lower().str.contains(patron, regex=True)
neg.groupby("app")[list(TEMAS)].mean().mul(100).round(1).T.to_csv(DATA / "temas_queja_por_app.csv")

extractor = yake.KeywordExtractor(lan="es", n=2, dedupLim=0.7, top=40)
if hasattr(extractor, "stopword_set"):
    extractor.stopword_set |= GENERICAS
filas = []
for (app, mes), g in neg.groupby(["app", "mes"]):
    if len(g) < 20:
        continue
    kws = extractor.extract_keywords(" . ".join(g["texto"].tolist())[:300000])
    kws = [(k, s) for k, s in kws if not all(t in GENERICAS for t in k.lower().split())][:10]
    filas += [{"app": app, "mes": mes, "palabra_clave": k, "puntaje_yake": s, "n_negativas": len(g)} for k, s in kws]
pd.DataFrame(filas).to_csv(DATA / "palabras_clave_negativas.csv", index=False)
print("Temas y palabras clave listos. Ahora: streamlit run app/app.py")
