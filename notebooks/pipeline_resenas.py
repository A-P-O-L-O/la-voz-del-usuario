# %% [markdown]
# # La voz del usuario: análisis de sentimiento de reseñas de apps bancarias colombianas
# **Proyecto Integrador · Inteligencia Artificial Avanzada · Ruta A · Reto A1**
#
# Este notebook corre de principio a fin y reproduce todo el proyecto:
#
# 1. Configuración y semillas
# 2. Descarga de reseñas de Google Play (`google-play-scraper`) y anonimización
# 3. Limpieza y etiqueta débil a partir de las estrellas
# 4. Análisis exploratorio (EDA)
# 5. Separación del conjunto de oro (300 reseñas: pre-etiquetado con LLM + revisión humana) y división train / val / test
# 6. Línea base: TF-IDF + regresión logística
# 7. Red profunda 1: BiLSTM con embeddings
# 8. Red profunda 2: ajuste fino de BETO
# 9. Referencia preentrenada: pysentimiento (sin ajuste)
# 10. Evaluación final sobre las 300 reseñas de oro (F1-macro)
# 11. Análisis de errores (10 o más casos)
# 12. Explicabilidad (coeficientes + SHAP)
# 13. Componente NLP: palabras clave (YAKE) de reseñas negativas por app y por mes
# 14. Exportar modelos y archivos para la demo
#
# **Recomendación:** ejecutar en Google Colab con GPU (Entorno de ejecución → Cambiar tipo → T4 GPU).
# El ajuste fino de BETO tarda ~15-25 min en una T4.

# %% [markdown]
# ## 1. Configuración

# %%
# En Colab: instalar dependencias que no vienen por defecto
import sys, subprocess
EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "google-play-scraper", "yake", "pysentimiento", "shap", "evaluate"], check=True)
    # Conectar Google Drive para que los datos y modelos no se borren al cerrar Colab.
    # La carpeta del proyecto debe estar en: Mi unidad / PI_RutaA_ResenasApps
    from google.colab import drive
    drive.mount("/content/drive")
    import os
    CARPETA_DRIVE = "/content/drive/MyDrive/PI_RutaA_ResenasApps"
    os.makedirs(CARPETA_DRIVE, exist_ok=True)
    os.chdir(CARPETA_DRIVE)
    print("Trabajando en:", os.getcwd())

# %%
import os, random, json, re, hashlib, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
os.environ["PYTHONHASHSEED"] = str(SEMILLA)

# Carpeta raíz del proyecto (funciona local y en Colab)
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
DATA_ETQ = RAIZ / "data" / "etiquetado"
MODELOS = RAIZ / "models"
FIGS = RAIZ / "reports" / "figures"
for p in [DATA_RAW, DATA_PROC, DATA_ETQ, MODELOS, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

# Apps del mismo sector (banca). Se pueden cambiar por otras del mismo sector.
APPS = {
    "Nequi": "com.nequi.MobileApp",
    "Daviplata": "com.davivienda.daviplataapp",
    "Bancolombia": "com.todo1.mobile",
}
RESENAS_POR_APP = 5000          # 3 apps x 5.000 = 15.000 (> 10.000 exigidas)
N_ORO = 300                     # reseñas que se etiquetan a mano
ETIQUETAS = ["negativa", "neutra", "positiva"]
ETQ2ID = {e: i for i, e in enumerate(ETIQUETAS)}

sns.set_theme(style="whitegrid")
print("Raíz del proyecto:", RAIZ)

# %% [markdown]
# ## 2. Descarga de reseñas y anonimización
# Se descargan las reseñas más recientes de cada app con `lang='es'` y `country='co'`.
# Si el archivo ya existe **no se vuelve a descargar**, para que el resultado sea reproducible
# (las reseñas de Google Play cambian todos los días). Se registra la fecha de descarga.
#
# **Ética (Ley 1581 de 2012):** se eliminan nombre de usuario e imagen de perfil y el id de la reseña se
# reemplaza por un hash. Nunca se publican nombres de usuarios.

# %%
ARCHIVO_RAW = DATA_RAW / "resenas_raw.csv"

def descargar_app(nombre, app_id, n):
    from google_play_scraper import reviews, Sort
    filas, token = [], None
    while len(filas) < n:
        lote, token = reviews(app_id, lang="es", country="co", sort=Sort.NEWEST,
                              count=min(200, n - len(filas)), continuation_token=token)
        if not lote:
            break
        filas.extend(lote)
        if token is None or token.token is None:
            break
        time.sleep(0.5)
    df = pd.DataFrame(filas)
    df["app"] = nombre
    return df

if ARCHIVO_RAW.exists():
    raw = pd.read_csv(ARCHIVO_RAW, parse_dates=["fecha"])
    print("Usando archivo existente:", ARCHIVO_RAW)
else:
    partes = [descargar_app(n, i, RESENAS_POR_APP) for n, i in APPS.items()]
    raw = pd.concat(partes, ignore_index=True)
    # Anonimización: se conservan solo columnas necesarias
    raw["id"] = raw["reviewId"].astype(str).map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])
    raw = raw.rename(columns={"content": "texto", "score": "estrellas", "at": "fecha",
                              "thumbsUpCount": "likes", "appVersion": "version_app"})
    raw = raw[["id", "app", "texto", "estrellas", "fecha", "likes", "version_app"]]
    raw.to_csv(ARCHIVO_RAW, index=False)
    meta = {"fecha_descarga_utc": datetime.now(timezone.utc).isoformat(),
            "fuente": "Google Play Store vía google-play-scraper", "lang": "es", "country": "co",
            "apps": APPS, "resenas_por_app": RESENAS_POR_APP}
    (DATA_RAW / "metadatos_descarga.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Descargadas:", len(raw))

print(json.loads((DATA_RAW / "metadatos_descarga.json").read_text()) if (DATA_RAW / "metadatos_descarga.json").exists() else "")
raw.head()

# %% [markdown]
# ## 3. Limpieza y etiqueta débil
# - Se eliminan textos vacíos y duplicados exactos (mismo texto y app).
# - Se enmascaran correos, teléfonos y números largos (posibles cédulas o cuentas) que algunos usuarios escriben.
# - **Etiqueta débil** a partir de las estrellas: 1–2 → negativa, 3 → neutra, 4–5 → positiva.
#   Es ruidosa (hay quien pone 5 estrellas y se queja), por eso la evaluación final se hace con etiquetas manuales.

# %%
def anonimizar_texto(t):
    t = re.sub(r"\S+@\S+\.\S+", "[CORREO]", t)
    t = re.sub(r"\+?\d[\d\s-]{6,}\d", "[NUMERO]", t)
    return t

df = raw.copy()
df["texto"] = df["texto"].astype(str).str.strip()
df = df[df["texto"].str.len() > 0]
df = df[df["texto"].str.lower() != "nan"]
df = df.drop_duplicates(subset=["app", "texto"]).reset_index(drop=True)
df["texto"] = df["texto"].map(anonimizar_texto)
df["fecha"] = pd.to_datetime(df["fecha"])
df["mes"] = df["fecha"].dt.to_period("M").astype(str)
df["etiqueta_debil"] = pd.cut(df["estrellas"], bins=[0, 2, 3, 5], labels=ETIQUETAS).astype(str)
df["n_palabras"] = df["texto"].str.split().str.len()
print(f"Registros tras limpieza: {len(df)} (descargados: {len(raw)})")

# %% [markdown]
# ## 4. Análisis exploratorio (EDA)

# %%
print("Tamaño:", df.shape)
print("\nFaltantes por columna:\n", raw.isna().sum())
print("\nReseñas por app:\n", df["app"].value_counts())
print("\nRango de fechas:", df["fecha"].min(), "→", df["fecha"].max())
print("\nDistribución de la etiqueta débil (%):\n", df["etiqueta_debil"].value_counts(normalize=True).round(3) * 100)

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(data=df, x="estrellas", hue="app", ax=ax[0]); ax[0].set_title("Estrellas por app")
sns.countplot(data=df, x="etiqueta_debil", order=ETIQUETAS, ax=ax[1]); ax[1].set_title("Desbalance de clases (etiqueta débil)")
sns.histplot(df["n_palabras"].clip(upper=100), bins=50, ax=ax[2]); ax[2].set_title("Longitud (palabras, recortada en 100)")
plt.tight_layout(); plt.savefig(FIGS / "eda_distribuciones.png", dpi=150); plt.show()
print(df["n_palabras"].describe())

# %%
mensual = (df.groupby(["mes", "app"]).size().rename("n").reset_index())
plt.figure(figsize=(12, 4))
sns.lineplot(data=mensual, x="mes", y="n", hue="app", marker="o")
plt.xticks(rotation=60); plt.title("Reseñas por mes y app"); plt.tight_layout()
plt.savefig(FIGS / "eda_resenas_por_mes.png", dpi=150); plt.show()

# %% [markdown]
# ## 5. Conjunto de oro y división train / val / test
# **Esto se hace ANTES de cualquier transformación** (vocabulario, TF-IDF, tokenización, balanceo).
#
# 1. Se apartan **300 reseñas** (estratificadas por app) como **conjunto de oro**. Se exportan a
#    `data/etiquetado/oro_para_etiquetar.csv` **sin las estrellas**, para etiquetarlas a mano sin sesgo
#    (ver `docs/guia_etiquetado.md`). Estas reseñas no se usan nunca para entrenar ni ajustar.
# 2. El resto se divide en 80 % train / 10 % val / 10 % test (estratificado por etiqueta débil, semilla 42).

# %%
from sklearn.model_selection import train_test_split

oro = (df.groupby("app", group_keys=False)
         .apply(lambda g: g.sample(n=N_ORO // df["app"].nunique(), random_state=SEMILLA)))
resto = df.drop(index=oro.index)

train, temp = train_test_split(resto, test_size=0.2, stratify=resto["etiqueta_debil"], random_state=SEMILLA)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["etiqueta_debil"], random_state=SEMILLA)
print(f"train={len(train)}  val={len(val)}  test={len(test)}  oro={len(oro)}")
assert set(oro["id"]).isdisjoint(train["id"]) and set(test["id"]).isdisjoint(train["id"])

ARCHIVO_ORO_PLANTILLA = DATA_ETQ / "oro_para_etiquetar.csv"
ARCHIVO_ORO_ETIQUETADO = DATA_ETQ / "oro_etiquetado.csv"
if not ARCHIVO_ORO_PLANTILLA.exists():
    plantilla = oro[["id", "app", "texto"]].sample(frac=1, random_state=SEMILLA).copy()
    plantilla["etiqueta_manual"] = ""   # negativa / neutra / positiva
    plantilla["comentario"] = ""
    plantilla.to_csv(ARCHIVO_ORO_PLANTILLA, index=False, encoding="utf-8-sig")
    print("Plantilla de etiquetado creada:", ARCHIVO_ORO_PLANTILLA)

for nombre, parte in [("train", train), ("val", val), ("test", test), ("oro", oro)]:
    parte.to_csv(DATA_PROC / f"{nombre}.csv", index=False)

# %% [markdown]
# ### Pre-etiquetado con LLM + revisión humana
# Las 300 reseñas no se etiquetan desde cero. El procedimiento es este (detalle en `docs/guia_etiquetado.md`):
# 1. Un modelo de lenguaje (Claude) lee **solo el texto** de `oro_para_etiquetar.csv` y escribe
#    `data/etiquetado/oro_etiquetado_llm.csv` con las columnas `id`, `etiqueta_llm` y `razon`.
# 2. Esta celda crea `oro_para_revisar.csv` con **todas las reseñas en las que el LLM no coincide con las estrellas**
#    y **30 coincidencias al azar** (control de calidad). La columna `etiqueta_revisada` viene llena con la del LLM:
#    solo se cambia cuando está mal.
# 3. Al volver a ejecutar, la celda une la revisión y genera `oro_etiquetado.csv`.

# %%
ARCHIVO_LLM = DATA_ETQ / "oro_etiquetado_llm.csv"
ARCHIVO_REVISAR = DATA_ETQ / "oro_para_revisar.csv"
ARCHIVO_REVISADO = DATA_ETQ / "oro_revisado.csv"   # oro_para_revisar.csv ya corregido y renombrado

if ARCHIVO_LLM.exists() and not ARCHIVO_ORO_ETIQUETADO.exists():
    llm = pd.read_csv(ARCHIVO_LLM)[["id", "etiqueta_llm", "razon"]]
    base_oro = oro[["id", "app", "texto", "etiqueta_debil"]].merge(llm, on="id", how="inner")
    desacuerdo = base_oro[base_oro["etiqueta_llm"] != base_oro["etiqueta_debil"]]
    acuerdo_muestra = base_oro[base_oro["etiqueta_llm"] == base_oro["etiqueta_debil"]].sample(
        n=min(30, (base_oro["etiqueta_llm"] == base_oro["etiqueta_debil"]).sum()), random_state=SEMILLA)
    if not ARCHIVO_REVISADO.exists():
        revisar = pd.concat([desacuerdo.assign(motivo="desacuerdo con estrellas"),
                             acuerdo_muestra.assign(motivo="control aleatorio")])
        revisar = revisar[["id", "app", "texto", "etiqueta_llm", "razon", "motivo"]].copy()
        revisar["etiqueta_revisada"] = revisar["etiqueta_llm"]
        revisar.to_csv(ARCHIVO_REVISAR, index=False, encoding="utf-8-sig")
        print(f"Para revisar a mano: {len(revisar)} reseñas ({len(desacuerdo)} desacuerdos + {len(acuerdo_muestra)} controles).")
        print(f"Corrige 'etiqueta_revisada' donde haga falta y guarda el archivo como {ARCHIVO_REVISADO.name}")
    else:
        rev = pd.read_csv(ARCHIVO_REVISADO)
        rev["etiqueta_revisada"] = rev["etiqueta_revisada"].str.strip().str.lower()
        assert rev["etiqueta_revisada"].isin(ETIQUETAS).all(), "Hay etiquetas revisadas vacías o mal escritas"
        final = base_oro.merge(rev[["id", "etiqueta_revisada"]], on="id", how="left")
        final["revisada_por_humano"] = final["etiqueta_revisada"].notna()
        final["etiqueta_manual"] = final["etiqueta_revisada"].fillna(final["etiqueta_llm"])
        corregidas = (final["revisada_por_humano"] & (final["etiqueta_manual"] != final["etiqueta_llm"])).sum()
        print(f"Revisadas por humano: {final['revisada_por_humano'].sum()} de {len(final)}. "
              f"Corregidas: {corregidas} ({corregidas / final['revisada_por_humano'].sum():.1%} de las revisadas).")
        final[["id", "app", "texto", "etiqueta_llm", "etiqueta_manual", "revisada_por_humano"]].to_csv(
            ARCHIVO_ORO_ETIQUETADO, index=False, encoding="utf-8-sig")
        print("Creado:", ARCHIVO_ORO_ETIQUETADO)

# %% [markdown]
# ### Cargar las etiquetas manuales
# Cuando termines de etiquetar, guarda el archivo como `data/etiquetado/oro_etiquetado.csv`.
# Si todavía no existe, el notebook sigue funcionando y evalúa el oro con la etiqueta débil
# (marcado como **provisional**); la nota final debe usar las etiquetas manuales.

# %%
if ARCHIVO_ORO_ETIQUETADO.exists():
    etq = pd.read_csv(ARCHIVO_ORO_ETIQUETADO)
    etq["etiqueta_manual"] = etq["etiqueta_manual"].str.strip().str.lower()
    assert etq["etiqueta_manual"].isin(ETIQUETAS).all(), "Hay etiquetas manuales vacías o mal escritas"
    oro = oro.merge(etq[["id", "etiqueta_manual"]], on="id", how="inner")
    oro["y_oro"] = oro["etiqueta_manual"]
    ORO_PROVISIONAL = False
    acuerdo = (oro["etiqueta_manual"] == oro["etiqueta_debil"]).mean()
    print(f"Etiquetas manuales cargadas: {len(oro)}. Acuerdo con la etiqueta débil (estrellas): {acuerdo:.1%}")
    print(pd.crosstab(oro["etiqueta_debil"], oro["etiqueta_manual"], margins=True))
else:
    oro["y_oro"] = oro["etiqueta_debil"]
    ORO_PROVISIONAL = True
    print("⚠ Aún no hay etiquetas manuales: se usa la etiqueta débil de forma PROVISIONAL.")

y_train = train["etiqueta_debil"].map(ETQ2ID).values
y_val = val["etiqueta_debil"].map(ETQ2ID).values
y_test = test["etiqueta_debil"].map(ETQ2ID).values
y_oro = oro["y_oro"].map(ETQ2ID).values

# %% [markdown]
# ## 6. Línea base: TF-IDF + regresión logística
# Se ajusta `C` con el conjunto de validación. `class_weight="balanced"` compensa el desbalance
# (la clase neutra es minoritaria). El vocabulario TF-IDF se construye **solo con train**.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

resultados_val = []
mejor_f1, mejor_C = -1, None
for C in [0.1, 0.3, 1, 3, 10]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEMILLA)),
    ])
    pipe.fit(train["texto"], y_train)
    f1 = f1_score(y_val, pipe.predict(val["texto"]), average="macro")
    resultados_val.append({"C": C, "f1_macro_val": f1})
    if f1 > mejor_f1:
        mejor_f1, mejor_C, base = f1, C, pipe
print(pd.DataFrame(resultados_val))
print(f"Mejor C={mejor_C} (F1-macro val={mejor_f1:.3f})")

# %% [markdown]
# ## 7. Red profunda 1: BiLSTM con embeddings
# `TextVectorization` se adapta **solo con train**. Arquitectura: Embedding(128) → BiLSTM(64) → Dropout → Dense(softmax).
# Pesos de clase para el desbalance y *early stopping* sobre la pérdida de validación.

# %%
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
# Evita que TensorFlow reserve toda la GPU (BETO con PyTorch la necesita después)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(SEMILLA)
keras.utils.set_random_seed(SEMILLA)

VOCAB, LARGO = 20000, 80
vectorizador = keras.layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=LARGO,
                                              standardize="lower_and_strip_punctuation")
vectorizador.adapt(train["texto"].values)

def construir_bilstm(emb=128, unidades=64, dropout=0.4):
    entrada = keras.Input(shape=(LARGO,), dtype="int64")
    x = keras.layers.Embedding(VOCAB, emb, mask_zero=True)(entrada)
    x = keras.layers.Bidirectional(keras.layers.LSTM(unidades))(x)
    x = keras.layers.Dropout(dropout)(x)
    salida = keras.layers.Dense(len(ETIQUETAS), activation="softmax")(x)
    m = keras.Model(entrada, salida)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

X_train_seq = vectorizador(train["texto"].values).numpy()
X_val_seq = vectorizador(val["texto"].values).numpy()
pesos = compute_class_weight("balanced", classes=np.arange(len(ETIQUETAS)), y=y_train)
pesos = dict(enumerate(pesos))

bilstm = construir_bilstm()
bilstm.summary()
hist = bilstm.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=10, batch_size=64,
                  class_weight=pesos, verbose=2,
                  callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

pd.DataFrame(hist.history)[["loss", "val_loss"]].plot(title="BiLSTM: curva de pérdida")
plt.savefig(FIGS / "bilstm_perdida.png", dpi=150); plt.show()

def predecir_bilstm(textos):
    return bilstm.predict(vectorizador(np.array(textos)).numpy(), verbose=0)

print("F1-macro val BiLSTM:", f1_score(y_val, predecir_bilstm(val["texto"].values).argmax(1), average="macro"))

# %% [markdown]
# ## 8. Red profunda 2: ajuste fino de BETO
# Modelo: `dccuchile/bert-base-spanish-wwm-cased` (Cañete et al., 2020).
# Hiperparámetros: lr 2e-5, 3 épocas, batch 32, longitud máxima 128, warmup 10 %, selección del mejor
# checkpoint por F1-macro en validación. Se usan pesos de clase en la pérdida.
# Para limitar el tiempo en Colab se entrena con hasta 12.000 reseñas de train (`MAX_TRAIN_BETO`).

# %%
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, set_seed)
set_seed(SEMILLA)
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", DISPOSITIVO)

BETO = "dccuchile/bert-base-spanish-wwm-cased"
MAX_TRAIN_BETO = 12000
tok = AutoTokenizer.from_pretrained(BETO)

def a_dataset(partes, y):
    ds = Dataset.from_dict({"text": list(partes["texto"]), "label": list(map(int, y))})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

tr_beto = train.sample(n=min(MAX_TRAIN_BETO, len(train)), random_state=SEMILLA)
ds_train = a_dataset(tr_beto, tr_beto["etiqueta_debil"].map(ETQ2ID).values)
ds_val = a_dataset(val, y_val)

pesos_t = torch.tensor(list(pesos.values()), dtype=torch.float)

class TrainerPonderado(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        etiquetas = inputs.pop("labels")
        salidas = model(**inputs)
        perdida = nn.CrossEntropyLoss(weight=pesos_t.to(salidas.logits.device))(salidas.logits, etiquetas)
        return (perdida, salidas) if return_outputs else perdida

def metricas(p):
    return {"f1_macro": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

modelo_beto = AutoModelForSequenceClassification.from_pretrained(
    BETO, num_labels=len(ETIQUETAS), id2label=dict(enumerate(ETIQUETAS)), label2id=ETQ2ID)

import math
pasos_totales = math.ceil(len(ds_train) / 32) * 3          # lotes por época x 3 épocas
pasos_warmup = int(0.1 * pasos_totales)                    # warmup del 10 %

args = TrainingArguments(
    output_dir=str(MODELOS / "beto_checkpoints"), learning_rate=2e-5, num_train_epochs=3,
    per_device_train_batch_size=32, per_device_eval_batch_size=64, warmup_steps=pasos_warmup, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1_macro", save_total_limit=1, seed=SEMILLA,
    fp16=torch.cuda.is_available(), report_to="none", logging_steps=50)

trainer = TrainerPonderado(model=modelo_beto, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                           data_collator=DataCollatorWithPadding(tok), compute_metrics=metricas)
trainer.train()
print(trainer.evaluate())

def predecir_beto(textos, lote=64):
    modelo_beto.eval(); probs = []
    for i in range(0, len(textos), lote):
        enc = tok(list(textos[i:i + lote]), truncation=True, max_length=128, padding=True,
                  return_tensors="pt").to(modelo_beto.device)
        with torch.no_grad():
            probs.append(torch.softmax(modelo_beto(**enc).logits, -1).cpu().numpy())
    return np.vstack(probs)

# %% [markdown]
# ## 9. Referencia preentrenada: pysentimiento (sin ajuste fino)
# `pysentimiento` (Pérez et al.) trae un RoBERTuito ya entrenado para sentimiento en español (POS/NEU/NEG).
# Sirve para responder: ¿vale la pena el ajuste fino propio frente a un modelo listo para usar?

# %%
from pysentimiento import create_analyzer
analizador = create_analyzer(task="sentiment", lang="es")
MAPA_PYS = {"NEG": "negativa", "NEU": "neutra", "POS": "positiva"}

def predecir_pysentimiento(textos):
    salida = analizador.predict(list(textos))
    return np.array([[o.probas["NEG"], o.probas["NEU"], o.probas["POS"]] for o in salida])

# %% [markdown]
# ## 10. Evaluación final
# **Métrica principal: F1-macro sobre las 300 reseñas etiquetadas a mano (conjunto de oro).**
# Se elige F1-macro porque las clases están desbalanceadas (la exactitud premiaría a un modelo que solo
# dice "negativa") y porque le da el mismo peso a la clase neutra, que es la más difícil.
# Como referencia secundaria se reporta el F1-macro sobre el test con etiqueta débil.

# %%
modelos = {
    "TF-IDF + RegLog (línea base)": lambda t: base.predict_proba(t),
    "BiLSTM": predecir_bilstm,
    "BETO ajuste fino": predecir_beto,
    "pysentimiento (sin ajuste)": predecir_pysentimiento,
}

filas, probas_oro = [], {}
for nombre, fn in modelos.items():
    p_oro = fn(oro["texto"].values); probas_oro[nombre] = p_oro
    p_test = fn(test["texto"].values)
    filas.append({
        "Modelo": nombre,
        "F1-macro ORO (principal)": f1_score(y_oro, p_oro.argmax(1), average="macro"),
        "Recall negativa ORO": (p_oro.argmax(1)[y_oro == 0] == 0).mean(),
        "Exactitud ORO": (p_oro.argmax(1) == y_oro).mean(),
        "F1-macro test débil": f1_score(y_test, p_test.argmax(1), average="macro"),
    })
tabla = pd.DataFrame(filas).round(3).sort_values("F1-macro ORO (principal)", ascending=False)
tabla.to_csv(RAIZ / "reports" / "tabla_resultados.csv", index=False)
if ORO_PROVISIONAL:
    print("⚠ RESULTADOS PROVISIONALES: el oro aún usa la etiqueta débil.")
tabla

# %%
fig, ejes = plt.subplots(1, len(modelos), figsize=(5 * len(modelos), 4))
for ax, (nombre, p) in zip(ejes, probas_oro.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_oro, p.argmax(1)), display_labels=ETIQUETAS).plot(ax=ax, colorbar=False)
    ax.set_title(nombre, fontsize=9)
plt.tight_layout(); plt.savefig(FIGS / "matrices_confusion_oro.png", dpi=150); plt.show()

MEJOR = tabla.iloc[0]["Modelo"]
print("Mejor modelo:", MEJOR)
print(classification_report(y_oro, probas_oro[MEJOR].argmax(1), target_names=ETIQUETAS, digits=3))

# %% [markdown]
# ## 11. Análisis de errores
# Se listan los errores del mejor modelo en el conjunto de oro, ordenados por confianza (los errores
# "seguros" son los más interesantes). Se exportan a `reports/analisis_errores.csv` con una columna
# `causa` para comentar **al menos 10** casos. Categorías sugeridas: sarcasmo/ironía, reseña mixta,
# queja sobre algo externo a la app, texto muy corto, error de etiquetado, jerga/ortografía, negación.

# %%
p = probas_oro[MEJOR]
errores = oro.assign(prediccion=[ETIQUETAS[i] for i in p.argmax(1)], confianza=p.max(1))
errores = errores[errores["prediccion"] != errores["y_oro"]].sort_values("confianza", ascending=False)
errores = errores[["id", "app", "texto", "estrellas", "y_oro", "prediccion", "confianza"]].copy()
errores["causa"] = ""
errores.to_csv(RAIZ / "reports" / "analisis_errores.csv", index=False, encoding="utf-8-sig")
print(f"Errores del mejor modelo en el oro: {len(errores)} de {len(oro)}")
pd.set_option("display.max_colwidth", 200)
errores.head(15)

# %% [markdown]
# ## 12. Explicabilidad
# **a) Línea base:** las palabras con mayor coeficiente por clase muestran qué términos empujan cada decisión.
# **b) BETO:** SHAP sobre un pipeline de Hugging Face resalta qué tokens aumentan la probabilidad de cada clase.

# %%
vocab = np.array(base.named_steps["tfidf"].get_feature_names_out())
coefs = base.named_steps["lr"].coef_
fig, ejes = plt.subplots(1, 3, figsize=(16, 5))
for i, ax in enumerate(ejes):
    top = np.argsort(coefs[i])[-15:]
    ax.barh(vocab[top], coefs[i][top]); ax.set_title(f"Términos → {ETIQUETAS[i]}")
plt.tight_layout(); plt.savefig(FIGS / "explicabilidad_coeficientes.png", dpi=150); plt.show()

# %%
try:
    import shap
    from transformers import pipeline as hf_pipeline
    clf = hf_pipeline("text-classification", model=modelo_beto, tokenizer=tok, top_k=None,
                      device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=128)
    explicador = shap.Explainer(clf)
    ejemplos = list(errores["texto"].head(3)) + list(oro["texto"].head(3))
    valores_shap = explicador(ejemplos)
    shap.plots.text(valores_shap)
except Exception as e:
    print("SHAP no se pudo ejecutar:", e)

# %% [markdown]
# ## 13. Componente NLP: ¿de qué se quejan los usuarios y mejora con el tiempo?
# 1. Se clasifica **todo** el corpus con el mejor modelo.
# 2. Se calcula el % de reseñas negativas por app y por mes.
# 3. Se extraen palabras clave con **YAKE** de las reseñas negativas, por app y por mes, quitando palabras
#    genéricas del dominio ("app", "aplicación", el nombre del banco…) que no dicen de qué se quejan.
# 4. Se agrupan las quejas en **temas** con reglas de palabras (acceso, actualización, compatibilidad, caídas…).
#
# Solo se grafican los meses con al menos 30 reseñas: con menos, un par de reseñas mueve el % de 0 a 100.

# %%
import yake
todas_p = modelos[MEJOR](df["texto"].values)
df["sentimiento"] = [ETIQUETAS[i] for i in todas_p.argmax(1)]

MIN_RESENAS_MES = 30
tendencia = (df.assign(neg=df["sentimiento"].eq("negativa"))
               .groupby(["app", "mes"])["neg"].agg(pct_negativas="mean", n_resenas="size").reset_index())
tendencia["pct_negativas"] *= 100
tendencia = tendencia[tendencia["n_resenas"] >= MIN_RESENAS_MES].sort_values("mes")
print(tendencia.pivot(index="mes", columns="app", values="n_resenas").fillna(0).astype(int))

plt.figure(figsize=(12, 4))
sns.lineplot(data=tendencia, x="mes", y="pct_negativas", hue="app", marker="o",
             hue_order=sorted(tendencia["app"].unique()))
plt.xticks(rotation=60); plt.ylabel("% negativas"); plt.ylim(0, 100)
plt.title(f"Reseñas negativas por mes (meses con ≥ {MIN_RESENAS_MES} reseñas)")
plt.tight_layout(); plt.savefig(FIGS / "tendencia_negativas.png", dpi=150); plt.show()

# %%
GENERICAS = {"app", "apps", "aplicación", "aplicacion", "aplicaciones", "nequi", "daviplata", "davivienda",
             "bancolombia", "banco", "deja", "dejar", "deja nada", "mala", "malo", "pésima", "pesima", "pésimo",
             "pesimo", "horrible", "terrible", "porquería", "servicio", "nada", "siempre", "ahora", "ser", "hacer",
             "puedo", "puede", "sirve", "funciona", "bien", "mal", "mejor", "peor", "gracias", "favor"}

def es_generica(frase):
    return all(t in GENERICAS for t in frase.lower().split())

extractor = yake.KeywordExtractor(lan="es", n=2, dedupLim=0.7, top=40)
if hasattr(extractor, "stopword_set"):
    extractor.stopword_set |= GENERICAS

def palabras_clave_de(textos, top=10):
    kws = extractor.extract_keywords(" . ".join(textos)[:300000])
    return [(k, s) for k, s in kws if not es_generica(k)][:top]

filas_kw = []
for (app, mes), g in df[df["sentimiento"] == "negativa"].groupby(["app", "mes"]):
    if len(g) < 20:
        continue
    for frase, puntaje in palabras_clave_de(g["texto"].tolist()):
        filas_kw.append({"app": app, "mes": mes, "palabra_clave": frase, "puntaje_yake": puntaje, "n_negativas": len(g)})
palabras_clave = pd.DataFrame(filas_kw)
palabras_clave.to_csv(DATA_PROC / "palabras_clave_negativas.csv", index=False)

# Palabras clave globales por app (para el informe)
for app, g in df[df["sentimiento"] == "negativa"].groupby("app"):
    print(f"\n{app} ({len(g)} negativas): " + ", ".join(k for k, _ in palabras_clave_de(g["texto"].tolist())))

# %% [markdown]
# ### Temas de queja
# Cada reseña negativa se asigna a uno o varios temas según palabras que contiene. Es un método transparente
# y fácil de explicar; el % por app muestra de qué se queja la gente en cada banco.

# %%
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
temas_app = neg.groupby("app")[list(TEMAS)].mean().mul(100).round(1).T
temas_app.to_csv(DATA_PROC / "temas_queja_por_app.csv")
print("% de reseñas negativas que mencionan cada tema:")
display(temas_app) if "display" in globals() else print(temas_app)

temas_app.plot(kind="barh", figsize=(10, 5))
plt.xlabel("% de reseñas negativas"); plt.title("Temas de queja por app"); plt.gca().invert_yaxis()
plt.tight_layout(); plt.savefig(FIGS / "temas_queja_por_app.png", dpi=150); plt.show()

# %% [markdown]
# **Evaluación de las palabras clave:** revisa a mano las 10 palabras clave globales de cada app y marca
# si corresponden a una queja real y concreta (precisión manual). Registra el resultado en el informe.

# %% [markdown]
# ## 14. Exportar para la demo

# %%
import joblib
joblib.dump(base, MODELOS / "linea_base_tfidf_lr.joblib")
modelo_beto.save_pretrained(MODELOS / "beto_sentimiento")
tok.save_pretrained(MODELOS / "beto_sentimiento")
df[["id", "app", "texto", "fecha", "mes", "estrellas", "sentimiento"]].to_csv(DATA_PROC / "resenas_clasificadas.csv", index=False)
tendencia.to_csv(DATA_PROC / "tendencia_negativas.csv", index=False)
print("Listo. Modelos en", MODELOS, "y datos para la demo en", DATA_PROC)
print("Para la demo: streamlit run app/app.py")
