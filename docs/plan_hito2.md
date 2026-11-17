# Plan de trabajo · Hito 2 (Clase 18 · Taller 16)

**Proyecto:** La voz del usuario: sentimiento y quejas en reseñas de apps bancarias colombianas
**Ruta:** A · **Reto:** A1 · **Integrante:** Elias Jaramillo

## 1. Problema
Nequi, Daviplata y Bancolombia reciben miles de reseñas al mes en Google Play _(dato del EDA)_. Hoy nadie las
analiza de forma sistemática: el área de experiencia de cliente no sabe cuáles son las quejas principales ni si
disminuyen con cada versión de la app.

## 2. Objetivos
**General:** construir y evaluar un sistema que clasifique el sentimiento de reseñas en español de apps
bancarias colombianas y resuma las quejas más frecuentes por app y por mes.

**Específicos:**
1. Construir un corpus de 15.000 reseñas anonimizadas y un conjunto de oro de 300 reseñas etiquetadas a mano.
2. Entrenar una línea base (TF-IDF + regresión logística) y dos redes profundas (BiLSTM y BETO con ajuste fino).
3. Superar la línea base en F1-macro sobre el conjunto de oro.
4. Extraer palabras clave de las reseñas negativas (YAKE) y mostrar su evolución mensual.
5. Analizar al menos 10 errores y explicar las predicciones (coeficientes y SHAP).
6. Publicar una demo en Streamlit y un repositorio reproducible.

## 3. Alcance
**Incluye:** 3 apps del sector bancario; reseñas en español de Colombia; 3 clases de sentimiento; palabras clave
de las reseñas negativas; demo local.

**No incluye:** respuestas automáticas a los usuarios; reseñas de App Store (iOS); detección de sarcasmo como
tarea aparte; despliegue en producción.

**Reto extra (si alcanza el tiempo):** modelado de temas con BERTopic y tablero ampliado.

## 4. Roles
| Rol | Responsable | Entregables |
|---|---|---|
| Datos | Elias Jaramillo | Descarga, anonimización, EDA, división, etiquetado de 300 reseñas |
| Modelo | Elias Jaramillo | Línea base, BiLSTM, BETO, ajuste de hiperparámetros |
| NLP | Elias Jaramillo | Palabras clave YAKE, tendencia mensual |
| Informe y presentación | Elias Jaramillo | Informe IEEE, diapositivas, demo |

## 5. Cronograma
| Semana / clase | Actividad | Estado |
|---|---|---|
| Clase 16 | Ruta y reto elegidos (Hito 0) | ☐ |
| Clase 17 | Ficha del pipeline (Hito 1). Descarga de datos y EDA | ☐ |
| Entre 17 y 18 | División train/val/test + oro. **Pre-etiquetado de las 300 reseñas con LLM + revisión manual de los casos dudosos** (≈ 30–45 min) | ☐ |
| Clase 18 | Este plan + plantilla IEEE con secciones (Hito 2) | ☐ |
| Entre 18 y 19 | Línea base y BiLSTM entrenados y evaluados | ☐ |
| Clase 19 | Primer resultado evaluado: línea base + red profunda (Hito 3) | ☐ |
| Entre 19 y 20 | BETO, análisis de errores, SHAP, YAKE, demo Streamlit, informe completo | ☐ |
| Clase 20 | Entrega final, presentación de 8–10 min y sustentación | ☐ |

## 6. Riesgos y mitigación
| Riesgo | Mitigación |
|---|---|
| Google Play bloquea o limita la descarga | Pausas entre lotes; el archivo se guarda una sola vez |
| Colab sin GPU o sesión cortada | Guardar checkpoints; reducir `MAX_TRAIN_BETO` |
| Etiquetado manual toma mucho tiempo | Pre-etiquetado con LLM y revisión humana solo de los casos dudosos; guía con criterios claros |
| El LLM etiqueta mal | Revisión manual de todos los desacuerdos con las estrellas y de una muestra de 30 acuerdos; se reporta el % corregido |
| Clase neutra con pocos ejemplos | Pesos de clase y F1-macro como métrica |
