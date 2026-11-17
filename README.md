# La voz del usuario: sentimiento y quejas en reseñas de apps bancarias colombianas

Proyecto Integrador · Inteligencia Artificial Avanzada · Momento 3 · **Ruta A · Reto A1**

**Repositorio:** https://github.com/A-P-O-L-O/la-voz-del-usuario · **Informe:** [docs/informe_ieee/informe_ieee.pdf](docs/informe_ieee/informe_ieee.pdf) · **Diapositivas:** [docs/diapositivas.pptx](docs/diapositivas.pptx)

## Problema
Las apps bancarias más usadas en Colombia (Nequi, Daviplata, Bancolombia) reciben miles de reseñas al mes en
Google Play. Nadie las lee todas, así que el equipo de producto no sabe de qué se quejan los usuarios ni si la
situación mejora después de cada versión. Este proyecto clasifica el sentimiento de cada reseña, extrae las
quejas más frecuentes por app y por mes y muestra la tendencia en un tablero.

## Qué hace
| Componente | Técnica |
|---|---|
| Línea base | TF-IDF (1–2 gramas) + regresión logística |
| Red profunda 1 | BiLSTM con embeddings entrenados desde cero (Keras) |
| Red profunda 2 | Ajuste fino de BETO (`dccuchile/bert-base-spanish-wwm-cased`) |
| Referencia | pysentimiento (RoBERTuito) sin ajuste |
| NLP | Sentimiento + palabras clave YAKE de las reseñas negativas, por app y por mes |
| Explicabilidad | Coeficientes de la regresión logística + SHAP sobre BETO |
| Métrica principal | **F1-macro sobre 300 reseñas de evaluación** (pre-etiquetadas con LLM y revisadas a mano) |
| Demo | Streamlit (`app/app.py`) |

## Estructura
```
PI_RutaA_ResenasApps/
├── notebooks/pipeline_resenas.ipynb   # todo el proyecto, de principio a fin
├── notebooks/pipeline_resenas.py      # mismo notebook en formato texto (# %%)
├── app/app.py                         # demo Streamlit
├── data/raw/                          # reseñas descargadas + metadatos (fecha de descarga)
├── data/processed/                    # train/val/test/oro, reseñas clasificadas, palabras clave
├── data/etiquetado/                   # 300 reseñas para etiquetar a mano
├── models/                            # línea base y BETO ajustado
├── reports/                           # tabla de resultados, análisis de errores, figuras
└── docs/                              # ficha Hito 1, plan Hito 2, guía de etiquetado, informe IEEE
```

## Cómo reproducirlo
### Opción A: Google Colab (recomendada, necesita GPU para BETO)
1. Sube la carpeta del proyecto a Google Drive o clona el repositorio en Colab.
2. Abre `notebooks/pipeline_resenas.ipynb` y elige **Entorno de ejecución → T4 GPU**.
3. Ejecuta todas las celdas. La primera vez descarga las reseñas y crea `data/etiquetado/oro_para_etiquetar.csv`.
4. Pre-etiqueta las 300 reseñas con el LLM y revisa los casos dudosos (ver `docs/guia_etiquetado.md`)
   y vuelve a ejecutar el notebook para obtener los resultados finales.

### Opción B: local
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/pipeline_resenas.ipynb
```

### Demo
Si corriste el notebook en Colab, copia `data/processed/` a tu PC. Luego:
```bash
python app/preparar_demo.py      # reentrena la línea base y calcula tendencia, temas y palabras clave
streamlit run app/app.py
```
La demo usa pysentimiento y la línea base. Para incluir BETO, copia `models/beto_sentimiento/` desde Drive.

## Datos
- **Fuente:** Google Play Store, descargadas con [`google-play-scraper`](https://pypi.org/project/google-play-scraper/)
  (`lang='es'`, `country='co'`, orden: más recientes). 5.000 reseñas por app, 15.000 en total.
- **Fecha de descarga:** queda registrada en `data/raw/metadatos_descarga.json`.
- **Licencia / uso:** contenido público de Google Play usado solo con fines académicos y no comerciales.
  Las reseñas pertenecen a sus autores y están sujetas a los términos de servicio de Google Play.
- **Anonimización (Ley 1581 de 2012):** se eliminan nombre e imagen del usuario, el id de la reseña se reemplaza
  por un hash y en el texto se enmascaran correos y números largos (teléfonos, cédulas, cuentas).

## Reproducibilidad
- Semilla fija `42` en Python, NumPy, TensorFlow, PyTorch y Transformers.
- El conjunto de oro (300) y la división train/val/test (80/10/10) se hacen **antes** de construir cualquier
  vocabulario o tokenizador.
- El archivo de reseñas crudas se guarda y no se vuelve a descargar, porque Google Play cambia cada día.

## Ética y límites
- La etiqueta por estrellas es ruidosa; por eso la evaluación final usa etiquetas manuales.
- Las reseñas de Google Play no representan a todos los usuarios: escriben más quienes están muy contentos
  o muy molestos.
- El modelo es una herramienta de apoyo y no reemplaza la atención al cliente.

## Uso de IA generativa
Se usó Claude (Anthropic) como asistente para estructurar el código y los documentos, y para pre-etiquetar las 300 reseñas de evaluación (revisadas a mano por el autor en todos los casos dudosos). Todo el código fue revisado,
ejecutado y entendido por el autor.

## Autor
Elias Jaramillo · Fundación Universitaria Compensar · Inteligencia Artificial Avanzada
