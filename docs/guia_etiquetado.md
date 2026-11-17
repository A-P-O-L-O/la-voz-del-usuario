# Guía de etiquetado del conjunto de oro (300 reseñas)

Las 300 reseñas de evaluación **no se etiquetan desde cero**. Se usa **pre-etiquetado con un modelo de lenguaje +
revisión humana de los casos dudosos**. Así se tarda unos 30–45 minutos en lugar de 2–3 horas, y la etiqueta
final sigue siendo responsabilidad del autor.

## Procedimiento
1. **Ejecutar el notebook** hasta la sección 5. Se crea `data/etiquetado/oro_para_etiquetar.csv`
   (300 reseñas **sin estrellas**).
2. **Pre-etiquetado con LLM.** Claude lee solo el texto y aplica las reglas de abajo. Escribe
   `data/etiquetado/oro_etiquetado_llm.csv` con `id`, `etiqueta_llm` y `razon` (una frase que justifica la etiqueta).
3. **Volver a ejecutar la sección 5.** Se crea `data/etiquetado/oro_para_revisar.csv` con:
   - todas las reseñas en las que el LLM **no coincide con las estrellas** (los casos dudosos), y
   - 30 reseñas al azar en las que sí coinciden (control de calidad).
4. **Revisión humana.** Abre `oro_para_revisar.csv` en Excel o Google Sheets. La columna `etiqueta_revisada` ya trae
   la etiqueta del LLM: **solo la cambias si está mal**. Guarda el archivo como
   **`data/etiquetado/oro_revisado.csv`** (CSV UTF-8).
5. **Volver a ejecutar el notebook completo.** Se genera `oro_etiquetado.csv` y el notebook reporta cuántas
   etiquetas corregiste. Ese número va en el informe.

### Cómo se justifica en el informe y en la sustentación
- El LLM que etiqueta (Claude) **no es ninguno de los modelos evaluados**, así que no hay evaluación circular.
- Lee solo el texto: no ve las estrellas ni las predicciones de los modelos.
- Todas las reseñas dudosas las revisa un humano, y también una muestra de control. El % de correcciones mide
  qué tan confiable fue el pre-etiquetado.
- Se declara en la sección "Uso de IA generativa".

## Etiquetas (escribir exactamente así)
| Etiqueta | Cuándo usarla | Ejemplos |
|---|---|---|
| `negativa` | Queja, problema, frustración o crítica dominante | "No me deja entrar desde la actualización", "pésima atención" |
| `neutra` | Pregunta, sugerencia sin emoción, texto informativo, o reseña mixta sin un lado dominante | "¿Cómo cambio la clave?", "Buena app pero le falta modo oscuro y a veces se cae" |
| `positiva` | Satisfacción o elogio dominante | "Excelente, muy fácil transferir", "Me encanta" |

## Casos difíciles (reglas fijas)
1. **Mixta:** si hay elogio y queja, gana la parte que ocupa más texto o la que va al final; si están parejas → `neutra`.
2. **Sarcasmo** ("Excelente, otra vez caída 🙃") → según la intención real: `negativa`.
3. **Solo emojis o una palabra** ("ok", "👍") → según el sentido evidente; si no se entiende → `neutra`.
4. **Queja sobre el banco, no sobre la app** (cobros, atención en oficina) → sigue siendo `negativa`.
5. **Texto en otro idioma o sin sentido** → `neutra`, y se anota "fuera de dominio" en `razon`.

Las reglas se fijan antes de etiquetar y no se cambian después de ver los resultados.
