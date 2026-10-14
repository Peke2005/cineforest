# 🌲 CineForest: chatbot de películas con Random Forest

Práctica de **Árboles de Decisión, Random Forest y Sistemas Expertos**. Es una IA propia, sin LLM ni APIs externas,, entrenada con scikit-learn sobre el dataset **MongoDB sample_mflix**: películas, comentarios, usuarios y cines. Se le hacen preguntas desde un chatbot web.

- **Frontend (GitHub Pages)**: `docs/`. HTML, CSS y JS estático.
- **Backend (Render)**: `backend/`. API FastAPI que carga `modelo.joblib`.

Es la arquitectura desacoplada de producción que se explica en clase: un frontend independiente y un backend de inferencia con FastAPI.

---

## 1) Estructura

```text
├── backend/
│   ├── train.py            # Entrena Árbol de Decisión + Random Forest + clasificador de intenciones
│   ├── features.py         # Ingeniería de variables (compartida entre train y API)
│   ├── intents.py          # Frases de entrenamiento del chatbot
│   ├── chatbot.py          # Sistema experto: motor de inferencia + base de conocimiento
│   ├── app.py              # API FastAPI (/chat, /info, /predict)
│   ├── modelo.joblib       # Modelos entrenados
│   ├── peliculas.csv.gz    # Base de conocimiento limpia
│   ├── cines.csv.gz
│   ├── usuarios_top.csv
│   └── requirements.txt
├── docs/                   # Frontend del chatbot (GitHub Pages)
│   ├── index.html · style.css · app.js
│   └── config.js           # URL de la API de Render
└── render.yaml             # Blueprint de Render
```

## 2) Cómo funciona la IA

1. **Modelo de calidad (supervisado, clasificación)**. Predice la categoría de la nota IMDb (**Mala < 5.5 ≤ Regular < 6.5 ≤ Buena < 7.5 ≤ Excelente**). Para ello usa el año, la duración, los votos, los premios, las nominaciones, los géneros, la clasificación por edades, los idiomas, los países y el número de comentarios.
   - `DecisionTreeClassifier` (Gini) frente a `RandomForestClassifier`. El bosque tiene 120 árboles, usa `max_features="sqrt"`, *bootstrap* y `oob_score=True`.
   - Resultados en test: Árbol ≈ 52,7 % y Random Forest ≈ 54,3 % (OOB ≈ 53,9 %). Con 4 clases, el azar ponderado estaría en torno al 30 %.
   - El chat muestra el **voto mayoritario**, es decir, cuántos árboles votan cada categoría, y la **importancia de las variables**.
   - Las **recomendaciones** se calculan por **proximidad del Random Forest**: dos películas se parecen si caen en las mismas hojas de los árboles.
2. **Clasificador de intenciones**. Usa TF-IDF y un Random Forest para decidir qué tipo de pregunta hace el usuario: info, predicción, ranking, director, actor, recomendación, estadísticas, cines, modelo o teoría.
3. **Sistema experto**. Unas reglas con palabras clave refuerzan al clasificador. Después, el motor de inferencia consulta la base de conocimiento (`peliculas.csv.gz`) o el modelo y redacta la respuesta.

### Ejemplos de preguntas
- *Háblame de The Matrix*
- *Predice la calidad de Inception*
- *Predice una película de terror de 2010 de 95 minutos con 2 premios*
- *Top 10 de ciencia ficción de los 90* / *Peores películas de terror*
- *Películas de Christopher Nolan* / *Películas con Tom Hanks*
- *Recomiéndame algo parecido a Toy Story*
- *Cuántas películas de comedia hay* / *Película más comentada* / *Cines en California*
- *Qué variables son más importantes* / *Qué es el bagging*

## 3) Reentrenar el modelo

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
# Descomprime archive.zip en la carpeta data\ (mflix.movies.csv, mflix.comments.csv, ...)
cd backend
python train.py --data-dir ..\data
```

Después haz `git push`: Render vuelve a desplegar la API con el modelo nuevo.

## 4) Despliegue

### Backend en Render
1. En Render: **New → Blueprint**. Elige este repositorio y Render lee `render.yaml`.
2. Comandos que configura:
   - Build: `pip install -r requirements.txt` (con `rootDir: backend`)
   - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Cada `git push` vuelve a desplegar. En el plan gratuito, el servidor tarda unos 50 s en arrancar si ha estado inactivo.

### Frontend en GitHub Pages
- Settings → Pages → *Deploy from a branch* → `main` / `docs`.
- Si Render te da otra URL, cámbiala en `docs/config.js`.
