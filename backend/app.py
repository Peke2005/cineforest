"""
API de inferencia (FastAPI) desplegada en Render.

Endpoints:
- GET  /          -> estado del servicio
- GET  /info      -> métricas del modelo y resumen del dataset
- POST /chat      -> {"message": "..."} -> respuesta del chatbot
- POST /predict   -> variables de una película -> predicción del Random Forest
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot import MovieBot

bot = MovieBot()

app = FastAPI(title="CineForest API", description="IA local de películas con Random Forest (dataset mflix)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str = Field(..., max_length=500)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "CineForest API", "docs": "/docs"}


@app.get("/info")
def info() -> dict:
    meta = bot.meta
    return {
        "metrics": meta["metrics"],
        "stats": meta["stats"],
        "categories": meta["categories"],
        "feature_importances": meta["feature_importances"][:10],
        "n_trees": bot.random_forest.n_estimators,
    }


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    return bot.answer(body.message)


@app.post("/predict")
def predict(body: ChatIn) -> dict:
    return bot._predict_custom(body.message)
