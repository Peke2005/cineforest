"""Ingeniería de variables compartida entre el entrenamiento (train.py) y la API (app.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["Mala", "Regular", "Buena", "Excelente"]

GENRES = [
    "Drama", "Comedy", "Romance", "Crime", "Thriller", "Action", "Adventure",
    "Documentary", "Horror", "Biography", "Family", "Mystery", "Fantasy", "Sci-Fi",
    "Animation", "History", "Music", "War", "Short", "Musical", "Sport", "Western",
    "Film-Noir",
]

RATED_GROUPS = {
    "R": "R", "PG-13": "PG-13", "PG": "PG", "G": "G",
    "APPROVED": "CLASICA", "PASSED": "CLASICA", "GP": "CLASICA", "M": "CLASICA",
    "TV-14": "TV", "TV-PG": "TV", "TV-MA": "TV", "TV-G": "TV", "TV-Y7": "TV", "TV-Y": "TV",
}
RATED_COLUMNS = ["R", "PG-13", "PG", "G", "CLASICA", "TV", "SIN_CLASIFICAR"]

NUMERIC_FEATURES = [
    "year", "runtime", "log_votes", "wins", "nominations", "n_genres",
    "n_languages", "n_countries", "is_usa", "is_english", "is_series", "num_comments",
]
FEATURE_NAMES = NUMERIC_FEATURES + [f"genre_{g}" for g in GENRES] + [f"rated_{r}" for r in RATED_COLUMNS]


def rating_to_category(rating: float) -> str:
    if rating < 5.5:
        return "Mala"
    if rating < 6.5:
        return "Regular"
    if rating < 7.5:
        return "Buena"
    return "Excelente"


def _split(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(lambda s: [x for x in s.split("|") if x])


def build_feature_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Convierte la tabla de películas en la matriz numérica X que usan los árboles."""
    genres = _split(movies["genres"])
    countries = _split(movies["countries"])
    languages = _split(movies["languages"])

    X = pd.DataFrame(index=movies.index)
    X["year"] = movies["year"]
    X["runtime"] = movies["runtime"]
    X["log_votes"] = np.log1p(movies["imdb_votes"])
    X["wins"] = movies["wins"]
    X["nominations"] = movies["nominations"]
    X["n_genres"] = genres.apply(len)
    X["n_languages"] = languages.apply(len)
    X["n_countries"] = countries.apply(len)
    X["is_usa"] = countries.apply(lambda c: int("USA" in c))
    X["is_english"] = languages.apply(lambda l: int("English" in l))
    X["is_series"] = (movies["type"] == "series").astype(int)
    X["num_comments"] = movies["num_comments"]
    for g in GENRES:
        X[f"genre_{g}"] = genres.apply(lambda gs, g=g: int(g in gs))
    rated = movies["rated"].fillna("").map(RATED_GROUPS).fillna("SIN_CLASIFICAR")
    for r in RATED_COLUMNS:
        X[f"rated_{r}"] = (rated == r).astype(int)

    X = X[FEATURE_NAMES].astype(float)
    # Los árboles no aceptan NaN en esta versión: se imputan con la mediana.
    return X.fillna(X.median()).fillna(0)
