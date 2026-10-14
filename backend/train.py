"""
Entrenamiento de la IA local del chatbot de películas (dataset MongoDB mflix).

Se generan dos modelos y una base de conocimiento:

1) Modelo de CALIDAD de películas (Árbol de Decisión vs Random Forest)
   - Clasificación supervisada: predice la categoría de la nota IMDb
     (Mala / Regular / Buena / Excelente) a partir de año, duración, géneros,
     clasificación por edades, premios, votos, idiomas, países...
2) Modelo de INTENCIONES del chatbot (TF-IDF + Random Forest)
   - Aprende a qué tipo de pregunta corresponde cada frase del usuario.
3) Base de conocimiento (peliculas.csv.gz, cines.csv.gz, usuarios_top.csv)
   - Tabla limpia que el motor de inferencia (sistema experto) consulta.

Uso:
    python train.py --data-dir ../data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier

from features import CATEGORIES, FEATURE_NAMES, GENRES, build_feature_matrix, rating_to_category
from intents import INTENTS

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena la IA local de películas (Random Forest).")
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data",
                        help="Carpeta con los CSV de mflix (descomprimidos de archive.zip).")
    parser.add_argument("--out-dir", type=Path, default=HERE,
                        help="Carpeta donde guardar modelo.joblib y la base de conocimiento.")
    return parser.parse_args()


def join_list_columns(df: pd.DataFrame, prefix: str, limit: int | None = None) -> pd.Series:
    """Las listas de MongoDB vienen aplanadas como genres[0], genres[1]... -> 'A|B|C'."""
    cols = [c for c in df.columns if c.startswith(prefix + "[")]
    cols.sort(key=lambda c: int(c[len(prefix) + 1:-1]))
    if limit:
        cols = cols[:limit]
    values = df[cols].astype("string")
    return values.apply(lambda row: "|".join(v for v in row if isinstance(v, str) and v), axis=1)


def build_knowledge_base(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    print("[1/6] Cargando dataset mflix...")
    raw = pd.read_csv(data_dir / "mflix.movies.csv", low_memory=False)
    comments = pd.read_csv(data_dir / "mflix.comments.csv", usecols=["name", "movie_id"])
    theaters = pd.read_csv(data_dir / "mflix.theaters.csv")
    users = pd.read_csv(data_dir / "mflix.users.csv", usecols=["name"])

    print("[2/6] Limpiando y construyendo la base de conocimiento...")
    movies = pd.DataFrame({
        "id": raw["_id"],
        "title": raw["title"].astype("string"),
        "year": pd.to_numeric(raw["year"], errors="coerce"),
        "runtime": pd.to_numeric(raw["runtime"], errors="coerce"),
        "genres": join_list_columns(raw, "genres"),
        "directors": join_list_columns(raw, "directors", limit=3),
        "cast": join_list_columns(raw, "cast"),
        "countries": join_list_columns(raw, "countries", limit=3),
        "languages": join_list_columns(raw, "languages", limit=3),
        "rated": raw["rated"].astype("string").str.upper(),
        "wins": pd.to_numeric(raw["awards.wins"], errors="coerce").fillna(0).astype(int),
        "nominations": pd.to_numeric(raw["awards.nominations"], errors="coerce").fillna(0).astype(int),
        "imdb_rating": pd.to_numeric(raw["imdb.rating"], errors="coerce"),
        "imdb_votes": pd.to_numeric(raw["imdb.votes"], errors="coerce"),
        "tomatoes_viewer": pd.to_numeric(raw["tomatoes.viewer.rating"], errors="coerce"),
        "metacritic": pd.to_numeric(raw["metacritic"], errors="coerce"),
        "type": raw["type"].astype("string"),
        "plot": raw["plot"].astype("string"),
        "poster": raw["poster"].astype("string"),
    })
    movies = movies.dropna(subset=["title"])
    comment_counts = comments["movie_id"].value_counts()
    movies["num_comments"] = movies["id"].map(comment_counts).fillna(0).astype(int)
    movies = movies.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    cines = pd.DataFrame({
        "city": theaters["location.address.city"],
        "state": theaters["location.address.state"],
        "street": theaters["location.address.street1"],
    })
    top_users = comments["name"].value_counts().head(50).rename_axis("name").reset_index(name="comments")

    stats = {
        "n_movies": int(len(movies)),
        "n_series": int((movies["type"] == "series").sum()),
        "n_comments": int(len(comments)),
        "n_users": int(len(users)),
        "n_theaters": int(len(cines)),
    }
    return movies, cines, top_users, stats


def train_quality_models(movies: pd.DataFrame) -> dict:
    labeled = movies.dropna(subset=["imdb_rating"]).copy()
    X = build_feature_matrix(labeled)
    y = labeled["imdb_rating"].apply(rating_to_category).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[3/6] Entrenando Árbol de Decisión (Gini)...")
    decision_tree = DecisionTreeClassifier(
        criterion="gini", max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    decision_tree.fit(X_train, y_train)

    print("[4/6] Entrenando Random Forest (Bagging + Feature Bagging + Voto Mayoritario)...")
    random_forest = RandomForestClassifier(
        n_estimators=120,
        criterion="gini",
        max_depth=16,
        max_features="sqrt",
        min_samples_leaf=3,
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    random_forest.fit(X_train, y_train)

    dt_acc = accuracy_score(y_test, decision_tree.predict(X_test))
    rf_pred = random_forest.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   Accuracy Árbol de Decisión: {dt_acc:.4f}")
    print(f"   Accuracy Random Forest   : {rf_acc:.4f}")
    print(f"   OOB Score Random Forest  : {random_forest.oob_score_:.4f}")
    print(classification_report(y_test, rf_pred, labels=CATEGORIES, digits=3))

    importances = sorted(
        zip(FEATURE_NAMES, random_forest.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )
    class_counts = pd.Series(y).value_counts().reindex(CATEGORIES).fillna(0).astype(int).to_dict()
    return {
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "metrics": {
            "decision_tree_accuracy": float(dt_acc),
            "random_forest_accuracy": float(rf_acc),
            "random_forest_oob_score": float(random_forest.oob_score_),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
        },
        "feature_importances": importances,
        "class_distribution": class_counts,
        "medians": X.median().to_dict(),
    }


def train_intent_model() -> tuple[Pipeline, float]:
    print("[5/6] Entrenando el clasificador de intenciones (TF-IDF + Random Forest)...")
    texts, labels = [], []
    for intent, phrases in INTENTS.items():
        for phrase in phrases:
            texts.append(phrase)
            labels.append(intent)

    def make_pipeline() -> Pipeline:
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), strip_accents="unicode", lowercase=True)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode", lowercase=True)),
            ])),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
        ])

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
    probe = make_pipeline().fit(X_tr, y_tr)
    acc = accuracy_score(y_te, probe.predict(X_te))
    print(f"   Accuracy intenciones (hold-out): {acc:.4f}")

    model = make_pipeline().fit(texts, labels)  # modelo final con todas las frases
    return model, float(acc)


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    movies, cines, top_users, stats = build_knowledge_base(args.data_dir)
    quality = train_quality_models(movies)
    intent_model, intent_acc = train_intent_model()

    print("[6/6] Guardando modelo y base de conocimiento...")
    artifact = {
        "decision_tree": quality["decision_tree"],
        "random_forest": quality["random_forest"],
        "intent_model": intent_model,
        "metadata": {
            "dataset": "MongoDB sample_mflix (Kaggle)",
            "target": "Categoría de nota IMDb",
            "categories": CATEGORIES,
            "feature_names": FEATURE_NAMES,
            "genres": GENRES,
            "metrics": {**quality["metrics"], "intent_accuracy": intent_acc},
            "feature_importances": quality["feature_importances"],
            "class_distribution": quality["class_distribution"],
            "medians": quality["medians"],
            "stats": stats,
        },
    }
    joblib.dump(artifact, out / "modelo.joblib", compress=3)
    movies.to_csv(out / "peliculas.csv.gz", index=False, compression="gzip")
    cines.to_csv(out / "cines.csv.gz", index=False, compression="gzip")
    top_users.to_csv(out / "usuarios_top.csv", index=False)
    size_mb = (out / "modelo.joblib").stat().st_size / 1e6
    print(f"Modelo guardado en {out / 'modelo.joblib'} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
