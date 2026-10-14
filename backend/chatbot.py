"""
Motor del chatbot: IA local sin LLM.

Arquitectura (sistema experto basado en reglas + Random Forest):
- Base de conocimiento: peliculas.csv.gz, cines.csv.gz, usuarios_top.csv
- Clasificador de intenciones: TF-IDF + Random Forest (modelo.joblib)
- Extracción de entidades: títulos, géneros, años, décadas, personas, números
- Motor de inferencia: reglas que, según intención + entidades, consultan el
  dataset o el Random Forest de calidad y redactan la respuesta.
"""

from __future__ import annotations

import random
import re
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, GENRES, build_feature_matrix

HERE = Path(__file__).resolve().parent

GENRE_ALIASES = {
    "drama": "Drama", "dramas": "Drama", "dramatica": "Drama", "dramaticas": "Drama",
    "comedia": "Comedy", "comedias": "Comedy", "comedy": "Comedy",
    "romance": "Romance", "romantica": "Romance", "romanticas": "Romance", "amor": "Romance",
    "crimen": "Crime", "crime": "Crime", "policiaca": "Crime", "policiacas": "Crime",
    "thriller": "Thriller", "thrillers": "Thriller", "suspense": "Thriller",
    "accion": "Action", "action": "Action",
    "aventura": "Adventure", "aventuras": "Adventure", "adventure": "Adventure",
    "documental": "Documentary", "documentales": "Documentary", "documentary": "Documentary",
    "terror": "Horror", "miedo": "Horror", "horror": "Horror",
    "biografia": "Biography", "biograficas": "Biography", "biografica": "Biography", "biopic": "Biography",
    "familiar": "Family", "familiares": "Family", "family": "Family",
    "misterio": "Mystery", "mystery": "Mystery",
    "fantasia": "Fantasy", "fantasy": "Fantasy",
    "ciencia ficcion": "Sci-Fi", "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "ciencia-ficcion": "Sci-Fi",
    "animacion": "Animation", "animadas": "Animation", "animada": "Animation", "dibujos": "Animation", "animation": "Animation",
    "historia": "History", "historicas": "History", "historica": "History", "history": "History",
    "musica": "Music", "music": "Music", "musical": "Musical", "musicales": "Musical",
    "guerra": "War", "belicas": "War", "belica": "War", "war": "War",
    "cortometraje": "Short", "cortometrajes": "Short", "cortos": "Short", "corto": "Short",
    "deporte": "Sport", "deportes": "Sport", "deportivas": "Sport", "sport": "Sport",
    "western": "Western", "westerns": "Western", "oeste": "Western", "vaqueros": "Western",
    "cine negro": "Film-Noir", "noir": "Film-Noir", "film-noir": "Film-Noir",
}
GENRE_ES = {
    "Drama": "drama", "Comedy": "comedia", "Romance": "romance", "Crime": "crimen",
    "Thriller": "thriller", "Action": "acción", "Adventure": "aventura",
    "Documentary": "documental", "Horror": "terror", "Biography": "biografía",
    "Family": "familiar", "Mystery": "misterio", "Fantasy": "fantasía", "Sci-Fi": "ciencia ficción",
    "Animation": "animación", "History": "historia", "Music": "música", "War": "guerra",
    "Short": "cortometraje", "Musical": "musical", "Sport": "deporte", "Western": "western",
    "Film-Noir": "cine negro", "News": "noticias", "Talk-Show": "talk-show",
}
CATEGORY_EMOJI = {"Mala": "🔴", "Regular": "🟠", "Buena": "🟢", "Excelente": "🏆"}

# Palabras de las plantillas de intención que no forman parte de títulos o nombres.
STOPWORDS = set("""
a al algo alguna algun como con cual cuales cuantas cuantos cuanto cuenta cuentame da dame de del dime donde el
en es esta este esto la las lo los me mi mas mejor mejores peor peores mucho muy no o para parecida parecido
parecidas parecidos pelicula peliculas por que quien sale salen se sobre su sus tiene tienen top un una unas uno
unos y ya ha han hecho dirigio dirigidas dirigida director directora actor actriz actua actuan protagonizadas
protagonizada reparto filmografia informacion info datos ficha sinopsis argumento trata habla hablame sabes busca
nota puntuacion ano anos salio dura duracion predice prediccion predecir clasifica calidad modelo random forest
categoria buena mala segun recomiendame recomienda recomiendas similares similar como quiero ver sugiereme
comentarios comentada comentadas obras trabaja hizo sobre acerca imdb hay cuanta the
peli pelis film films serie series tal bueno buenas buenos larga largas largo cortas corta antiguas antigua
recientes reciente nuevas nueva populares votadas famosas conocidas puedes hacer haces sabes ver quiero
recomendar algun alguna esa ese eso mas menos todas todos
""".split())

SPANISH_TITLES = {
    "el padrino": "the godfather", "el padrino 2": "the godfather part ii", "cadena perpetua": "the shawshank redemption",
    "el club de la lucha": "fight club", "el caballero oscuro": "the dark knight", "origen": "inception",
    "el senor de los anillos": "the lord of the rings the fellowship of the ring", "la comunidad del anillo": "the lord of the rings the fellowship of the ring",
    "el retorno del rey": "the lord of the rings the return of the king", "las dos torres": "the lord of the rings the two towers",
    "la guerra de las galaxias": "star wars episode iv - a new hope", "el imperio contraataca": "star wars episode v - the empire strikes back",
    "el rey leon": "the lion king", "regreso al futuro": "back to the future", "tiburon": "jaws", "psicosis": "psycho",
    "el resplandor": "the shining", "la lista de schindler": "schindler's list", "salvar al soldado ryan": "saving private ryan",
    "el silencio de los corderos": "the silence of the lambs", "lo que el viento se llevo": "gone with the wind",
    "ciudadano kane": "citizen kane", "buscando a nemo": "finding nemo", "parque jurasico": "jurassic park",
    "el exorcista": "the exorcist", "la vida es bella": "life is beautiful", "los vengadores": "the avengers",
    "el viaje de chihiro": "spirited away", "la naranja mecanica": "a clockwork orange", "el sexto sentido": "the sixth sense",
    "el gran dictador": "the great dictator", "tiempos modernos": "modern times", "el laberinto del fauno": "pan's labyrinth",
    "seven": "se7en", "el pianista": "the pianist", "los intocables": "the untouchables", "intocable": "the intouchables",
    "el gladiador": "gladiator", "la milla verde": "the green mile", "el truco final": "the prestige",
    "malditos bastardos": "inglourious basterds", "toy story 3": "toy story 3", "monstruos sa": "monsters inc",
    "piratas del caribe": "pirates of the caribbean the curse of the black pearl", "harry potter": "harry potter and the sorcerer's stone",
    "el hobbit": "the hobbit an unexpected journey", "indiana jones": "raiders of the lost ark", "en busca del arca perdida": "raiders of the lost ark",
    "el lobo de wall street": "the wolf of wall street", "atrapame si puedes": "catch me if you can", "la red social": "the social network",
}

COUNTRY_ALIASES = {
    "espanolas": "Spain", "espanola": "Spain", "espana": "Spain", "espanoles": "Spain",
    "francesas": "France", "francesa": "France", "francia": "France",
    "italianas": "Italy", "italiana": "Italy", "italia": "Italy",
    "japonesas": "Japan", "japonesa": "Japan", "japon": "Japan",
    "alemanas": "Germany", "alemana": "Germany", "alemania": "Germany",
    "britanicas": "UK", "britanica": "UK", "inglesas": "UK", "inglesa": "UK", "reino unido": "UK",
    "americanas": "USA", "estadounidenses": "USA", "estados unidos": "USA", "eeuu": "USA", "hollywood": "USA",
    "mexicanas": "Mexico", "mexicana": "Mexico", "mexico": "Mexico",
    "coreanas": "South Korea", "coreana": "South Korea", "corea": "South Korea",
    "indias": "India", "india": "India", "bollywood": "India",
    "chinas": "China", "china": "China", "argentinas": "Argentina", "argentina": "Argentina",
    "suecas": "Sweden", "suecia": "Sweden", "rusas": "Russia", "rusia": "Russia",
    "canadienses": "Canada", "canada": "Canada", "australianas": "Australia", "australia": "Australia",
}
COUNTRY_ES = {"Spain": "españolas", "France": "francesas", "Italy": "italianas", "Japan": "japonesas",
              "Germany": "alemanas", "UK": "británicas", "USA": "estadounidenses", "Mexico": "mexicanas",
              "South Korea": "coreanas", "India": "indias", "China": "chinas", "Argentina": "argentinas",
              "Sweden": "suecas", "Russia": "rusas", "Canada": "canadienses", "Australia": "australianas"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\-' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_movie_line(row: pd.Series) -> str:
    year = f" ({int(row['year'])})" if pd.notna(row["year"]) else ""
    rating = f" — ⭐ {row['imdb_rating']:.1f}" if pd.notna(row["imdb_rating"]) else ""
    return f"**{row['title']}**{year}{rating}"


class MovieBot:
    def __init__(self, base_dir: Path = HERE) -> None:
        artifact = joblib.load(base_dir / "modelo.joblib")
        self.decision_tree = artifact["decision_tree"]
        self.random_forest = artifact["random_forest"]
        self.intent_model = artifact["intent_model"]
        self.meta = artifact["metadata"]

        self.movies = pd.read_csv(base_dir / "peliculas.csv.gz")
        self.cines = pd.read_csv(base_dir / "cines.csv.gz")
        self.top_users = pd.read_csv(base_dir / "usuarios_top.csv")
        self.X_all = build_feature_matrix(self.movies)
        self._leaves: np.ndarray | None = None

        self.movies["title_norm"] = self.movies["title"].map(normalize)
        self.movies["genre_list"] = self.movies["genres"].fillna("").str.split("|")
        # Índice de títulos (prioriza la versión con más votos si hay remakes).
        ordered = self.movies.sort_values("imdb_votes", ascending=False)
        self.title_index: dict[str, int] = {}
        self.title_alias: dict[str, int] = {}
        for idx, t in zip(ordered.index, ordered["title_norm"]):
            self.title_index.setdefault(t, idx)
        # Alias: "matrix" -> The Matrix, "star wars" -> Star Wars: Episode ...
        for idx, t in zip(ordered.index, ordered["title_norm"]):
            short = re.sub(r"^(the|a|an) ", "", t)
            if len(short) >= 4:
                self.title_alias.setdefault(short, idx)
        for idx, title in zip(ordered.index, ordered["title"]):
            if ":" in title:
                pre = normalize(title.split(":")[0])
                if len(pre) >= 4:
                    self.title_alias.setdefault(pre, idx)
                    self.title_alias.setdefault(re.sub(r"^(the|a|an) ", "", pre), idx)
        self.title_keys = sorted(self.title_index, key=len, reverse=True)
        self.alias_keys = sorted(self.title_alias, key=len, reverse=True)

        self.directors = self._person_index("directors")
        self.actors = self._person_index("cast")
        self.surnames = self._surname_index()
        self.cities = {normalize(c): c for c in self.cines["city"].dropna().unique()}
        self.states = {normalize(s): s for s in self.cines["state"].dropna().unique()}

    # ------------------------------------------------------------------ utils
    def _person_index(self, column: str) -> dict[str, str]:
        names = self.movies[column].dropna().str.split("|").explode().dropna()
        counts = names.value_counts()
        return {normalize(n): n for n in counts[counts >= 2].index}

    def _surname_index(self) -> dict[str, tuple[str, str]]:
        """Apellido -> (nombre completo, rol) del cineasta con más películas (ej. almodovar)."""
        out: dict[str, tuple[str, str, int]] = {}
        for column, role in (("cast", "cast"), ("directors", "directors")):
            names = self.movies[column].dropna().str.split("|").explode().dropna()
            for name, n in names.value_counts().items():
                if n < 3:
                    break
                last = normalize(name).split()[-1] if normalize(name) else ""
                if len(last) < 5 or last in STOPWORDS or last in GENRE_ALIASES:
                    continue
                bonus = 3 if role == "directors" else 0  # el director pesa más (Almodóvar, Kubrick...)
                if last not in out or n + bonus > out[last][2]:
                    out[last] = (name, role, n + bonus)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def find_surname(self, q: str) -> tuple[str, str] | None:
        words = [w for w in normalize(q).split()
                 if w not in STOPWORDS and w not in GENRE_ALIASES and w not in COUNTRY_ALIASES]
        for w in words:
            if w in self.surnames:
                return self.surnames[w]
        for w in words:
            if len(w) >= 6:
                match = get_close_matches(w, list(self.surnames), n=1, cutoff=0.88)
                if match:
                    return self.surnames[match[0]]
        return None

    def find_title(self, q: str, fuzzy: bool = True) -> int | None:
        quoted = re.findall(r"[\"'“”«»](.+?)[\"'“”«»]", q)
        for cand in quoted:
            idx = self.title_index.get(normalize(cand))
            if idx is not None:
                return idx
        nq = f" {normalize(q)} "
        for es, en in sorted(SPANISH_TITLES.items(), key=lambda kv: -len(kv[0])):
            if f" {es} " in nq:
                idx = self.title_index.get(en) or self.title_alias.get(en)
                if idx is not None:
                    return idx
        best: tuple[str, int] | None = None
        for key in self.title_keys:  # coincidencia exacta más larga dentro de la frase
            if len(key) < 4 and not quoted:
                continue
            if f" {key} " in nq:
                if key.split()[0] in STOPWORDS and len(key.split()) == 1:
                    continue
                best = (key, self.title_index[key])
                break
        for key in self.alias_keys:  # "matrix", "star wars"... gana si es más larga
            if best and len(key) <= len(best[0]):
                break
            if f" {key} " in nq and not (len(key.split()) == 1 and key in STOPWORDS):
                best = (key, self.title_alias[key])
                break
        if best:
            return best[1]
        if not fuzzy:
            return None
        # Búsqueda aproximada con lo que queda al quitar palabras de la plantilla.
        rest = " ".join(w for w in normalize(q).split()
                        if w not in STOPWORDS and w not in GENRE_ALIASES and w not in COUNTRY_ALIASES and not w.isdigit())
        if len(rest) >= 4:
            match = get_close_matches(rest, self.title_keys, n=1, cutoff=0.8)
            if match:
                return self.title_index[match[0]]
        return None

    def find_person(self, q: str, index: dict[str, str]) -> str | None:
        nq = f" {normalize(q)} "
        best = None
        for key, name in index.items():
            if " " in key and f" {key} " in nq and (best is None or len(key) > len(best[0])):
                best = (key, name)
        if best:
            return best[1]
        rest = " ".join(w for w in normalize(q).split() if w not in STOPWORDS)
        if len(rest.split()) >= 2:
            match = get_close_matches(rest, list(index), n=1, cutoff=0.85)
            if match:
                return index[match[0]]
        return None

    @staticmethod
    def find_country(q: str) -> str | None:
        nq = f" {normalize(q)} "
        for alias in sorted(COUNTRY_ALIASES, key=len, reverse=True):
            if f" {alias} " in nq:
                return COUNTRY_ALIASES[alias]
        return None

    @staticmethod
    def find_genre(q: str) -> str | None:
        nq = f" {normalize(q)} "
        for alias in sorted(GENRE_ALIASES, key=len, reverse=True):
            if f" {alias} " in nq:
                return GENRE_ALIASES[alias]
        return None

    @staticmethod
    def find_years(q: str) -> tuple[int, int] | None:
        nq = normalize(q)
        dec =re.search(r"\b(?:los|anos|decada de los|decada de)\s+(?:19)?([2-9])0\b", nq)
        if dec:
            start = 1900 + int(dec.group(1)) * 10
            return start, start + 9
        m = re.search(r"\b(?:los |decada de )(19|20)([0-9])0s?\b", nq) or re.search(r"\b(19|20)([0-9])0s\b", nq)
        if m:
            start = int(m.group(1) + m.group(2) + "0")
            return start, start + 9
        y = re.search(r"\b(18[89][0-9]|19[0-9]{2}|20[0-2][0-9])\b", nq)
        if y:
            return int(y.group(1)), int(y.group(1))
        return None

    @staticmethod
    def find_top_n(q: str, default: int = 5) -> int:
        nq = normalize(q)
        m = (re.search(r"\btop\s+(\d{1,2})\b", nq) or re.search(r"\b(\d{1,2})\s+(?:mejores|peores|peliculas)\b", nq)
             or re.search(r"\b(?:dame|las)\s+(\d{1,2})\b(?! minutos)", nq))
        return max(1, min(int(m.group(1)), 20)) if m else default

    # ------------------------------------------------------------ inference
    def detect_intent(self, q: str) -> tuple[str, float, bool]:
        proba = self.intent_model.predict_proba([q])[0]
        classes = self.intent_model.classes_
        best = int(np.argmax(proba))
        intent, conf = classes[best], float(proba[best])

        # Reglas del sistema experto: palabras clave muy fiables tienen prioridad.
        nq = normalize(q)
        rules = [
            (r"\b(predic|predec|predi|clasific|que categoria|segun el (modelo|random)|votan los arboles)", "predecir"),
            (r"\bque es\b|\bexplica|\bdiferencia entre\b|\bque significa\b|\bcomo funciona (el|la|un|una)\b", "teoria"),
            (r"\b(importan|accuracy|precision|metricas|oob score|hiperparametro|features usa|cuantos arboles)", "modelo"),
            (r"\b(recomiend|recomienda|parecid|similar|sugier|que veo|no se que ver|algo para ver|(pelicula|peli) buena|algo bueno)", "recomendar"),
            (r"\bcines?\b", "cines"),
            (r"\bcomentad|\bcomentarios\b|\bcomenta\b", "comentarios"),
            (r"\b(dirigid|dirigio|director|directora)\b", "director"),
            (r"\b(actua|actor|actriz|protagoniz|sale |salen )", "actor"),
            (r"\b(mejores|peores|top|ranking|mejor valorad|peor valorad|mejor puntuad|mejor pelicula)"
             r"|\bmas (larga|cort|votad|popular|famos|conocid|antigua|viej|reciente|nueva|premi)", "top_genero"),
            (r"\b(cuantas|cuantos|estadistic|media|resumen del dataset|que generos)", "estadisticas"),
        ]
        for pattern, rule_intent in rules:
            if re.search(pattern, nq):
                if (rule_intent == "teoria" and self.find_title(q, fuzzy=False) is not None
                        and not any(k in nq for keys, _ in KNOWLEDGE for k in keys)):
                    continue
                return rule_intent, (max(conf, 0.9) if rule_intent == intent else 0.9), True
        return intent, conf, False

    def answer(self, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            return {"intent": "vacío", "confidence": 1.0, "answer": "Escríbeme una pregunta sobre las películas 🎬"}
        intent, conf, by_rule = self.detect_intent(question)
        if not by_rule:
            intent = self._route_by_entities(question, intent, conf)

        handler = getattr(self, f"_h_{intent}", self._h_ayuda)
        result = handler(question)
        if isinstance(result, str):
            result = {"answer": result}
        result.update({"intent": intent, "confidence": round(conf, 3)})
        return result

    def _route_by_entities(self, q: str, intent: str, conf: float) -> str:
        """Motor de inferencia: las entidades encontradas corrigen la intención del Random Forest."""
        if intent == "saludo" and conf >= 0.7 or intent == "despedida" and conf >= 0.6:
            return intent
        d, a = self.find_person(q, self.directors), self.find_person(q, self.actors)
        if d and a:
            nd, na = self._count_person(d, "directors"), self._count_person(a, "cast")
            return "director" if nd >= na else "actor"
        if d:
            return "director"
        if a:
            return "actor"
        if self.find_title(q, fuzzy=False) is not None:
            return "info_pelicula" if intent not in ("predecir", "recomendar", "comentarios") else intent
        sur = self.find_surname(q)
        if sur:
            return "director" if sur[1] == "directors" else "actor"
        if self.find_genre(q) or self.find_country(q) or self.find_years(q):
            return intent if intent in ("estadisticas", "recomendar", "predecir", "top_genero") else "top_genero"
        if intent in ("info_pelicula", "teoria") and self.find_title(q) is not None:
            return "info_pelicula"
        if (conf < 0.3 and intent not in ("despedida", "ayuda")) or (intent == "saludo" and conf < 0.7):
            return "no_entiendo"
        return intent

    def _count_person(self, person: str, column: str) -> int:
        return int(self.movies[column].fillna("").str.split("|").apply(lambda xs: person in xs).sum())

    # ------------------------------------------------------------- handlers
    def _h_no_entiendo(self, q: str) -> str:
        return "No te he entendido del todo 🤔. " + self._h_ayuda(q)

    def _h_saludo(self, q: str) -> str:
        return random.choice([
            "¡Hola! 👋 Soy **CineForest**, una IA entrenada con Random Forest sobre el dataset de películas mflix. Pregúntame por películas, directores, actores, rankings o pídeme que prediga la calidad de una película.",
            "¡Buenas! 🎬 Conozco más de 21.000 películas del dataset mflix. Prueba: *«mejores películas de terror de los 80»* o *«predice la calidad de Titanic»*.",
        ])

    def _h_despedida(self, q: str) -> str:
        return "¡Hasta luego! 🍿 Que disfrutes de la peli."

    def _h_ayuda(self, q: str) -> str:
        return (
            "Puedo responder a preguntas como:\n"
            "- 🎞️ **Info**: *háblame de The Matrix*\n"
            "- 🌲 **Predicción con Random Forest**: *predice la calidad de Inception* o *predice una película de terror de 2010 de 95 minutos*\n"
            "- 🏆 **Rankings**: *top 10 de ciencia ficción de los 90*\n"
            "- 🎬 **Directores / actores**: *películas de Christopher Nolan*, *películas con Tom Hanks*\n"
            "- 💡 **Recomendaciones**: *recomiéndame algo parecido a Toy Story*\n"
            "- 📊 **Estadísticas**: *cuántas películas de comedia hay*, *película más comentada*, *cines en California*\n"
            "- 🧠 **El modelo**: *qué variables son más importantes*, *qué es el bagging*"
        )

    def _h_info_pelicula(self, q: str) -> dict | str:
        idx = self.find_title(q)
        if idx is None:
            person = self.find_person(q, self.directors) or self.find_person(q, self.actors)
            if person:
                return self._person_movies(person, "directors" if person in self.directors.values() else "cast")
            return "No he encontrado esa película en el dataset 🤔. Prueba a escribir el título original en inglés, por ejemplo *háblame de \"The Godfather\"*."
        m = self.movies.loc[idx]
        pred = self.predict_row(idx)
        lines = [f"🎬 **{m['title']}** ({int(m['year']) if pd.notna(m['year']) else '¿?'})"]
        if pd.notna(m["genres"]):
            lines.append(f"- **Géneros:** {', '.join(GENRE_ES.get(g, g) for g in m['genre_list'])}")
        if pd.notna(m["directors"]):
            lines.append(f"- **Dirección:** {m['directors'].replace('|', ', ')}")
        if pd.notna(m["cast"]):
            lines.append(f"- **Reparto:** {m['cast'].replace('|', ', ')}")
        if pd.notna(m["runtime"]):
            lines.append(f"- **Duración:** {int(m['runtime'])} min")
        if pd.notna(m["imdb_rating"]):
            lines.append(f"- **Nota IMDb:** ⭐ {m['imdb_rating']:.1f} ({fmt_int(m['imdb_votes'] if pd.notna(m['imdb_votes']) else 0)} votos)")
        if pd.notna(m["metacritic"]):
            lines.append(f"- **Metacritic:** {int(m['metacritic'])}")
        lines.append(f"- **Premios:** {m['wins']} ganados, {m['nominations']} nominaciones")
        if m["num_comments"]:
            lines.append(f"- **Comentarios en mflix:** {m['num_comments']}")
        if pd.notna(m["plot"]):
            lines.append(f"\n> {m['plot']}")
        lines.append(f"\n🌲 El Random Forest la clasifica como **{CATEGORY_EMOJI[pred['category']]} {pred['category']}** ({pred['confidence']:.0%} de los votos).")
        val = lambda x: None if pd.isna(x) else x  # noqa: E731
        detail = {
            "title": m["title"],
            "year": int(m["year"]) if pd.notna(m["year"]) else None,
            "genres": [GENRE_ES.get(g, g) for g in m["genre_list"] if g],
            "directors": val(m["directors"]).split("|") if val(m["directors"]) else [],
            "cast": val(m["cast"]).split("|") if val(m["cast"]) else [],
            "runtime": int(m["runtime"]) if pd.notna(m["runtime"]) else None,
            "rating": float(m["imdb_rating"]) if pd.notna(m["imdb_rating"]) else None,
            "votes": int(m["imdb_votes"]) if pd.notna(m["imdb_votes"]) else None,
            "metacritic": int(m["metacritic"]) if pd.notna(m["metacritic"]) else None,
            "wins": int(m["wins"]), "nominations": int(m["nominations"]),
            "comments": int(m["num_comments"]),
            "rated": val(m["rated"]),
            "countries": val(m["countries"]).split("|") if val(m["countries"]) else [],
            "plot": val(m["plot"]),
        }
        return {"answer": "\n".join(lines), "poster": m["poster"] if pd.notna(m["poster"]) else None,
                "prediction": pred, "detail": detail}

    def poster_wall(self, n: int = 36) -> list[dict]:
        top = self.movies[self.movies["poster"].notna() & (self.movies["type"] != "series")]
        top = top.sort_values("imdb_votes", ascending=False).head(n)
        return [{"title": t, "poster": p} for t, p in zip(top["title"], top["poster"])]

    def _h_predecir(self, q: str) -> dict | str:
        idx = self.find_title(q)
        if idx is not None and not self.find_genre(q):
            m = self.movies.loc[idx]
            pred = self.predict_row(idx)
            real = m["imdb_rating"]
            real_txt = ""
            if pd.notna(real):
                from features import rating_to_category
                real_cat = rating_to_category(real)
                ok = "✅ ¡acierta!" if real_cat == pred["category"] else "❌ falla"
                real_txt = f"\n\nNota real en IMDb: ⭐ {real:.1f} → **{real_cat}**. El bosque {ok}"
            return {
                "answer": f"🌲 **Predicción del Random Forest para {m['title']}**: {CATEGORY_EMOJI[pred['category']]} **{pred['category']}** "
                          f"({pred['votes'][pred['category']]} de {pred['n_trees']} árboles votan así).\n"
                          f"🌳 El Árbol de Decisión individual dice: **{pred['tree_category']}**.{real_txt}",
                "prediction": pred,
                "poster": m["poster"] if pd.notna(m["poster"]) else None,
            }
        return self._predict_custom(q)

    def _predict_custom(self, q: str) -> dict:
        nq = normalize(q)
        med = self.meta["medians"]
        row = {name: 0.0 for name in FEATURE_NAMES}
        for k in ("year", "runtime", "log_votes", "wins", "nominations", "n_languages", "n_countries", "num_comments"):
            row[k] = float(med[k])
        row.update({"is_usa": 1.0, "is_english": 1.0, "n_genres": 1.0, "rated_SIN_CLASIFICAR": 1.0})
        used = []
        genres = [g for alias, g in GENRE_ALIASES.items() if f" {alias} " in f" {nq} "]
        genres = list(dict.fromkeys(g for g in genres if g in GENRES))
        if genres:
            for g in genres:
                row[f"genre_{g}"] = 1.0
            row["n_genres"] = float(len(genres))
            used.append("géneros: " + ", ".join(GENRE_ES[g] for g in genres))
        years = self.find_years(q)
        if years:
            row["year"] = float((years[0] + years[1]) // 2)
            used.append(f"año: {int(row['year'])}")
        m = re.search(r"(\d{2,3})\s*(?:min|minutos)", nq)
        if m:
            row["runtime"] = float(m.group(1))
            used.append(f"duración: {m.group(1)} min")
        m = re.search(r"(\d+)\s*(?:premios|oscars?|galardones)", nq)
        if m:
            row["wins"] = float(m.group(1))
            used.append(f"premios: {m.group(1)}")
        m = re.search(r"(\d+)\s*nominacion", nq)
        if m:
            row["nominations"] = float(m.group(1))
            used.append(f"nominaciones: {m.group(1)}")
        m = re.search(r"(\d[\d\.]*)\s*votos", nq)
        if m:
            row["log_votes"] = float(np.log1p(float(m.group(1).replace(".", ""))))
            used.append(f"votos: {m.group(1)}")
        if not used:
            return {"answer": "Para predecir necesito algún dato. Dime un título (*predice la calidad de Alien*) "
                              "o describe la película: *predice una película de comedia de 2005 de 95 minutos con 2 premios*."}
        X = pd.DataFrame([row])[FEATURE_NAMES]
        pred = self._predict_X(X)
        return {
            "answer": f"🌲 Con estos datos ({'; '.join(used)}) el Random Forest predice: {CATEGORY_EMOJI[pred['category']]} **{pred['category']}** "
                      f"({pred['votes'][pred['category']]} de {pred['n_trees']} árboles).\n"
                      f"🌳 El Árbol de Decisión dice: **{pred['tree_category']}**.\n\n"
                      f"_Lo que no me has dicho lo relleno con la mediana del dataset._",
            "prediction": pred,
        }

    def predict_row(self, idx: int) -> dict:
        return self._predict_X(self.X_all.loc[[idx]])

    def _predict_X(self, X: pd.DataFrame) -> dict:
        classes = list(self.random_forest.classes_)
        votes = {c: 0 for c in self.meta["categories"]}
        values = X.to_numpy()
        for est in self.random_forest.estimators_:  # voto mayoritario explícito
            votes[classes[int(est.predict(values)[0])]] += 1
        proba = self.random_forest.predict_proba(X)[0]
        category = classes[int(np.argmax(proba))]
        return {
            "category": category,
            "confidence": float(np.max(proba)),
            "probabilities": {c: float(p) for c, p in zip(classes, proba)},
            "votes": votes,
            "n_trees": len(self.random_forest.estimators_),
            "tree_category": str(self.decision_tree.predict(X)[0]),
        }

    def _filter(self, q: str) -> tuple[pd.DataFrame, list[str]]:
        df, desc = self.movies, []
        genre = self.find_genre(q)
        if genre:
            df = df[df["genre_list"].apply(lambda gs: genre in gs)]
            desc.append(f"de {GENRE_ES.get(genre, genre)}")
        years = self.find_years(q)
        if years:
            df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
            desc.append(f"de {years[0]}" if years[0] == years[1] else f"de {years[0]}-{years[1]}")
        country = self.find_country(q)
        if country:
            df = df[df["countries"].fillna("").str.split("|").apply(lambda cs: country in cs)]
            desc.insert(0, COUNTRY_ES.get(country, f"de {country}"))
        if re.search(r"\bseries?\b", normalize(q)):
            df = df[df["type"] == "series"]
            desc.insert(0, "·series·")
        else:
            df = df[df["type"] != "series"]
        return df, desc

    def _h_top_genero(self, q: str) -> str:
        df, desc = self._filter(q)
        nq = normalize(q)
        n = self.find_top_n(q, 1 if re.search(r"\b(la mejor|mejor pelicula|mejor peli|la peor|peor pelicula)\b", nq) else 5)
        worst = bool(re.search(r"\b(peor|peores)\b", nq))
        df = df.dropna(subset=["imdb_rating", "imdb_votes"])
        df = df[df["imdb_votes"] >= (300 if len(df) < 300 else 1000)]
        if df.empty:
            return "No encuentro películas con esos filtros 🤷."
        sorts = [
            (r"mas larga|mas largas|mas largo|dura mas", "runtime", False, "⏱️ Películas más largas", "runtime"),
            (r"mas corta|mas cortas|dura menos", "runtime", True, "⏱️ Películas más cortas", "runtime"),
            (r"mas votad|mas popular|mas famos|mas conocid", "imdb_votes", False, "🔥 Películas más populares", "imdb_votes"),
            (r"mas antigua|mas vieja", "year", True, "📜 Películas más antiguas", None),
            (r"mas reciente|mas nueva|ultimas", "year", False, "🆕 Películas más recientes", None),
            (r"mas premi|mas oscar|mas galardon", "wins", False, "🏅 Películas con más premios", "wins"),
        ]
        for pattern, col, asc, label, extra in sorts:
            if re.search(pattern, nq):
                df = df[df["imdb_votes"] >= 2000] if col in ("year", "runtime") and len(df) > 50 else df
                df = df.sort_values([col, "imdb_votes"], ascending=[asc, False]).head(n)
                unit = {"runtime": " min", "imdb_votes": " votos", "wins": " premios"}
                cards = self._cards(df, lambda r: f"{fmt_int(r[extra])}{unit[extra]}" if extra else None)
                desc = [d for d in desc if d != "·series·"]
                head = f"{label} {' '.join(desc)}".strip()
                return {"answer": f"**{head}**", "movies": cards}
        # Nota ponderada (estilo IMDb Top 250): evita que gane una película con 300 votos.
        C, m = self.movies["imdb_rating"].mean(), 10000
        v, R = df["imdb_votes"], df["imdb_rating"]
        df = df.assign(score=(v / (v + m)) * R + (m / (v + m)) * C)
        df = df.sort_values("score", ascending=worst).head(n)
        noun = "series" if "·series·" in desc else "películas"
        desc = [d for d in desc if d != "·series·"]
        title = f"{'👎 Peores' if worst else '🏆 Mejores'} {noun} {' '.join(desc)}".strip()
        return {"answer": f"**{title}**\n_Ordenadas por nota IMDb ponderada por número de votos._", "movies": self._cards(df)}

    def _cards(self, df: pd.DataFrame, extra=None) -> list[dict]:
        """Tarjetas de película para el frontend (póster, año, nota, géneros)."""
        cards = []
        for _, r in df.iterrows():
            cards.append({
                "title": r["title"],
                "year": int(r["year"]) if pd.notna(r["year"]) else None,
                "rating": round(float(r["imdb_rating"]), 1) if pd.notna(r["imdb_rating"]) else None,
                "poster": r["poster"] if pd.notna(r["poster"]) else None,
                "genres": [GENRE_ES.get(g, g) for g in r["genre_list"] if g][:2],
                "extra": extra(r) if extra else None,
            })
        return cards

    def _person_movies(self, person: str, column: str) -> dict:
        df = self.movies[self.movies[column].fillna("").str.split("|").apply(lambda xs: person in xs)]
        df = df.sort_values(["imdb_rating", "imdb_votes"], ascending=False)
        role = "dirigidas por" if column == "directors" else "con"
        avg = df["imdb_rating"].mean()
        return {
            "answer": f"🎬 **{len(df)} películas {role} {person}** en el dataset · nota media ⭐ {avg:.1f}\nLas mejor valoradas:",
            "movies": self._cards(df.head(10)),
        }

    def _h_director(self, q: str) -> str:
        person = self.find_person(q, self.directors)
        actor = self.find_person(q, self.actors)
        if actor and (not person or not re.search(r"dirig|director", normalize(q))
                      and self._count_person(actor, "cast") > self._count_person(person, "directors")):
            return self._person_movies(actor, "cast")
        sur = self.find_surname(q)
        if not person and sur:
            return self._person_movies(*sur)
        if not person and self.find_title(q) is not None:
            return self._h_info_pelicula(q)
        if not person:
            return "No reconozco a ese director en el dataset. Escribe nombre y apellido, p. ej. *películas de Steven Spielberg*."
        return self._person_movies(person, "directors")

    def _h_actor(self, q: str) -> str:
        person = self.find_person(q, self.actors) or self.find_person(q, self.directors)
        sur = self.find_surname(q)
        if not person and sur:
            return self._person_movies(*sur)
        if not person and self.find_title(q) is not None:
            return self._h_info_pelicula(q)
        if not person:
            return "No reconozco a ese actor o actriz. Escribe nombre y apellido, p. ej. *películas con Meryl Streep*."
        return self._person_movies(person, "cast")

    def _h_recomendar(self, q: str) -> str:
        idx = self.find_title(q)
        if idx is not None:
            # Proximidad del Random Forest: dos películas son parecidas si caen en las mismas hojas.
            if self._leaves is None:
                self._leaves = self.random_forest.apply(self.X_all).astype(np.int32)
            leaves = self._leaves
            same = (leaves == leaves[idx]).mean(axis=1)
            base = self.movies.loc[idx]
            gset = set(base["genre_list"])
            genre_sim = self.movies["genre_list"].apply(lambda gs: len(gset & set(gs)) / max(len(gset | set(gs)), 1))
            score = same * 0.6 + genre_sim.to_numpy() * 0.4
            cand = self.movies.assign(score=score)
            cand = cand[(cand.index != idx) & (cand["imdb_votes"] >= 1000) & (cand["title_norm"] != base["title_norm"])]
            cand = cand.sort_values(["score", "imdb_rating"], ascending=False).head(8)
            return {"answer": f"💡 Si te gustó **{base['title']}**, el bosque te recomienda:\n"
                              f"_Similitud = proximidad en las hojas del Random Forest + géneros en común._",
                    "movies": self._cards(cand, lambda r: f"{r['score']:.0%} similar")}
        df, desc = self._filter(q)
        df = df[(df["imdb_rating"] >= 7.3) & (df["imdb_votes"] >= 20000)]
        if len(df) < 5:
            df, desc = self._filter(q)
            df = df[(df["imdb_rating"] >= 7.0) & (df["imdb_votes"] >= 2000)]
        if df.empty:
            return "No tengo recomendaciones con esos filtros."
        picks = df.sample(min(6, len(df)), random_state=random.randint(0, 9999)).sort_values("imdb_rating", ascending=False)
        desc = [d for d in desc if d != "·series·"]
        head = f"Te recomiendo estas películas {' '.join(desc)}".strip()
        return {"answer": f"💡 **{head}**\n_Pídemelo otra vez y te saco otras distintas._",
                "movies": self._cards(picks)}

    def _h_estadisticas(self, q: str) -> str:
        nq = normalize(q)
        s = self.meta["stats"]
        if "usuario" in nq:
            return f"👥 Hay **{s['n_users']} usuarios** registrados y **{fmt_int(s['n_comments'])} comentarios** en mflix."
        if "comentario" in nq:
            return self._h_comentarios(q)
        if "serie" in nq:
            return f"📺 Hay **{s['n_series']} series** en el dataset."
        if re.search(r"que generos|distribucion|genero tiene mas|generos hay", nq):
            counts = self.movies["genre_list"].explode().value_counts().head(12)
            return "📊 **Películas por género:**\n" + "\n".join(f"- {GENRE_ES.get(g, g)}: {fmt_int(c)}" for g, c in counts.items() if g)
        if re.search(r"ano con mas", nq):
            y = self.movies["year"].value_counts().idxmax()
            return f"📅 El año con más películas es **{int(y)}** ({self.movies['year'].value_counts().max()} películas)."
        df, desc = self._filter(q)
        if desc:
            return (f"📊 Hay **{fmt_int(len(df))} películas** {' '.join(desc)}. Nota IMDb media: ⭐ {df['imdb_rating'].mean():.2f}, "
                    f"duración media: {df['runtime'].mean():.0f} min.")
        return (
            f"📊 **Resumen del dataset mflix**\n"
            f"- Películas: **{fmt_int(s['n_movies'])}** (de ellas {s['n_series']} series)\n"
            f"- Años: {int(self.movies['year'].min())} – {int(self.movies['year'].max())}\n"
            f"- Nota IMDb media: ⭐ {self.movies['imdb_rating'].mean():.2f}\n"
            f"- Duración media: {self.movies['runtime'].mean():.0f} min\n"
            f"- Comentarios: {fmt_int(s['n_comments'])} · Usuarios: {s['n_users']} · Cines: {fmt_int(s['n_theaters'])}"
        )

    def _h_comentarios(self, q: str) -> str:
        nq = normalize(q)
        idx = self.find_title(q)
        if idx is not None:
            m = self.movies.loc[idx]
            return f"💬 **{m['title']}** tiene **{m['num_comments']} comentarios** en mflix."
        if re.search(r"usuario|quien", nq):
            top = self.top_users.head(5)
            return "💬 **Usuarios que más comentan:**\n" + "\n".join(f"{i}. {r['name']} — {r['comments']} comentarios" for i, (_, r) in enumerate(top.iterrows(), 1))
        df = self.movies.sort_values("num_comments", ascending=False).head(8)
        return {"answer": "💬 **Películas más comentadas en mflix**",
                "movies": self._cards(df, lambda r: f"{r['num_comments']} comentarios")}

    def _h_cines(self, q: str) -> str:
        nq = f" {normalize(q)} "
        for name, code in sorted(US_STATES.items(), key=lambda kv: -len(kv[0])):
            if f" {name} " in nq and not (name == "new york" and "ciudad" in nq):
                df = self.cines[self.cines["state"] == code]
                top = df["city"].value_counts().head(5)
                return (f"🎦 Hay **{len(df)} cines** en el estado de {name.title()} ({code}). Ciudades con más cines:\n"
                        + "\n".join(f"- {c}: {n}" for c, n in top.items()))
        for key, city in sorted(self.cities.items(), key=lambda kv: -len(kv[0])):
            if len(key) > 3 and f" {key} " in nq:
                df = self.cines[self.cines["city"] == city]
                return f"🎦 Hay **{len(df)} cines** en {city}:\n" + "\n".join(f"- {r['street']} ({r['state']})" for _, r in df.head(8).iterrows())
        for key, st in self.states.items():
            if f" {key} " in nq:
                return f"🎦 Hay **{(self.cines['state'] == st).sum()} cines** en el estado {st}."
        if re.search(r"ciudad", nq):
            top = self.cines["city"].value_counts().head(5)
            return "🎦 Ciudades con más cines:\n" + "\n".join(f"- {c}: {n}" for c, n in top.items())
        top = self.cines["state"].value_counts().head(5)
        return (f"🎦 El dataset tiene **{fmt_int(len(self.cines))} cines** en EE. UU. Estados con más cines:\n"
                + "\n".join(f"- {s}: {c}" for s, c in top.items()))

    def _h_modelo(self, q: str) -> str:
        m = self.meta["metrics"]
        rf = self.random_forest
        imp = self.meta["feature_importances"][:8]
        return (
            "🌲 **Modelo de calidad (Random Forest)**\n"
            f"- Objetivo: clasificar la nota IMDb en **{' / '.join(self.meta['categories'])}**\n"
            f"- {rf.n_estimators} árboles · criterio **{rf.criterion}** · max_features **{rf.max_features}** · max_depth {rf.max_depth}\n"
            f"- Accuracy Random Forest: **{m['random_forest_accuracy']:.1%}** · OOB score: **{m['random_forest_oob_score']:.1%}**\n"
            f"- Accuracy Árbol de Decisión: **{m['decision_tree_accuracy']:.1%}** → el bosque reduce la varianza 🌳➡️🌲\n"
            f"- Entrenado con {fmt_int(m['train_size'])} películas, test con {fmt_int(m['test_size'])}\n\n"
            + "**Variables más importantes (feature importance):**\n"
            + "\n".join(f"{i}. `{n}` — {v:.1%}" for i, (n, v) in enumerate(imp, 1))
            + f"\n\n🧠 Clasificador de intenciones del chat: TF-IDF + Random Forest ({m['intent_accuracy']:.0%} en test, reforzado con reglas)."
        )

    def _h_teoria(self, q: str) -> str:
        nq = normalize(q)
        for keys, text in KNOWLEDGE:
            if any(k in nq for k in keys):
                return text
        return KNOWLEDGE[0][1]


US_STATES = {
    "california": "CA", "texas": "TX", "florida": "FL", "nueva york": "NY", "new york": "NY", "illinois": "IL",
    "pensilvania": "PA", "pennsylvania": "PA", "ohio": "OH", "georgia": "GA", "michigan": "MI", "arizona": "AZ",
    "washington": "WA", "massachusetts": "MA", "nueva jersey": "NJ", "new jersey": "NJ", "virginia": "VA",
    "colorado": "CO", "nevada": "NV", "oregon": "OR", "tennessee": "TN", "carolina del norte": "NC",
    "north carolina": "NC", "maryland": "MD", "minnesota": "MN", "missouri": "MO", "indiana": "IN", "utah": "UT",
}

KNOWLEDGE: list[tuple[tuple[str, ...], str]] = [
    (("random forest", "bosque"),
     "🌲 **Random Forest** es un ensamble de muchos árboles de decisión independientes. Cada árbol se entrena con una muestra *bootstrap* "
     "(con reemplazo) y en cada nodo solo ve un subconjunto aleatorio de variables (*feature bagging*). La predicción final es el **voto mayoritario** "
     "(clasificación) o la **media** (regresión). Así se reduce la varianza y el sobreajuste de un árbol solo."),
    (("arbol de decision", "arbol"),
     "🌳 Un **árbol de decisión** es un modelo supervisado que divide los datos con reglas *si X entonces Y*. Tiene **nodo raíz**, **nodos internos**, "
     "**ramas** y **hojas** (la predicción). Es interpretable, no necesita escalado y sirve para clasificación y regresión."),
    (("gini",),
     "📐 La **impureza de Gini** mide la probabilidad de clasificar mal un dato al azar según las proporciones del nodo. Gini = 0 es puro; 0.5 es la máxima "
     "mezcla en binario. Es el criterio por defecto de scikit-learn (CART) y no usa logaritmos, por eso es más rápido que la entropía."),
    (("entropia", "ganancia de informacion"),
     "📐 La **entropía** (Shannon) mide el desorden de un nodo: 0 = pureza total, 1 = caos (50/50). La **ganancia de información** es cuánto baja la "
     "entropía tras una pregunta; la usan ID3 y C4.5."),
    (("bagging y boosting", "boosting"),
     "⚖️ **Bagging** entrena modelos en paralelo sobre muestras bootstrap y promedia (reduce varianza, ej. Random Forest). **Boosting** entrena árboles en "
     "secuencia, cada uno corrigiendo los errores del anterior (reduce sesgo, ej. XGBoost, AdaBoost)."),
    (("feature bagging", "subespacio", "max_features"),
     "🎲 **Feature bagging**: en cada nodo el árbol solo puede elegir entre un subconjunto aleatorio de variables (`max_features='sqrt'`). Evita que una "
     "variable dominante haga a todos los árboles iguales y los descorrelaciona."),
    (("bagging", "bootstrap"),
     "🎒 **Bootstrapping / Bagging**: se crean N bolsas del mismo tamaño muestreando con reemplazo; cada árbol se entrena aislado en su bolsa y al final "
     "se agregan las salidas. Cada bolsa deja fuera ~36,8 % de los datos."),
    (("oob", "out of bag", "out-of-bag"),
     "🧪 **Out-Of-Bag (OOB)**: el ~36,8 % de datos que no entra en la bolsa de un árbol sirve como test interno, sin necesidad de validación cruzada. "
     "Este modelo usa `oob_score=True`."),
    (("voto mayoritario", "voto"),
     "🗳️ **Voto mayoritario**: cada árbol del bosque vota una clase y gana la más votada. Cuando pido una predicción te enseño cuántos árboles votan cada categoría."),
    (("n_estimators", "hiperparametro"),
     "⚙️ Hiperparámetros clave: **n_estimators** (nº de árboles, más = más estable), **max_features** ('sqrt' en clasificación), **max_depth** y "
     "**min_samples_leaf** (controlan el tamaño de los árboles)."),
    (("sobreajuste", "overfitting"),
     "📉 **Sobreajuste**: el modelo memoriza el ruido del entrenamiento y falla con datos nuevos. Un árbol solo tiende a ello; el Random Forest lo controla promediando."),
    (("feature selection", "seleccion de caracteristicas"),
     "🔎 **Feature selection**: métodos *Filter* (correlación, chi², ANOVA), *Wrapper* (RFE, forward/backward) y *Embedded* (Lasso, importancia del Random Forest)."),
    (("pca", "lda"),
     "📉 **PCA** proyecta los datos en componentes de máxima varianza sin mirar etiquetas (no supervisado). **LDA** busca ejes que maximicen la separación entre clases (supervisado)."),
    (("sistema experto", "sistemas expertos"),
     "🧠 Un **sistema experto** replica el juicio de un especialista con una **base de conocimiento** y un **motor de inferencia**. Este chatbot es uno: "
     "reglas + Random Forest que consultan la base de conocimiento de películas. Tipos: basados en reglas, en casos, lógica difusa, redes bayesianas y marcos."),
    (("render",),
     "☁️ **Render** es un PaaS: despliegue automático con cada push a GitHub, HTTPS gratis. Plan free: 512 MB de RAM y *cold start* de ~50 s tras inactividad. "
     "Aquí aloja la API FastAPI con el modelo."),
    (("streamlit",),
     "🖥️ **Streamlit** convierte scripts de Python en webs interactivas. Ideal para prototipos; para producción se usa una arquitectura desacoplada "
     "(frontend + API FastAPI), como este proyecto."),
]
