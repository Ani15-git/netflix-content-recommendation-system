"""Content-based recommender system for the Netflix titles dataset.

The recommender merges the ``listed_in`` (genres), ``director``, ``country``,
``type`` and ``rating`` columns into a single text feature per title, converts
that feature into TF-IDF vectors and ranks other titles by cosine similarity.

Public API
----------
recommend(title, n=5)
    Return the ``n`` most similar titles to ``title``. Every result is a dict
    with the title's ``type``, ``genre``, ``year``, ``rating``, ``description``
    and the ``similarity`` (match score) against the input title.
evaluate(...)
    Measure how often recommendations share at least one genre (Precision@n).
ContentRecommender
    Reusable object when you want to control the dataset path or inspect state.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "dataset", "neteflix_titles.csv")

# Columns combined into the single text feature used by TF-IDF.
FEATURE_COLUMNS = ["listed_in", "director", "country", "type", "rating"]

# Keep hyphenated tokens such as "TV-MA" and "PG-13" intact and allow
# single-character ratings such as "R".
TOKEN_PATTERN = r"(?u)\b\w[\w-]*\b"


class TitleNotFoundError(ValueError):
    """Raised when the requested title is not present in the dataset."""


def _split_values(value) -> list[str]:
    """Split a comma separated cell into trimmed, non-empty parts."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _collect_unique(values: Iterable) -> list[str]:
    """Collect the unique comma separated parts across a group of cells."""
    seen: list[str] = []
    for value in values:
        for part in _split_values(value):
            if part not in seen:
                seen.append(part)
    return seen


def _join_values(values: Iterable) -> str:
    """Join the unique parts across a group with spaces (for the text feature)."""
    return " ".join(_collect_unique(values))


def _join_genres(values: Iterable) -> str:
    """Join the unique genres across a group with commas (for display/eval)."""
    return ", ".join(_collect_unique(values))


def parse_genres(value) -> set[str]:
    """Return the set of genres contained in a ``listed_in`` cell."""
    return set(_split_values(value))


def load_dataframe(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the dataset and collapse duplicate titles into one row each.

    Rows that share the same title (case insensitive, ignoring surrounding
    whitespace) are merged: their genres/directors/countries are unioned so no
    information is lost, while display columns keep the first occurrence.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = set(FEATURE_COLUMNS + ["title"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]
    # Normalised lookup key: same title with different casing/spacing collapses.
    df["_key"] = df["title"].str.lower()

    grouped = df.groupby("_key", as_index=False, sort=False).agg(
        show_id=("show_id", "first"),
        title=("title", "first"),
        type=("type", _join_values),
        director=("director", _join_values),
        country=("country", _join_values),
        rating=("rating", _join_values),
        release_year=("release_year", "first"),
        duration=("duration", "first"),
        listed_in=("listed_in", _join_genres),
        # Comma separated variants used only when building the display
        # description (the space separated ``text`` feature above stays as is).
        director_display=("director", _join_genres),
        country_display=("country", _join_genres),
    )

    # Single combined text feature used for vectorisation.
    grouped["text"] = (
        grouped[FEATURE_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    )
    return grouped


def _format_description(row: pd.Series) -> str:
    """Compose a short, human-readable description from a title's metadata.

    The source dataset has no synopsis column, so the description is derived
    from the fields that *are* available (type, country, director, duration and
    genres) instead of inventing content that is not in the data.
    """
    segments: list[str] = []

    title_type = str(row.get("type", "") or "").strip() or "Title"
    country = str(row.get("country_display", "") or "").strip()
    director = str(row.get("director_display", "") or "").strip()
    duration = str(row.get("duration", "") or "").strip()

    # Sentence 1: "<type> from <country>."
    head = f"{title_type} from {country}" if country else title_type
    segments.append(f"{head}.")
    if director:
        segments.append(f"Directed by {director}.")
    if duration:
        segments.append(f"Runtime: {duration}.")

    genres = _collect_unique([row.get("listed_in")])
    if genres:
        segments.append("Genres: " + ", ".join(genres) + ".")

    return " ".join(segments)


class ContentRecommender:
    """TF-IDF + cosine similarity content-based recommender."""

    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.csv_path = csv_path
        self.df = load_dataframe(csv_path)
        self._fit()

    def _fit(self) -> None:
        self.vectorizer = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=True,
            norm="l2",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])
        # Map every known (normalised) title to its row position.
        self._title_to_index = {key: i for i, key in enumerate(self.df["_key"])}

    @property
    def titles(self) -> list[str]:
        """All distinct titles available for recommendation."""
        return self.df["title"].tolist()

    def _resolve_index(self, title: str) -> int:
        if not isinstance(title, str) or not title.strip():
            raise TitleNotFoundError("Title must be a non-empty string.")
        index = self._title_to_index.get(title.strip().lower())
        if index is None:
            raise TitleNotFoundError(
                f"'{title}' was not found in the dataset. "
                f"Use ContentRecommender.titles to browse available titles."
            )
        return index

    def similarity_row(self, title: str) -> np.ndarray:
        """Similarity of ``title`` against every title in the dataset.

        This is a single row of the full N x N cosine similarity matrix. It is
        computed on demand (query vector vs. the whole matrix) rather than
        materialising the full matrix, which keeps memory usage low for the
        ~8.8k titles in this dataset. The input title's own score is forced to
        ``-1`` so it can never recommend itself.
        """
        index = self._resolve_index(title)
        similarities = cosine_similarity(
            self.tfidf_matrix[index], self.tfidf_matrix
        ).ravel()
        similarities[index] = -1.0  # never recommend the title itself
        return similarities

    def _build_result(self, row: pd.Series, score: float) -> dict:
        """Shape one recommended row into the public result dictionary."""
        genres = _collect_unique([row["listed_in"]])
        year = row["release_year"]
        rating = row["rating"]
        return {
            "title": row["title"],
            "type": str(row["type"]).strip(),
            "genre": ", ".join(genres),
            "genres": genres,
            "year": int(year) if pd.notna(year) else None,
            "rating": str(rating).strip() if pd.notna(rating) else None,
            "description": _format_description(row),
            "similarity": round(float(score), 4),
            # ``listed_in`` carries the same value as ``genre`` and is kept so
            # the existing evaluation harness keeps working unchanged.
            "listed_in": row["listed_in"],
        }

    def rank(self, similarities: np.ndarray, n: int = 5) -> list[dict]:
        """Turn a similarity row into the top ``n`` shaped result dictionaries.

        Zero-similarity titles are omitted, so fewer than ``n`` results may be
        returned.
        """
        if n <= 0:
            return []

        ranking = np.argsort(-similarities, kind="stable")[:n]

        results: list[dict] = []
        for position in ranking:
            score = float(similarities[position])
            if score <= 0.0:
                break
            results.append(self._build_result(self.df.iloc[position], score))
        return results

    def recommend(self, title: str, n: int = 5) -> list[dict]:
        """Return up to ``n`` titles most similar to ``title``.

        Each result is a dict containing ``title``, ``type``, ``genre``,
        ``genres``, ``year``, ``rating``, ``description`` and ``similarity``.

        Raises
        ------
        TitleNotFoundError
            If ``title`` is empty or not present in the dataset.
        """
        if n <= 0:
            return []
        return self.rank(self.similarity_row(title), n)

    def title_info(self, title: str) -> dict:
        """Metadata for a single title, shaped like a recommendation entry.

        Used by the UI to render the "selected title" panel (no similarity).
        """
        index = self._resolve_index(title)
        info = self._build_result(self.df.iloc[index], 1.0)
        info.pop("similarity", None)
        return info


@lru_cache(maxsize=None)
def _get_recommender(csv_path: str) -> ContentRecommender:
    """Build (and cache) a recommender for the given dataset path."""
    return ContentRecommender(csv_path)


def recommend(title: str, n: int = 5, csv_path: str = DEFAULT_CSV_PATH) -> list[dict]:
    """Convenience wrapper around :meth:`ContentRecommender.recommend`."""
    return _get_recommender(csv_path).recommend(title, n=n)


def evaluate(
    recommender: ContentRecommender | None = None,
    n: int = 5,
    sample_size: int | None = 200,
    seed: int = 42,
) -> dict:
    """Evaluate recommendations by genre overlap.

    For each sampled input title, ``n`` recommendations are generated and each
    one is checked for sharing at least one genre with the input. Returns a
    summary dict with the average number of matching recommendations, the
    average overlap ratio and precision@n.
    """
    rec = recommender or _get_recommender(DEFAULT_CSV_PATH)
    titles = rec.titles
    total = len(titles)

    if sample_size is None or sample_size >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(total, size=sample_size, replace=False).tolist())

    evaluated = 0
    skipped = 0
    total_recs = 0
    total_hits = 0
    per_title: list[dict] = []

    for position in indices:
        input_title = rec.df.iloc[position]["title"]
        input_genres = parse_genres(rec.df.iloc[position]["listed_in"])
        if not input_genres:
            skipped += 1
            continue

        try:
            recommendations = rec.recommend(input_title, n=n)
        except TitleNotFoundError:
            skipped += 1
            continue

        if not recommendations:
            skipped += 1
            continue

        hits = sum(
            1
            for item in recommendations
            if input_genres & parse_genres(item["listed_in"])
        )
        evaluated += 1
        total_hits += hits
        total_recs += len(recommendations)
        per_title.append(
            {
                "title": input_title,
                "recommended": len(recommendations),
                "genre_matches": hits,
            }
        )

    summary = {
        "evaluated_titles": evaluated,
        "skipped_titles": skipped,
        "n": n,
        "avg_genre_matches": round(total_hits / evaluated, 3) if evaluated else 0.0,
        "avg_overlap_ratio": round(total_hits / total_recs, 3) if total_recs else 0.0,
        "precision_at_n": round(total_hits / total_recs, 3) if total_recs else 0.0,
        "per_title": per_title,
    }
    return summary


def _print_recommendations(title: str, results: list[dict]) -> None:
    print(f"\nBecause you watched: {title}")
    if not results:
        print("  No similar titles found.")
        return
    for rank, item in enumerate(results, start=1):
        print(
            f"  {rank}. {item['title']} "
            f"({item['year']}, {item['type']}, {item['rating']}) "
            f"score={item['similarity']} | {item['genre']}"
        )


if __name__ == "__main__":
    engine = ContentRecommender()

    sample = engine.titles[0]
    try:
        _print_recommendations(sample, engine.recommend(sample, n=5))
    except TitleNotFoundError as error:
        print(f"Error: {error}")

    # Demonstrate handling of an unknown title.
    try:
        engine.recommend("This Title Does Not Exist")
    except TitleNotFoundError as error:
        print(f"\nHandled missing title: {error}")

    print("\nRunning evaluation...")
    report = evaluate(engine, n=5, sample_size=200)
    print(
        f"  Evaluated {report['evaluated_titles']} titles "
        f"(skipped {report['skipped_titles']}, n={report['n']})"
    )
    print(f"  Average recommendations sharing a genre: {report['avg_genre_matches']}")
    print(f"  Average overlap ratio: {report['avg_overlap_ratio']}")
    print(f"  Precision@n: {report['precision_at_n']}")
