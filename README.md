# Netflix Content-Based Recommender

A content-based recommender for the Netflix titles dataset, wrapped in a
Netflix-style Streamlit web app. Each title is described by a single text
feature built from its genres, director, country, type and rating; that text is
vectorised with **TF-IDF** and other titles are ranked by **cosine similarity**.

The interface is a dark "Netflix" theme (background `#141414`, cards `#1f1f1f`,
accent red `#E50914`) with a hero header, sidebar controls, a metrics row and a
responsive grid of recommendation cards.

## Highlights

- **Precision@5 = 0.861** on 200 sampled titles (see [Results](#results)).
- `recommend(title, n=5)` returns for each match: title, type, genre, year,
  rating, description and similarity score.
- Streamlit UI with cached data loading and cached similarity lookups, so
  interactions stay snappy.
- Works on desktop and mobile widths.

## Dataset

`dataset/neteflix_titles.csv` (~8.8k rows) with the columns:

`show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in`

Rows that share the same title (case-insensitive, ignoring whitespace) are
merged into a single row so a title is never recommended twice.

> The dataset has no synopsis column, so the `description` shown in the app is
> derived from the available metadata (type, country, director, duration and
> genres) rather than inventing content.

## Approach

1. **Feature construction** – `listed_in`, `director`, `country`, `type` and
   `rating` are concatenated into one `text` field per title.
2. **Vectorisation** – `TfidfVectorizer` converts the text into TF-IDF vectors.
   Hyphenated tokens such as `TV-MA` and `PG-13` are preserved, and single
   character ratings such as `R` are kept.
3. **Similarity** – cosine similarity between the query title and all other
   titles. It is computed on demand (query vector vs. the whole matrix) instead
   of materialising the full 8.8k × 8.8k matrix, keeping memory usage low.
   Titles with zero similarity are skipped; the input title is never returned.
4. **Missing titles** – an unknown or empty title raises `TitleNotFoundError`.

## Project structure

```
Neteflix-Recommendor/
├── app.py                      # Streamlit UI (Netflix-style front-end)
├── recommender.py              # TF-IDF + cosine similarity engine and evaluation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Dark Netflix theme for Streamlit
└── dataset/
    └── neteflix_titles.csv     # Netflix titles dataset
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (by default **http://localhost:8501**) — open it in
your browser. In the sidebar, search for a title, choose how many
recommendations you want (3–10), optionally filter by type (Movie / TV Show /
All) and press **Recommend**.

## Run the model / evaluation only

```bash
python recommender.py
```

This prints a small demo, handles a deliberately missing title, and then runs
the Precision@5 evaluation.

## Programmatic usage

```python
from recommender import recommend

recommend("Ganglands", n=5)
```

Each result is a dict with `title`, `type`, `genre`, `genres`, `year`, `rating`,
`description` and `similarity`:

```python
[
    {
        "title": "Sentinelle",
        "type": "Movie",
        "genre": "Action & Adventure, Dramas, International Movies",
        "genres": ["Action & Adventure", "Dramas", "International Movies"],
        "year": 2021,
        "rating": "TV-MA",
        "description": "Movie from France. Directed by Julien Leclercq. "
                       "Runtime: 80 min. Genres: Action & Adventure, Dramas.",
        "similarity": 0.7563,
    },
    ...
]
```

Handling a title that is not in the dataset:

```python
from recommender import TitleNotFoundError, recommend

try:
    recommend("This Title Does Not Exist")
except TitleNotFoundError as error:
    print(error)  # 'This Title Does Not Exist' was not found in the dataset. ...
```

Lookups are case-insensitive and ignore surrounding whitespace, so
`recommend("ganglands")` and `recommend("Ganglands")` return the same results.

### Reusing the model

For repeated calls, use a single instance instead of the module-level wrapper:

```python
from recommender import ContentRecommender

engine = ContentRecommender()
engine.titles               # list of all distinct titles
engine.title_info("Lupin")  # metadata for a single title
engine.recommend("Lupin", n=10)
```

## Results

`evaluate()` checks how many recommended titles share at least one genre with the
input title (genres are read from `listed_in`).

```python
from recommender import evaluate

report = evaluate(n=5, sample_size=200)
print(report["precision_at_n"])     # share of recommendations sharing a genre
print(report["avg_genre_matches"])  # average matches per input title
```

Example output (`python recommender.py`, 200 sampled titles, n=5):

```
Evaluated 200 titles (skipped 0, n=5)
Average recommendations sharing a genre: 4.305
Average overlap ratio: 0.861
Precision@n: 0.861
```

**Precision@5 = 0.861** — on average ~4.3 of the 5 recommendations share at least
one genre with the input title. The same value is pinned as a metric in the app
so you can see the model quality at a glance.

`skipped_titles` counts sampled titles that had no genres to compare against or
that produced no recommendations.
