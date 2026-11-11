"""Netflix-style Streamlit front-end for the content-based recommender.

Run with::

    streamlit run app.py

The heavy lifting (TF-IDF vectorisation + cosine similarity) lives in
``recommender.py``; this module is only responsible for presentation. The
recommendation logic and the Precision@5 = 0.861 evaluation are untouched.
"""

from __future__ import annotations

import hashlib
import html

import numpy as np
import pandas as pd
import streamlit as st

from recommender import (
    DEFAULT_CSV_PATH,
    ContentRecommender,
    TitleNotFoundError,
    load_dataframe,
)

# Precision@5 measured by `python recommender.py` on 200 sampled titles.
# Pinned here so the UI can surface the model quality without re-running the
# (relatively slow) evaluation on every page load.
PRECISION_AT_5 = 0.861

# Number of cards rendered per row in the results grid.
CARDS_PER_ROW = 3

# Banner gradients for the cards; picked deterministically from the title.
BANNER_GRADIENTS = [
    ("#E50914", "#7A0007"),
    ("#8E2DE2", "#4A00E0"),
    ("#0F2027", "#2C5364"),
    ("#FF512F", "#DD2476"),
    ("#11998E", "#0B6E63"),
    ("#41295A", "#2F0743"),
    ("#0F0C29", "#302B63"),
    ("#B3212D", "#3A0409"),
]


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_dataframe(csv_path: str) -> pd.DataFrame:
    """Load (and cache) the cleaned dataset used for the metrics row."""
    return load_dataframe(csv_path)


@st.cache_resource(show_spinner=False)
def get_recommender(csv_path: str) -> ContentRecommender:
    """Build (and cache) the model: the TF-IDF matrix lives here.

    ``cache_resource`` keeps a single shared instance alive across reruns and
    sessions instead of refitting the vectoriser on every interaction.
    """
    return ContentRecommender(csv_path)


@st.cache_data(show_spinner=False)
def get_similarities(csv_path: str, title: str) -> np.ndarray:
    """Cache the similarity vector of ``title`` against every other title.

    This is the relevant slice (one row) of the full similarity matrix, so
    repeated requests for the same title are served instantly.
    """
    return get_recommender(csv_path).similarity_row(title)


# --------------------------------------------------------------------------- #
# Page config + global CSS
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* ---------- Global theme ---------- */
      .stApp {
        background:
          radial-gradient(1200px 620px at 50% -12%, #2a0a0d 0%, rgba(20,20,20,0) 62%),
          #141414;
        color: #ffffff;
      }
      html, body, [class*="css"] {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      }
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }
      header[data-testid="stHeader"] { background: transparent; }
      .block-container {
        padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px;
      }
      section[data-testid="stSidebar"] {
        background: #181818; border-right: 1px solid rgba(255,255,255,0.07);
      }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

      /* ---------- Hero ---------- */
      .hero {
        position: relative; overflow: hidden; border-radius: 18px;
        padding: 38px 34px; margin-bottom: 22px;
        border: 1px solid rgba(229,9,20,0.35);
        background: linear-gradient(135deg, #1a0203 0%, #3b0509 45%, #141414 100%);
      }
      .hero::after {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background: radial-gradient(620px 250px at 12% 18%, rgba(229,9,20,0.35), transparent 70%);
      }
      .hero-title {
        position: relative; margin: 0; line-height: 1; letter-spacing: 3px;
        font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 900; color: #E50914;
        text-shadow: 0 0 26px rgba(229,9,20,0.55);
      }
      .hero-tagline {
        position: relative; margin-top: 14px; color: #fff;
        font-size: clamp(1rem, 2.4vw, 1.5rem); font-weight: 600;
      }
      .hero-sub {
        position: relative; margin-top: 8px; color: #b3b3b3; font-size: 0.95rem;
      }

      /* ---------- Metrics ---------- */
      .metric-card {
        height: 100%; border-radius: 14px; padding: 16px 18px;
        background: linear-gradient(160deg, #232323 0%, #1a1a1a 100%);
        border: 1px solid rgba(255,255,255,0.07);
      }
      .metric-value { font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1.1; }
      .metric-label {
        margin-top: 4px; color: #b3b3b3; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 1px;
      }
      .metric-accent .metric-value { color: #E50914; }

      /* ---------- Selected title panel ---------- */
      .panel {
        border-radius: 14px; padding: 20px 22px; margin-bottom: 22px;
        background: linear-gradient(135deg, #1f1f1f 0%, #171717 100%);
        border: 1px solid rgba(229,9,20,0.25);
        border-left: 4px solid #E50914;
      }
      .panel-kicker {
        color: #E50914; font-size: 0.72rem; font-weight: 800;
        letter-spacing: 1.5px; text-transform: uppercase;
      }
      .panel-title { margin-top: 6px; font-size: 1.6rem; font-weight: 800; color: #fff; }
      .panel-desc {
        margin-top: 10px; color: #cfcfcf; font-size: 0.95rem; line-height: 1.55;
      }

      /* ---------- Badges & chips ---------- */
      .badge {
        display: inline-block; margin: 0 6px 6px 0; padding: 3px 10px;
        border-radius: 999px; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.4px; color: #eaeaea;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.18);
      }
      .badge-type {
        color: #ff6b6b; background: rgba(229,9,20,0.18);
        border-color: rgba(229,9,20,0.6);
      }
      .chip {
        display: inline-block; margin: 4px 6px 0 0; padding: 3px 9px;
        border-radius: 6px; font-size: 0.7rem; color: #d6d6d6;
        background: #2a2a2a; border: 1px solid rgba(255,255,255,0.08);
      }

      /* ---------- Result cards ---------- */
      .card {
        height: 100%; overflow: hidden; border-radius: 16px; background: #1f1f1f;
        border: 1px solid rgba(255,255,255,0.07); margin-bottom: 18px;
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
      }
      .card:hover {
        transform: translateY(-6px);
        border-color: rgba(229,9,20,0.55);
        box-shadow: 0 14px 34px rgba(229,9,20,0.40);
      }
      .card-banner {
        position: relative; min-height: 96px; padding: 18px 16px;
        display: flex; align-items: flex-end;
      }
      .card-banner::before {
        content: ""; position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.05) 30%, rgba(0,0,0,0.6) 100%);
      }
      .card-title {
        position: relative; font-size: 1.05rem; font-weight: 800; color: #fff;
        line-height: 1.25; text-shadow: 0 2px 8px rgba(0,0,0,0.65);
      }
      .card-body { padding: 14px 16px 16px; }
      .card-desc {
        margin: 10px 0 12px; color: #b8b8b8; font-size: 0.82rem; line-height: 1.45;
        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
        overflow: hidden;
      }
      .match {
        font-size: 0.78rem; font-weight: 700; color: #ff6b6b; letter-spacing: 0.4px;
      }
      .bar {
        margin-top: 6px; width: 100%; height: 7px; border-radius: 999px;
        background: #333333; overflow: hidden;
      }
      .bar-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #E50914, #ff4b52);
      }

      /* ---------- Empty state & footer ---------- */
      .empty-state {
        border-radius: 16px; padding: 46px 24px; text-align: center;
        background: #1a1a1a; border: 1px dashed rgba(255,255,255,0.14);
        color: #9a9a9a;
      }
      .empty-state .big { font-size: 1.15rem; color: #e5e5e5; font-weight: 700; }
      .footer {
        margin-top: 30px; padding: 24px 10px 8px; text-align: center;
        color: #8c8c8c; font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.08);
      }
      .footer b { color: #E50914; }

      /* ---------- Primary button (Recommend) ---------- */
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E50914, #b0060f);
        border: none; border-radius: 10px; font-weight: 800;
        letter-spacing: 0.5px; padding: 0.6rem 1rem;
        box-shadow: 0 6px 18px rgba(229,9,20,0.35);
      }
      .stButton > button[kind="primary"]:hover { filter: brightness(1.12); }

      /* ---------- Mobile ---------- */
      @media (max-width: 640px) {
        .hero { padding: 26px 20px; }
        .hero-title { letter-spacing: 2px; }
        .panel-title { font-size: 1.3rem; }
        .metric-value { font-size: 1.5rem; }
        .card-title { font-size: 1rem; }
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #
def _escape(value) -> str:
    """HTML-escape any value coming from the dataset (titles contain &, quotes)."""
    return html.escape(str(value)) if value is not None else ""


def _gradient_for(title: str) -> tuple[str, str]:
    """Pick a stable banner gradient for a title."""
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()
    return BANNER_GRADIENTS[int(digest, 16) % len(BANNER_GRADIENTS)]


def _badges_html(item: dict) -> str:
    """Type / rating / year badges shared by the panel and the cards."""
    badges = [f'<span class="badge badge-type">{_escape(item.get("type") or "Title")}</span>']
    if item.get("rating"):
        badges.append(f'<span class="badge">{_escape(item["rating"])}</span>')
    if item.get("year"):
        badges.append(f'<span class="badge">{_escape(item["year"])}</span>')
    return "".join(badges)


def _chips_html(genres: list[str], limit: int = 3) -> str:
    """Genre chips; extra genres are collapsed into a ``+N`` chip."""
    if not genres:
        return ""
    shown = genres[:limit]
    chips = "".join(f'<span class="chip">{_escape(g)}</span>' for g in shown)
    if len(genres) > limit:
        chips += f'<span class="chip">+{len(genres) - limit}</span>'
    return chips


def hero_html() -> str:
    return """
    <div class="hero">
      <div class="hero-title">NETFLIX</div>
      <div class="hero-tagline">Find your next watch</div>
      <div class="hero-sub">
        Content-based recommendations &middot; TF-IDF + cosine similarity
      </div>
    </div>
    """


def metric_card_html(value: str, label: str, accent: bool = False) -> str:
    cls = "metric-card metric-accent" if accent else "metric-card"
    return (
        f'<div class="{cls}">'
        f'<div class="metric-value">{_escape(value)}</div>'
        f'<div class="metric-label">{_escape(label)}</div>'
        f"</div>"
    )


def panel_html(item: dict) -> str:
    """Styled card describing the currently selected title."""
    return (
        f'<div class="panel">'
        f'<div class="panel-kicker">Your selection</div>'
        f'<div class="panel-title">{_escape(item["title"])}</div>'
        f'<div style="margin-top:10px">{_badges_html(item)}</div>'
        f'<div>{_chips_html(item.get("genres") or [])}</div>'
        f'<div class="panel-desc">{_escape(item.get("description") or "")}</div>'
        f"</div>"
    )


def card_html(item: dict) -> str:
    """One recommendation card: banner, badges, chips, description, match bar."""
    start, end = _gradient_for(item["title"])
    percent = max(0, min(100, round(float(item["similarity"]) * 100)))
    return (
        f'<div class="card">'
        f'<div class="card-banner" style="background:linear-gradient(135deg,{start},{end});">'
        f'<div class="card-title">{_escape(item["title"])}</div>'
        f"</div>"
        f'<div class="card-body">'
        f"<div>{_badges_html(item)}</div>"
        f"<div>{_chips_html(item.get('genres') or [])}</div>"
        f'<div class="card-desc">{_escape(item.get("description") or "")}</div>'
        f'<div class="match">{percent}% match</div>'
        f'<div class="bar"><div class="bar-fill" style="width:{percent}%;"></div></div>'
        f"</div>"
        f"</div>"
    )


def empty_state_html(message: str, hint: str = "") -> str:
    hint_html = f'<div style="margin-top:8px;font-size:0.88rem">{_escape(hint)}</div>' if hint else ""
    return (
        f'<div class="empty-state">'
        f'<div class="big">{_escape(message)}</div>{hint_html}'
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.35rem;font-weight:900;color:#E50914;'
        'letter-spacing:1px;">NETFLIX</div>'
        '<div style="color:#b3b3b3;font-size:0.82rem;margin-bottom:14px;">'
        "Recommender controls</div>",
        unsafe_allow_html=True,
    )

    try:
        _recommender = get_recommender(DEFAULT_CSV_PATH)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()

    all_titles = sorted(_recommender.titles)

    selected_title = st.selectbox(
        "Search a title",
        options=all_titles,
        index=0,
        help="Start typing to search the catalogue.",
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    type_filter = st.radio(
        "Filter by type",
        options=["All", "Movie", "TV Show"],
        index=0,
        horizontal=True,
    )

    st.write("")
    recommend_clicked = st.button("Recommend", type="primary", use_container_width=True)

    st.caption(
        "Similarity is computed with TF-IDF vectorisation of genres, director, "
        "country, type and rating."
    )

# The button only stores the request; the main area renders it so that changing
# a control afterwards does not silently drop the results.
if recommend_clicked:
    st.session_state["request"] = {
        "title": selected_title,
        "n": top_n,
        "type_filter": type_filter,
    }


# --------------------------------------------------------------------------- #
# Hero + metrics row
# --------------------------------------------------------------------------- #
st.markdown(hero_html(), unsafe_allow_html=True)

try:
    _df = get_dataframe(DEFAULT_CSV_PATH)
    _types = _df["type"].fillna("").astype(str)
    total_titles = len(_df)
    movie_count = int(_types.str.contains("Movie", regex=False).sum())
    tv_count = int(_types.str.contains("TV Show", regex=False).sum())
except (FileNotFoundError, ValueError):
    total_titles = movie_count = tv_count = 0

metric_columns = st.columns(4, gap="medium")
metric_columns[0].markdown(metric_card_html(f"{total_titles:,}", "Titles"), unsafe_allow_html=True)
metric_columns[1].markdown(metric_card_html(f"{movie_count:,}", "Movies"), unsafe_allow_html=True)
metric_columns[2].markdown(metric_card_html(f"{tv_count:,}", "TV Shows"), unsafe_allow_html=True)
metric_columns[3].markdown(
    metric_card_html(f"{PRECISION_AT_5:.3f}", "Model Precision@5", accent=True),
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Main results area
# --------------------------------------------------------------------------- #
request = st.session_state.get("request")

if not request:
    st.markdown(
        empty_state_html(
            "Pick a title from the sidebar and hit Recommend.",
            "We'll surface the most similar shows and movies from the catalogue.",
        ),
        unsafe_allow_html=True,
    )
else:
    title = request["title"]
    n = request["n"]
    active_filter = request["type_filter"]

    try:
        # Selected title panel.
        st.markdown(panel_html(_recommender.title_info(title)), unsafe_allow_html=True)

        with st.spinner(f"Finding titles similar to \u201c{title}\u201d\u2026"):
            # Cached similarity vector, then rank the top candidates. A slightly
            # larger pool is fetched so the type filter still yields ``n`` cards.
            pool = min(max(n * 5, n), 60)
            similarities = get_similarities(DEFAULT_CSV_PATH, title)
            results = _recommender.rank(similarities, pool)

        if active_filter != "All":
            results = [item for item in results if item["type"] == active_filter]

        results = results[:n]

        if not results:
            st.markdown(
                empty_state_html(
                    "No recommendations found.",
                    "Try switching the type filter to \u201cAll\u201d or choosing another title.",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="margin:6px 0 14px;color:#e5e5e5;font-size:1.15rem;'
                f'font-weight:800;">More like '
                f'<span style="color:#E50914">{_escape(title)}</span></div>',
                unsafe_allow_html=True,
            )

            # Responsive grid: Streamlit columns stack on narrow screens.
            for start in range(0, len(results), CARDS_PER_ROW):
                columns = st.columns(CARDS_PER_ROW, gap="medium")
                for column, item in zip(columns, results[start:start + CARDS_PER_ROW]):
                    column.markdown(card_html(item), unsafe_allow_html=True)

            st.toast(
                f"{len(results)} recommendation{'s' if len(results) != 1 else ''} ready",
                icon="\U0001F3AC",
            )

    except TitleNotFoundError:
        st.error(
            f"\u201c{title}\u201d is not in the dataset. "
            "Please pick a title from the sidebar search."
        )
    except Exception as error:  # pragma: no cover - defensive UI guard
        st.error(f"Something went wrong while computing recommendations: {error}")

st.markdown(
    '<div class="footer">'
    "Built by <b>Aniket Mishra</b> &nbsp;|&nbsp; ML Internship Task 1, Auspify Technologies"
    "</div>",
    unsafe_allow_html=True,
)
