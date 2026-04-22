"""
NextWatch — Movie Recommendation App
Cinematic UI · Hybrid recommendation engine · Streamlit
"""

import os
import pickle
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NextWatch", page_icon="🎬", layout="wide")

try:
    API_KEY: str = st.secrets["OMDB_API_KEY"]
except KeyError:
    st.error("⚠️ OMDB API Key missing from `.streamlit/secrets.toml`.")
    st.stop()

PLACEHOLDER = "https://via.placeholder.com/300x450/0d0f14/c9a84c?text=No+Poster"
COLS_PER_ROW = 5

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #08090a !important;
    font-family: 'DM Sans', sans-serif;
    color: #ede9e0;
}
[data-testid="stSidebar"] { background: #0d0f14 !important; border-right: 1px solid #1c1f26; }
[data-testid="stSidebar"] * { color: #ede9e0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDeployButton"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

/* ── Masthead ── */
.nw-masthead {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding: 2rem 0 0.25rem;
    border-bottom: 1px solid #1c1f26;
    margin-bottom: 2rem;
}
.nw-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: -0.02em;
    line-height: 1;
}
.nw-tagline {
    font-size: 0.8rem;
    color: #5a5a4e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}

/* ── Section header ── */
.nw-section {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.nw-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1c1f26;
}

/* ── Movie Card ── */
.nw-card {
    display: block;
    text-decoration: none !important;
    color: inherit !important;
    cursor: pointer;
}
.nw-card-inner {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: #0d0f14;
    transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.28s ease;
    will-change: transform;
}
.nw-card:hover .nw-card-inner {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.7), 0 0 0 1px #c9a84c44;
}
.nw-card-inner img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    object-position: top center;
    display: block;
    transition: opacity 0.3s ease;
    opacity: 0.88;
}
.nw-card:hover .nw-card-inner img { opacity: 1; }

/* Gold shimmer on hover */
.nw-card-inner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, #c9a84c18 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.28s ease;
    pointer-events: none;
}
.nw-card:hover .nw-card-inner::after { opacity: 1; }

/* Rating badge */
.nw-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(8,9,10,0.85);
    backdrop-filter: blur(8px);
    border: 1px solid #c9a84c66;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #c9a84c;
    letter-spacing: 0.04em;
}

/* Card title */
.nw-card-title {
    padding: 0.55rem 0.4rem 0.2rem;
    font-size: 0.78rem;
    font-weight: 400;
    color: #b0aa9a;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    transition: color 0.2s ease;
}
.nw-card:hover .nw-card-title { color: #ede9e0; }

/* Stagger-reveal animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.nw-grid-item {
    animation: fadeUp 0.45s ease both;
}

/* ── Detail panel ── */
.nw-detail {
    background: #0d0f14;
    border: 1px solid #1c1f26;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2.5rem;
    display: flex;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}
.nw-detail::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top left, #c9a84c0a 0%, transparent 60%);
    pointer-events: none;
}
.nw-detail-poster {
    flex-shrink: 0;
    width: 160px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.nw-detail-poster img {
    width: 100%;
    display: block;
    aspect-ratio: 2/3;
    object-fit: cover;
}
.nw-detail-body { flex: 1; min-width: 0; }
.nw-detail-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #ede9e0;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}
.nw-detail-meta {
    font-size: 0.75rem;
    color: #5a5a4e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nw-chip {
    background: #1c1f26;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #8a8478;
}
.nw-chip.gold { background: #c9a84c1a; color: #c9a84c; border: 1px solid #c9a84c33; }
.nw-detail-plot {
    font-size: 0.88rem;
    color: #a09a8c;
    line-height: 1.65;
    margin-bottom: 1rem;
}
.nw-detail-credits {
    font-size: 0.78rem;
    color: #5a5a4e;
    line-height: 1.8;
}
.nw-detail-credits strong { color: #8a8478; }

/* ── Trailer link ── */
.nw-trailer-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #c9a84c;
    color: #08090a !important;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.45rem 1.1rem;
    border-radius: 4px;
    text-decoration: none !important;
    transition: background 0.2s ease;
    margin-top: 0.75rem;
}
.nw-trailer-btn:hover { background: #e0be6a; }

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d0f14 !important;
    border: 1px solid #1c1f26 !important;
    border-radius: 6px !important;
    color: #ede9e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] label { color: #8a8478 !important; font-size: 0.78rem !important; }

.stButton > button {
    background: transparent !important;
    color: #c9a84c !important;
    border: 1px solid #c9a84c55 !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: #c9a84c14 !important;
    border-color: #c9a84c !important;
}

/* Sidebar number input */
[data-testid="stNumberInput"] input {
    background: #0d0f14 !important;
    color: #ede9e0 !important;
    border-color: #1c1f26 !important;
}
[data-testid="stNumberInput"] label { color: #8a8478 !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets() -> tuple[pd.DataFrame, np.ndarray]:
    """Load movie metadata and similarity matrix from disk."""
    movies: pd.DataFrame = pickle.load(open("data/moviess.pkl", "rb"))

    if Path("data/similarities.pkl").exists():
        similarity: np.ndarray = pickle.load(open("data/similarities.pkl", "rb"))
    elif Path("data/similarities_part1.pkl").exists():
        p1 = pickle.load(open("data/similarities_part1.pkl", "rb"))
        p2 = pickle.load(open("data/similarities_part2.pkl", "rb"))
        similarity = np.concatenate((p1, p2), axis=0) if not isinstance(p1, list) else p1 + p2
    else:
        st.error("Similarity data not found in `data/` directory.")
        st.stop()

    return movies, similarity


try:
    movies, similarity = load_assets()
except FileNotFoundError:
    st.error("Data files not found in `data/` directory.")
    st.stop()


# ── OMDB helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(title: str) -> dict | None:
    """Fetch full OMDB metadata for a single title. Cached for 1 hour."""
    try:
        resp = requests.get(
            "https://www.omdbapi.com/",
            params={"t": title, "apikey": API_KEY},
            timeout=5,
        ).json()
        return resp if resp.get("Response") == "True" else None
    except Exception:
        return None


def safe_poster(details: dict | None) -> str:
    """Return a valid poster URL, falling back to placeholder."""
    poster = (details or {}).get("Poster", "")
    return poster if poster not in (None, "N/A", "") else PLACEHOLDER


# ── Recommendation engine ───────────────────────────────────────────────────────
def recommend(movie_name: str, n: int) -> list[dict]:
    """
    Hybrid recommender: blends cosine similarity (60 %) with
    normalised TMDB vote_average (40 %) over the top-40 similar candidates.
    Returns a list of dicts with 'title', 'score', 'rating'.
    """
    idx: int = movies[movies["title"] == movie_name].index[0]
    sims: np.ndarray = np.array(similarity[idx])

    # Candidate pool: top 40 by cosine sim (excluding self)
    top_idxs = np.argsort(sims)[::-1][1:41]

    # Normalise similarity scores within the pool
    pool_sims = sims[top_idxs]
    sim_min, sim_max = pool_sims.min(), pool_sims.max()
    norm_sims = (pool_sims - sim_min) / (sim_max - sim_min + 1e-9)

    # Normalise ratings within the pool
    ratings = np.array(
        [movies.iloc[i].get("vote_average", 0) for i in top_idxs], dtype=float
    )
    r_min, r_max = ratings.min(), ratings.max()
    norm_ratings = (ratings - r_min) / (r_max - r_min + 1e-9)

    hybrid_scores = 0.60 * norm_sims + 0.40 * norm_ratings

    # Sort by hybrid score and return top-n
    ranked = np.argsort(hybrid_scores)[::-1][:n]
    results: list[dict] = []
    for rank_i in ranked:
        movie_idx = top_idxs[rank_i]
        row = movies.iloc[movie_idx]
        results.append({
            "title": row["title"],
            "rating": round(float(row.get("vote_average", 0)), 1),
            "score": round(float(hybrid_scores[rank_i]), 3),
        })
    return results


# ── UI helpers ──────────────────────────────────────────────────────────────────
def _card_html(title: str, poster_url: str, rating: float = 0.0, delay_ms: int = 0) -> str:
    link = f"?movie={urllib.parse.quote(title)}"
    badge = f'<div class="nw-badge">★ {rating}</div>' if rating else ""
    return f"""
    <div class="nw-grid-item" style="animation-delay:{delay_ms}ms">
      <a href="{link}" target="_self" class="nw-card">
        <div class="nw-card-inner">
          <img src="{poster_url}" alt="{title}"
               onerror="this.src='{PLACEHOLDER}'">
          {badge}
        </div>
        <div class="nw-card-title">{title}</div>
      </a>
    </div>"""


def render_grid(items: list[dict]) -> None:
    """Render a 5-column responsive grid from a list of {title, poster, rating} dicts."""
    for row_start in range(0, len(items), COLS_PER_ROW):
        row = items[row_start : row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW, gap="small")
        for col_i, (col, item) in enumerate(zip(cols, row)):
            with col:
                st.markdown(
                    _card_html(
                        item["title"],
                        item["poster"],
                        item.get("rating", 0.0),
                        delay_ms=(row_start // COLS_PER_ROW * 60) + col_i * 40,
                    ),
                    unsafe_allow_html=True,
                )


# ── Session state defaults ──────────────────────────────────────────────────────
st.session_state.setdefault("trending_offset", 0)
st.session_state.setdefault("selected_movie_name", None)

# Sync URL → state
if "movie" in st.query_params:
    url_movie = st.query_params["movie"]
    if url_movie in movies["title"].values:
        st.session_state["selected_movie_name"] = url_movie
        st.session_state["movie_selectbox"] = url_movie

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Playfair Display,serif;font-size:1.1rem;
                color:#c9a84c;margin-bottom:1.5rem;padding-bottom:0.75rem;
                border-bottom:1px solid #1c1f26'>
        Settings
    </div>""", unsafe_allow_html=True)

    num_rec: int = st.number_input(
        "Recommendations", min_value=1, max_value=30, value=10
    )

    if st.session_state["selected_movie_name"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Trending", use_container_width=True):
            st.session_state["selected_movie_name"] = None
            st.query_params.clear()
            st.rerun()

# ── Masthead ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nw-masthead">
  <span class="nw-logo">NextWatch</span>
  <span class="nw-tagline">Your next obsession, curated</span>
</div>
""", unsafe_allow_html=True)

# ── Search bar ───────────────────────────────────────────────────────────────────
_, col_search, _ = st.columns([1, 2, 1])
with col_search:
    movie_titles = list(movies["title"].values)
    current = st.session_state.get("selected_movie_name")
    default_idx = movie_titles.index(current) if current in movie_titles else None

    selected_movie: str | None = st.selectbox(
        "Search for a film:",
        movie_titles,
        index=default_idx,
        placeholder="Start typing a title...",
        key="movie_selectbox",
    )

# Keep state in sync with manual selectbox change
if selected_movie != st.session_state["selected_movie_name"]:
    st.session_state["selected_movie_name"] = selected_movie

# Reflect in URL
if selected_movie:
    st.query_params["movie"] = selected_movie
elif "movie" in st.query_params:
    del st.query_params["movie"]

st.markdown("<div style='margin-bottom:2rem'></div>", unsafe_allow_html=True)

# ── MAIN VIEWS ───────────────────────────────────────────────────────────────────
if selected_movie:
    # ── Detail panel ──────────────────────────────────────────────────────────
    with st.spinner(""):
        info = fetch_movie_details(selected_movie)

    if info:
        poster_url = safe_poster(info)
        rating = info.get("imdbRating", "—")
        trailer_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(selected_movie + ' official trailer')}"
        genres = [g.strip() for g in info.get("Genre", "").split(",")]

        st.markdown(f"""
        <div class="nw-detail">
            <div class="nw-detail-poster">
                <img src="{poster_url}" alt="{selected_movie}"
                     onerror="this.src='{PLACEHOLDER}'">
            </div>
            <div class="nw-detail-body">
                <div class="nw-detail-title">{selected_movie}</div>
                <div class="nw-detail-meta">
                    <span>{info.get("Year","")}</span>
                    <span>·</span>
                    <span>{info.get("Runtime","")}</span>
                    <span>·</span>
                    <span class="nw-chip gold">★ {rating}</span>
                    {"".join(f'<span class="nw-chip">{g}</span>' for g in genres)}
                </div>
                <div class="nw-detail-plot">{info.get("Plot","")}</div>
                <div class="nw-detail-credits">
                    <strong>Director</strong> {info.get("Director","—")}<br>
                    <strong>Cast</strong> {info.get("Actors","—")}
                </div>
                <a href="{trailer_url}" target="_blank" class="nw-trailer-btn">
                    ▶ Watch Trailer
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="nw-section">Because you watched this</div>',
        unsafe_allow_html=True,
    )

    recs = recommend(selected_movie, num_rec)
    grid_items = []
    for rec in recs:
        details = fetch_movie_details(rec["title"])
        grid_items.append({
            "title": rec["title"],
            "poster": safe_poster(details),
            "rating": rec["rating"],
        })

    render_grid(grid_items)

else:
    # ── Trending view ──────────────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([6, 1], gap="small")
    with hdr_col:
        st.markdown('<div class="nw-section">Trending Now</div>', unsafe_allow_html=True)
    with btn_col:
        if st.button("Refresh ↻", use_container_width=True):
            st.session_state["trending_offset"] += COLS_PER_ROW * 2
            st.rerun()

    # Pick 10 movies from a rotating slice of top-rated titles
    sorted_movies = (
        movies.sort_values("vote_average", ascending=False)
        if "vote_average" in movies.columns
        else movies
    )
    total = len(sorted_movies)
    offset = st.session_state["trending_offset"] % max(total - 10, 1)
    trending_titles: list[str] = sorted_movies.iloc[offset : offset + 10]["title"].tolist()

    grid_items = []
    for title in trending_titles:
        details = fetch_movie_details(title)
        rating = float(
            movies.loc[movies["title"] == title, "vote_average"].values[0]
            if "vote_average" in movies.columns and title in movies["title"].values
            else 0
        )
        grid_items.append({
            "title": title,
            "poster": safe_poster(details),
            "rating": round(rating, 1),
        })

    render_grid(grid_items)

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#2a2a24;font-size:0.72rem;
            letter-spacing:0.12em;text-transform:uppercase;
            margin-top:4rem;padding-top:1.5rem;border-top:1px solid #1c1f26'>
    NextWatch · Portfolio Project · 2026
</div>
""", unsafe_allow_html=True)
