"""
ReelReview - Film-Review-Analyse-Tool (neues Design)

Eine Streamlit-Website, die Film-Daten von The Movie Database (TMDB) sowie eine
Sentiment-Analyse von Film-Reviews mithilfe eines Hugging Face ML-Modells
bereitstellt.
"""

import os
import streamlit as st
import torch
import jinja2

# ---------------------------------------------------------------------------
# Pfade (nur im lokalen Dev-Kontext, nicht für Deployment)
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(REPO_DIR, "static")
TEMPLATE_DIR = os.path.join(REPO_DIR, "templates")

# ---------------------------------------------------------------------------
# Konfiguration / Secrets
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    TMDB_API_KEY = st.secrets.get("tmdb_api_key", "")
    if not TMDB_API_KEY:
        TMDB_API_KEY = st.secrets.get("general", {}).get("tmdb_api_key", "")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TMDB_MOVIE_URL = "https://www.themoviedb.org/movie/"

# Hugging Face Sentiment-Modell
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Maximale Zeichenlaenge eines Reviews vor der Tokenizer-Analyse.
# Das Modell hat eine Token-Grenze von 512; diese Zahl ist sicher darueber.
REVIEW_MAX_CHARS = 1600

# Maximale Anzahl an Reviews, die analysiert werden
MAX_REVIEWS = 10


# ---------------------------------------------------------------------------
# Sentiment-Modell (once, spaetester Cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    """Lade das Sentiment-Modell beim ersten Aufruf in den Arbeitsspeicher.

    Die Lade-Nachricht wird nur beim wirklich ersten Aufruf gezeigt. Da
    st.cache_resource den Funktionsrumpf nach dem ersten Aufruf nicht
    wiederholt, genuegt ein einmaliger Hinweis; zusatzlich pruefen wir
    st.session_state als Fallback gegen wiederholte Anzeige.
    """
    import torch
    if not st.session_state.get("sentiment_model_loaded", False):
        st.info("Sentiment-Modell wird geladen (erst einmalige Ladezeit, ~270 MB)...")
        st.session_state.sentiment_model_loaded = True

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Sentiment-Analyse
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str) -> dict:
    """Fuehrt eine Sentiment-Analyse auf einem einzelnen Text durch."""
    if not text.strip():
        return {"label": "UNKNOWN", "score": 0.5}
    try:
        tokenizer, model = load_sentiment_model()
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        label = model.config.id2label[pred]
        score = float(torch.softmax(logits, dim=-1)[0][pred].item())
        return {"label": label, "score": round(score, 4)}
    except Exception as e:
        st.warning(f"Analyse fehlgeschlagen: {e}")
        return {"label": "ERROR", "score": 0.0}


def batch_sentiment(texts: list) -> list:
    """Fuehrt eine batchweise Sentiment-Analyse fuer eine Liste von Texen durch."""
    if not texts:
        return []
    try:
        tokenizer, model = load_sentiment_model()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        results = []
        for i, pid in enumerate(pred_ids):
            label = model.config.id2label[int(pid.item())]
            score = float(probs[i][pid].item())
            results.append({"label": label, "score": round(score, 4)})
        return results
    except Exception as e:
        st.warning(f"Batch-Analyse fehlgeschlagen: {e}")
        return [{"label": "ERROR", "score": 0.0}] * len(texts)


def summarize_sentiment(reviews: list) -> dict:
    """Fuehrt eine Sentiment-Analyse auf einer Liste von Reviews durch und
    gibt ein Aggregat zurueck."""
    texts = [r.get("content", "") for r in reviews if (r.get("content") or "").strip()]
    if not texts:
        return {
            "positive": 0, "negative": 0, "neutral": 0, "unknown": 0, "error": 0,
            "total": 0, "avg_score": 0.0, "sentiment": "Keine Texte verfuegbar",
        }
    analysis = batch_sentiment(texts)
    positive = sum(1 for a in analysis if a["label"] == "POSITIVE")
    negative = sum(1 for a in analysis if a["label"] == "NEGATIVE")
    unknown = sum(1 for a in analysis if a["label"] == "UNKNOWN")
    error = sum(1 for a in analysis if a["label"] == "ERROR")
    total = len(analysis)
    avg_score = sum(a["score"] for a in analysis) / total if total else 0.0
    return {
        "positive": positive,
        "negative": negative,
        "unknown": unknown,
        "error": error,
        "total": total,
        "avg_score": round(avg_score, 4),
        "sentiment": f"{positive} positiv, {negative} negativ, {unknown} unklar, {error} fehlerhaft",
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_sentiment_to_10(score_01: float) -> str:
    """Map a 0..1 score onto a 1..10 scale, rounded to one decimal.

    Args:
        score_01: value between 0.0 and 1.0 (e.g. softmax confidence)

    Returns:
        string with one decimal place, using a comma as the decimal separator.
        Example: 0.0 -> "1,0", 0.5 -> "5,5", 1.0 -> "10,0".
    """
    v = 1.0 + score_01 * 9.0                 # 0 -> 1.0 , 1 -> 10.0
    return f"{v:.1f}".replace(".", ",")


def format_german_decimal(value: float) -> str:
    """Format a float with one decimal place, using a comma as the decimal separator."""
    return f"{value:.1f}".replace(".", ",")


# ---------------------------------------------------------------------------
# API-Helfer
# ---------------------------------------------------------------------------
def fetch_json(url: str, params: dict) -> dict:
    """Rueckgabe von JSON-Daten einer TMDB-HTTP-Anfrage."""
    import requests
    headers = {"accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_movies(query: str) -> list:
    """Suche nach Filmen ueber den TMDB-Search-Endpunkt."""
    try:
        data = fetch_json(f"{TMDB_BASE_URL}/search/movie", {
            "api_key": TMDB_API_KEY,
            "query": query,
            "include_adult": "false",
            "language": "en-US",
        })
        return data.get("results", [])
    except Exception as e:
        st.error(f"Suche fehlgeschlagen: {e}")
        return []


def get_movie_details(movie_id: int) -> dict:
    """Hole detaillierte Infos zu einem Film per TMDB-Movie-Endpunkt."""
    try:
        data = fetch_json(f"{TMDB_BASE_URL}/movie/{movie_id}", {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
        })
        return data
    except Exception as e:
        st.error(f"Details-Anfrage fehlgeschlagen: {e}")
        return {}


def get_movie_reviews(movie_id: int, page: int = 1) -> list:
    """Hole Top-Reviews ueber den TMDB-Reviews-Endpunkt."""
    try:
        data = fetch_json(f"{TMDB_BASE_URL}/movie/{movie_id}/reviews", {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page,
        })
        return data.get("results", [])
    except Exception as e:
        st.error(f"Reviews-Anfrage fehlgeschlagen: {e}")
        return []


def get_random_movies(count: int = 6) -> list:
    """Lade zufaellige Filme von TMDB Popular-Liste."""
    try:
        data = fetch_json(f"{TMDB_BASE_URL}/movie/popular", {
            "api_key": TMDB_API_KEY,
            "page": 1,
            "language": "en-US",
            "sort_by": "popularity.desc",
        })
        return data.get("results", [])[:count]
    except Exception as e:
        st.error(f"Zufalls-Filme konnten nicht geladen werden: {e}")
        return []


# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Lastet main.css aus static/ in das Streamlit-Dokument."""
    css_path = os.path.join(STATIC_DIR, "cinematic.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS-Datei nicht gefunden: {css_path}")


# ---------------------------------------------------------------------------
# Templates (Jinja2)
# ---------------------------------------------------------------------------
def _get_template_env() -> jinja2.Environment:
    """Erzeugt eine Jinja2-Umgebung mit den HTML-Templates."""
    loader = jinja2.FileSystemLoader(TEMPLATE_DIR)
    return jinja2.Environment(loader=loader, autoescape=True)


def render_movie_overview(movie: dict, path_status: str = None) -> str:
    """Rendert movie_overview.html und gibt das HTML zurueck."""
    template_env = _get_template_env()
    template = template_env.get_template("movie_overview.html")

    poster = movie.get("poster_path")
    poster_url = f"{TMDB_IMAGE_BASE}{poster}" if poster else ""

    # Sentiment-Werte
    reviews = get_movie_reviews(movie.get("id"), page=1)
    top_reviews = reviews[:MAX_REVIEWS]
    reviews_with_text = [r for r in top_reviews if (r.get("content") or "").strip()]

    if reviews_with_text:
        analysis = batch_sentiment([r.get("content", "") for r in reviews_with_text])
        avg_score = sum(a["score"] for a in analysis) / len(analysis)
        pos = sum(1 for a in analysis if a["label"] == "POSITIVE")
        neg = sum(1 for a in analysis if a["label"] == "NEGATIVE")
        neutral = sum(1 for a in analysis if a["label"] == "NEUTRAL")
        unknown = sum(1 for a in analysis if a["label"] == "UNKNOWN")
        err = sum(1 for a in analysis if a["label"] == "ERROR")
    else:
        avg_score = 0.0
        pos = neg = neutral = unknown = err = 0

    # Normalisieren auf 1..10, 1 Dezimalstelle (deutsche Schreibweise)
    sentiment_score_10 = normalize_sentiment_to_10(1.0 + avg_score * 9.0)

    critic_10 = format_german_decimal(movie.get("vote_average", 0.0)) if movie.get("vote_average") else "—"

    context = {
        "title": movie.get("title", "Unbekannt"),
        "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "—",
        "vote_average": critic_10,
        "vote_count": movie.get("vote_count", 0),
        "overview": movie.get("overview", "Keine Beschreibung verfügbar."),
        "genres": [g.get("name") for g in movie.get("genres", [])] or ["—"],
        "poster_url": poster_url,
        "tmdb_url": f"{TMDB_MOVIE_URL}{movie.get('id')}",
        "score_1_label": "Sentiment-Score",
        "score_1": sentiment_score_10,
        "path_status": path_status,
        "total_reviews": len(reviews_with_text),
        "positive": pos,
        "negative": neg,
        "neutral": neutral,
        "unknown": unknown,
        "error": err,
    }
    return template.render(**context)


# ---------------------------------------------------------------------------
# UI-Komponenten
# ---------------------------------------------------------------------------
def render_logo():
    """Bildet das Logo als geschwungenen Text mit modernem Design."""
    st.markdown(
        '<div style="text-align:center; margin-bottom:8px;">'
        '<span style="'
        'background:linear-gradient(90deg, #FF6B6B, #F5A623, #FF6B6B); '
        'color:#fff; '
        'font-size:30px; font-weight:bold; '
        'padding:10px 30px; '
        'border-radius:16px; '
        'box-shadow:0 6px 16px rgba(245,166,35,0.35), 0 2px 6px rgba(0,0,0,0.15); '
        'letter-spacing:0.4px; '
        'font-family: "Inter", "Helvetica Neue", Arial, sans-serif;'
        '>'
        'ReelReview'
        '</span>'
        '</div>'
        '<div style="font-size:13px; color:#888; margin-top:8px; '
        'font-family: "Inter", "Helvetica Neue", Arial, sans-serif;">'
        'Film-Meinungen & Sentiment-Analyse'
        '</div>',
        unsafe_allow_html=True
    )


def search_movie():
    """Haupt-Suchinteraktion im Layout."""
    query = st.text_input(
        "Film suchen",
        placeholder="z.B. Inception, The Matrix, Interstellar...",
        label_visibility="collapsed",
        key="film_suche",
    )
    if query:
        with st.spinner("Film wird gesucht..."):
            results = search_movies(query.strip())
        if results:
            # Section heading for results (styled via CSS, no inline HTML)
            st.markdown(
                '<div class="section-title"><span class="eyebrow">Ergebnisse</span></div>',
                unsafe_allow_html=True
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                for i, m in enumerate(results[:5]):
                    movie_id = m.get("id")
                    title = m.get("title", "Unbekannt")
                    year = m.get("release_date", "Unbekannt")[:4]
                    overview = m.get("overview", "")
                    st.subheader(f"{title} ({year})")
                    if overview:
                        st.caption(overview[:220] + "..." if len(overview) > 220 else overview)
                    st.page_link(
                        f"{TMDB_MOVIE_URL}{movie_id}",
                        label="→ TMDB-Seite des Films anzeigen",
                        width=260
                    )
                    st.divider()
            with col2:
                st.markdown(
                    '<div class="section-action-label">Auswahl:</div>',
                    unsafe_allow_html=True
                )
                selected = st.selectbox(
                    "Film auswaehlen",
                    results,
                    format_func=lambda x: f"{x.get('title')} ({x.get('release_date','')[:4]})",
                )
                if st.button("Analyse anzeigen", key="analyze_btn"):
                    return selected
        else:
            st.info("Keine Treffer gefunden. Bitte geben Sie einen anderen Film-Namen ein.")
    return None


def display_movie_overview(movie: dict):
    """Zeigt eine detaillierte Film-Übersicht inkl. Reviews und Sentiment.

    Layout: links das Cover, oben rechts die beiden Scores, darunter die
    Beschreibung mit weiteren Infos.

    Beide HTML-Blöcke (Film-Detail-Karte und Reviews) werden via Jinja2
    Template gerendert; keine rohen HTML-Strings mehr im Python-Code.
    """
    if movie is None or not movie.get("id"):
        return

    movie_id = movie.get("id")
    poster = movie.get("poster_path")
    poster_url = f"{TMDB_IMAGE_BASE}{poster}" if poster else ""

    reviews = get_movie_reviews(movie_id, page=1)
    top_reviews = reviews[:MAX_REVIEWS]
    reviews_with_text = [r for r in top_reviews if (r.get("content") or "").strip()]

    # Sentiment-Analyse
    if reviews_with_text:
        analysis = batch_sentiment([r.get("content", "") for r in reviews_with_text])
        avg_score = sum(a["score"] for a in analysis) / len(analysis)
        pos = sum(1 for a in analysis if a["label"] == "POSITIVE")
        neg = sum(1 for a in analysis if a["label"] == "NEGATIVE")
        unknown = sum(1 for a in analysis if a["label"] == "UNKNOWN")
        error = sum(1 for a in analysis if a["label"] == "ERROR")
        neutral = sum(1 for a in analysis if a["label"] == "NEUTRAL")
    else:
        analysis = []
        avg_score = 0.0
        pos = neg = unknown = error = neutral = 0

    # Normalisiere Sentiment auf 1..10
    sentiment_score_10 = normalize_sentiment_to_10(1.0 + avg_score * 9.0)

    # --- 1. Film-Detail-Karte (Jinja2) ---
    # Das Template bindet score_1 (Sentiment-Score) und score_2 (Kritiker-Bewertung)
    # sowie den Beschreibungsblock und die Metadaten.
    html = render_movie_overview(movie, path_status=None)
    st.markdown(html, unsafe_allow_html=True)

    # --- 2. Reviews / Sentiment-Analyse (Jinja2) ---
    # Nur die Reviews mit echtem Textinhalt werden analysiert.
    if reviews_with_text:
        # Berechne die Einzel-Analyse für jede Review
        details = []
        for r in reviews_with_text:
            content = r.get("content", "")
            if len(content) > REVIEW_MAX_CHARS:
                content = content[:REVIEW_MAX_CHARS].rsplit(" ", 1)[0] + "..."
            res = analyze_sentiment(content)
            details.append({
                "author": r.get("author", "Unknown"),
                "content": content,
                "label": res["label"],
                "score": res["score"],
            })

        # Aggregat (Gleich wie zuvor: Positiv/Negativ/etc.)
        sentiment_summary = summarize_sentiment(reviews_with_text)

        # Template-Daten sammeln
        review_ctx = {
            "total_reviews": len(reviews_with_text),
            "positive": sentiment_summary["positive"],
            "negative": sentiment_summary["negative"],
            "neutral": sentiment_summary["neutral"],
            "unknown": sentiment_summary["unknown"],
            "error": sentiment_summary["error"],
            "details": details,
        }

        # Rendert den Reviews-Block als eigenes Template
        t = _get_template_env().get_template("reviews_section.html")
        st.markdown(t.render(**review_ctx), unsafe_allow_html=True)
    else:
        st.info("Keine Reviews mit Text fuer diesen Film gefunden "
                "(Der Film hat sehr wenige Reviews oder diese enthalten keinen "
                "auszuwertenden Text).")


def display_home():
    """Erster Seiten-Start: Logo, Suchleiste + zufällige Empfehlungen."""
    render_logo()
    st.title("Film-Übersicht & Meinungen")
    st.write("Suche nach einem Film, um die Meinungen der Top-Reviews zu analysieren.")

    st.markdown("---")
    selected = search_movie()

    if selected:
        st.markdown("---")
        with st.spinner("Film wird analysiert..."):
            display_movie_overview(selected)
    else:
        # Startseite mit Empfehlungen
        # Section heading for recommendations (styled via CSS, no inline HTML)
        st.markdown(
            '<div class="section-title"><span class="eyebrow">Empfohlene Filme</span></div>',
            unsafe_allow_html=True
        )
        with st.spinner("Lade zufällige Filme..."):
            movies = get_random_movies(6)

        if movies:
            cols = st.columns(3)
            for idx, m in enumerate(movies):
                with cols[idx % 3]:
                    poster = m.get("poster_path")
                    title = m.get("title", "Unbekannt")
                    year = m.get("release_date", "")[:4]
                    rating = m.get("vote_average", 0)

                    st.subheader(f"{title} ({year})")
                    st.caption(f"{year} | {rating:.1f}/10")

                    if poster:
                        try:
                            st.image(f"{TMDB_IMAGE_BASE}{poster}", width=180)
                        except Exception:
                            pass

                    st.page_link(
                        f"{TMDB_MOVIE_URL}{m.get('id')}",
                        label="Details anzeigen",
                        width=180
                    )
        else:
            st.info("Keine Empfehlungen verfügbar.")


def main():
    """Haupt-Anwendungslogik."""
    st.set_page_config(
        page_title="ReelReview - Film-Meinungen",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # CSS einbinden
    inject_css()

    # Startseite
    display_home()


if __name__ == "__main__":
    main()
