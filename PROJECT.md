# ReelReview – Projekt-Übersicht für Agents

## Kurzbeschreibung
Eine Streamlit-Website, die Film-Daten von **The Movie Database (TMDB)** anzeigt und
die **Top-Reviews** eines Films einer **Sentiment-Analyse** mit einem
**Hugging Face ML-Modell** unterzieht. Das Design ist ein **cinematisches
Dark-Mode-Design** mit minimalistischer UI und warmen Akzenten (Gold/Amber) sowie
editorialer Typografie (Playfair Display + Inter).

## Dateistruktur

```
Repo/
├── app.py                  # Haupt-Streamlit-Anwendung (keine CSS/HTML-Strings mehr)
├── requirements.txt        # Abhängigkeiten inkl. jinja2
├── .streamlit/
│   └── secrets.toml        # TMDB-API-Schlüssel (flat, top-level)
├── static/
│   └── cinematic.css       # Eigenes stylesheet: Säulen, Variablen, Komponenten
├── templates/
│   └── movie_overview.html # Jinja2-Vorlage: Layout, Scores, Beschreibung
├── verify.py               # Verifikationsskript (nicht für Deployment)
├── .agents/                # Agent-Infrastruktur (nicht für Deployment)
└── README.md               # Installationsanleitung
```

## Technologien
- **Streamlit** 1.66+ (UI-Framework)
- **requests** (HTTP-Client für TMDB-API)
- **transformers** (Hugging Face – Sentiment-Modell)
- **torch** (PyTorch – Modell-Backend)
- **pandas** (Datenverarbeitung)
- **jinja2** (HTML-Templates)

## Abhängigkeiten (`requirements.txt`)
```
streamlit>=1.40.0
requests>=2.31.0
pandas>=2.2.0
transformers>=4.40.0
torch>=2.3.0
jinja2>=3.1.0
```

## API-Endpunkte (TMDB)
- **Suche:** `GET /search/movie` – Query-Parameter: `api_key`, `query`, `include_adult`, `language`
- **Film-Details:** `GET /movie/{id}` – Parameter: `api_key`, `language`
- **Reviews:** `GET /movie/{id}/reviews` – Parameter: `api_key`, `language`, `page`
- **Zufallsfilme:** `GET /movie/popular` – Parameter: `api_key`, `page`, `language`, `sort_by`

## Hugging Face Modell
- **Modell:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Klasse:** `AutoModelForSequenceClassification`
- **Tokenizer:** `AutoTokenizer`
- **Cache:** `@st.cache_resource` – wird nur beim ersten Aufruf geladen (~270 MB)

## Wichtige Funktionen (`app.py`)
| Funktion | Beschreibung |
|---|---|
| `load_sentiment_model()` | Lädt Tokenizer + Modell (gecacht) |
| `analyze_sentiment(text)` | Sentiment für einen einzelnen Text |
| `batch_sentiment(texts)` | Batch-Sentiment für Liste von Texten |
| `summarize_sentiment(reviews)` | Aggregat (positiv/negativ/neutral/unklar/fehlerhaft) |
| `search_movies(query)` | TMDB-Suche |
| `get_movie_reviews(movie_id, page)` | Top-Reviews eines Films |
| `get_random_movies(count)` | Zufällige Filme (Popular) |
| `render_logo()` | Cinematic-Logo (Gradient-Text) |
| `search_movie()` | Suchleiste + Ergebnisliste |
| `display_movie_overview(movie)` | Film-Details + Reviews + Sentiment (neu: Layout-Komponente) |
| `display_home()` | Startseite: Logo + Titel + Suchleiste + Empfehlungen |
| `main()` | Einstiegspunkt (ruft nur `display_home()` auf) |
| `normalize_sentiment_to_10(score_01)` | Mapper 0..1 ↦ 1..10 mit 1 Nachkommastelle |
| `format_german_decimal(value)` | Float mit 1 Nachkommastelle, Komma als Dezimaltrenner |
| `inject_css()` | Lädt `static/cinematic.css` in die Streamlit-App |
| `render_movie_overview(movie)` | Rendert `templates/movie_overview.html` mit Jinja2 |

## Design-System
- **Farben:** Deep Black (`#0a0a0a`), Surface (`#141414`), Gold (`#d4a574`), Amber (`#c8843a`)
- **Fonts:** Playfair Display (Headlines), Inter (Body), JetBrains Mono (Meta)
- **CSS:** Ein eigenes stylesheet `static/cinematic.css`; per `<style>`-Tag injiziert.
- **HTML:** Das Detail-Layout wird via Jinja2-Template `templates/movie_overview.html`
  gerendert.
- **Effekte:** Film-Grain-Overlay, Vignette, Glow-on-Hover, Fade-in-Animationen

## Wichtige Hinweise für Agents
1. **Streamlit-Element-IDs:** Jedes `st.text_input` braucht einen eindeutigen `key`
   (z. B. `key="film_suche"`), sonst `StreamlitDuplicateElementId`.
2. **Links:** IMMER `st.page_link()` verwenden, nie rohe HTML-Links (`<a href=...>`),
   da diese als Text gerendert werden.
3. **Logo:** `render_logo()` wird **nur in `display_home()`** aufgerufen, nie in `main()`,
   um Doppel-Rendering zu vermeiden.
4. **Sentiment-Zähler:** `summarize_sentiment()` filtert leere/None-Texte automatisch
   heraus. `total` stimmt exakt mit der Anzahl der Textsätze überein.
5. **Model-Caching:** `@st.cache_resource` hält das Modell nach dem ersten Aufruf im
   Speicher. Die `st.info`-Ladeanzeige erscheint nur beim ersten Laden.
6. **TMDB-Reviews:** `get_movie_reviews()` gibt max. 10 Reviews (Top-Reviews) zurück.
   Reviews mit leerem Text werden gefiltert.
7. **Template-Engine:** Das Detail-Design nutzt Jinja2: `templates/movie_overview.html`.
   `render_movie_overview()` rendert die Vorlage und liefert das HTML zurück.
8. **CSS-Injektion:** `inject_css()` liest `static/cinematic.css` und schreibt es in
   `st.markdown(unsafe_allow_html=True)`.
9. **Requirements:** `jinja2>=3.1.0` ist nun in `requirements.txt` enthalten.
10. **Scores:** Scores werden normalisiert (0..1 ↦ 1..10), mit einer Nachkommastelle.
    Die Eingabe- und Ausgabeformate verwenden das Deutsche Dezimalformat
    (Komma statt Punkt). Die Gruppierung (Tausender-Trennzeichen) wird nicht
    eingesetzt. Näheres unter `normalize_sentiment_to_10()` und
    `format_german_decimal()`.

## Bekannte Einschränkungen
- Das Hugging Face-Modell lädt ~270 MB beim ersten Start (einmalig).
- TMDB-API-Schlüssel ist erforderlich (kostenlos erhältlich).
- Die App läuft lokal unter `http://localhost:8501`.

## Starten
```bash
cd Repo
pip install -r requirements.txt
# secrets.toml anpassen
streamlit run app.py
```
