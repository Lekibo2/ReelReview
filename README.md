# ReelReview – Film-Review-Analyse

Eine einfache Streamlit-Website, mit der du Film-Daten aus der **The Movie Database (TMDB)** 
verfolgen und die **Top-Reviews** einer Sentiment-Analyse mit einem **Hugging Face ML-Modell** 
unterziehen kannst.

## Features

- 🔍 **Film-Suche**: Suche nach Filmen via TMDB-Search-Endpunkt
- 🖼️ **Film-Übersicht**: Titel, Jahr, Bewertung, Beschreibung, Poster
- 📊 **Sentiment-Analyse**: Analyse der Top-Reviews mit dem Modell 
  `distilbert-base-uncased-finetuned-sst-2-english` von Hugging Face
- 🎲 **Zufalls-Filme**: Angezeigt unterhalb der Suchleiste
- 📱 **Responsives Layout**: Suchleiste oben, Logo, zufällige Filme unten

## Dateien

```
Repo/
├── app.py                 # Haupt-Streamlit-Anwendung
├── requirements.txt       # Python-Abhängigkeiten
├── .streamlit/
│   └── secrets.toml       # TMDB-API-Schlüssel (Template)
└── assets/
    └── logo.png           # Logo-Bild (optional)
```

## Voraussetzungen

- Python 3.10 oder höher
- Ein **TMDB API-Schlüssel** (kostenlos): https://www.themoviedb.org/settings/api

## Installation & Ausführen

1. Repository klonen:
   ```bash
   git clone <repo-url>
   cd Repo
   ```

2. Virtuelle Umgebung erstellen (optional, aber empfohlen):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. TMDB-API-Schlüssel konfigurieren:
   - Lege eine Datei `.streamlit/secrets.toml` an (siehe Template oben)
   - Ersetze `DEIN_TMDB_API_SCHLUESSEL` durch deinen tatsächlichen Schlüssel

5. App starten:
   ```bash
   streamlit run app.py
   ```

Die Website ist dann unter `http://localhost:8501` erreichbar.

## Hugging Face Modell

Das Sentiment-Analyse-Modell ist `distilbert-base-uncased-finetuned-sst-2-english` 
(von Hugging Face). Es benötigt **keinen API-Schlüssel**, nur eine Authentifizierung 
für hohe Rate-Limits. Die erste Nutzung lädt das Modell herunter (bis zu ~270 MB).

## Hinweise

- Die erste Ausführung kann etwas langsam sein, da die Sentiment-Modell-Dateien 
  initial geladen werden müssen.
- Verwende den `TMDB_API_KEY` als Umgebungsvariable oder in `.streamlit/secrets.toml`.
- Reviews werden maximal 10 Stück für die Sentiment-Analyse heruntergeladen und analysiert.
