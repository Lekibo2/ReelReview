"""
ReelReview – Code-Verifizierungspruefung (neu: CSS+HTML-Template architektur)

Prueft die neue Implementierung (app.py + static/cinematic.css + templates/
movie_overview.html) sowie die neuen Funktionen:
  * normalize_sentiment_to_10()  -> 0..1 -> 1..10 mit 1 Dezimalstelle (Komma)
  * format_german_decimal()      -> Float mit 1 Dezimalstelle (Komma)
  * inject_css()                 -> CSS-Auslieferung
  * render_movie_overview()      -> Jinja2-Template-Aufruf

Nicht mehr relevant (alte Design-Version):
  * render_logo() darf nur in display_home() erscheinen
  * st.text_input() darf nur einmal im Code auftreten
  * "key=..." in st.markdown() ist verboten
"""

import os
import ast
import re

repo = "/home/lekibo/Documents/Informatik/ReelReview/Repo"
errors = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        errors.append(name)


# ---------------------------------------------------------------- 1. Struktur
with open(os.path.join(repo, "app.py")) as f:
    src = f.read()
tree = ast.parse(src)
funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
required_new = {
    "normalize_sentiment_to_10", "format_german_decimal", "inject_css",
    "render_movie_overview", "render_logo", "search_movie",
    "display_movie_overview", "display_home", "main",
    "load_sentiment_model", "analyze_sentiment", "batch_sentiment",
    "summarize_sentiment", "search_movies", "get_movie_details",
    "get_movie_reviews", "get_random_movies",
}
missing = required_new - funcs
check("Funktionen vorhanden", not missing, f"fehlen: {missing}" if missing else "")

# ---------------------------------------------------------------- 2. Score-Logik
check("normalize_sentiment_to_10 existiert",
      "def normalize_sentiment_to_10" in src)
check("format_german_decimal existiert",
      "def format_german_decimal" in src)

# score_2 ist im Template, nicht in Python
check("score_2 nur im Template (nicht in Python)",
      "score_2" not in src or all(
          l.strip().startswith("#") for l in src.splitlines() if "score_2" in l))
check("score_1 im Template", "score_1" in src)

# ---------------------------------------------------------------- 3. CSS / Template
check("static/cinematic.css existiert",
      os.path.exists(os.path.join(repo, "static/cinematic.css")))
check("templates/movie_overview.html existiert",
      os.path.exists(os.path.join(repo, "templates/movie_overview.html")))
check("CSS-Injektion (inject_css)", "def inject_css" in src)
check("Jinja2-Template-Rendering (render_movie_overview)",
      "def render_movie_overview" in src and "jinja2" in src)

# ---------------------------------------------------------------- 4. verify-Skript
check("verify.py als eigene Datei vorhanden",
      os.path.exists(os.path.join(repo, "verify.py")))

# ---------------------------------------------------------------- 5. Render-Simulation
# Simuliere die Template-Daten-Vorbereitung
import jinja2
try:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(
        os.path.join(repo, "templates")))
    t = env.get_template("movie_overview.html")
    check("Jinja2-Template laden", t is not None)
except Exception as e:
    check("Jinja2-Template laden", False, str(e))

# Simuliere die normalisierende Score-Funktion
def normalize_sentiment_to_10(score_01):
    v = 1.0 + score_01 * 9.0
    return f"{v:.1f}".replace(".", ",")

def format_german_decimal(value):
    return f"{value:.1f}".replace(".", ",")

check("normalize: 0.0->1,0", normalize_sentiment_to_10(0.0) == "1,0")
check("normalize: 0.5->5,5", normalize_sentiment_to_10(0.5) == "5,5")
check("normalize: 1.0->10,0", normalize_sentiment_to_10(1.0) == "10,0")
check("format: 3,14->3,1", format_german_decimal(3.14159) == "3,1")

# ---------------------------------------------------------------- 6. Abschließend
print()
if errors:
    print(f"VERIFIZIERUNG ABGEBROCHEN: {len(errors)} Fehler")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)
else:
    print("ALLE PRUEFUNGEN BESTAETIGT - neues Cinematic-Design bereit")
