# LiaScript Paper – Projektkontext

Empirische Analyse des LiaScript-Ökosystems. Referenz für KI-Assistenten und Entwickler.

## 1. Projektübersicht

### Ziel
Empirische Analyse von LiaScript-Kursen aus GitHub-Repositories. Das Projekt umfasst:
- **DELFI 2026 Paper** (aktiv): Feature-Adoption-Analyse – „Designed but Not Used?"
- **Journal Paper**: Fokus auf Kollaboration und Wiederverwendung (älterer Stand)
- **Conference Paper**: Fokus auf User-Segmentierung (älterer Stand)
- **LiaScript-Kurs**: Datengetriebene Präsentation der Ergebnisse (älterer Stand)

### Autoren
- Sebastian Zug (TU Bergakademie Freiberg, Institut für Informatik)
- André Dietrich (TU Bergakademie Freiberg, Institut für Informatik)
- Ines Aubel (TU Bergakademie Freiberg, Institut für Technische Chemie)
- Volker Göhler (TU Bergakademie Freiberg, Institut für Informatik)

---

## 2. Verzeichnisstruktur

```
LiaScript_Paper/
├── PROJECT_KONTEXT.md       # Diese Dokumentation
├── run_pipeline.py          # Pipeline für ältere Papers (journal, conference, course)
├── Pipfile / Pipfile.lock   # Python-Abhängigkeiten (pipenv)
│
├── pipeline/                # Kern-Pipeline-Module (für ältere Papers)
│   ├── data_loader.py       # Laden und Mergen der Datensätze
│   ├── analysis_runner.py   # Ausführung aller Analysen
│   └── paper_builder.py     # Jinja2-basierte Dokumentgenerierung
│
├── analyses/                # Analyse-Module
│   ├── descriptive_stats.py
│   ├── temporal_analysis.py
│   ├── collaboration_analysis.py
│   ├── feature_analysis.py
│   ├── license_analysis.py
│   ├── topic_clustering.py
│   ├── network_analysis.py
│   ├── user_segmentation.py
│   └── three_group_analysis.py
│
├── scripts/                 # Hilfs-Skripte
│   ├── usage_patterns_analysis.py        # Analyse für DELFI2026
│   ├── generate_all_figures.py           # Generiert Visualisierungen
│   ├── update_cumulative_growth_figure.py
│   ├── indicator_frequency_analysis.py
│   └── quick_test.py
│
├── papers/
│   ├── DELFI2026_usage_patterns/   # ← Aktives Paper
│   │   ├── main.tex                # LaTeX-Hauptdatei
│   │   ├── Makefile                # `make pdf` kompiliert direkt
│   │   ├── sections/               # LaTeX-Sections (Quelltext)
│   │   └── static_figures/         # Manuell erstellte Abbildungen
│   │
│   ├── shared/
│   │   ├── literatur.bib           # Gemeinsame Bibliographie
│   │   └── figures/                # Generierte Abbildungen (PNG)
│   │
│   ├── journal_collaboration/      # Älteres Journal Paper
│   ├── conference_development/     # Älteres Conference Paper
│   ├── liascript_course/           # Älterer LiaScript-Kurs
│   └── author_map/                 # Autoren-Karte (Geo-Visualisierung)
│
└── data_march_2026/                # Symlink → externe Daten (nicht im Repo)
    └── liascript_march_2026 → /media/sz/Data/Connected_Lecturers/liascript_march_2026
```

---

## 3. Datenstruktur

### Rohdaten (extern, nicht im Repo)
Liegen unter `/media/sz/Data/Connected_Lecturers/liascript_march_2026/raw/` (~1,4 GB):

| Datei | Beschreibung |
|-------|--------------|
| `LiaScript_files_validated.p` | Hauptdatensatz aller Kurse mit Validierungsstatus |
| `LiaScript_consolidated.p` | Zusammengeführter Datensatz |
| `LiaScript_commits.p` | Git-Commit-Historie pro Kurs |
| `LiaScript_metadata.p` | LiaScript-Header-Metadaten |
| `LiaScript_content.p` | Textinhalt und Wortstatistiken |
| `LiaScript_ai_meta.p` | KI-generierte Metadaten (DDC, Bildungsstufe) |
| `LiaScript_features.p` | Erkannte LiaScript-Features pro Kurs |
| `LiaScript_feature_statistics.p` | Aggregierte Feature-Statistiken |
| `LiaScript_repositories.p` | Repository-Level-Daten |
| `LiaScript_user_profiles.csv` | Nutzerprofile |

### Wichtige Spalten im Hauptdatensatz

```python
# Identifikation
'pipe:ID', 'repo_id', 'repo_full_name', 'path'

# Validierung
'pipe:is_valid_liascript'  # True = validierter Kurs

# Zeitlich
'created_at', 'updated_at', 'created_year'

# Autoren
'contributors', 'author_count', 'commit_count'

# Metadaten
'meta:language', 'meta:author', 'meta:version'

# KI-Klassifikation
'ai:ddc_code', 'ai:education_level'

# Features (Beispiele)
'feat:has_import', 'feat:has_narrator', 'feat:has_version'
```

---

## 4. DELFI 2026 Paper – Build-Prozess

Das aktive Paper wird direkt mit LaTeX kompiliert, ohne Jinja2-Zwischenschritt:

```bash
cd papers/DELFI2026_usage_patterns/
make pdf    # pdflatex + biber + pdflatex ×2
make clean  # LaTeX-Artefakte entfernen
```

Die `main.tex` bindet Sections direkt per `\input{sections/...}` ein.
Bibliographie: `papers/shared/literatur.bib`.

---

## 5. Ältere Pipeline (journal, conference, course)

Die Jinja2-basierte Pipeline für die älteren Papers:

```bash
# Vollständige Pipeline
python run_pipeline.py --paper all

# Einzelnes Paper
python run_pipeline.py --paper course --skip-analysis
```

Diese Papers nutzen `build/`-Verzeichnisse für generierte Ausgaben.

---

## 6. Git-Konventionen

Im Repo ignoriert (`.gitignore`):
- `data_march_2026/` – externe Daten (3,4 GB, per Symlink)
- `**/build/` – generierte Build-Artefakte
- LaTeX-Zwischendateien (`*.aux`, `*.bbl`, etc.)
- `pipeline.log`
- `__pycache__/`, `.vscode/`, `.claude/`

---

*Letzte Aktualisierung: März 2026*
