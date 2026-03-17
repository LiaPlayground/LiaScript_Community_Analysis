"""
Usage Patterns Analysis – LiaScript March 2026
Erzeugt results dict für Jinja2-Rendering + Figuren in build/figures/

Aufruf:  python scripts/usage_patterns_analysis.py
Output:  papers/DELFI2026_usage_patterns/build/results.json
         papers/DELFI2026_usage_patterns/build/figures/*.pdf + *.png
"""

import pickle
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
DATA      = Path("/media/sz/Data/Connected_Lecturers/liascript_march_2026/raw")
PAPER_DIR = Path(__file__).parent.parent / "papers" / "DELFI2026_usage_patterns"
BUILD_DIR = PAPER_DIR / "build"
FIG_DIR   = BUILD_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
print("Lade Daten ...")
with open(DATA / "LiaScript_consolidated.p", "rb") as f:
    df = pickle.load(f)
profiles = pd.read_csv(DATA / "LiaScript_user_profiles.csv")

df = df.merge(profiles[["login", "type"]], left_on="repo_user",
              right_on="login", how="left")
n_total = len(df)
print(f"  {n_total} Kurse, {df['repo_name'].nunique()} Repos geladen.")

# ---------------------------------------------------------------------------
# Feature-Taxonomie
# ---------------------------------------------------------------------------
TAXONOMY = {
    "Presentation": {
        "color": "#4393c3",
        "features": {
            "Narrator":        "feature:has_narrator",
            "TTS fragments":   "feature:has_tts_fragments",
            "TTS blocks":      "feature:has_tts_blocks",
            "Animations":      "feature:has_animation_fragments",
            "Anim. blocks":    "feature:has_animation_blocks",
            "Animated CSS":    "feature:has_animated_css",
            "Effects":         "feature:has_effects",
            "Galleries":       "feature:has_galleries",
        }
    },
    "Interaction": {
        "color": "#d6604d",
        "features": {
            "Any quiz":        "feature:has_quiz",
            "Single choice":   "feature:has_single_choice",
            "Multiple choice": "feature:has_multiple_choice",
            "Text quiz":       "feature:has_text_quiz",
            "Matrix quiz":     "feature:has_matrix_quiz",
            "Quiz hints":      "feature:has_quiz_hints",
            "Surveys":         "feature:has_any_survey",
            "Task lists":      "feature:has_task_lists",
        }
    },
    "Reuse": {
        "color": "#4dac26",
        "features": {
            "Macros (any)":    "feature:has_macros",
            "Custom macros":   "feature:has_custom_macro_defs",
            "Imports":         "feature:has_imports",
            "Ext. scripts":    "feature:has_external_scripts",
        }
    },
    "Embedding": {
        "color": "#998ec3",
        "features": {
            "Exec. code":      "feature:has_executable_code",
            "Code projects":   "feature:has_code_projects",
            "WebApps":         "feature:has_webapp",
            "Script tags":     "feature:has_script_tags",
            "ASCII diagrams":  "feature:has_ascii_diagrams",
            "Math":            "feature:has_math",
            "HTML embeds":     "feature:has_html_embeds",
        }
    },
}

# ---------------------------------------------------------------------------
# Adoptionsraten berechnen
# ---------------------------------------------------------------------------
adoption = {}
for cat, cat_data in TAXONOMY.items():
    for label, col in cat_data["features"].items():
        if col in df.columns:
            adoption[label] = round(df[col].mean() * 100, 1)
        else:
            adoption[label] = None
            print(f"  WARN: Spalte '{col}' nicht gefunden ({label})")

# ---------------------------------------------------------------------------
# Drei-Gruppen-Segmentierung
# ---------------------------------------------------------------------------
INTERNAL_ACCOUNTS = [
    "SebastianZug", "andre-dietrich", "LiaPlayground",
    "LiaScript", "LiaBooks", "LiaTemplates", "TUBAF-IfI-LiaScript",
]
MINT_ACCOUNT = "MINT-the-GAP"

def assign_group(user):
    if user in INTERNAL_ACCOUNTS:
        return "Internal"
    if user == MINT_ACCOUNT:
        return "MINT-the-GAP"
    return "Community"

df["group"] = df["repo_user"].apply(assign_group)

group_counts = df["group"].value_counts()
print("\n=== Drei-Gruppen-Segmentierung ===")
for g in ["Internal", "MINT-the-GAP", "Community"]:
    n_acc = df.loc[df["group"] == g, "repo_user"].nunique()
    print(f"  {g:15s} {group_counts.get(g, 0):5d} Kurse, {n_acc} Accounts")

# ---------------------------------------------------------------------------
# Drei-Gruppen-Daten
# ---------------------------------------------------------------------------
groups = ["Internal", "MINT-the-GAP", "Community"]
group_colors = {"Internal": "#e41a1c", "MINT-the-GAP": "#377eb8", "Community": "#4daf4a"}
categories = list(TAXONOMY.keys())

group_cat_means = {}
for g in groups:
    gdf = df[df["group"] == g]
    means = []
    for cat, cat_data in TAXONOMY.items():
        vals = []
        for label, col in cat_data["features"].items():
            if col in gdf.columns:
                vals.append(gdf[col].mean() * 100)
        means.append(round(np.mean(vals), 1) if vals else 0)
    group_cat_means[g] = means

# ---------------------------------------------------------------------------
# Figur: Per-Feature Adoption nach Gruppe (gruppierte Balken, Backup)
# ---------------------------------------------------------------------------
all_features = []
all_cols = []
for cat, cat_data in TAXONOMY.items():
    for label, col in cat_data["features"].items():
        if col in df.columns:
            all_features.append(label)
            all_cols.append(col)

x = np.arange(len(all_features))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
for i, g in enumerate(groups):
    gdf = df[df["group"] == g]
    vals = [gdf[c].mean() * 100 for c in all_cols]
    ax.bar(x + i * width, vals, width, label=g, color=group_colors[g], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Adoption rate (%)")
ax.set_title("Per-feature adoption rates by group")
ax.legend()
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIG_DIR / f"three_group_features.{ext}",
                bbox_inches="tight", dpi=150)
plt.close()
print("Figur 2 gespeichert: three_group_features")

# ---------------------------------------------------------------------------
# Per-Feature Gruppenvergleich (für Tabelle im Paper)
# ---------------------------------------------------------------------------
print("\n=== Per-Feature Gruppenvergleich ===")
print(f"{'Feature':20s} {'Category':14s} {'Internal':>9s} {'MINT':>9s} {'Community':>10s} {'Max-Min':>8s}")
print("-" * 75)

group_feature_rates = []
for cat, cat_data in TAXONOMY.items():
    for label, col in cat_data["features"].items():
        if col not in df.columns:
            continue
        rates = {}
        for g in groups:
            gdf = df[df["group"] == g]
            rates[g] = round(gdf[col].mean() * 100, 1)
        spread = max(rates.values()) - min(rates.values())
        group_feature_rates.append({
            "feature": label, "category": cat,
            "Internal": rates["Internal"],
            "MINT-the-GAP": rates["MINT-the-GAP"],
            "Community": rates["Community"],
            "spread": round(spread, 1),
        })
        marker = " ***" if spread > 20 else " *" if spread > 10 else ""
        print(f"{label:20s} {cat:14s} {rates['Internal']:8.1f}% {rates['MINT-the-GAP']:8.1f}% {rates['Community']:9.1f}% {spread:7.1f}{marker}")

# Export als CSV
gf_df = pd.DataFrame(group_feature_rates)
gf_df.to_csv(BUILD_DIR / "group_feature_comparison.csv", index=False)
print(f"\nGruppenvergleich → {BUILD_DIR / 'group_feature_comparison.csv'}")

# ---------------------------------------------------------------------------
# Results-Dict
# ---------------------------------------------------------------------------
results = {
    # Korpus
    "n_courses":           n_total,
    "n_repos":             df["repo_name"].nunique(),

    # Presentation
    "pct_narrator":        adoption.get("Narrator"),
    "pct_tts_fragments":   adoption.get("TTS fragments"),
    "pct_tts_blocks":      adoption.get("TTS blocks"),
    "pct_animations":      adoption.get("Animations"),
    "pct_animated_css":    adoption.get("Animated CSS"),
    "pct_effects":         adoption.get("Effects"),
    "pct_galleries":       adoption.get("Galleries"),

    # Interaction
    "pct_quiz":            adoption.get("Any quiz"),
    "pct_single_choice":   adoption.get("Single choice"),
    "pct_multiple_choice": adoption.get("Multiple choice"),
    "pct_text_quiz":       adoption.get("Text quiz"),
    "pct_matrix_quiz":     adoption.get("Matrix quiz"),
    "pct_surveys":         adoption.get("Surveys"),
    "pct_task_lists":      adoption.get("Task lists"),

    # Reuse
    "pct_macros":          adoption.get("Macros (any)"),
    "pct_custom_macros":   adoption.get("Custom macros"),
    "pct_imports":         adoption.get("Imports"),
    "pct_ext_scripts":     adoption.get("Ext. scripts"),

    # Embedding
    "pct_exec_code":       adoption.get("Exec. code"),
    "pct_code_projects":   adoption.get("Code projects"),
    "pct_webapp":          adoption.get("WebApps"),
    "pct_ascii":           adoption.get("ASCII diagrams"),
    "pct_math":            adoption.get("Math"),

    # Kategorie-Mittelwerte pro Gruppe
    "group_cat_means":     group_cat_means,
}

with open(BUILD_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Ausgabe
print("\n=== Zentrale Adoptionsraten ===")
for k, v in results.items():
    if k.startswith("pct_") and v is not None:
        print(f"  {k:30s} {v:6.1f}%")
print(f"\nresults.json → {BUILD_DIR / 'results.json'}")
