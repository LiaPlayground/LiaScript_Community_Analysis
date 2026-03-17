#!/usr/bin/env python3
"""
Update fig_cumulative_growth.png using the March 2026 dataset.
Loads LiaScript_consolidated.p, derives first_commit_year, and recreates the figure.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import yaml
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/home/sz/Desktop/Python/LiaScript_Paper")
DATA_PATH     = PROJECT_ROOT / "data_march_2026/liascript_march_2026/raw/LiaScript_consolidated.p"
CONFIG_PATH   = PROJECT_ROOT / "papers/shared/config.yaml"
FIGURES_DIR   = PROJECT_ROOT / "papers/shared/figures"

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
fig_cfg   = config["figures"]
colors    = fig_cfg["colors"]
fonts     = fig_cfg["fonts"]["sizes"]
weights   = fig_cfg["fonts"]["weights"]
line_cfg  = fig_cfg.get("lines", {})
style_cfg = fig_cfg.get("style", {})
legend_cfg = fig_cfg.get("legend", {})

sns.set_style(style_cfg.get("seaborn_style", "whitegrid"))
plt.rcParams["font.family"]    = fig_cfg["fonts"]["family"]
plt.rcParams["savefig.dpi"]    = fig_cfg["output"]["dpi"]
plt.rcParams["savefig.facecolor"] = fig_cfg["output"]["facecolor"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading March 2026 dataset …")
with open(DATA_PATH, "rb") as f:
    df = pickle.load(f)

df["first_commit"] = pd.to_datetime(df["first_commit"], errors="coerce")
df["first_commit_year"] = df["first_commit"].dt.year

# Filter 2018–2026
df = df[(df["first_commit_year"] >= 2018) & (df["first_commit_year"] <= 2026)].copy()
print(f"  {len(df):,} courses after filtering (2018–2026)")

# ── Cumulative metrics per year ───────────────────────────────────────────────
years = sorted(df["first_commit_year"].dropna().unique().astype(int))
cumulative_courses = []
cumulative_users   = []
all_users_seen     = set()

for year in years:
    subset = df[df["first_commit_year"] <= year]
    cumulative_courses.append(len(subset))
    for contributors in subset["contributors_list"]:
        if isinstance(contributors, list):
            all_users_seen.update(c.strip() for c in contributors if c.strip())
    cumulative_users.append(len(all_users_seen))

print("  Year | Courses | Users")
for y, c, u in zip(years, cumulative_courses, cumulative_users):
    print(f"  {int(y)}  |  {c:5d}  |  {u:4d}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=fig_cfg["dimensions"]["wide"])

color_courses = colors["primary"]
color_users   = colors["secondary"]
lw = line_cfg.get("linewidth", 2) + 1
ms = line_cfg.get("marker_size", 8) + 2
mec = line_cfg.get("marker_edge_color", "white")
mew = line_cfg.get("marker_edge_width", 1.5)

ax1.set_xlabel("Year",              fontsize=fonts["axis_label"], fontweight=weights.get("axis_label", "bold"))
ax1.set_ylabel("Cumulative Courses", fontsize=fonts["axis_label"], fontweight=weights.get("axis_label", "bold"),
               color=color_courses)
line1 = ax1.plot(years, cumulative_courses,
                 marker="o", linewidth=lw, color=color_courses,
                 markersize=ms, markeredgecolor=mec, markeredgewidth=mew,
                 label="Courses (cumulative)", zorder=3)
ax1.fill_between(years, cumulative_courses, alpha=0.2, color=color_courses, zorder=1)
ax1.tick_params(axis="y", labelcolor=color_courses)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.grid(True, alpha=style_cfg.get("grid_alpha", 0.3),
         linestyle=style_cfg.get("grid_linestyle", "--"),
         linewidth=style_cfg.get("grid_linewidth", 0.8))

ax2 = ax1.twinx()
ax2.set_ylabel("Cumulative Users (Committers)",
               fontsize=fonts["axis_label"], fontweight=weights.get("axis_label", "bold"),
               color=color_users)
line2 = ax2.plot(years, cumulative_users,
                 marker="s", linewidth=lw, color=color_users,
                 markersize=ms, markeredgecolor=mec, markeredgewidth=mew,
                 label="Users (cumulative)", zorder=3)
ax2.fill_between(years, cumulative_users, alpha=0.2, color=color_users, zorder=1)
ax2.tick_params(axis="y", labelcolor=color_users)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.title("Cumulative Development: LiaScript Courses and Users (2018–2026)",
          fontsize=fonts["title"] + 2, fontweight=weights.get("title", "bold"), pad=20)

lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left",
           fontsize=legend_cfg.get("fontsize", 10) + 1,
           framealpha=legend_cfg.get("framealpha", 0.95))

ax1.set_xticks(years)
ax1.set_axisbelow(True)
fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = FIGURES_DIR / "fig_cumulative_growth.png"
fig.savefig(out_path, dpi=fig_cfg["output"]["dpi"], bbox_inches="tight",
            facecolor=fig_cfg["output"]["facecolor"])
print(f"\nSaved: {out_path}")
plt.close()
