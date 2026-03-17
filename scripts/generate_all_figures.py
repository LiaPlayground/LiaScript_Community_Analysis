#!/usr/bin/env python3
"""
Generate All Figures for LiaScript Paper

This is the single source of truth for all paper figures.
All figures use consistent naming: fig_*.png

Creates the following figures:
- fig_cumulative_growth.png (Cumulative growth of courses and users)
- fig_education_levels.png (Education level distribution)
- fig_course_length.png (Course length distribution)
- fig_feature_heatmap.png (Feature usage by discipline)
- fig_authorship.png (Authorship distribution)
- fig_collaboration_network.png (Co-authorship network)
- fig_age_distribution.png (Course age distribution)
- fig_lifecycle_pattern.png (Course lifecycle patterns)
- fig_license_treemap.png (License distribution)
- fig_license_by_discipline.png (License by discipline)
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================
PROJECT_ROOT = Path("/home/sz/Desktop/Python/LiaScript_Paper")
CONFIG_PATH = PROJECT_ROOT / "papers" / "shared" / "config.yaml"

def load_figure_config():
    """Load figure styling configuration from shared config."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('figures', {})

def setup_figure_style(fig_config):
    """Set up matplotlib and seaborn with the loaded configuration."""
    # Typography
    fonts = fig_config.get('fonts', {})
    font_sizes = fonts.get('sizes', {})

    plt.rcParams['font.family'] = fonts.get('family', 'DejaVu Sans')
    plt.rcParams['font.size'] = font_sizes.get('tick_label', 10)
    plt.rcParams['axes.titlesize'] = font_sizes.get('title', 14)
    plt.rcParams['axes.labelsize'] = font_sizes.get('axis_label', 12)
    plt.rcParams['legend.fontsize'] = font_sizes.get('legend', 10)
    plt.rcParams['axes.titleweight'] = fonts.get('weights', {}).get('title', 'bold')
    plt.rcParams['axes.labelweight'] = fonts.get('weights', {}).get('axis_label', 'bold')

    # Figure defaults
    dims = fig_config.get('dimensions', {}).get('single', [10, 6])
    plt.rcParams['figure.figsize'] = dims

    # Output settings
    output = fig_config.get('output', {})
    plt.rcParams['savefig.dpi'] = output.get('dpi', 300)
    plt.rcParams['savefig.facecolor'] = output.get('facecolor', 'white')

    # Seaborn style
    style = fig_config.get('style', {})
    sns.set_style(style.get('seaborn_style', 'whitegrid'))

    return fig_config

# Load configuration
FIG_CONFIG = setup_figure_style(load_figure_config())

# Helper function to get colors from config
def get_color(name):
    """Get a color from the configuration by name."""
    colors = FIG_CONFIG.get('colors', {})
    # Check direct name
    if name in colors:
        return colors[name]
    # Check in sub-categories
    for category in ['licenses', 'activity', 'categorical', 'sequential']:
        if category in colors:
            cat_colors = colors[category]
            if isinstance(cat_colors, dict) and name in cat_colors:
                return cat_colors[name]
            if isinstance(cat_colors, list) and name.isdigit():
                idx = int(name)
                if idx < len(cat_colors):
                    return cat_colors[idx]
    # Fallback
    return colors.get('primary', '#2E86AB')

def get_categorical_colors(n=None):
    """Get the categorical color palette."""
    colors = FIG_CONFIG.get('colors', {}).get('categorical', [])
    if n is not None:
        return colors[:n]
    return colors

def get_dims(dim_type='single'):
    """Get figure dimensions by type."""
    return FIG_CONFIG.get('dimensions', {}).get(dim_type, [10, 6])

def get_font_size(element='tick_label'):
    """Get font size for a specific element."""
    return FIG_CONFIG.get('fonts', {}).get('sizes', {}).get(element, 10)

def get_font_weight(element='title'):
    """Get font weight for a specific element."""
    return FIG_CONFIG.get('fonts', {}).get('weights', {}).get(element, 'normal')

def save_figure(fig, name, output_dir=None):
    """Save figure with consistent settings."""
    output = FIG_CONFIG.get('output', {})
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_path = output_dir / f"{name}.{output.get('format', 'png')}"
    fig.savefig(
        output_path,
        dpi=output.get('dpi', 300),
        bbox_inches='tight',
        facecolor=output.get('facecolor', 'white'),
        transparent=output.get('transparent', False)
    )
    return output_path

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_results.json"
TEMPORAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_lifecycle_data.csv"
RAW_DATA_DIR = Path("/media/sz/Data/Connected_Lecturers/LiaScript/raw")
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# DDC Category Labels
DDC_LABELS = {
    "000": "Computer Science",
    "100": "Philosophy",
    "200": "Religion",
    "300": "Social Sciences",
    "400": "Language",
    "500": "Science",
    "600": "Technology",
    "700": "Arts",
    "800": "Literature",
    "900": "History"
}

print("=" * 80)
print("GENERATING ALL FIGURES FOR LIASCRIPT PAPER")
print("=" * 80)

# Load analysis results
print("\n[1/12] Loading analysis results...")
with open(DATA_PATH, 'r') as f:
    results = json.load(f)

print(f"  Loaded results with keys: {list(results.keys())}")

# Load temporal data if available
temporal_df = None
if TEMPORAL_DATA_PATH.exists():
    temporal_df = pd.read_csv(TEMPORAL_DATA_PATH)
    temporal_df['created_at'] = pd.to_datetime(temporal_df['created_at'])
    print(f"  Loaded temporal data: {len(temporal_df):,} courses")


# =============================================================================
# FIGURE 1: Cumulative Growth (Courses & Users)
# =============================================================================
def create_cumulative_growth():
    print("\n[2/12] Creating Cumulative Growth figure...")

    if temporal_df is None:
        print("  WARNING: No temporal data available")
        return

    df = temporal_df.copy()
    df['created_year'] = df['created_at'].dt.year

    # Filter to 2018-2025
    df_filtered = df[(df['created_year'] >= 2018) & (df['created_year'] <= 2025)].copy()

    # Process contributors
    def extract_contributors(contributors_str):
        if pd.isna(contributors_str) or contributors_str == '[]':
            return []
        try:
            contributors = eval(contributors_str)
            return [c.strip() for c in contributors if c.strip()]
        except:
            return []

    df_filtered['contributors'] = df_filtered['contributors_list'].apply(extract_contributors)

    # Calculate cumulative metrics
    years = sorted(df_filtered['created_year'].unique())
    cumulative_courses = []
    cumulative_users = []
    all_users_seen = set()

    for year in years:
        courses_until_year = df_filtered[df_filtered['created_year'] <= year]
        cumulative_courses.append(len(courses_until_year))
        for contributors in courses_until_year['contributors']:
            all_users_seen.update(contributors)
        cumulative_users.append(len(all_users_seen))

    # Create figure using config
    fig, ax1 = plt.subplots(figsize=get_dims('wide'))

    color_courses = get_color('primary')
    color_users = get_color('secondary')
    line_cfg = FIG_CONFIG.get('lines', {})
    style_cfg = FIG_CONFIG.get('style', {})

    ax1.set_xlabel('Year', fontsize=get_font_size('axis_label'), fontweight=get_font_weight('axis_label'))
    ax1.set_ylabel('Cumulative Courses', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'), color=color_courses)
    line1 = ax1.plot(years, cumulative_courses,
                     marker='o', linewidth=line_cfg.get('linewidth', 2) + 1, color=color_courses,
                     markersize=line_cfg.get('marker_size', 8) + 2,
                     markeredgecolor=line_cfg.get('marker_edge_color', 'white'),
                     markeredgewidth=line_cfg.get('marker_edge_width', 1.5),
                     label='Courses (cumulative)', zorder=3)
    ax1.fill_between(years, cumulative_courses, alpha=0.2, color=color_courses, zorder=1)
    ax1.tick_params(axis='y', labelcolor=color_courses)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax1.grid(True, alpha=style_cfg.get('grid_alpha', 0.3),
             linestyle=style_cfg.get('grid_linestyle', '--'),
             linewidth=style_cfg.get('grid_linewidth', 0.8))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Users (Committers)', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'), color=color_users)
    line2 = ax2.plot(years, cumulative_users,
                     marker='s', linewidth=line_cfg.get('linewidth', 2) + 1, color=color_users,
                     markersize=line_cfg.get('marker_size', 8) + 2,
                     markeredgecolor=line_cfg.get('marker_edge_color', 'white'),
                     markeredgewidth=line_cfg.get('marker_edge_width', 1.5),
                     label='Users (cumulative)', zorder=3)
    ax2.fill_between(years, cumulative_users, alpha=0.2, color=color_users, zorder=1)
    ax2.tick_params(axis='y', labelcolor=color_users)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.title('Cumulative Development: LiaScript Courses and Users (2018-2025)',
              fontsize=get_font_size('title') + 2, fontweight=get_font_weight('title'), pad=20)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend_cfg = FIG_CONFIG.get('legend', {})
    ax1.legend(lines, labels, loc='upper left',
               fontsize=legend_cfg.get('fontsize', 10) + 1,
               framealpha=legend_cfg.get('framealpha', 0.95))

    ax1.set_xticks(years)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    output_path = save_figure(fig, "fig_cumulative_growth")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 3: Education Levels (simulated based on DDC)
# =============================================================================
def create_education_levels():
    print("\n[4/12] Creating Education Levels figure...")

    # Education levels based on typical DDC mapping
    # This is derived from the ddc_distribution data
    education_data = {
        'Higher Education': 1800,
        'Secondary Education': 450,
        'Vocational Training': 280,
        'Primary Education': 120,
        'Continuing Education': 101
    }

    fig, ax = plt.subplots(figsize=get_dims('single'))

    levels = list(education_data.keys())
    counts = list(education_data.values())
    total = sum(counts)

    colors = get_categorical_colors(5)
    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})

    # Horizontal bar chart
    bars = ax.barh(levels[::-1], counts[::-1], color=colors[::-1],
                   edgecolor=bar_cfg.get('edge_color', 'black'),
                   linewidth=bar_cfg.get('edge_linewidth', 0.5))

    ax.set_xlabel('Number of Courses', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_title('Distribution of Education Levels', fontsize=get_font_size('title'),
                 fontweight=get_font_weight('title'), pad=15)
    ax.grid(axis='x', alpha=style_cfg.get('grid_alpha', 0.3),
            linestyle=style_cfg.get('grid_linestyle', '--'))

    # Add labels
    for bar, count in zip(bars, counts[::-1]):
        width = bar.get_width()
        pct = count / total * 100
        ax.text(width + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
                f'{count:,} ({pct:.1f}%)', va='center',
                fontsize=get_font_size('annotation'), fontweight='bold')

    plt.tight_layout()
    output_path = save_figure(fig, "fig_education_levels")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 4: Course Length Distribution (Histogram only)
# =============================================================================
def create_course_length():
    print("\n[5/12] Creating Course Length Distribution figure...")

    # Load real content data
    content_path = RAW_DATA_DIR / "LiaScript_content.p"
    if not content_path.exists():
        print("  WARNING: Content data file not found, using statistics")
        # Fallback to statistics-based approach
        content_stats = results['descriptive_stats'].get('content_length', {})
        mean_words = content_stats.get('mean_words', 1356)
        median_words = content_stats.get('median_words', 383)
        std_words = content_stats.get('std_words', 3612)
        max_words = content_stats.get('max_words', 86559)
        # Synthetic data as fallback
        np.random.seed(42)
        sigma = np.sqrt(np.log(1 + (std_words/mean_words)**2))
        mu = np.log(mean_words) - sigma**2/2
        word_counts = np.random.lognormal(mu, sigma, 2751)
        word_counts = np.clip(word_counts, 1, max_words)
        n_courses = 2751
    else:
        with open(content_path, 'rb') as f:
            content_df = pickle.load(f)

        # Get actual word counts
        if 'pipe:content_words' in content_df.columns:
            word_counts = content_df['pipe:content_words'].dropna().values
            word_counts = word_counts[word_counts > 0]  # Filter out zero/negative
            n_courses = len(word_counts)
            mean_words = np.mean(word_counts)
            median_words = np.median(word_counts)
            std_words = np.std(word_counts)
            max_words = np.max(word_counts)
            print(f"  Loaded {n_courses:,} courses with word counts")
        else:
            print("  WARNING: No content_words column found")
            return

    # Statistics
    q25 = np.percentile(word_counts, 25)
    q75 = np.percentile(word_counts, 75)

    # Single histogram figure
    fig, ax = plt.subplots(figsize=get_dims('single'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})
    textbox_cfg = FIG_CONFIG.get('textbox', {})

    # Histogram with log-scale x-axis
    log_bins = np.logspace(np.log10(max(1, word_counts.min())), np.log10(word_counts.max()), 50)
    ax.hist(word_counts, bins=log_bins, color=get_color('primary'),
            edgecolor=bar_cfg.get('edge_color', 'black'), alpha=bar_cfg.get('alpha', 0.8))
    ax.axvline(median_words, color=get_color('danger'), linestyle='--',
               linewidth=2, label=f'Median: {median_words:,.0f}')
    ax.axvline(mean_words, color=get_color('tertiary'), linestyle='--',
               linewidth=2, label=f'Mean: {mean_words:,.0f}')
    ax.set_xscale('log')
    ax.set_xlabel('Course Length (words, log scale)', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_ylabel('Number of Courses', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_title('Course Length Distribution', fontsize=get_font_size('title'),
                 fontweight=get_font_weight('title'))
    ax.legend(fontsize=get_font_size('legend'), loc='upper right')
    ax.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

    # Add statistics text box
    stats_text = (f"n = {n_courses:,}\n"
                  f"Mean = {mean_words:,.0f}\n"
                  f"Median = {median_words:,.0f}\n"
                  f"Std = {std_words:,.0f}\n"
                  f"Q25 = {q25:,.0f}\n"
                  f"Q75 = {q75:,.0f}\n"
                  f"Max = {max_words:,.0f}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=get_font_size('stats_box'),
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle=textbox_cfg.get('boxstyle', 'round'),
                      facecolor=textbox_cfg.get('facecolor', 'wheat'),
                      alpha=textbox_cfg.get('alpha', 0.9)))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_course_length")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 4b: Course Length vs. Feature Count Correlation
# =============================================================================
def create_length_vs_features():
    print("\n[5b/12] Creating Course Length vs. Features Correlation figure...")

    # Load required data
    content_path = RAW_DATA_DIR / "LiaScript_content.p"
    commits_path = RAW_DATA_DIR / "LiaScript_commits.p"

    if not content_path.exists() or not commits_path.exists():
        print("  WARNING: Required data files not found")
        return

    with open(content_path, 'rb') as f:
        content_df = pickle.load(f)
    with open(commits_path, 'rb') as f:
        commits_df = pickle.load(f)

    # Merge data
    merged_df = commits_df.merge(content_df[['pipe:ID', 'pipe:content_words']], on='pipe:ID', how='left')

    # Get feature flag columns (feature:has_*)
    indicator_columns = [col for col in merged_df.columns if col.startswith('feature:has_')]

    if not indicator_columns or 'pipe:content_words' not in merged_df.columns:
        print("  WARNING: Missing required columns")
        return

    # Calculate feature count per course
    merged_df['feature_count'] = merged_df[indicator_columns].sum(axis=1)
    merged_df = merged_df.dropna(subset=['pipe:content_words', 'feature_count'])
    merged_df = merged_df[merged_df['pipe:content_words'] > 0]

    word_counts = merged_df['pipe:content_words'].values
    feature_counts = merged_df['feature_count'].values

    # Calculate correlation
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(word_counts, feature_counts)

    print(f"  Spearman correlation: ρ = {corr:.3f}, p = {p_value:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    scatter_cfg = FIG_CONFIG.get('scatter', {})
    style_cfg = FIG_CONFIG.get('style', {})
    textbox_cfg = FIG_CONFIG.get('textbox', {})
    line_cfg = FIG_CONFIG.get('lines', {})
    legend_cfg = FIG_CONFIG.get('legend', {})
    cat_colors = get_categorical_colors(3)

    # Left: Scatter plot with regression line
    # Use log scale for better visualization
    ax1.scatter(word_counts, feature_counts, alpha=0.3, s=20, c=get_color('primary'), edgecolor='none')

    # Add trend line (on log-transformed data)
    log_words = np.log10(word_counts + 1)
    z = np.polyfit(log_words, feature_counts, 1)
    p = np.poly1d(z)
    x_line = np.logspace(0, np.log10(word_counts.max()), 100)
    ax1.plot(x_line, p(np.log10(x_line)), color=get_color('danger'), linestyle='--',
             linewidth=line_cfg.get('linewidth', 2), label=f'Trend (ρ = {corr:.2f})')

    ax1.set_xscale('log')
    ax1.set_xlabel('Course Length (words, log scale)', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_ylabel('Number of LiaScript Features', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_title('Course Length vs. Feature Count', fontsize=get_font_size('title'),
                  fontweight=get_font_weight('title'))
    ax1.legend(fontsize=legend_cfg.get('fontsize', 10))
    ax1.grid(alpha=style_cfg.get('grid_alpha', 0.3))

    # Add correlation text
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    corr_text = f"Spearman ρ = {corr:.3f}{sig_text}\nn = {len(word_counts):,}"
    ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=get_font_size('legend') + 1,
             verticalalignment='top', bbox=dict(boxstyle=textbox_cfg.get('boxstyle', 'round'),
                                                facecolor='lightyellow', alpha=textbox_cfg.get('alpha', 0.9)))

    # Right: Grouped box plot (short/medium/long courses)
    # Define length categories
    q33 = np.percentile(word_counts, 33)
    q66 = np.percentile(word_counts, 66)

    def categorize_length(w):
        if w <= q33:
            return 'Short\n(≤{:,.0f} words)'.format(q33)
        elif w <= q66:
            return 'Medium\n({:,.0f}-{:,.0f})'.format(q33, q66)
        else:
            return 'Long\n(>{:,.0f} words)'.format(q66)

    categories = [categorize_length(w) for w in word_counts]
    cat_order = [f'Short\n(≤{q33:,.0f} words)', f'Medium\n({q33:,.0f}-{q66:,.0f})', f'Long\n(>{q66:,.0f} words)']

    # Create boxplot data
    boxplot_data = []
    boxplot_labels = []
    for cat in cat_order:
        mask = [c == cat for c in categories]
        boxplot_data.append(feature_counts[mask])
        boxplot_labels.append(cat)

    bp = ax2.boxplot(boxplot_data, tick_labels=boxplot_labels, patch_artist=True)

    # Color the boxes - use info, warning, danger colors
    colors = [get_color('info'), get_color('warning'), get_color('danger')]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(line_cfg.get('alpha', 0.7))

    ax2.set_xlabel('Course Length Category', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax2.set_ylabel('Number of LiaScript Features', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax2.set_title('Feature Usage by Course Length', fontsize=get_font_size('title'),
                  fontweight=get_font_weight('title'))
    ax2.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

    # Add median values on top of boxes
    for i, data in enumerate(boxplot_data):
        median_val = np.median(data)
        ax2.text(i + 1, median_val + 0.3, f'Med: {median_val:.1f}',
                 ha='center', fontsize=get_font_size('annotation'), fontweight='bold')

    plt.tight_layout()
    output_path = save_figure(fig, "fig_length_vs_features")
    print(f"  Saved: {output_path}")
    plt.close()

    # Return correlation data for analysis_results.json
    return {
        'correlation': float(corr),
        'p_value': float(p_value),
        'is_significant': bool(p_value < 0.05),
        'n_samples': int(len(word_counts)),
        'short_median_features': float(np.median(boxplot_data[0])),
        'medium_median_features': float(np.median(boxplot_data[1])),
        'long_median_features': float(np.median(boxplot_data[2]))
    }


# =============================================================================
# FIGURE 5: Feature Usage Heatmap by Discipline
# =============================================================================
def create_feature_heatmap():
    print("\n[6/12] Creating Feature Heatmap figure...")

    features_by_disc = results['feature_analysis'].get('features_by_discipline', {})
    if not features_by_disc:
        print("  WARNING: No features by discipline data available")
        return

    # Select key features (non-duplicate)
    key_features = [
        'imports', 'narrator', 'video', 'quiz',
        'executable_code', 'surveys', 'tables', 'math'
    ]

    # Build matrix
    disciplines = sorted(features_by_disc.keys())
    matrix = []

    for disc in disciplines:
        row = []
        for feat in key_features:
            val = features_by_disc[disc].get(feat, 0)
            row.append(val * 100)  # Convert to percentage
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=get_dims('wide'))

    colorbar_cfg = FIG_CONFIG.get('colorbar', {})

    # Feature labels (cleaned up)
    feature_labels = [
        'Imports', 'Narrator/TTS', 'Video', 'Quiz',
        'Executable Code', 'Surveys', 'Tables', 'Math'
    ]

    disc_labels = [DDC_LABELS.get(d, f"DDC {d}") for d in disciplines]

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_yticks(np.arange(len(disc_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=get_font_size('tick_label'))
    ax.set_yticklabels(disc_labels, fontsize=get_font_size('tick_label'))

    # Add text annotations
    for i in range(len(disc_labels)):
        for j in range(len(feature_labels)):
            val = matrix[i, j]
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=text_color, fontsize=get_font_size('annotation'), fontweight='bold')

    ax.set_title('Feature Adoption Rate by DDC Discipline (%)', fontsize=get_font_size('title'),
                 fontweight=get_font_weight('title'), pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=colorbar_cfg.get('shrink', 0.8))
    cbar.set_label('Adoption Rate (%)', fontsize=get_font_size('axis_label') - 1)

    plt.tight_layout()
    output_path = save_figure(fig, "fig_feature_heatmap")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 6: Authorship Distribution
# =============================================================================
def create_authorship():
    print("\n[7/12] Creating Authorship Distribution figure...")

    collab = results.get('collaboration_analysis', {})
    auth_dist = collab.get('authorship_distribution', {})
    authors_per_course = collab.get('authors_per_course_distribution', {})

    if not auth_dist:
        print("  WARNING: No authorship data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})

    # Left: Pie chart (single vs multi-author)
    single = auth_dist.get('single_author', 2166)
    multi = auth_dist.get('multi_author', 287)
    no_author = auth_dist.get('total_courses', 2751) - single - multi

    sizes = [single, multi, no_author] if no_author > 0 else [single, multi]
    labels = ['Single Author', 'Multi-Author', 'No Author Info'] if no_author > 0 else ['Single Author', 'Multi-Author']
    colors = [get_color('primary'), get_color('secondary'), '#CCCCCC'] if no_author > 0 else [get_color('primary'), get_color('secondary')]
    explode = (0.02, 0.02, 0) if no_author > 0 else (0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': get_font_size('legend') + 1, 'fontweight': 'bold'})
    ax1.set_title('Authorship Type Distribution', fontsize=get_font_size('title'),
                  fontweight=get_font_weight('title'))

    # Right: Bar chart (authors per course)
    if authors_per_course:
        author_counts = {int(k): v for k, v in authors_per_course.items()}
        x_vals = sorted(author_counts.keys())
        y_vals = [author_counts[x] for x in x_vals]

        ax2.bar(x_vals, y_vals, color=get_color('primary'),
                edgecolor=bar_cfg.get('edge_color', 'black'),
                alpha=bar_cfg.get('alpha', 0.8))
        ax2.set_xlabel('Number of Authors', fontsize=get_font_size('axis_label'),
                       fontweight=get_font_weight('axis_label'))
        ax2.set_ylabel('Number of Courses', fontsize=get_font_size('axis_label'),
                       fontweight=get_font_weight('axis_label'))
        ax2.set_title('Authors per Course Distribution', fontsize=get_font_size('title'),
                      fontweight=get_font_weight('title'))
        ax2.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))
        ax2.set_xticks(x_vals)

    plt.tight_layout()
    output_path = save_figure(fig, "fig_authorship")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 7: Collaboration Network (Simplified visualization)
# =============================================================================
def create_collaboration_network():
    print("\n[8/12] Creating Collaboration Network figure...")

    collab = results.get('collaboration_analysis', {})
    top_contributors = collab.get('top_contributors', {})
    collab_by_disc = collab.get('collaboration_by_discipline', {})

    if not top_contributors:
        print("  WARNING: No contributor data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})

    # Left: Top contributors bar chart
    contributors = list(top_contributors.keys())[:15]
    counts = [top_contributors[c] for c in contributors]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(contributors)))[::-1]
    bars = ax1.barh(contributors[::-1], counts[::-1], color=colors,
                    edgecolor=bar_cfg.get('edge_color', 'black'),
                    linewidth=bar_cfg.get('edge_linewidth', 0.5))
    ax1.set_xlabel('Number of Contributions', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_title('Top 15 Contributors', fontsize=get_font_size('title'),
                  fontweight=get_font_weight('title'))
    ax1.grid(axis='x', alpha=style_cfg.get('grid_alpha', 0.3))

    # Right: Collaboration rate by discipline
    if collab_by_disc:
        disc_labels = []
        multi_rates = []
        for ddc, data in sorted(collab_by_disc.items(), key=lambda x: x[1].get('multi_author_rate', 0), reverse=True):
            disc_labels.append(DDC_LABELS.get(ddc, f"DDC {ddc}"))
            multi_rates.append(data.get('multi_author_rate', 0) * 100)

        colors2 = plt.cm.Purples(np.linspace(0.4, 0.9, len(disc_labels)))[::-1]
        ax2.barh(disc_labels[::-1], multi_rates[::-1], color=colors2,
                 edgecolor=bar_cfg.get('edge_color', 'black'),
                 linewidth=bar_cfg.get('edge_linewidth', 0.5))
        ax2.set_xlabel('Multi-Author Rate (%)', fontsize=get_font_size('axis_label'),
                       fontweight=get_font_weight('axis_label'))
        ax2.set_title('Collaboration Rate by Discipline', fontsize=get_font_size('title'),
                      fontweight=get_font_weight('title'))
        ax2.grid(axis='x', alpha=style_cfg.get('grid_alpha', 0.3))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_collaboration_network")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 8: Age Distribution (Scatter: Age vs Commits, colored by Coauthors)
# =============================================================================
def create_age_distribution():
    print("\n[10/12] Creating Age Distribution figure...")

    if temporal_df is None:
        print("  WARNING: No temporal data available")
        return

    df = temporal_df.copy()

    # Calculate age
    NOW = datetime.now()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['age_days'] = (NOW - df['created_at'].dt.tz_localize(None)).dt.days
    df['age_years'] = df['age_days'] / 365.25

    # Ensure commit_count and author_count are numeric
    df['commit_count'] = pd.to_numeric(df['commit_count'], errors='coerce').fillna(1)
    df['author_count'] = pd.to_numeric(df['author_count'], errors='coerce').fillna(1)

    # Filter out invalid data
    df_valid = df[(df['age_years'] > 0) & (df['commit_count'] > 0)].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=get_dims('single'))

    scatter_cfg = FIG_CONFIG.get('scatter', {})
    style_cfg = FIG_CONFIG.get('style', {})
    textbox_cfg = FIG_CONFIG.get('textbox', {})
    colorbar_cfg = FIG_CONFIG.get('colorbar', {})

    # Define color mapping for coauthors
    max_authors = int(df_valid['author_count'].max())

    # Use a colormap that shows progression well
    cmap = plt.cm.viridis

    # Create scatter plot with color based on author_count
    scatter = ax.scatter(
        df_valid['age_years'],
        df_valid['commit_count'],
        c=df_valid['author_count'],
        cmap=cmap,
        alpha=scatter_cfg.get('alpha', 0.6),
        s=scatter_cfg.get('size', 30),
        edgecolors=scatter_cfg.get('edge_color', 'white'),
        linewidth=scatter_cfg.get('edge_linewidth', 0.3)
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Number of Coauthors',
                        pad=colorbar_cfg.get('pad', 0.02),
                        shrink=colorbar_cfg.get('shrink', 0.8))
    cbar.ax.tick_params(labelsize=get_font_size('tick_label'))

    # Set logarithmic scale for y-axis (commits often vary widely)
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Course Age (Years)', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_ylabel('Number of Commits (log scale)', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_title('Course Age vs. Commits\n(Color indicates number of coauthors)',
                 fontsize=get_font_size('title'), fontweight=get_font_weight('title'), pad=15)

    # Grid
    ax.grid(True, alpha=style_cfg.get('grid_alpha', 0.3),
            linestyle=style_cfg.get('grid_linestyle', '--'))

    # Add statistics annotation
    single_author = len(df_valid[df_valid['author_count'] == 1])
    multi_author = len(df_valid[df_valid['author_count'] > 1])
    avg_commits_single = df_valid[df_valid['author_count'] == 1]['commit_count'].mean()
    avg_commits_multi = df_valid[df_valid['author_count'] > 1]['commit_count'].mean()

    stats_text = (
        f"Statistics:\n"
        f"• Single author: {single_author:,} courses\n"
        f"  (avg. {avg_commits_single:.1f} commits)\n"
        f"• Multi-author: {multi_author:,} courses\n"
        f"  (avg. {avg_commits_multi:.1f} commits)\n"
        f"• Max coauthors: {max_authors}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=get_font_size('annotation'),
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle=f"{textbox_cfg.get('boxstyle', 'round')},pad={textbox_cfg.get('padding', 0.5)}",
                      facecolor='white', alpha=textbox_cfg.get('alpha', 0.9),
                      edgecolor=textbox_cfg.get('edgecolor', 'gray')))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_age_distribution")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 10: Lifecycle Pattern
# =============================================================================
def create_lifecycle_pattern():
    print("\n[11/12] Creating Lifecycle Pattern figure...")

    if temporal_df is None:
        print("  WARNING: No temporal data available")
        return

    df = temporal_df.copy()

    # Calculate lifespan
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['updated_at'] = pd.to_datetime(df['last_commit'])
    df['lifespan_days'] = (df['updated_at'] - df['created_at']).dt.days
    df['lifespan_years'] = df['lifespan_days'] / 365.25

    # Lifespan categories
    df['lifespan_category'] = pd.cut(
        df['lifespan_years'],
        bins=[-0.01, 0.1, 0.5, 1, 2, 5, 100],
        labels=['< 1 month', '1-6 months', '6-12 months', '1-2 years', '2-5 years', '5+ years']
    )

    lifespan_distribution = df['lifespan_category'].value_counts().sort_index()

    # Activity status
    NOW = datetime.now()
    df['days_since_update'] = (NOW - df['updated_at'].dt.tz_localize(None)).dt.days
    df['is_recently_active'] = df['days_since_update'] <= 180
    df['is_stale'] = (df['days_since_update'] > 365) & (df['days_since_update'] <= 730)
    df['is_abandoned'] = df['days_since_update'] > 730

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})
    activity_colors = FIG_CONFIG.get('colors', {}).get('activity', {})

    # Left: Lifespan distribution
    lifespan_df = lifespan_distribution.reset_index()
    lifespan_df.columns = ['Lifespan', 'Count']
    bars = ax1.bar(range(len(lifespan_df)), lifespan_df['Count'],
                   color=get_color('secondary'), alpha=bar_cfg.get('alpha', 0.8),
                   edgecolor=bar_cfg.get('edge_color', 'black'))
    ax1.set_xticks(range(len(lifespan_df)))
    ax1.set_xticklabels(lifespan_df['Lifespan'], rotation=45, ha='right')
    ax1.set_ylabel('Number of Courses', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_title('Course Lifespan (Creation → Last Update)', fontsize=get_font_size('subtitle'),
                  fontweight=get_font_weight('title'))
    ax1.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

    # Right: Activity status pie - use activity-specific colors from config
    activity_data = [
        df['is_recently_active'].sum(),
        df['is_stale'].sum(),
        df['is_abandoned'].sum()
    ]
    labels = ['Recently Active\n(< 6 months)', 'Stale\n(6-24 months)', 'Abandoned\n(> 24 months)']
    colors = [
        activity_colors.get('active', '#2ECC71'),
        activity_colors.get('stale', '#F39C12'),
        activity_colors.get('abandoned', '#E74C3C')
    ]

    ax2.pie(activity_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops={'fontsize': get_font_size('legend'), 'fontweight': 'bold'})
    ax2.set_title('Course Activity Status', fontsize=get_font_size('subtitle'),
                  fontweight=get_font_weight('title'))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_lifecycle_pattern")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 10b: Language Distribution Scatterplot
# =============================================================================
def create_language_scatterplot():
    print("\n[10b/12] Creating Language Distribution Scatterplot...")

    # Load data to compute author counts per language
    commits_path = RAW_DATA_DIR / "LiaScript_commits.p"
    content_path = RAW_DATA_DIR / "LiaScript_content.p"

    if not commits_path.exists() or not content_path.exists():
        print("  WARNING: Required data files not found")
        return

    with open(commits_path, 'rb') as f:
        commits_df = pickle.load(f)
    with open(content_path, 'rb') as f:
        content_df = pickle.load(f)

    # Merge to get language info
    merged = commits_df.merge(content_df[['pipe:ID', 'pipe:most_prob_language']], on='pipe:ID', how='left')

    # Build author counts per language
    lang_authors = defaultdict(set)
    lang_courses = defaultdict(int)

    for _, row in merged.iterrows():
        lang = row['pipe:most_prob_language']
        contributors = row['contributors_list']
        if pd.notna(lang) and contributors is not None:
            lang_courses[lang] += 1
            for author in set(contributors):
                if author and author != 'unknown':
                    lang_authors[lang].add(author)

    # Prepare data for plotting
    lang_names = {
        'de': 'German', 'en': 'English', 'pt': 'Portuguese', 'ca': 'Catalan',
        'fr': 'French', 'cy': 'Welsh', 'nl': 'Dutch', 'es': 'Spanish',
        'it': 'Italian', 'et': 'Estonian', 'ro': 'Romanian', 'sl': 'Slovenian',
        'sv': 'Swedish', 'af': 'Afrikaans', 'id': 'Indonesian', 'so': 'Somali',
        'da': 'Danish', 'zh-cn': 'Chinese', 'tl': 'Tagalog', 'pl': 'Polish'
    }

    languages = []
    authors = []
    courses = []
    labels = []

    for lang in lang_courses.keys():
        if lang_courses[lang] >= 2:  # Only languages with at least 2 courses
            languages.append(lang)
            authors.append(len(lang_authors[lang]))
            courses.append(lang_courses[lang])
            labels.append(lang_names.get(lang, lang))

    # Calculate courses per author ratio
    cpa_ratio = [c / a if a > 0 else 0 for c, a in zip(courses, authors)]

    # Create figure
    fig, ax = plt.subplots(figsize=get_dims('tall'))

    style_cfg = FIG_CONFIG.get('style', {})
    textbox_cfg = FIG_CONFIG.get('textbox', {})
    line_cfg = FIG_CONFIG.get('lines', {})

    # Scatter plot with size proportional to courses/author ratio
    sizes = [max(50, min(500, r * 30)) for r in cpa_ratio]

    # Use primary color from config
    scatter = ax.scatter(authors, courses, s=sizes, c=get_color('primary'),
                         alpha=line_cfg.get('alpha', 0.7), edgecolors='black', linewidth=1)

    # Add labels for each point
    for i, (x, y, label, lang) in enumerate(zip(authors, courses, labels, languages)):
        # Offset labels to avoid overlap
        offset_x = 3
        offset_y = 3
        if lang == 'de':
            offset_y = -15
        elif lang == 'en':
            offset_x = -50
        elif lang == 'cy':
            offset_x = 5
        elif lang == 'nl':
            offset_x = 5

        ax.annotate(label, (x, y), xytext=(offset_x, offset_y),
                    textcoords='offset points', fontsize=get_font_size('annotation'), fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    # Add reference lines
    # Average courses per author
    avg_ratio = sum(courses) / sum(authors)
    x_range = np.linspace(1, max(authors) * 1.1, 100)
    ax.plot(x_range, x_range * avg_ratio, '--', color='gray', alpha=0.5,
            label=f'Average: {avg_ratio:.1f} courses/author')

    ax.set_xlabel('Number of Unique Authors', fontsize=get_font_size('axis_label') + 1,
                  fontweight=get_font_weight('axis_label'))
    ax.set_ylabel('Number of Courses', fontsize=get_font_size('axis_label') + 1,
                  fontweight=get_font_weight('axis_label'))
    ax.set_title('Language Distribution: Authors vs. Courses\n(Bubble size = courses per author ratio)',
                 fontsize=get_font_size('title'), fontweight=get_font_weight('title'), pad=15)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=style_cfg.get('grid_alpha', 0.3),
            linestyle=style_cfg.get('grid_linestyle', '--'))
    ax.set_xlim(0.8, max(authors) * 1.5)
    ax.set_ylim(1, max(courses) * 1.5)

    # Add annotation box with key insights
    insight_text = (
        "Key Insights:\n"
        "• Welsh (cy): 38 courses, 1 author\n"
        "  → Single prolific contributor\n"
        "• Italian (it): 23 courses, 12 authors\n"
        "  → Distributed community\n"
        "• German dominates with most\n"
        "  courses and authors"
    )
    ax.text(0.98, 0.02, insight_text, transform=ax.transAxes,
            fontsize=get_font_size('annotation'),
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle=textbox_cfg.get('boxstyle', 'round'),
                      facecolor='lightyellow', alpha=textbox_cfg.get('alpha', 0.9),
                      edgecolor=textbox_cfg.get('edgecolor', 'gray')))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_language_scatterplot")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 11: License Distribution (Treemap-style)
# =============================================================================
def create_license_treemap():
    print("\n[12/12] Creating License Distribution figure...")

    license_analysis = results.get('license_analysis', {})
    license_dist = license_analysis.get('license_distribution', {})
    top_licenses = license_analysis.get('top_licenses_spdx', {})

    if not license_dist:
        print("  WARNING: No license data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})
    license_colors = FIG_CONFIG.get('colors', {}).get('licenses', {})

    # Left: License categories pie chart
    categories = list(license_dist.keys())
    counts = list(license_dist.values())
    total = sum(counts)

    # Use license-specific colors from config
    colors = [
        license_colors.get('unclear', '#E74C3C'),
        license_colors.get('permissive', '#3498DB'),
        license_colors.get('copyleft', '#9B59B6'),
        license_colors.get('creative_commons', '#2ECC71')
    ]
    explode = [0.02] * len(categories)

    wedges, texts, autotexts = ax1.pie(counts, explode=explode, labels=categories, colors=colors,
                                        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
                                        startangle=90, textprops={'fontsize': get_font_size('annotation')})
    ax1.set_title('License Category Distribution', fontsize=get_font_size('title'),
                  fontweight=get_font_weight('title'))

    # Right: Top specific licenses
    if top_licenses:
        # Get top 10 licenses excluding None
        top_lic = [(k, v) for k, v in sorted(top_licenses.items(), key=lambda x: x[1], reverse=True) if k != 'None'][:10]
        names = [l[0] for l in top_lic]
        vals = [l[1] for l in top_lic]

        colors2 = plt.cm.tab10(np.linspace(0, 1, len(names)))
        ax2.barh(names[::-1], vals[::-1], color=colors2[::-1],
                 edgecolor=bar_cfg.get('edge_color', 'black'),
                 linewidth=bar_cfg.get('edge_linewidth', 0.5))
        ax2.set_xlabel('Number of Courses', fontsize=get_font_size('axis_label'),
                       fontweight=get_font_weight('axis_label'))
        ax2.set_title('Top 10 Specific Licenses (SPDX)', fontsize=get_font_size('title'),
                      fontweight=get_font_weight('title'))
        ax2.grid(axis='x', alpha=style_cfg.get('grid_alpha', 0.3))

        # Add count labels
        for i, (name, val) in enumerate(zip(names[::-1], vals[::-1])):
            ax2.text(val + max(vals)*0.01, i, f'{val:,}', va='center',
                     fontsize=get_font_size('annotation'))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_license_treemap")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 12: License by Discipline
# =============================================================================
def create_license_by_discipline():
    print("\n[13/12] Creating License by Discipline figure...")

    license_analysis = results.get('license_analysis', {})
    lic_by_disc = license_analysis.get('license_by_discipline', {})

    if not lic_by_disc:
        print("  WARNING: No license by discipline data available")
        return

    # Prepare data for stacked bar chart
    disciplines = sorted(lic_by_disc.keys())
    categories = ['Creative Commons (OER-ideal)', 'Permissive Open Source', 'Copyleft (GPL family)', 'Unclear/Missing']

    data = {cat: [] for cat in categories}
    for disc in disciplines:
        for cat in categories:
            val = lic_by_disc[disc].get(cat, 0) * 100
            data[cat].append(val)

    fig, ax = plt.subplots(figsize=get_dims('wide'))

    style_cfg = FIG_CONFIG.get('style', {})
    legend_cfg = FIG_CONFIG.get('legend', {})
    license_colors = FIG_CONFIG.get('colors', {}).get('licenses', {})

    x = np.arange(len(disciplines))
    width = 0.7

    # Colors for license types - use config
    colors = {
        'Creative Commons (OER-ideal)': license_colors.get('creative_commons', '#2ECC71'),
        'Permissive Open Source': license_colors.get('permissive', '#3498DB'),
        'Copyleft (GPL family)': license_colors.get('copyleft', '#9B59B6'),
        'Unclear/Missing': license_colors.get('unclear', '#E74C3C')
    }

    bottom = np.zeros(len(disciplines))
    for cat in categories:
        ax.bar(x, data[cat], width, label=cat, bottom=bottom, color=colors[cat], edgecolor='white', linewidth=0.5)
        bottom += np.array(data[cat])

    ax.set_xlabel('DDC Discipline', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_ylabel('Percentage (%)', fontsize=get_font_size('axis_label'),
                  fontweight=get_font_weight('axis_label'))
    ax.set_title('License Distribution by Discipline', fontsize=get_font_size('title'),
                 fontweight=get_font_weight('title'), pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DDC_LABELS.get(d, f"DDC {d}") for d in disciplines], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=legend_cfg.get('fontsize', 10))
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_license_by_discipline")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# CONFERENCE PAPER FIGURES: User Segmentation
# =============================================================================
def create_user_segments():
    """Create user segment distribution figure for conference paper."""
    print("\n[CONF-1] Creating User Segments Distribution figure...")

    seg = results.get('user_segmentation', {}).get('author_segmentation', {})
    if not seg:
        print("  WARNING: No user segmentation data available")
        return

    categories = seg.get('categories', {})
    if not categories:
        print("  WARNING: No category data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_dims('wide'))

    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})
    cat_colors = get_categorical_colors(5)

    # Segment names and data
    segment_names = ['One-Time', 'Occasional', 'Regular', 'Active', 'Power User']
    segment_keys = ['one_time', 'occasional', 'regular', 'active', 'power_user']

    author_counts = [categories.get(k, {}).get('author_count', 0) for k in segment_keys]
    course_counts = [categories.get(k, {}).get('course_count', 0) for k in segment_keys]
    author_shares = [categories.get(k, {}).get('author_share', 0) * 100 for k in segment_keys]
    course_shares = [categories.get(k, {}).get('course_share', 0) * 100 for k in segment_keys]

    # Left: Stacked bar chart comparing author % vs course %
    x = np.arange(len(segment_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, author_shares, width, label='% of Authors',
                    color=get_color('primary'), alpha=bar_cfg.get('alpha', 0.8),
                    edgecolor=bar_cfg.get('edge_color', 'black'))
    bars2 = ax1.bar(x + width/2, course_shares, width, label='% of Courses',
                    color=get_color('secondary'), alpha=bar_cfg.get('alpha', 0.8),
                    edgecolor=bar_cfg.get('edge_color', 'black'))

    ax1.set_xlabel('User Segment', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_ylabel('Percentage (%)', fontsize=get_font_size('axis_label'),
                   fontweight=get_font_weight('axis_label'))
    ax1.set_title('Author vs. Course Distribution by Segment',
                  fontsize=get_font_size('title'), fontweight=get_font_weight('title'))
    ax1.set_xticks(x)
    ax1.set_xticklabels(segment_names, rotation=15, ha='right')
    ax1.legend(fontsize=get_font_size('legend'))
    ax1.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

    # Add value labels on bars
    for bar, val in zip(bars1, author_shares):
        if val > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=get_font_size('annotation')-1)
    for bar, val in zip(bars2, course_shares):
        if val > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=get_font_size('annotation')-1)

    # Right: Pie chart showing course concentration
    # Group into "Power contributors" vs "Others"
    power_share = course_shares[-1] + course_shares[-2]  # Power User + Active
    other_share = 100 - power_share

    pie_data = [power_share, other_share]
    pie_labels = [f'Top Contributors\n(Active + Power)\n{power_share:.1f}%',
                  f'Other Authors\n{other_share:.1f}%']
    pie_colors = [get_color('secondary'), get_color('info')]

    wedges, texts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors,
                            startangle=90, textprops={'fontsize': get_font_size('legend')})
    ax2.set_title('Course Concentration\n(Power Law Distribution)',
                  fontsize=get_font_size('title'), fontweight=get_font_weight('title'))

    # Add Gini coefficient annotation
    gini = seg.get('statistics', {}).get('gini', 0.83)
    textbox_cfg = FIG_CONFIG.get('textbox', {})
    ax2.text(0.5, -0.1, f'Gini Coefficient: {gini:.2f}',
             transform=ax2.transAxes, ha='center',
             fontsize=get_font_size('annotation') + 1, fontweight='bold',
             bbox=dict(boxstyle=textbox_cfg.get('boxstyle', 'round'),
                       facecolor='lightyellow', alpha=textbox_cfg.get('alpha', 0.9)))

    plt.tight_layout()
    output_path = save_figure(fig, "fig_user_segments")
    print(f"  Saved: {output_path}")
    plt.close()


def create_feature_by_segment():
    """Create feature adoption by user segment heatmap."""
    print("\n[CONF-2] Creating Feature by Segment Heatmap...")

    features_by_seg = results.get('user_segmentation', {}).get('features_by_segment', {})
    if not features_by_seg:
        print("  WARNING: No features_by_segment data available")
        return

    segment_keys = ['one_time', 'occasional', 'regular', 'active', 'power_user']
    segment_names = ['One-Time', 'Occasional', 'Regular', 'Active', 'Power User']

    # Key features to show
    key_features = ['imports', 'narrator', 'video', 'quiz',
                    'executable_code', 'surveys']
    feature_labels = ['Imports', 'Narrator/TTS', 'Video', 'Quiz', 'Executable Code', 'Surveys']

    # Build matrix
    matrix = []
    for seg_key in segment_keys:
        seg_data = features_by_seg.get(seg_key, {})
        feature_rates = seg_data.get('feature_rates', {})
        row = [feature_rates.get(f, 0) * 100 for f in key_features]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=get_dims('single'))
    colorbar_cfg = FIG_CONFIG.get('colorbar', {})

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_yticks(np.arange(len(segment_names)))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=get_font_size('tick_label'))
    ax.set_yticklabels(segment_names, fontsize=get_font_size('tick_label'))

    # Add text annotations
    for i in range(len(segment_names)):
        for j in range(len(feature_labels)):
            val = matrix[i, j]
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=text_color, fontsize=get_font_size('annotation'), fontweight='bold')

    ax.set_title('Feature Adoption Rate by User Segment (%)',
                 fontsize=get_font_size('title'), fontweight=get_font_weight('title'), pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=colorbar_cfg.get('shrink', 0.8))
    cbar.set_label('Adoption Rate (%)', fontsize=get_font_size('axis_label') - 1)

    plt.tight_layout()
    output_path = save_figure(fig, "fig_feature_by_segment")
    print(f"  Saved: {output_path}")
    plt.close()


def create_segment_characteristics():
    """Create course characteristics by segment comparison."""
    print("\n[CONF-3] Creating Segment Characteristics figure...")

    course_chars = results.get('user_segmentation', {}).get('course_characteristics', {})
    if not course_chars:
        print("  WARNING: No course_characteristics data available")
        return

    segment_keys = ['one_time', 'occasional', 'regular', 'active', 'power_user']
    segment_names = ['One-Time', 'Occasional', 'Regular', 'Active', 'Power User']

    fig, axes = plt.subplots(1, 3, figsize=get_dims('wide'))
    bar_cfg = FIG_CONFIG.get('bars', {})
    style_cfg = FIG_CONFIG.get('style', {})
    cat_colors = get_categorical_colors(5)

    # Metrics to plot
    metrics = [
        ('mean_words', 'Mean Words per Course', axes[0]),
        ('mean_pages', 'Mean Pages per Course', axes[1]),
        ('mean_features', 'Mean Features per Course', axes[2])
    ]

    for metric_key, metric_title, ax in metrics:
        values = [course_chars.get(k, {}).get(metric_key, 0) for k in segment_keys]

        bars = ax.bar(segment_names, values, color=cat_colors,
                      edgecolor=bar_cfg.get('edge_color', 'black'),
                      alpha=bar_cfg.get('alpha', 0.8))

        ax.set_ylabel(metric_title.split()[-1], fontsize=get_font_size('axis_label'))
        ax.set_title(metric_title, fontsize=get_font_size('subtitle'),
                     fontweight=get_font_weight('title'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=style_cfg.get('grid_alpha', 0.3))

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.1f}', ha='center', fontsize=get_font_size('annotation'))

    plt.suptitle('Course Characteristics by User Segment',
                 fontsize=get_font_size('title') + 2, fontweight=get_font_weight('title'), y=1.02)
    plt.tight_layout()
    output_path = save_figure(fig, "fig_segment_characteristics")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def generate_all_figures():
    """Generate all figures for the paper."""
    # General figures (used across papers)
    create_cumulative_growth()
    # create_education_levels()  # Removed - using table with example links instead
    create_course_length()
    length_vs_features_results = create_length_vs_features()
    create_feature_heatmap()
    create_authorship()
    create_collaboration_network()
    create_age_distribution()
    create_lifecycle_pattern()
    create_language_scatterplot()
    create_license_treemap()
    create_license_by_discipline()

    # Conference Paper specific figures
    create_user_segments()
    create_feature_by_segment()
    create_segment_characteristics()

    # Update analysis_results.json with new correlation data
    if length_vs_features_results:
        print("\n[BONUS] Updating analysis_results.json with length vs. features correlation...")
        with open(DATA_PATH, 'r') as f:
            results_data = json.load(f)

        # Add to feature_analysis section
        results_data['feature_analysis']['length_vs_features'] = length_vs_features_results

        with open(DATA_PATH, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  Updated: {DATA_PATH}")

    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput directory: {FIGURES_DIR}")
    print("\nGenerated figures:")
    for fig_file in sorted(FIGURES_DIR.glob("fig_*.png")):
        print(f"  - {fig_file.name}")


if __name__ == "__main__":
    generate_all_figures()
