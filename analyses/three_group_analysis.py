"""
Three-Group Segmentation Analysis
Compares Internal, MINT-the-GAP, and Community courses.

Groups:
  Internal (7 accounts):  Core developers — Sebastian Zug & André Dietrich,
                           plus organisational accounts they maintain.
  MINT-the-GAP (1 account): Highly productive school-focused power user
                             (Martin Lommatzsch), collaborates with the team
                             but has a distinct, template-heavy usage profile.
  Community (all others):   Independent adopters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy.stats import spearmanr, kruskal
import logging

logger = logging.getLogger(__name__)

# ── group definitions (by repo_user / GitHub org) ───────────────────────
INTERNAL_ACCOUNTS = [
    'SebastianZug',
    'andre-dietrich',
    'LiaPlayground',
    'LiaScript',
    'LiaBooks',
    'LiaTemplates',
    'TUBAF-IfI-LiaScript',
]

MINT_THE_GAP_ACCOUNTS = [
    'MINT-the-GAP',
]

GROUP_LABELS = {
    'internal': 'Internal',
    'mint_the_gap': 'MINT-the-GAP',
    'community': 'Community',
}


def _assign_group(repo_user: str) -> str:
    """Map a repo_user value to one of the three groups."""
    if repo_user in INTERNAL_ACCOUNTS:
        return 'internal'
    if repo_user in MINT_THE_GAP_ACCOUNTS:
        return 'mint_the_gap'
    return 'community'


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Three-group segmentation: Internal vs MINT-the-GAP vs Community.

    Args:
        df: Main dataset (validated courses)
        config: Configuration dictionary

    Returns:
        Dictionary with three-group analysis results
    """
    logger.info("Running three-group segmentation analysis...")

    if 'repo_user' not in df.columns:
        logger.warning("repo_user column not found")
        return {'error': 'repo_user column not available'}

    results = {}

    # ── 1. Assign groups ────────────────────────────────────────────────
    df = df.copy()
    df['_group'] = df['repo_user'].apply(_assign_group)

    group_sizes = df['_group'].value_counts().to_dict()
    unique_users = df.groupby('_group')['repo_user'].nunique().to_dict()

    results['group_overview'] = {
        g: {
            'label': GROUP_LABELS[g],
            'courses': int(group_sizes.get(g, 0)),
            'accounts': int(unique_users.get(g, 0)),
        }
        for g in GROUP_LABELS
    }
    results['group_overview']['total_courses'] = int(len(df))

    # ── 2. Feature usage per group ──────────────────────────────────────
    flag_cols = sorted([c for c in df.columns if c.startswith('feature:has_')])
    if not flag_cols:
        logger.warning("No feature:has_ columns found")
        return results

    feature_rates: Dict[str, Dict[str, float]] = {}
    for group in GROUP_LABELS:
        gdf = df[df['_group'] == group]
        if len(gdf) == 0:
            continue
        rates = {}
        for col in flag_cols:
            fname = col.replace('feature:has_', '')
            rates[fname] = round(float(gdf[col].mean()), 4)
        feature_rates[group] = rates

    results['feature_rates_by_group'] = feature_rates

    # ── 3. Feature diversity (count of distinct features used) ──────────
    df['_feature_count'] = df[flag_cols].sum(axis=1)
    diversity: Dict[str, Any] = {}
    for group in GROUP_LABELS:
        gdf = df[df['_group'] == group]
        if len(gdf) == 0:
            continue
        fc = gdf['_feature_count']
        diversity[group] = {
            'mean': round(float(fc.mean()), 2),
            'median': float(fc.median()),
            'std': round(float(fc.std()), 2),
            'max': int(fc.max()),
        }
    results['feature_diversity'] = diversity

    # Kruskal-Wallis test across groups
    groups_data = [
        df[df['_group'] == g]['_feature_count'].values
        for g in GROUP_LABELS if (df['_group'] == g).any()
    ]
    if len(groups_data) >= 2 and all(len(g) > 0 for g in groups_data):
        stat, p = kruskal(*groups_data)
        results['feature_diversity_kruskal'] = {
            'statistic': round(float(stat), 4),
            'p_value': float(p),
            'is_significant': bool(p < 0.05),
        }

    # ── 4. Course characteristics per group ─────────────────────────────
    char_results: Dict[str, Dict[str, Any]] = {}
    for group in GROUP_LABELS:
        gdf = df[df['_group'] == group]
        if len(gdf) == 0:
            continue
        chars: Dict[str, Any] = {}

        if 'pipe:content_words' in gdf.columns:
            w = gdf['pipe:content_words'].dropna()
            if len(w) > 0:
                chars['median_words'] = float(w.median())
                chars['mean_words'] = round(float(w.mean()), 1)

        if 'pipe:content_pages' in gdf.columns:
            p = gdf['pipe:content_pages'].dropna()
            if len(p) > 0:
                chars['median_pages'] = float(p.median())
                chars['mean_pages'] = round(float(p.mean()), 1)

        if 'commit_count' in gdf.columns:
            c = gdf['commit_count'].dropna()
            if len(c) > 0:
                chars['median_commits'] = float(c.median())
                chars['mean_commits'] = round(float(c.mean()), 1)

        if 'author_count' in gdf.columns:
            a = gdf['author_count'].dropna()
            if len(a) > 0:
                chars['mean_authors'] = round(float(a.mean()), 2)

        char_results[group] = chars
    results['course_characteristics'] = char_results

    # ── 5. Top distinguishing features ──────────────────────────────────
    # For each group: features where that group's rate is most above the
    # overall rate (= features that characterise the group).
    overall_rates = {
        col.replace('feature:has_', ''): float(df[col].mean())
        for col in flag_cols
    }

    distinguishing: Dict[str, List[Dict[str, Any]]] = {}
    for group in GROUP_LABELS:
        if group not in feature_rates:
            continue
        diffs = []
        for fname, rate in feature_rates[group].items():
            overall = overall_rates.get(fname, 0)
            if overall > 0.01:  # skip extremely rare features
                ratio = rate / overall
                diffs.append({
                    'feature': fname,
                    'group_rate': round(rate, 4),
                    'overall_rate': round(overall, 4),
                    'ratio': round(ratio, 2),
                })
        diffs.sort(key=lambda x: x['ratio'], reverse=True)
        distinguishing[group] = diffs[:10]

    results['distinguishing_features'] = distinguishing

    # ── 6. Interactivity profile per group ──────────────────────────────
    interactivity_keywords = [
        'animation', 'tts', 'quiz', 'executable', 'classroom', 'survey'
    ]
    interactive_cols = [
        c for c in flag_cols
        if any(k in c for k in interactivity_keywords)
    ]
    if interactive_cols:
        df['_interactivity'] = df[interactive_cols].sum(axis=1)
        inter_results: Dict[str, Dict[str, Any]] = {}
        for group in GROUP_LABELS:
            gdf = df[df['_group'] == group]
            if len(gdf) == 0:
                continue
            ic = gdf['_interactivity']
            inter_results[group] = {
                'mean': round(float(ic.mean()), 2),
                'median': float(ic.median()),
                'pct_with_3plus': round(float((ic >= 3).mean()) * 100, 1),
                'pct_with_none': round(float((ic == 0).mean()) * 100, 1),
            }
        results['interactivity_profile'] = inter_results

    # ── 7. Template & extension usage ───────────────────────────────────
    ext_features = ['imports', 'external_scripts', 'external_css', 'macros']
    ext_results: Dict[str, Dict[str, float]] = {}
    for group in GROUP_LABELS:
        if group not in feature_rates:
            continue
        ext_results[group] = {
            f: feature_rates[group].get(f, 0.0)
            for f in ext_features
            if f in feature_rates[group]
        }
    if ext_results:
        results['extensibility_profile'] = ext_results

    # ── 8. Education level distribution per group ───────────────────────
    if 'ai:education_level' in df.columns:
        edu_results: Dict[str, Dict[str, int]] = {}
        for group in GROUP_LABELS:
            gdf = df[df['_group'] == group]
            if len(gdf) == 0:
                continue
            edu = gdf['ai:education_level'].value_counts()
            edu_results[group] = {str(k): int(v) for k, v in edu.items()}
        results['education_by_group'] = edu_results

    # ── 9. Discipline distribution per group ────────────────────────────
    if 'ddc_toplevel' in df.columns and df['ddc_toplevel'].notna().sum() > 0:
        disc_results: Dict[str, Dict[str, int]] = {}
        for group in GROUP_LABELS:
            gdf = df[df['_group'] == group]
            if len(gdf) == 0:
                continue
            ddc = gdf['ddc_toplevel'].value_counts()
            disc_results[group] = {str(k): int(v) for k, v in ddc.items()}
        results['discipline_by_group'] = disc_results

    logger.info("Three-group segmentation analysis complete")
    return results
