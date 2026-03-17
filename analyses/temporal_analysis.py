"""
Temporal Analysis
Analyzes temporal patterns, adoption trends, and lifecycle.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime, timedelta
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes temporal patterns and lifecycle.

    Args:
        df: Main dataset
        config: Configuration dictionary

    Returns:
        Dictionary with temporal analysis results
    """
    logger.info("Running temporal analysis...")

    results = {}
    results['status'] = 'implemented_full'

    # 1. Adoption Timeline (using first_commit_year = actual course creation date)
    # Note: created_year is the repository creation date, not the course creation
    timeline_col = 'first_commit_year' if 'first_commit_year' in df.columns else 'created_year'
    if timeline_col in df.columns:
        yearly_counts = df[timeline_col].value_counts().sort_index()
        results['adoption_timeline'] = {
            int(year): int(count) for year, count in yearly_counts.items() if pd.notna(year)
        }

        # Cumulative adoption
        cumulative = yearly_counts.sort_index().cumsum()
        results['cumulative_adoption'] = {
            int(year): int(count) for year, count in cumulative.items()
        }

    # 2. Age Distribution
    if 'age_years' in df.columns:
        age = df['age_years'].dropna()
        results['age_distribution'] = {
            'mean_years': float(age.mean()),
            'median_years': float(age.median()),
            'min_years': float(age.min()),
            'max_years': float(age.max()),
            'less_than_1_year': int((age < 1).sum()),
            'less_than_1_year_pct': float((age < 1).mean()),
            'less_than_6_months': int((age < 0.5).sum()),
            'less_than_6_months_pct': float((age < 0.5).mean())
        }

    # 3. Lifecycle Metrics
    if 'lifespan_years' in df.columns:
        lifespan = df['lifespan_years'].dropna()
        results['lifecycle_metrics'] = {
            'mean_years': float(lifespan.mean()),
            'median_years': float(lifespan.median()),
            'less_than_1_month': int((lifespan < (1/12)).sum()),
            'less_than_1_month_pct': float((lifespan < (1/12)).mean()),
            'one_shot_courses': int((lifespan <= (7/365)).sum()),
            'one_shot_pct': float((lifespan <= (7/365)).mean())
        }

    # 4. Commit Distribution
    if 'total_commits' in df.columns:
        commits = df['total_commits'].dropna()
        results['commit_distribution'] = {
            'mean_commits': float(commits.mean()),
            'median_commits': float(commits.median()),
            'max_commits': int(commits.max()),
            'single_commit': int((commits == 1).sum()),
            'single_commit_pct': float((commits == 1).mean()),
            'five_or_less': int((commits <= 5).sum()),
            'five_or_less_pct': float((commits <= 5).mean()),
            'fifty_or_more': int((commits >= 50).sum()),
            'fifty_or_more_pct': float((commits >= 50).mean())
        }

    # 5. Activity Status (enhanced)
    if 'months_since_update' in df.columns:
        months = df['months_since_update'].dropna()
        results['activity_status'] = {
            'recently_active': int((months <= 6).sum()),
            'active_rate': float((months <= 6).mean()),
            'stale': int(((months > 12) & (months <= 24)).sum()),
            'stale_rate': float(((months > 12) & (months <= 24)).mean()),
            'abandoned': int((months > 24).sum()),
            'abandoned_rate': float((months > 24).mean())
        }
    elif 'is_recently_active' in df.columns:
        results['activity_status'] = {
            'recently_active': int(df['is_recently_active'].sum()),
            'inactive': int((~df['is_recently_active']).sum()),
            'active_rate': float(df['is_recently_active'].mean())
        }

    # 6. Sample Quality Metrics
    if 'total_commits' in df.columns:
        courses_with_history = df['total_commits'].notna().sum()
        total_courses = len(df)
        results['sample_quality'] = {
            'courses_with_history': int(courses_with_history),
            'total_courses': int(total_courses),
            'coverage_rate': float(courses_with_history / total_courses) if total_courses > 0 else 0
        }

    # 7. Stars Analysis (RQ4.5: Community Recognition and Longevity)
    if 'stars' in df.columns:
        stars_results = {}
        stars = df['stars'].dropna()

        # 7a. Stars vs. Lifespan correlation
        if 'lifespan_years' in df.columns:
            valid_mask = df['stars'].notna() & df['lifespan_years'].notna()
            if valid_mask.sum() > 10:
                stars_data = df.loc[valid_mask, 'stars']
                lifespan_data = df.loc[valid_mask, 'lifespan_years']
                corr, p_val = spearmanr(stars_data, lifespan_data)
                stars_results['stars_vs_lifespan'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'is_significant': p_val < 0.05,
                    'n_samples': int(valid_mask.sum())
                }

        # 7b. Stars vs. Commit Count correlation
        if 'total_commits' in df.columns:
            valid_mask = df['stars'].notna() & df['total_commits'].notna()
            if valid_mask.sum() > 10:
                stars_data = df.loc[valid_mask, 'stars']
                commits_data = df.loc[valid_mask, 'total_commits']
                corr, p_val = spearmanr(stars_data, commits_data)
                stars_results['stars_vs_commits'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'is_significant': p_val < 0.05,
                    'n_samples': int(valid_mask.sum())
                }

        # 7c. Stars by Activity Status
        if 'months_since_update' in df.columns:
            stars_by_status = {}
            valid_df = df[df['stars'].notna() & df['months_since_update'].notna()]

            # Active (< 6 months)
            active_mask = valid_df['months_since_update'] <= 6
            if active_mask.sum() > 0:
                active_stars = valid_df.loc[active_mask, 'stars']
                stars_by_status['active'] = {
                    'median': float(active_stars.median()),
                    'mean': float(active_stars.mean()),
                    'n': int(active_mask.sum())
                }

            # Stale (6-24 months)
            stale_mask = (valid_df['months_since_update'] > 6) & (valid_df['months_since_update'] <= 24)
            if stale_mask.sum() > 0:
                stale_stars = valid_df.loc[stale_mask, 'stars']
                stars_by_status['stale'] = {
                    'median': float(stale_stars.median()),
                    'mean': float(stale_stars.mean()),
                    'n': int(stale_mask.sum())
                }

            # Abandoned (> 24 months)
            abandoned_mask = valid_df['months_since_update'] > 24
            if abandoned_mask.sum() > 0:
                abandoned_stars = valid_df.loc[abandoned_mask, 'stars']
                stars_by_status['abandoned'] = {
                    'median': float(abandoned_stars.median()),
                    'mean': float(abandoned_stars.mean()),
                    'n': int(abandoned_mask.sum())
                }

            if stars_by_status:
                stars_results['stars_by_activity_status'] = stars_by_status

        if stars_results:
            results['stars_analysis'] = stars_results
            logger.info(f"Stars analysis complete - {len(stars_results)} sub-metrics computed")

    logger.info(f"Temporal analysis complete - {len(results)} metrics computed")
    return results
