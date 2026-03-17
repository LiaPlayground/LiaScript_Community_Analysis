"""
Collaboration Analysis
Analyzes authorship and collaboration patterns in LiaScript courses.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes collaboration patterns.

    Args:
        df: Main dataset
        config: Configuration dictionary

    Returns:
        Dictionary with collaboration analysis results
    """
    logger.info("Running collaboration analysis...")

    results = {}

    if 'author_count' not in df.columns:
        logger.warning("author_count column not found")
        return {'error': 'Collaboration data not available'}

    # 1. Authorship Distribution
    author_counts = df['author_count'].dropna()
    results['authorship_distribution'] = {
        'total_courses': int(len(author_counts)),
        'single_author': int((author_counts == 1).sum()),
        'multi_author': int((author_counts > 1).sum()),
        'single_author_rate': float((author_counts == 1).mean()),
        'multi_author_rate': float((author_counts > 1).mean()),
        'mean_authors': float(author_counts.mean()),
        'median_authors': float(author_counts.median()),
        'max_authors': int(author_counts.max())
    }

    # Distribution by author count
    author_dist = author_counts.value_counts().sort_index()
    results['authors_per_course_distribution'] = {
        int(k): int(v) for k, v in author_dist.items()
    }

    # 2. Collaboration vs. Course Complexity
    if 'pipe:content_words' in df.columns:
        # Correlation between author count and course length
        valid_data = df[['author_count', 'pipe:content_words']].dropna()

        if len(valid_data) > 10:
            corr, p_value = spearmanr(valid_data['author_count'], valid_data['pipe:content_words'])
            results['collaboration_vs_length'] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'interpretation': 'positive' if corr > 0 else 'negative' if corr < 0 else 'none'
            }

            # Compare median course length
            single_author_words = df[df['author_count'] == 1]['pipe:content_words'].dropna()
            multi_author_words = df[df['author_count'] > 1]['pipe:content_words'].dropna()

            results['length_by_authorship'] = {
                'single_author_median': float(single_author_words.median()) if len(single_author_words) > 0 else None,
                'multi_author_median': float(multi_author_words.median()) if len(multi_author_words) > 0 else None
            }

    # 3. Collaboration vs. Feature Diversity
    feature_cols = [col for col in df.columns if col.startswith('feature:has_')]
    if feature_cols:
        df_temp = df.copy()
        df_temp['feature_count'] = df_temp[feature_cols].sum(axis=1)

        valid_data = df_temp[['author_count', 'feature_count']].dropna()

        if len(valid_data) > 10:
            corr, p_value = spearmanr(valid_data['author_count'], valid_data['feature_count'])
            results['collaboration_vs_features'] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05)
            }

    # 4. Commit Activity
    if 'commit_count' in df.columns:
        commits = df['commit_count'].dropna()
        results['commit_activity'] = {
            'mean_commits': float(commits.mean()),
            'median_commits': float(commits.median()),
            'max_commits': int(commits.max()),
            'courses_with_multiple_commits': int((commits > 1).sum())
        }

        # Correlation: authors vs. commits
        valid_data = df[['author_count', 'commit_count']].dropna()
        if len(valid_data) > 10:
            corr, p_value = spearmanr(valid_data['author_count'], valid_data['commit_count'])
            results['authors_vs_commits'] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05)
            }

    # 5. Collaboration by Discipline
    if 'ddc_toplevel' in df.columns and df['ddc_toplevel'].notna().sum() > 0:
        collab_by_discipline = {}

        for ddc in df['ddc_toplevel'].dropna().unique():
            ddc_df = df[df['ddc_toplevel'] == ddc]
            ddc_authors = ddc_df['author_count'].dropna()

            if len(ddc_authors) > 0:
                collab_by_discipline[str(ddc)] = {
                    'total_courses': int(len(ddc_authors)),
                    'multi_author_rate': float((ddc_authors > 1).mean()),
                    'mean_authors': float(ddc_authors.mean())
                }

        results['collaboration_by_discipline'] = collab_by_discipline

    # 6. Most Collaborative Authors (if contributors_list available)
    if 'contributors_list' in df.columns:
        # Count how many courses each author contributed to
        all_contributors = []
        for contrib_list in df['contributors_list'].dropna():
            if isinstance(contrib_list, list):
                all_contributors.extend(contrib_list)

        if all_contributors:
            from collections import Counter
            contrib_counts = Counter(all_contributors)
            top_contributors = contrib_counts.most_common(20)

            results['top_contributors'] = {
                author: int(count) for author, count in top_contributors
            }

    logger.info("Collaboration analysis complete")
    return results
