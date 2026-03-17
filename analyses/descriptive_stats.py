"""
Descriptive Statistics Analysis
Generates comprehensive descriptive statistics for the LiaScript corpus.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ISO 639-1 to full language name mapping
LANGUAGE_NAMES = {
    'de': 'German',
    'en': 'English',
    'pt': 'Portuguese',
    'ca': 'Catalan',
    'fr': 'French',
    'cy': 'Welsh',
    'nl': 'Dutch',
    'es': 'Spanish',
    'it': 'Italian',
    'et': 'Estonian',
    'pl': 'Polish',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
}


def get_language_name(code: str) -> str:
    """Convert ISO 639-1 language code to full name."""
    return LANGUAGE_NAMES.get(code, code)


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs descriptive statistical analysis.

    Args:
        df: Main dataset
        config: Configuration dictionary

    Returns:
        Dictionary with descriptive statistics
    """
    logger.info("Running descriptive statistics analysis...")

    results = {}

    # 1. Corpus Overview
    results['corpus'] = {
        'total_courses': int(len(df)),
        'total_repositories': int(df['repo_url'].nunique()) if 'repo_url' in df.columns else None,
        'total_authors': int(df['repo_user'].nunique()) if 'repo_user' in df.columns else None,
        'validated_rate': float(df['pipe:is_valid_liascript'].mean()) if 'pipe:is_valid_liascript' in df.columns else None
    }

    # 2. Content Statistics
    if 'pipe:content_words' in df.columns:
        words = df['pipe:content_words'].dropna()
        results['content_length'] = {
            'mean_words': float(words.mean()),
            'median_words': float(words.median()),
            'std_words': float(words.std()),
            'min_words': int(words.min()),
            'max_words': int(words.max()),
            'q25': float(words.quantile(0.25)),
            'q75': float(words.quantile(0.75))
        }

    if 'pipe:content_pages' in df.columns:
        pages = df['pipe:content_pages'].dropna()
        results['content_pages'] = {
            'mean_pages': float(pages.mean()),
            'median_pages': float(pages.median()),
            'total_pages': int(pages.sum())
        }

    # 3. Language Distribution
    if 'pipe:most_prob_language' in df.columns:
        lang_dist = df['pipe:most_prob_language'].value_counts().head(10)
        results['language_distribution'] = {
            str(k): int(v) for k, v in lang_dist.items()
        }
        results['language_diversity'] = {
            'unique_languages': int(df['pipe:most_prob_language'].nunique()),
            'dominant_language': get_language_name(str(lang_dist.index[0])) if len(lang_dist) > 0 else None,
            'dominant_language_pct': float(lang_dist.iloc[0] / len(df)) if len(lang_dist) > 0 else None
        }

    # 4. Declared vs. Detected Language
    if 'language_first' in df.columns and 'pipe:most_prob_language' in df.columns:
        declared = df['language_first'].dropna()
        detected = df.loc[declared.index, 'pipe:most_prob_language']
        match_rate = (declared == detected).mean()
        results['language_agreement'] = {
            'match_rate': float(match_rate),
            'declared_available_rate': float(df['language_first'].notna().mean())
        }

    # 5. Repository Statistics
    if 'stars' in df.columns:
        stars = df['stars'].dropna()
        results['popularity'] = {
            'mean_stars': float(stars.mean()),
            'median_stars': float(stars.median()),
            'max_stars': int(stars.max()),
            'courses_with_stars': int((stars > 0).sum())
        }

    if 'forks' in df.columns:
        forks = df['forks'].dropna()
        results['reuse'] = {
            'mean_forks': float(forks.mean()),
            'median_forks': float(forks.median()),
            'max_forks': int(forks.max()),
            'courses_with_forks': int((forks > 0).sum())
        }

    # 6. DDC Distribution (if available)
    if 'ddc_toplevel' in df.columns:
        ddc_dist = df['ddc_toplevel'].value_counts().head(10)
        results['ddc_distribution'] = {
            str(k): int(v) for k, v in ddc_dist.items()
        }
        results['ddc_coverage'] = {
            'courses_with_ddc': int(df['ddc_toplevel'].notna().sum()),
            'coverage_rate': float(df['ddc_toplevel'].notna().mean())
        }

    # 7. Institutional vs. Individual
    if 'internal' in df.columns:
        results['repository_type'] = {
            'institutional': int((df['internal'] == True).sum()),
            'individual': int((df['internal'] == False).sum()),
            'institutional_rate': float((df['internal'] == True).mean())
        }

    # 8. Author Concentration Analysis
    results['author_concentration'] = _calculate_author_concentration(df)

    # 9. Top Authors
    results['top_authors'] = _get_top_authors(df, n=10)

    # 10. Education Level Distribution (RQ1.5)
    if 'ai:education_level' in df.columns:
        edu_dist = df['ai:education_level'].value_counts()
        results['education_levels'] = {
            str(k): int(v) for k, v in edu_dist.items()
        }
        results['education_level_coverage'] = {
            'courses_with_education_level': int(df['ai:education_level'].notna().sum()),
            'coverage_rate': float(df['ai:education_level'].notna().mean())
        }
        logger.info(f"Education levels: {len(edu_dist)} categories found")

    # 11. Target Audience (RQ1.5)
    if 'ai:target_audience' in df.columns:
        results['target_audience_coverage'] = {
            'courses_with_target_audience': int(df['ai:target_audience'].notna().sum()),
            'coverage_rate': float(df['ai:target_audience'].notna().mean())
        }

    logger.info(f"Descriptive statistics complete: {len(results)} categories analyzed")
    return results


def _parse_contributors_list(contributors_str) -> list:
    """
    Parse contributors list from various formats.

    Args:
        contributors_str: String representation of contributors

    Returns:
        List of author names
    """
    import ast

    # Handle None, NaN, and empty values
    if contributors_str is None:
        return []
    try:
        if pd.isna(contributors_str):
            return []
    except (ValueError, TypeError):
        # pd.isna can fail on arrays/lists
        pass

    contrib_str = str(contributors_str).strip()

    # Try to parse as Python list
    try:
        if contrib_str.startswith('['):
            authors = ast.literal_eval(contrib_str)
            if isinstance(authors, list):
                return [str(a).strip() for a in authors if a]
    except (ValueError, SyntaxError):
        pass

    # Fall back to comma-separated parsing
    authors = [a.strip().strip("'\"[]") for a in contrib_str.split(',')]
    return [a for a in authors if a and a != 'nan' and a != 'unknown']


def _calculate_author_concentration(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate author concentration metrics.

    Analyzes the distribution of courses per author and computes
    concentration metrics including Gini coefficient.

    Args:
        df: Main dataset

    Returns:
        Dictionary with concentration metrics
    """
    from collections import Counter

    # Determine which column to use for contributors
    if 'contributors_list' in df.columns:
        contrib_col = 'contributors_list'
    elif 'repo_user' in df.columns:
        contrib_col = 'repo_user'
    else:
        logger.warning("No author column found for concentration analysis")
        return {}

    # Count unique courses per author
    author_courses = {}

    for idx, row in df.iterrows():
        course_id = row.get('pipe:ID', idx)

        if contrib_col == 'contributors_list':
            authors = _parse_contributors_list(row.get(contrib_col))
        else:
            author = row.get(contrib_col)
            authors = [author] if pd.notna(author) and author != 'unknown' else []

        for author in authors:
            if author not in author_courses:
                author_courses[author] = set()
            author_courses[author].add(course_id)

    if not author_courses:
        return {}

    # Calculate course counts per author
    course_counts = {author: len(courses) for author, courses in author_courses.items()}
    counts_list = sorted(course_counts.values(), reverse=True)

    total_authors = len(course_counts)
    total_courses = len(df)

    # Calculate cumulative shares
    cumsum = np.cumsum(counts_list)

    # Gini coefficient
    n = len(counts_list)
    sorted_asc = sorted(counts_list)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(sorted_asc)) - (n + 1) * sum(sorted_asc)) / (n * sum(sorted_asc))

    # Author categories
    categories = {
        'tester': sum(1 for c in counts_list if c == 1),
        'occasional': sum(1 for c in counts_list if 2 <= c <= 5),
        'regular': sum(1 for c in counts_list if 6 <= c <= 20),
        'active': sum(1 for c in counts_list if 21 <= c <= 50),
        'power_user': sum(1 for c in counts_list if c > 50)
    }

    # Course contributions per category
    course_contributions = {
        'tester_courses': sum(c for c in counts_list if c == 1),
        'occasional_courses': sum(c for c in counts_list if 2 <= c <= 5),
        'regular_courses': sum(c for c in counts_list if 6 <= c <= 20),
        'active_courses': sum(c for c in counts_list if 21 <= c <= 50),
        'power_user_courses': sum(c for c in counts_list if c > 50)
    }

    # Use sum of contributed courses as the base for percentage calculations
    total_contributed_courses = sum(counts_list)

    return {
        'total_unique_authors': total_authors,
        'total_courses_with_authors': total_contributed_courses,
        'top5_share': float(cumsum[min(4, len(cumsum) - 1)] / total_contributed_courses) if len(cumsum) >= 5 else None,
        'top10_share': float(cumsum[min(9, len(cumsum) - 1)] / total_contributed_courses) if len(cumsum) >= 10 else None,
        'top1pct_share': float(cumsum[max(0, int(total_authors * 0.01) - 1)] / total_contributed_courses),
        'top5pct_share': float(cumsum[max(0, int(total_authors * 0.05) - 1)] / total_contributed_courses),
        'top10pct_share': float(cumsum[max(0, int(total_authors * 0.10) - 1)] / total_contributed_courses),
        'single_course_authors': categories['tester'],
        'single_course_rate': float(categories['tester'] / total_authors),
        'gini_coefficient': float(gini),
        'median_courses_per_author': float(np.median(counts_list)),
        'mean_courses_per_author': float(np.mean(counts_list)),
        'max_courses_per_author': int(max(counts_list)),
        # Percentiles
        'p25_courses': float(np.percentile(counts_list, 25)),
        'p50_courses': float(np.percentile(counts_list, 50)),
        'p75_courses': float(np.percentile(counts_list, 75)),
        'p90_courses': float(np.percentile(counts_list, 90)),
        'p95_courses': float(np.percentile(counts_list, 95)),
        # Author categories
        'author_categories': categories,
        'author_category_rates': {
            'tester_rate': float(categories['tester'] / total_authors),
            'occasional_rate': float(categories['occasional'] / total_authors),
            'regular_rate': float(categories['regular'] / total_authors),
            'active_rate': float(categories['active'] / total_authors),
            'power_user_rate': float(categories['power_user'] / total_authors)
        },
        # Course contributions per category
        'course_contributions': course_contributions,
        'course_contribution_rates': {
            'tester_course_rate': float(course_contributions['tester_courses'] / total_contributed_courses),
            'occasional_course_rate': float(course_contributions['occasional_courses'] / total_contributed_courses),
            'regular_course_rate': float(course_contributions['regular_courses'] / total_contributed_courses),
            'active_course_rate': float(course_contributions['active_courses'] / total_contributed_courses),
            'power_user_course_rate': float(course_contributions['power_user_courses'] / total_contributed_courses)
        }
    }


def _get_top_authors(df: pd.DataFrame, n: int = 10) -> Dict[str, int]:
    """
    Get top N authors by course count.

    Args:
        df: Main dataset
        n: Number of top authors to return

    Returns:
        Dictionary mapping author names to course counts
    """
    # Determine which column to use
    if 'contributors_list' in df.columns:
        contrib_col = 'contributors_list'
    elif 'repo_user' in df.columns:
        contrib_col = 'repo_user'
    else:
        return {}

    # Count unique courses per author
    author_courses = {}

    for idx, row in df.iterrows():
        course_id = row.get('pipe:ID', idx)

        if contrib_col == 'contributors_list':
            authors = _parse_contributors_list(row.get(contrib_col))
        else:
            author = row.get(contrib_col)
            authors = [author] if pd.notna(author) and author != 'unknown' else []

        for author in authors:
            if author not in author_courses:
                author_courses[author] = set()
            author_courses[author].add(course_id)

    # Sort by count and return top N (excluding 'unknown')
    sorted_authors = sorted(
        [(author, len(courses)) for author, courses in author_courses.items()
         if author and author.lower() != 'unknown'],
        key=lambda x: x[1],
        reverse=True
    )

    return {author: count for author, count in sorted_authors[:n]}
