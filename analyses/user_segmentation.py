"""
User Segmentation Analysis
Identifies user groups and analyzes feature usage per group.

This analysis creates actionable insights for LiaScript development by:
1. Segmenting authors by productivity level
2. Analyzing feature preferences per segment
3. Identifying development priorities for each user group
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Author categories with boundaries
AUTHOR_CATEGORIES = {
    'one_time': (1, 1),       # Exactly 1 course
    'occasional': (2, 5),      # 2-5 courses
    'regular': (6, 20),        # 6-20 courses
    'active': (21, 100),       # 21-100 courses
    'power_user': (101, float('inf'))  # >100 courses
}

CATEGORY_LABELS = {
    'one_time': 'One-Time Author',
    'occasional': 'Occasional Author',
    'regular': 'Regular Author',
    'active': 'Active Author',
    'power_user': 'Power User'
}


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs user segmentation analysis.

    Args:
        df: Main dataset with all merged data
        config: Configuration dictionary

    Returns:
        Dictionary with user segmentation results
    """
    logger.info("Running user segmentation analysis...")

    results = {}

    # 1. Build author-course mapping
    author_courses = _build_author_course_mapping(df)
    if not author_courses:
        logger.warning("No author data available for segmentation")
        return {'error': 'No author data available'}

    # 2. Categorize authors by productivity
    author_categories = _categorize_authors(author_courses)
    results['author_segmentation'] = author_categories

    # 3. Get feature columns (feature:has_* boolean flags)
    indicator_columns = [col for col in df.columns if col.startswith('feature:has_')]
    if not indicator_columns:
        logger.warning("No feature flag columns (feature:has_*) found")
        return results

    # 4. Analyze feature usage per author category
    results['features_by_segment'] = _analyze_features_by_segment(
        df, author_courses, indicator_columns
    )

    # 5. Identify segment-specific feature preferences
    results['segment_preferences'] = _identify_segment_preferences(
        results['features_by_segment']
    )

    # 6. Analyze education levels by segment
    if 'ai:education_level' in df.columns:
        results['education_by_segment'] = _analyze_education_by_segment(
            df, author_courses
        )

    # 7. Analyze course characteristics by segment
    results['course_characteristics'] = _analyze_course_characteristics(
        df, author_courses
    )

    # 8. Generate development recommendations
    results['development_recommendations'] = _generate_recommendations(
        results['features_by_segment'],
        results.get('segment_preferences', {}),
        author_categories
    )

    # 9. Feature clusters (users grouped by feature usage patterns)
    results['feature_clusters'] = _cluster_by_features(df, indicator_columns)

    logger.info("User segmentation analysis complete")
    return results


def _build_author_course_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build mapping from authors to their course IDs.

    Args:
        df: Main dataset

    Returns:
        Dictionary mapping author names to list of course IDs
    """
    author_courses = defaultdict(list)

    # Prefer contributors_list, fall back to repo_user
    if 'contributors_list' in df.columns:
        for idx, row in df.iterrows():
            course_id = row.get('pipe:ID', str(idx))
            contributors = _parse_contributors(row.get('contributors_list'))
            for author in contributors:
                if author and author.lower() not in ['unknown', 'nan']:
                    author_courses[author].append(course_id)
    elif 'repo_user' in df.columns:
        for idx, row in df.iterrows():
            course_id = row.get('pipe:ID', str(idx))
            author = row.get('repo_user')
            if pd.notna(author) and str(author).lower() not in ['unknown', 'nan']:
                author_courses[str(author)].append(course_id)

    logger.info(f"Built author mapping: {len(author_courses)} authors")
    return dict(author_courses)


def _parse_contributors(contributors_str) -> List[str]:
    """Parse contributors from various formats."""
    import ast

    if contributors_str is None:
        return []
    try:
        if pd.isna(contributors_str):
            return []
    except (ValueError, TypeError):
        pass

    contrib_str = str(contributors_str).strip()

    try:
        if contrib_str.startswith('['):
            authors = ast.literal_eval(contrib_str)
            if isinstance(authors, list):
                return [str(a).strip() for a in authors if a]
    except (ValueError, SyntaxError):
        pass

    return [a.strip().strip("'\"[]") for a in contrib_str.split(',')
            if a.strip() and a.strip() not in ['nan', 'unknown']]


def _categorize_authors(author_courses: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Categorize authors by their productivity level.

    Args:
        author_courses: Mapping from author to course IDs

    Returns:
        Dictionary with categorization results
    """
    categories = {cat: [] for cat in AUTHOR_CATEGORIES.keys()}
    course_counts = []

    for author, courses in author_courses.items():
        count = len(courses)
        course_counts.append(count)

        for cat_name, (min_val, max_val) in AUTHOR_CATEGORIES.items():
            if min_val <= count <= max_val:
                categories[cat_name].append((author, count))
                break

    total_authors = len(author_courses)
    total_courses = sum(len(courses) for courses in author_courses.values())

    result = {
        'total_authors': total_authors,
        'total_courses_attributed': total_courses,
        'categories': {}
    }

    for cat_name, authors in categories.items():
        cat_authors = len(authors)
        cat_courses = sum(count for _, count in authors)

        result['categories'][cat_name] = {
            'label': CATEGORY_LABELS[cat_name],
            'author_count': cat_authors,
            'author_share': float(cat_authors / total_authors) if total_authors > 0 else 0,
            'course_count': cat_courses,
            'course_share': float(cat_courses / total_courses) if total_courses > 0 else 0,
            'top_authors': sorted(authors, key=lambda x: x[1], reverse=True)[:5]
        }

    # Summary statistics
    result['statistics'] = {
        'mean_courses': float(np.mean(course_counts)),
        'median_courses': float(np.median(course_counts)),
        'std_courses': float(np.std(course_counts)),
        'max_courses': int(max(course_counts)),
        'gini': float(_calculate_gini(course_counts))
    }

    return result


def _calculate_gini(values: List[int]) -> float:
    """Calculate Gini coefficient."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def _analyze_features_by_segment(
    df: pd.DataFrame,
    author_courses: Dict[str, List[str]],
    indicator_columns: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze feature usage rates for each author segment.

    Args:
        df: Main dataset
        author_courses: Author to course mapping
        indicator_columns: List of feature indicator columns

    Returns:
        Dictionary mapping segments to feature usage rates
    """
    # Create course_id to segment mapping
    course_segments = {}
    for author, courses in author_courses.items():
        count = len(courses)
        for cat_name, (min_val, max_val) in AUTHOR_CATEGORIES.items():
            if min_val <= count <= max_val:
                for course_id in courses:
                    course_segments[course_id] = cat_name
                break

    # Add segment to dataframe
    df_copy = df.copy()
    df_copy['_segment'] = df_copy['pipe:ID'].map(course_segments)

    results = {}
    for segment in AUTHOR_CATEGORIES.keys():
        segment_df = df_copy[df_copy['_segment'] == segment]
        if len(segment_df) == 0:
            continue

        feature_rates = {}
        for col in indicator_columns:
            feature_name = col.replace('feature:has_', '')
            rate = float(segment_df[col].mean()) if col in segment_df.columns else 0
            feature_rates[feature_name] = rate

        results[segment] = {
            'course_count': len(segment_df),
            'feature_rates': feature_rates
        }

    return results


def _identify_segment_preferences(
    features_by_segment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Identify which features are preferred by each segment.

    Args:
        features_by_segment: Feature rates per segment

    Returns:
        Dictionary with segment-specific preferences
    """
    if not features_by_segment:
        return {}

    # Calculate average feature rate across all segments
    all_rates = defaultdict(list)
    for segment_data in features_by_segment.values():
        for feature, rate in segment_data.get('feature_rates', {}).items():
            all_rates[feature].append(rate)

    avg_rates = {f: np.mean(rates) for f, rates in all_rates.items()}

    results = {}
    for segment, segment_data in features_by_segment.items():
        feature_rates = segment_data.get('feature_rates', {})

        # Features used more than average
        preferred = []
        avoided = []

        for feature, rate in feature_rates.items():
            avg = avg_rates.get(feature, 0)
            if avg > 0:
                ratio = rate / avg
                if ratio > 1.2:  # 20% above average
                    preferred.append((feature, rate, ratio))
                elif ratio < 0.8:  # 20% below average
                    avoided.append((feature, rate, ratio))

        results[segment] = {
            'preferred_features': sorted(preferred, key=lambda x: x[2], reverse=True)[:5],
            'avoided_features': sorted(avoided, key=lambda x: x[2])[:5]
        }

    return results


def _analyze_education_by_segment(
    df: pd.DataFrame,
    author_courses: Dict[str, List[str]]
) -> Dict[str, Dict[str, int]]:
    """
    Analyze education level distribution per segment.

    Args:
        df: Main dataset
        author_courses: Author to course mapping

    Returns:
        Dictionary mapping segments to education level distributions
    """
    # Create course_id to segment mapping
    course_segments = {}
    for author, courses in author_courses.items():
        count = len(courses)
        for cat_name, (min_val, max_val) in AUTHOR_CATEGORIES.items():
            if min_val <= count <= max_val:
                for course_id in courses:
                    course_segments[course_id] = cat_name
                break

    df_copy = df.copy()
    df_copy['_segment'] = df_copy['pipe:ID'].map(course_segments)

    results = {}
    for segment in AUTHOR_CATEGORIES.keys():
        segment_df = df_copy[df_copy['_segment'] == segment]
        if len(segment_df) == 0:
            continue

        if 'ai:education_level' in segment_df.columns:
            edu_dist = segment_df['ai:education_level'].value_counts()
            results[segment] = {
                str(k): int(v) for k, v in edu_dist.items()
            }

    return results


def _analyze_course_characteristics(
    df: pd.DataFrame,
    author_courses: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze course characteristics (length, complexity) per segment.

    Args:
        df: Main dataset
        author_courses: Author to course mapping

    Returns:
        Dictionary mapping segments to course characteristics
    """
    # Create course_id to segment mapping
    course_segments = {}
    for author, courses in author_courses.items():
        count = len(courses)
        for cat_name, (min_val, max_val) in AUTHOR_CATEGORIES.items():
            if min_val <= count <= max_val:
                for course_id in courses:
                    course_segments[course_id] = cat_name
                break

    df_copy = df.copy()
    df_copy['_segment'] = df_copy['pipe:ID'].map(course_segments)

    results = {}
    for segment in AUTHOR_CATEGORIES.keys():
        segment_df = df_copy[df_copy['_segment'] == segment]
        if len(segment_df) == 0:
            continue

        chars = {}

        # Content length
        if 'pipe:content_words' in segment_df.columns:
            words = segment_df['pipe:content_words'].dropna()
            if len(words) > 0:
                chars['mean_words'] = float(words.mean())
                chars['median_words'] = float(words.median())

        # Number of pages/sections
        if 'pipe:content_pages' in segment_df.columns:
            pages = segment_df['pipe:content_pages'].dropna()
            if len(pages) > 0:
                chars['mean_pages'] = float(pages.mean())

        # Activity span
        if 'lifespan_years' in segment_df.columns:
            lifespan = segment_df['lifespan_years'].dropna()
            if len(lifespan) > 0:
                chars['mean_lifespan_years'] = float(lifespan.mean())

        # Feature count
        indicator_cols = [c for c in segment_df.columns if c.startswith('feature:has_')]
        if indicator_cols:
            feature_counts = segment_df[indicator_cols].sum(axis=1)
            chars['mean_features'] = float(feature_counts.mean())

        results[segment] = chars

    return results


def _generate_recommendations(
    features_by_segment: Dict[str, Any],
    segment_preferences: Dict[str, Any],
    author_categories: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate development recommendations based on segment analysis.

    Args:
        features_by_segment: Feature usage per segment
        segment_preferences: Preferences per segment
        author_categories: Author categorization data

    Returns:
        List of recommendations
    """
    recommendations = []

    categories = author_categories.get('categories', {})

    # Recommendation 1: Support one-time authors (largest group)
    one_time = categories.get('one_time', {})
    if one_time.get('author_share', 0) > 0.4:
        recommendations.append({
            'priority': 'high',
            'target': 'one_time',
            'title': 'Lower Entry Barrier for New Authors',
            'rationale': f"One-time authors represent {one_time.get('author_share', 0):.1%} of all authors but only contribute {one_time.get('course_share', 0):.1%} of courses.",
            'suggestions': [
                'Simplified getting-started templates',
                'Better onboarding documentation',
                'Quick-start wizard in LiaScript editor',
                'Template gallery with one-click import'
            ]
        })

    # Recommendation 2: Empower power users (highest impact)
    power = categories.get('power_user', {})
    if power.get('course_share', 0) > 0.3:
        recommendations.append({
            'priority': 'high',
            'target': 'power_user',
            'title': 'Advanced Tools for Power Users',
            'rationale': f"Power users ({power.get('author_share', 0):.1%} of authors) create {power.get('course_share', 0):.1%} of all courses.",
            'suggestions': [
                'Batch course management',
                'CI/CD integration for automated publishing',
                'Advanced template inheritance',
                'Course analytics dashboard'
            ]
        })

    # Recommendation 3: Feature-specific based on usage patterns
    if features_by_segment:
        # Find underutilized features
        all_rates = {}
        for segment_data in features_by_segment.values():
            for feature, rate in segment_data.get('feature_rates', {}).items():
                if feature not in all_rates:
                    all_rates[feature] = []
                all_rates[feature].append(rate)

        underutilized = [
            (f, np.mean(rates))
            for f, rates in all_rates.items()
            if np.mean(rates) < 0.1 and f not in ['comments', 'links']
        ]

        if underutilized:
            recommendations.append({
                'priority': 'medium',
                'target': 'all',
                'title': 'Improve Discoverability of Underutilized Features',
                'rationale': f"{len(underutilized)} features are used by <10% of courses.",
                'suggestions': [
                    f"Better documentation for: {', '.join([f[0] for f in underutilized[:5]])}",
                    'Feature spotlight in release notes',
                    'Interactive feature demos'
                ]
            })

    # Recommendation 4: Template ecosystem
    if segment_preferences:
        # Check if template import is preferred by active users
        for segment in ['regular', 'active', 'power_user']:
            prefs = segment_preferences.get(segment, {}).get('preferred_features', [])
            if any('import' in f[0].lower() for f in prefs):
                recommendations.append({
                    'priority': 'medium',
                    'target': 'regular+',
                    'title': 'Expand Template Ecosystem',
                    'rationale': 'Active authors prefer template imports, indicating strong template adoption.',
                    'suggestions': [
                        'Template repository/registry',
                        'Template versioning and updates',
                        'Domain-specific template packs (STEM, Language, etc.)'
                    ]
                })
                break

    return recommendations


def _cluster_by_features(
    df: pd.DataFrame,
    indicator_columns: List[str],
    n_clusters: int = 5
) -> Dict[str, Any]:
    """
    Cluster courses by feature usage patterns.

    Args:
        df: Main dataset
        indicator_columns: Feature indicator columns
        n_clusters: Number of clusters

    Returns:
        Dictionary with clustering results
    """
    if len(indicator_columns) < 3:
        return {'error': 'Not enough features for clustering'}

    # Prepare feature matrix
    feature_matrix = df[indicator_columns].fillna(0).astype(int)

    # Simple clustering: group by feature profile
    profiles = feature_matrix.apply(lambda row: tuple(row), axis=1)
    profile_counts = profiles.value_counts()

    # Identify dominant patterns
    top_profiles = profile_counts.head(10)

    # Name the clusters based on active features
    named_clusters = []
    for profile, count in top_profiles.items():
        active_features = [
            col.replace('feature:has_', '')
            for col, val in zip(indicator_columns, profile)
            if val == 1
        ]

        if len(active_features) == 0:
            name = "Minimalist (no special features)"
        elif len(active_features) <= 2:
            name = f"Basic: {', '.join(active_features[:2])}"
        else:
            name = f"Advanced: {len(active_features)} features"

        named_clusters.append({
            'name': name,
            'course_count': int(count),
            'share': float(count / len(df)),
            'active_features': active_features[:5]
        })

    return {
        'unique_profiles': len(profile_counts),
        'top_clusters': named_clusters,
        'concentration': float(top_profiles.sum() / len(df))
    }
