"""
License Analysis
Comprehensive analysis of licensing patterns in LiaScript courses.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import chi2_contingency, mannwhitneyu
import logging

logger = logging.getLogger(__name__)


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs comprehensive license analysis.

    Args:
        df: Main dataset with license columns
        config: Configuration dictionary

    Returns:
        Dictionary with license analysis results
    """
    logger.info("Running license analysis...")

    results = {}

    # Ensure license categorization
    if 'license_category' not in df.columns:
        logger.warning("license_category not found, analysis may be incomplete")
        return {'error': 'License columns not available'}

    # 1. Overall License Distribution
    license_dist = df['license_category'].value_counts()
    results['license_distribution'] = {
        str(k): int(v) for k, v in license_dist.items()
    }

    # Raw SPDX distribution
    if 'repo_license_spdx' in df.columns:
        spdx_dist = df['repo_license_spdx'].value_counts(dropna=False).head(15)
        results['top_licenses_spdx'] = {
            str(k): int(v) for k, v in spdx_dist.items()
        }

    # 2. OER Compliance
    if 'is_oer_compliant' in df.columns:
        results['oer_compliance'] = {
            'total_oer_compliant': int(df['is_oer_compliant'].sum()),
            'oer_compliance_rate': float(df['is_oer_compliant'].mean()),
            'non_compliant': int((~df['is_oer_compliant']).sum())
        }

        # Breakdown by category
        cc_licenses = df['repo_license_spdx'].isin(['CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0'])
        permissive_os = df['repo_license_spdx'].isin(['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'Unlicense'])

        results['oer_compliance']['creative_commons_rate'] = float(cc_licenses.mean())
        results['oer_compliance']['permissive_opensource_rate'] = float(permissive_os.mean())

    # 3. License by Discipline (DDC)
    if 'ddc_toplevel' in df.columns and df['ddc_toplevel'].notna().sum() > 0:
        logger.info("Analyzing license distribution by discipline...")

        # Create crosstab
        crosstab = pd.crosstab(
            df['ddc_toplevel'],
            df['license_category'],
            normalize='index'
        )

        results['license_by_discipline'] = {
            str(ddc): {str(lic): float(val) for lic, val in row.items()}
            for ddc, row in crosstab.iterrows()
        }

        # Chi-square test for independence
        crosstab_counts = pd.crosstab(df['ddc_toplevel'], df['license_category'])
        if crosstab_counts.size > 0:
            chi2, p_value, dof, expected = chi2_contingency(crosstab_counts)
            results['discipline_independence_test'] = {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'is_significant': bool(p_value < 0.05)
            }

    # 4. License by Repository Type (Institutional vs. Individual)
    if 'internal' in df.columns:
        logger.info("Analyzing licensing by repository type...")

        institutional = df[df['internal'] == True]
        individual = df[df['internal'] == False]

        if len(institutional) > 0 and len(individual) > 0:
            results['license_by_repo_type'] = {
                'institutional': {
                    'total': int(len(institutional)),
                    'oer_compliant_rate': float(institutional['is_oer_compliant'].mean()) if 'is_oer_compliant' in institutional.columns else None,
                    'license_distribution': {
                        str(k): int(v) for k, v in institutional['license_category'].value_counts().items()
                    }
                },
                'individual': {
                    'total': int(len(individual)),
                    'oer_compliant_rate': float(individual['is_oer_compliant'].mean()) if 'is_oer_compliant' in individual.columns else None,
                    'license_distribution': {
                        str(k): int(v) for k, v in individual['license_category'].value_counts().items()
                    }
                }
            }

    # 5. Temporal Trends in Licensing
    if 'created_year' in df.columns:
        logger.info("Analyzing temporal licensing trends...")

        yearly_oer = df.groupby('created_year')['is_oer_compliant'].agg(['sum', 'count', 'mean'])
        results['licensing_temporal_trend'] = {
            int(year): {
                'total_courses': int(row['count']),
                'oer_compliant': int(row['sum']),
                'oer_rate': float(row['mean'])
            }
            for year, row in yearly_oer.iterrows()
            if pd.notna(year)
        }

    # 6. License Impact on Engagement
    if 'is_oer_compliant' in df.columns:
        logger.info("Analyzing license impact on engagement...")

        oer_courses = df[df['is_oer_compliant'] == True]
        non_oer_courses = df[df['is_oer_compliant'] == False]

        impact_metrics = {}

        # Stars
        if 'stars' in df.columns:
            oer_stars = oer_courses['stars'].dropna()
            non_oer_stars = non_oer_courses['stars'].dropna()

            if len(oer_stars) > 0 and len(non_oer_stars) > 0:
                stat, p_val = mannwhitneyu(oer_stars, non_oer_stars, alternative='two-sided')
                impact_metrics['stars'] = {
                    'oer_median': float(oer_stars.median()),
                    'oer_mean': float(oer_stars.mean()),
                    'non_oer_median': float(non_oer_stars.median()),
                    'non_oer_mean': float(non_oer_stars.mean()),
                    'mann_whitney_u': float(stat),
                    'p_value': float(p_val),
                    'is_significant': bool(p_val < 0.05)
                }

        # Forks
        if 'forks' in df.columns:
            oer_forks = oer_courses['forks'].dropna()
            non_oer_forks = non_oer_courses['forks'].dropna()

            if len(oer_forks) > 0 and len(non_oer_forks) > 0:
                stat, p_val = mannwhitneyu(oer_forks, non_oer_forks, alternative='two-sided')
                impact_metrics['forks'] = {
                    'oer_median': float(oer_forks.median()),
                    'oer_mean': float(oer_forks.mean()),
                    'non_oer_median': float(non_oer_forks.median()),
                    'non_oer_mean': float(non_oer_forks.mean()),
                    'mann_whitney_u': float(stat),
                    'p_value': float(p_val),
                    'is_significant': bool(p_val < 0.05)
                }

        # Contributors
        if 'author_count' in df.columns:
            oer_authors = oer_courses['author_count'].dropna()
            non_oer_authors = non_oer_courses['author_count'].dropna()

            if len(oer_authors) > 0 and len(non_oer_authors) > 0:
                stat, p_val = mannwhitneyu(oer_authors, non_oer_authors, alternative='two-sided')
                impact_metrics['collaboration'] = {
                    'oer_median': float(oer_authors.median()),
                    'oer_mean': float(oer_authors.mean()),
                    'non_oer_median': float(non_oer_authors.median()),
                    'non_oer_mean': float(non_oer_authors.mean()),
                    'mann_whitney_u': float(stat),
                    'p_value': float(p_val),
                    'is_significant': bool(p_val < 0.05)
                }

        results['license_impact'] = impact_metrics

    # 7. Missing License Analysis
    missing = df['repo_license_spdx'].isna() | (df['repo_license_spdx'] == 'NOASSERTION')
    results['missing_license_analysis'] = {
        'total_missing': int(missing.sum()),
        'missing_rate': float(missing.mean()),
        'missing_by_year': {}
    }

    if 'created_year' in df.columns:
        missing_by_year = df.groupby('created_year').apply(
            lambda x: (x['repo_license_spdx'].isna() | (x['repo_license_spdx'] == 'NOASSERTION')).mean()
        )
        results['missing_license_analysis']['missing_by_year'] = {
            int(year): float(rate) for year, rate in missing_by_year.items() if pd.notna(year)
        }

    # 8. NEW: Content License vs Repository License Comparison
    if 'lia:content_license' in df.columns:
        logger.info("Analyzing content license vs repository license...")

        has_content_license = df['lia:content_license'].notna()
        has_repo_license = df['repo_license_spdx'].notna() & (df['repo_license_spdx'] != 'NOASSERTION')

        results['content_vs_repo_license'] = {
            'courses_with_content_license': int(has_content_license.sum()),
            'content_license_rate': float(has_content_license.mean()),
            'courses_with_repo_license': int(has_repo_license.sum()),
            'repo_license_rate': float(has_repo_license.mean()),
            'courses_with_both': int((has_content_license & has_repo_license).sum()),
            'courses_with_either': int((has_content_license | has_repo_license).sum()),
            'courses_with_neither': int((~has_content_license & ~has_repo_license).sum())
        }

        # Distribution of content licenses
        content_license_dist = df['lia:content_license'].value_counts().head(15)
        results['content_vs_repo_license']['content_license_distribution'] = {
            str(k): int(v) for k, v in content_license_dist.items()
        }

        # Compare for courses that have both
        both_df = df[has_content_license & has_repo_license].copy()
        if len(both_df) > 0:
            # Normalize license strings for comparison
            both_df['content_lic_norm'] = both_df['lia:content_license'].str.upper().str.replace('-', '').str.replace('_', '')
            both_df['repo_lic_norm'] = both_df['repo_license_spdx'].str.upper().str.replace('-', '').str.replace('_', '')

            matches = both_df['content_lic_norm'].str.contains(both_df['repo_lic_norm'], na=False, regex=False) | \
                     both_df['repo_lic_norm'].str.contains(both_df['content_lic_norm'], na=False, regex=False)

            results['content_vs_repo_license']['license_agreement'] = {
                'total_with_both': int(len(both_df)),
                'matching_licenses': int(matches.sum()) if hasattr(matches, 'sum') else 0,
                'discrepancy_rate': float(1 - (matches.sum() / len(both_df))) if len(both_df) > 0 and hasattr(matches, 'sum') else None
            }

    # 9. NEW: Content License URL Usage
    if 'lia:content_license_url' in df.columns:
        has_url = df['lia:content_license_url'].notna()
        results['content_license_url_usage'] = {
            'courses_with_license_url': int(has_url.sum()),
            'license_url_rate': float(has_url.mean())
        }

    logger.info(f"License analysis complete: {len(results)} result categories")
    return results
