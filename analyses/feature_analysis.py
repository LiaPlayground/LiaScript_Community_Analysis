"""
Feature Analysis
Analyzes usage patterns of LiaScript features based on the feature: column schema.

Feature columns follow the naming convention:
  - feature:has_X  (boolean flag)
  - feature:X_count (integer count)

Feature categories (from LiaScript_Features_Patterns.md):
  Multimedia:  video (!?[](url)), audio (?[](url)), webapp (??[](url)), images (![](url))
  Quiz:        text_quiz ([[answer]]), single_choice ([(X)]), multiple_choice ([[X]]),
               selection_quiz ([[a|(b)|c]]), matrix_quiz, quiz_hints ([[?]])
  Survey:      surveys ([(text)]), survey_text
  Code:        code_blocks, executable_code (@input), code_projects (```lang+),
               script_tags (<script>)
  Animation:   animation_fragments ({{n}}), animation_blocks ({{n}} + ****),
               tts_fragments (--{{n}}--), tts_blocks (--{{n}}-- + ****),
               animated_css, effects
  Tables:      tables (standard MD), lia_viz_tables (data-type charts)
  Math:        inline_math ($...$), display_math ($$...$$)
  Header:      imports (import:), external_scripts (script:), external_css (link:),
               logo, icon, narrator (narrator:)
  Special:     qr_codes, ascii_diagrams, galleries, footnotes, classroom (@classroom),
               macros (@macro), custom_macro_defs, html_embeds, task_lists, links, comments
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Prefix for boolean feature flags
FEATURE_FLAG_PREFIX = 'feature:has_'
FEATURE_COUNT_PREFIX = 'feature:'


def run_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes LiaScript feature usage patterns.

    Args:
        df: Main dataset
        config: Configuration dictionary

    Returns:
        Dictionary with feature analysis results
    """
    logger.info("Running feature analysis...")

    results = {}

    # Boolean feature flag columns (feature:has_*)
    flag_columns = [col for col in df.columns if col.startswith(FEATURE_FLAG_PREFIX)]

    if not flag_columns:
        logger.warning("No feature flag columns (feature:has_*) found")
        return {'error': 'No feature columns available'}

    # 1. Overall Feature Usage
    feature_usage = {}
    for col in flag_columns:
        feature_name = col.replace(FEATURE_FLAG_PREFIX, '')
        usage_count = int(df[col].sum())
        usage_rate = float(df[col].mean())

        feature_usage[feature_name] = {
            'count': usage_count,
            'rate': usage_rate,
            'percentage': round(usage_rate * 100, 2)
        }

    results['feature_usage'] = feature_usage

    # 2. Template Usage (from feature:has_imports)
    if 'feature:has_imports' in df.columns:
        has_imports = df['feature:has_imports']
        results['template_usage'] = {
            'courses_using_templates': int(has_imports.sum()),
            'template_usage_rate': float(has_imports.mean()),
            'percentage': round(float(has_imports.mean() * 100), 2)
        }

    # 3. Custom JavaScript Usage (from feature:has_external_scripts)
    if 'feature:has_external_scripts' in df.columns:
        has_scripts = df['feature:has_external_scripts']
        results['javascript_usage'] = {
            'courses_using_js': int(has_scripts.sum()),
            'js_usage_rate': float(has_scripts.mean()),
            'percentage': round(float(has_scripts.mean() * 100), 2)
        }

    # 3a. Custom CSS Usage (from feature:has_external_css)
    if 'feature:has_external_css' in df.columns:
        has_css = df['feature:has_external_css']
        results['css_usage'] = {
            'courses_using_css': int(has_css.sum()),
            'css_usage_rate': float(has_css.mean()),
            'percentage': round(float(has_css.mean() * 100), 2)
        }

    # 4. Text-to-Speech / Narrator Usage
    if 'feature:has_narrator' in df.columns:
        has_narrator = df['feature:has_narrator']
        results['tts_usage'] = {
            'courses_using_tts': int(has_narrator.sum()),
            'tts_usage_rate': float(has_narrator.mean())
        }

    # 5. Video Integration
    if 'feature:has_video' in df.columns:
        has_video = df['feature:has_video']
        results['video_usage'] = {
            'courses_with_videos': int(has_video.sum()),
            'video_usage_rate': float(has_video.mean())
        }

    # 6. Feature Diversity (how many features per course)
    feature_counts = df[flag_columns].sum(axis=1)
    results['feature_diversity'] = {
        'mean_features_per_course': float(feature_counts.mean()),
        'median_features_per_course': float(feature_counts.median()),
        'max_features': int(feature_counts.max()),
        'courses_with_no_features': int((feature_counts == 0).sum())
    }

    # 7. Feature Co-occurrence (which features are used together?)
    if len(flag_columns) >= 2:
        feature_corr = df[flag_columns].corr()
        correlations = []
        for i, col1 in enumerate(flag_columns):
            for col2 in flag_columns[i+1:]:
                corr_val = feature_corr.loc[col1, col2]
                if not pd.isna(corr_val):
                    correlations.append({
                        'feature1': col1.replace(FEATURE_FLAG_PREFIX, ''),
                        'feature2': col2.replace(FEATURE_FLAG_PREFIX, ''),
                        'correlation': float(corr_val)
                    })

        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        results['feature_correlations'] = correlations[:10]

    # 8. Features by Discipline
    if 'ddc_toplevel' in df.columns and df['ddc_toplevel'].notna().sum() > 0:
        features_by_discipline = {}

        for ddc in df['ddc_toplevel'].dropna().unique():
            ddc_df = df[df['ddc_toplevel'] == ddc]
            ddc_features = {}

            for col in flag_columns:
                feature_name = col.replace(FEATURE_FLAG_PREFIX, '')
                ddc_features[feature_name] = float(ddc_df[col].mean())

            features_by_discipline[str(ddc)] = ddc_features

        results['features_by_discipline'] = features_by_discipline

    # 9. Visual Elements Usage (logo, icon — from feature flags)
    visual_features = {}
    if 'feature:has_logo' in df.columns:
        has_logo = df['feature:has_logo']
        visual_features['logo'] = {
            'count': int(has_logo.sum()),
            'rate': float(has_logo.mean()),
            'percentage': round(float(has_logo.mean() * 100), 2)
        }
    if 'feature:has_icon' in df.columns:
        has_icon = df['feature:has_icon']
        visual_features['icon'] = {
            'count': int(has_icon.sum()),
            'rate': float(has_icon.mean()),
            'percentage': round(float(has_icon.mean() * 100), 2)
        }
    if visual_features:
        results['visual_features'] = visual_features

    # 10. Tagging Usage
    if 'lia:tags' in df.columns:
        has_tags = df['lia:tags'].notna()
        results['tagging_usage'] = {
            'courses_with_tags': int(has_tags.sum()),
            'tagging_rate': float(has_tags.mean()),
            'percentage': round(float(has_tags.mean() * 100), 2)
        }

    # 11. Coding/Programming Features
    coding_features = {}
    if 'feature:has_executable_code' in df.columns:
        coding_features['courses_with_executable_code'] = int(df['feature:has_executable_code'].sum())
    if 'feature:has_code_projects' in df.columns:
        coding_features['courses_with_code_projects'] = int(df['feature:has_code_projects'].sum())
    if 'feature:code_language_count' in df.columns:
        coding_features['mean_code_languages'] = float(df['feature:code_language_count'].mean())
    if coding_features:
        results['coding_features'] = coding_features

    # 12. Mode Usage (Presentation vs Textbook)
    if 'lia:mode' in df.columns:
        mode_counts = df['lia:mode'].value_counts().to_dict()
        results['mode_usage'] = {
            'distribution': {str(k): int(v) for k, v in mode_counts.items()},
            'has_mode_set': int(df['lia:mode'].notna().sum())
        }

    # 13. Quiz & Survey aggregates
    quiz_survey = {}
    if 'feature:has_quiz' in df.columns:
        quiz_survey['courses_with_any_quiz'] = int(df['feature:has_quiz'].sum())
    if 'feature:has_any_survey' in df.columns:
        quiz_survey['courses_with_any_survey'] = int(df['feature:has_any_survey'].sum())
    if 'feature:total_quiz_count' in df.columns:
        quiz_survey['total_quiz_items'] = int(df['feature:total_quiz_count'].sum())
    if quiz_survey:
        results['quiz_survey'] = quiz_survey

    # 14. Interactivity score (animation + TTS + quiz + executable code)
    interactivity_cols = [c for c in flag_columns if any(
        k in c for k in ['animation', 'tts', 'quiz', 'executable', 'classroom', 'survey']
    )]
    if interactivity_cols:
        interactivity = df[interactivity_cols].sum(axis=1)
        results['interactivity'] = {
            'mean_interactive_features': float(interactivity.mean()),
            'courses_with_none': int((interactivity == 0).sum()),
            'courses_with_3plus': int((interactivity >= 3).sum())
        }

    logger.info(f"Feature analysis complete: {len(flag_columns)} feature flags analyzed")
    return results
