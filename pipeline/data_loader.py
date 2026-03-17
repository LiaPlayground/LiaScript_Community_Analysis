"""
Data Loader Module
Handles loading and merging of all LiaScript datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class LiaScriptDataLoader:
    """Loads and merges all LiaScript datasets."""

    def __init__(self, base_path: str, raw_folder: str = "raw"):
        """
        Initialize the data loader.

        Args:
            base_path: Base path to LiaScript data directory
            raw_folder: Name of raw data subfolder
        """
        self.base_path = Path(base_path) / raw_folder
        self.datasets = {}
        logger.info(f"Initialized DataLoader with base path: {self.base_path}")

    def load_all(self, validated_only: bool = True) -> pd.DataFrame:
        """
        Loads all datasets and merges them via pipe:ID.

        Args:
            validated_only: If True, filters for validated LiaScript courses only

        Returns:
            Merged DataFrame with all data
        """
        logger.info("Loading all datasets...")

        # Load core datasets
        # Use validated file which contains pipe:is_valid_liascript column
        self.datasets['files'] = self._load_pickle('LiaScript_files_validated.p')
        self.datasets['commits'] = self._load_pickle('LiaScript_commits.p')
        self.datasets['metadata'] = self._load_pickle('LiaScript_metadata.p')
        self.datasets['content'] = self._load_pickle('LiaScript_content.p')
        self.datasets['ai_meta'] = self._load_pickle('LiaScript_ai_meta.p')
        self.datasets['repositories'] = self._load_pickle('LiaScript_repositories.p')
        self.datasets['features'] = self._load_pickle('LiaScript_features.p')
        self.datasets['feature_statistics'] = self._load_pickle('LiaScript_feature_statistics.p')

        # Start with files dataset
        df_base = self.datasets['files'].copy()

        # Filter for validated courses if requested
        if validated_only and 'pipe:is_valid_liascript' in df_base.columns:
            logger.info(f"Filtering for validated courses...")
            df_base = df_base[df_base['pipe:is_valid_liascript'] == True].copy()
            logger.info(f"Retained {len(df_base)} validated courses")

        # Merge all datasets
        logger.info("Merging datasets...")
        df_full = df_base

        for name, df in [
            ('commits', self.datasets['commits']),
            ('metadata', self.datasets['metadata']),
            ('content', self.datasets['content']),
            ('ai_meta', self.datasets['ai_meta']),
            ('features', self.datasets['features'])
        ]:
            if df is not None and 'pipe:ID' in df.columns:
                logger.info(f"Merging {name}...")
                df_full = df_full.merge(
                    df,
                    on='pipe:ID',
                    how='left',
                    suffixes=('', f'_{name}')
                )

        # Merge repository-level data (created_at, etc.)
        if self.datasets['repositories'] is not None and 'repo_url' in self.datasets['repositories'].columns:
            logger.info("Merging repository metadata...")
            if 'repo_url' in df_full.columns:
                df_full = df_full.merge(
                    self.datasets['repositories'],
                    on='repo_url',
                    how='left',
                    suffixes=('', '_repo')
                )

        logger.info(f"Final dataset: {len(df_full)} rows, {len(df_full.columns)} columns")
        return df_full

    def get_repository_stats(self) -> pd.DataFrame:
        """Returns repository-level statistics."""
        return self.datasets.get('repositories')

    def get_feature_statistics(self) -> Dict:
        """Returns aggregated feature statistics dictionary."""
        return self.datasets.get('feature_statistics')

    def get_author_concentration(self, df: pd.DataFrame) -> Dict:
        """
        Calculates author concentration metrics.

        Args:
            df: DataFrame with repo_user column

        Returns:
            Dictionary with concentration metrics
        """
        if 'repo_user' not in df.columns:
            logger.warning("repo_user column not found")
            return {}

        author_counts = df['repo_user'].value_counts()
        total_courses = len(df)
        total_authors = len(author_counts)

        return {
            'total_authors': total_authors,
            'top5_share': author_counts.head(5).sum() / total_courses,
            'top10_share': author_counts.head(10).sum() / total_courses,
            'single_course_authors': (author_counts == 1).sum(),
            'single_course_rate': (author_counts == 1).sum() / total_authors,
            'top_authors': author_counts.head(10).to_dict(),
            'gini_coefficient': self._calculate_gini(author_counts.values)
        }

    def _calculate_gini(self, values) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        import numpy as np
        values = np.array(values, dtype=float)
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

    def get_education_level_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Extracts education level distribution from AI metadata.

        Args:
            df: DataFrame with ai:education_level column

        Returns:
            Dictionary with education level counts
        """
        if 'ai:education_level' not in df.columns:
            logger.warning("ai:education_level column not found")
            return {}

        return df['ai:education_level'].value_counts().to_dict()

    def categorize_licenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorizes licenses by OER compliance and type.

        Args:
            df: DataFrame with license columns

        Returns:
            DataFrame with added license_category and is_oer_compliant columns
        """
        logger.info("Categorizing licenses...")

        def categorize(spdx):
            if pd.isna(spdx) or spdx == 'NOASSERTION':
                return 'Unclear/Missing'
            elif spdx in ['CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0']:
                return 'Creative Commons (OER-ideal)'
            elif spdx in ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'Unlicense']:
                return 'Permissive Open Source'
            elif spdx in ['GPL-3.0', 'GPL-2.0', 'AGPL-3.0']:
                return 'Copyleft (GPL family)'
            else:
                return 'Other Open Source'

        if 'repo_license_spdx' in df.columns:
            df['license_category'] = df['repo_license_spdx'].apply(categorize)

            # OER-Compliance Flag
            df['is_oer_compliant'] = df['repo_license_spdx'].isin([
                'CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0',
                'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'Unlicense'
            ])

            logger.info(f"OER compliance rate: {df['is_oer_compliant'].mean():.1%}")
        else:
            logger.warning("No license columns found in dataset")

        return df

    def extract_dewey_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts DDC categories from ai:dewey field.

        Args:
            df: DataFrame with ai:dewey column

        Returns:
            DataFrame with additional DDC columns
        """
        logger.info("Extracting Dewey categories...")

        if 'ai:dewey' not in df.columns:
            logger.warning("ai:dewey column not found")
            return df

        def extract_primary_ddc(dewey_list):
            """Extract primary (highest-scored) DDC notation."""
            if not dewey_list or not isinstance(dewey_list, list) or len(dewey_list) == 0:
                return None
            return dewey_list[0].get('notation') if isinstance(dewey_list[0], dict) else None

        def extract_ddc_toplevel(notation):
            """Extract DDC top-level category (e.g., '000' from '004.6')."""
            if pd.isna(notation):
                return None
            notation_str = str(notation)
            if len(notation_str) >= 3:
                return notation_str[:1] + '00'
            return None

        df['ddc_primary'] = df['ai:dewey'].apply(extract_primary_ddc)
        df['ddc_toplevel'] = df['ddc_primary'].apply(extract_ddc_toplevel)

        logger.info(f"Extracted DDC for {df['ddc_primary'].notna().sum()} courses")
        return df

    def extract_first_from_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts first element from list-type metadata columns.

        Args:
            df: DataFrame with list-type columns

        Returns:
            DataFrame with additional _first columns
        """
        logger.info("Extracting first elements from list columns...")

        # Expanded list of metadata fields from new structure
        list_columns = [
            'lia:author', 'lia:email', 'lia:comment', 'lia:tags',
            'lia:import', 'lia:link', 'lia:script',
            'lia:current_version_description', 'lia:long_description'
        ]

        for col in list_columns:
            if col in df.columns:
                new_col = col.replace(':', '_first')
                df[new_col] = df[col].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if isinstance(x, str) else None)
                )
                logger.debug(f"Created {new_col} from {col}")

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds temporal features derived from timestamps.

        Args:
            df: DataFrame with timestamp columns

        Returns:
            DataFrame with additional temporal columns
        """
        logger.info("Adding temporal features...")

        # Extract year, month from timestamps
        if 'created_at' in df.columns:
            df['created_year'] = pd.to_datetime(df['created_at'], errors='coerce').dt.year
            df['created_month'] = pd.to_datetime(df['created_at'], errors='coerce').dt.month

        if 'first_commit' in df.columns:
            df['first_commit_year'] = pd.to_datetime(df['first_commit'], errors='coerce').dt.year

        if 'last_commit' in df.columns:
            df['last_commit_year'] = pd.to_datetime(df['last_commit'], errors='coerce').dt.year

        # Calculate age and activity duration
        from datetime import datetime, timedelta
        import pytz

        now = datetime.now(pytz.UTC)

        if 'first_commit' in df.columns:
            first = pd.to_datetime(df['first_commit'], errors='coerce')

            # Age in years (time since first commit)
            df['age_years'] = (now - first).dt.total_seconds() / (365.25 * 24 * 3600)

        if 'first_commit' in df.columns and 'last_commit' in df.columns:
            first = pd.to_datetime(df['first_commit'], errors='coerce')
            last = pd.to_datetime(df['last_commit'], errors='coerce')

            # Activity duration (first to last commit)
            df['activity_duration_days'] = (last - first).dt.days

            # Lifespan in years
            df['lifespan_years'] = (last - first).dt.total_seconds() / (365.25 * 24 * 3600)

            # Months since last update
            df['months_since_update'] = (now - last).dt.total_seconds() / (30.44 * 24 * 3600)

            # Recent activity flag (within last 6 months)
            cutoff = now - timedelta(days=180)
            # Make cutoff timezone-aware if last is timezone-aware
            if last.dt.tz is not None:
                if cutoff.tzinfo is None:
                    cutoff = pytz.UTC.localize(cutoff)
            else:
                cutoff = cutoff.replace(tzinfo=None)
            df['is_recently_active'] = last > cutoff

        logger.info("Temporal features added")
        return df

    def _load_pickle(self, filename: str):
        """
        Loads a pickle file from the data directory.

        Args:
            filename: Name of pickle file

        Returns:
            DataFrame, Dict, or None if file not found
        """
        filepath = self.base_path / filename
        try:
            logger.info(f"Loading {filename}...")

            # Handle ai_meta.p which may require dummy module for unpickling
            if 'ai_meta' in filename:
                data = self._load_pickle_with_dummy_modules(filepath)
            else:
                data = pd.read_pickle(filepath)

            if isinstance(data, pd.DataFrame):
                logger.info(f"  Loaded {len(data)} rows")
            elif isinstance(data, dict):
                logger.info(f"  Loaded dict with {len(data)} keys")
            else:
                logger.info(f"  Loaded {type(data).__name__}")
            return data
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None

    def _load_pickle_with_dummy_modules(self, filepath: Path):
        """
        Load pickle files that may have missing module dependencies.
        Creates dummy modules to allow unpickling.
        """
        import sys
        import types

        # Create dummy module for checkAuthorNames if not available
        if 'checkAuthorNames' not in sys.modules:
            dummy_module = types.ModuleType('checkAuthorNames')
            # Name class needs __reduce__ to be unpickleable
            dummy_module.Name = type('Name', (), {
                '__reduce__': lambda self: (str, (str(self),))
            })
            sys.modules['checkAuthorNames'] = dummy_module
            logger.debug("Created dummy checkAuthorNames module for unpickling")

        return pd.read_pickle(filepath)

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generates summary statistics for the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_courses': len(df),
            'total_repositories': df['repo_url'].nunique() if 'repo_url' in df.columns else None,
            'total_authors': df['repo_user'].nunique() if 'repo_user' in df.columns else None,
            'median_course_length': df['pipe:content_words'].median() if 'pipe:content_words' in df.columns else None,
            'languages': df['pipe:most_prob_language'].value_counts().to_dict() if 'pipe:most_prob_language' in df.columns else {},
        }

        if 'author_count' in df.columns:
            stats['single_author_rate'] = (df['author_count'] == 1).mean()
            stats['multi_author_rate'] = (df['author_count'] > 1).mean()

        if 'license_category' in df.columns:
            stats['license_distribution'] = df['license_category'].value_counts().to_dict()

        return stats
