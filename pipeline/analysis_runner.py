"""
Analysis Runner Module
Orchestrates all analyses and caches results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Orchestrates all analyses and caches results."""

    def __init__(self, config: Dict[str, Any], df: pd.DataFrame):
        """
        Initialize the analysis runner.

        Args:
            config: Configuration dictionary from paper_config.yaml
            df: Main dataset with all merged data
        """
        self.config = config
        self.df = df
        self.results = {}
        self.enabled_analyses = config.get('analyses', {}).get('enabled', [])
        logger.info(f"Initialized AnalysisRunner with {len(self.enabled_analyses)} enabled analyses")

    def run_all(self) -> Dict[str, Any]:
        """
        Runs all enabled analyses.

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting analysis pipeline...")

        for analysis_name in self.enabled_analyses:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {analysis_name}")
                logger.info(f"{'='*60}")

                result = self._run_analysis(analysis_name)
                self.results[analysis_name] = result

                logger.info(f"✓ {analysis_name} completed successfully")
            except Exception as e:
                logger.error(f"✗ {analysis_name} failed: {e}", exc_info=True)
                self.results[analysis_name] = {'error': str(e)}

        logger.info(f"\n{'='*60}")
        logger.info(f"Analysis pipeline complete: {len(self.results)} analyses run")
        logger.info(f"{'='*60}\n")

        return self.results

    def _run_analysis(self, analysis_name: str) -> Dict[str, Any]:
        """
        Runs a single analysis by importing its module.

        Args:
            analysis_name: Name of the analysis module

        Returns:
            Analysis results dictionary
        """
        # Import the analysis module dynamically
        module_name = f"analyses.{analysis_name}"

        try:
            import importlib
            module = importlib.import_module(module_name)

            # Call the run_analysis function
            if hasattr(module, 'run_analysis'):
                return module.run_analysis(self.df, self.config)
            else:
                raise AttributeError(f"Module {module_name} has no run_analysis function")
        except ModuleNotFoundError:
            logger.warning(f"Module {module_name} not found, skipping...")
            return {'status': 'not_implemented'}

    def save_cache(self, output_path: str):
        """
        Saves analysis results to JSON cache.

        Args:
            output_path: Path to cache file
        """
        cache_path = Path(output_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving analysis cache to {cache_path}")

        # Convert non-serializable objects
        serializable_results = self._make_serializable(self.results)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Cache saved successfully")

    def load_cache(self, input_path: str) -> Dict[str, Any]:
        """
        Loads analysis results from JSON cache.

        Args:
            input_path: Path to cache file

        Returns:
            Cached results dictionary
        """
        cache_path = Path(input_path)

        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return {}

        logger.info(f"Loading analysis cache from {cache_path}")

        with open(cache_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        logger.info(f"Loaded {len(self.results)} cached analyses")
        return self.results

    def _make_serializable(self, obj):
        """
        Recursively converts objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        import numpy as np
        import pandas as pd
        from datetime import datetime, date

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_summary(self) -> str:
        """
        Generates a text summary of all analysis results.

        Returns:
            Formatted summary string
        """
        summary_lines = [
            "="*60,
            "ANALYSIS RESULTS SUMMARY",
            "="*60,
            ""
        ]

        for analysis_name, result in self.results.items():
            summary_lines.append(f"\n{analysis_name.upper()}:")
            summary_lines.append("-" * 40)

            if isinstance(result, dict):
                if 'error' in result:
                    summary_lines.append(f"  ERROR: {result['error']}")
                elif 'status' in result and result['status'] == 'not_implemented':
                    summary_lines.append(f"  Status: Not implemented")
                else:
                    # Show first few key-value pairs
                    for i, (key, value) in enumerate(result.items()):
                        if i >= 5:  # Limit to first 5 items
                            summary_lines.append(f"  ... ({len(result) - 5} more items)")
                            break
                        summary_lines.append(f"  {key}: {value}")
            else:
                summary_lines.append(f"  {result}")

        summary_lines.append("\n" + "="*60)
        return "\n".join(summary_lines)

    def export_to_excel(self, output_path: str):
        """
        Exports all analysis results to Excel with multiple sheets.

        Args:
            output_path: Path to Excel file
        """
        logger.info(f"Exporting results to Excel: {output_path}")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for analysis_name, result in self.results.items():
                if isinstance(result, dict) and 'error' not in result:
                    # Try to convert dict to DataFrame
                    try:
                        df_result = pd.DataFrame([result]) if not isinstance(list(result.values())[0], (list, dict)) else pd.DataFrame(result)
                        df_result.to_excel(writer, sheet_name=analysis_name[:31], index=False)
                    except Exception as e:
                        logger.warning(f"Could not export {analysis_name} to Excel: {e}")

        logger.info("Excel export complete")
