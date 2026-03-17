#!/usr/bin/env python3
"""
LiaScript Paper Pipeline - Main Entry Point

This script orchestrates the complete pipeline:
1. Load and merge all LiaScript datasets
2. Run all configured analyses
3. Generate paper document(s) from templates
4. Export in configured formats (Markdown, LaTeX, PDF)

Usage:
    python run_pipeline.py [--paper PAPER_ID] [--skip-analysis] [--skip-paper]

Options:
    --paper PAPER_ID    Paper to generate: journal, conference, or all (default: all)
    --config PATH       Path to main config file (default: config/paper_config.yaml)
    --skip-analysis     Skip analysis phase (use cached results)
    --skip-paper        Skip paper generation (analysis only)
    --cache PATH        Path to analysis cache file
    --log-level LEVEL   Logging level (DEBUG, INFO, WARNING, ERROR)

Available Papers:
    journal     - "Collaboration and Reuse" (collaborative authoring, RQ1-3)
    conference  - "User Segmentation" (user groups, development priorities)
    course      - "LiaScript Ecosystem Overview" (interactive LiaScript course)
    all         - Generate all three outputs
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.data_loader import LiaScriptDataLoader
from pipeline.analysis_runner import AnalysisRunner
from pipeline.paper_builder import PaperBuilder


# Paper configurations
PAPERS = {
    'journal': {
        'id': 'journal_collaboration',
        'config_path': 'papers/journal_collaboration/config.yaml',
        'template_dir': 'papers/journal_collaboration/sections',
        'output_dir': 'papers/journal_collaboration/build',
        'title': 'Collaboration and Reuse (Journal)'
    },
    'conference': {
        'id': 'conference_development',
        'config_path': 'papers/conference_development/config.yaml',
        'template_dir': 'papers/conference_development/sections',
        'output_dir': 'papers/conference_development/build',
        'title': 'User Segmentation (Conference)'
    },
    'course': {
        'id': 'liascript_course',
        'config_path': 'papers/liascript_course/config.yaml',
        'template_dir': 'papers/liascript_course/sections',
        'output_dir': 'papers/liascript_course/build',
        'title': 'LiaScript Ecosystem Overview (Interactive Course)'
    }
}


def setup_logging(log_level: str = "INFO"):
    """
    Configure logging for the pipeline.

    Args:
        log_level: Logging level string
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from {config_path}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base_config: dict, paper_config: dict) -> dict:
    """
    Merge paper-specific config with base config.

    Args:
        base_config: Main configuration from paper_config.yaml
        paper_config: Paper-specific configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    # Override paper section with paper-specific settings
    if 'paper' in paper_config:
        merged['paper'] = {**merged.get('paper', {}), **paper_config['paper']}

    # Keep research_questions from base config
    if 'research_questions' not in merged:
        merged['research_questions'] = base_config.get('research_questions', {})

    return merged


def generate_paper(paper_id: str, analysis_results: dict, base_config: dict, logger):
    """
    Generate a specific paper.

    Args:
        paper_id: Paper identifier (journal or conference)
        analysis_results: Analysis results dictionary
        base_config: Base configuration
        logger: Logger instance
    """
    if paper_id not in PAPERS:
        logger.error(f"Unknown paper: {paper_id}")
        return

    paper_info = PAPERS[paper_id]
    logger.info(f"\n{'='*70}")
    logger.info(f"GENERATING: {paper_info['title']}")
    logger.info(f"{'='*70}")

    # Load paper-specific config
    paper_config_path = Path(paper_info['config_path'])
    if paper_config_path.exists():
        paper_config = load_config(str(paper_config_path))
        merged_config = merge_configs(base_config, paper_config)
    else:
        logger.warning(f"Paper config not found: {paper_config_path}, using base config")
        merged_config = base_config

    # Create output directory
    output_dir = Path(paper_info['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy figures to paper-specific directory
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Link or copy figures from scripts output or existing builds
    import shutil
    # Try multiple source locations
    for source_dir in ['scripts/figures', 'papers/journal_collaboration/build/figures']:
        main_figures = Path(source_dir)
        if main_figures.exists():
            for fig in main_figures.glob('*.png'):
                dest = figures_dir / fig.name
                if not dest.exists():
                    shutil.copy(fig, dest)
            break

    # Build paper
    builder = PaperBuilder(
        template_dir=paper_info['template_dir'],
        results=analysis_results,
        config=merged_config
    )

    logger.info(f"Building paper document...")
    builder.export_all(str(output_dir))

    logger.info(f"Paper generated: {output_dir}/paper.pdf")


def main():
    """Main pipeline execution."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='LiaScript Paper Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available papers:
  journal     - "Collaboration and Reuse" (collaborative authoring, templates, forks)
  conference  - "User Segmentation" (user groups, development priorities)
  course      - "LiaScript Ecosystem Overview" (interactive LiaScript course)
  all         - Generate all three outputs

Examples:
  python run_pipeline.py --paper course
  python run_pipeline.py --paper journal --skip-analysis
  python run_pipeline.py --paper all
        """
    )
    parser.add_argument('--paper', default='all',
                        choices=['journal', 'conference', 'course', 'all'],
                        help='Paper to generate (default: all)')
    parser.add_argument('--config', default='papers/shared/config.yaml',
                        help='Path to main configuration file')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip analysis phase (use cached results)')
    parser.add_argument('--skip-paper', action='store_true',
                        help='Skip paper generation')
    parser.add_argument('--cache', default='data/processed/analysis_results.json',
                        help='Path to analysis cache file')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("LiaScript Paper Pipeline - Starting")
    logger.info(f"Target paper(s): {args.paper}")
    logger.info("="*70)

    # Load main configuration
    config = load_config(args.config)
    paper_config = config.get('paper', {})

    # Initialize analysis results
    analysis_results = {}

    # PHASE 1: Data Loading
    if not args.skip_analysis:
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("="*70)

        data_config = config.get('data', {})
        loader = LiaScriptDataLoader(
            base_path=data_config.get('base_path'),
            raw_folder=data_config.get('raw_folder', 'raw')
        )

        logger.info("Loading and merging datasets...")
        df_full = loader.load_all(validated_only=True)

        logger.info(f"Loaded {len(df_full)} validated courses")

        # Apply data transformations
        logger.info("Applying data transformations...")
        df_full = loader.categorize_licenses(df_full)
        df_full = loader.extract_dewey_categories(df_full)
        df_full = loader.extract_first_from_list_columns(df_full)
        df_full = loader.add_temporal_features(df_full)

        # Show summary statistics
        summary = loader.get_summary_statistics(df_full)
        logger.info(f"\nDataset Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    # PHASE 2: Analysis
    if not args.skip_analysis:
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: ANALYSIS")
        logger.info("="*70)

        runner = AnalysisRunner(config, df_full)
        analysis_results = runner.run_all()

        # Save cache
        if config.get('data', {}).get('cache_processed', True):
            logger.info(f"\nSaving analysis cache to {args.cache}")
            runner.save_cache(args.cache)

        # Show summary
        logger.info("\n" + runner.get_summary())

        # Export to Excel
        excel_path = Path(args.cache).parent / 'analysis_results.xlsx'
        try:
            runner.export_to_excel(str(excel_path))
            logger.info(f"Analysis results exported to Excel: {excel_path}")
        except Exception as e:
            logger.warning(f"Could not export to Excel: {e}")

    else:
        # Load cached results
        import pandas as pd
        logger.info(f"\nLoading cached analysis results from {args.cache}")
        runner = AnalysisRunner(config, pd.DataFrame())  # Empty df for cache loading
        analysis_results = runner.load_cache(args.cache)

    # PHASE 3: Figure Generation
    if not args.skip_paper:
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: FIGURE GENERATION")
        logger.info("="*70)

        logger.info("Generating all figures...")
        try:
            # Import only when needed to avoid loading data at import time
            from scripts.generate_all_figures import generate_all_figures
            generate_all_figures()
            logger.info("Figure generation complete")
        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")

    # PHASE 4: Paper Generation
    if not args.skip_paper:
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: PAPER GENERATION")
        logger.info("="*70)

        # Determine which papers to generate
        if args.paper == 'all':
            papers_to_generate = ['course', 'journal', 'conference']
        else:
            papers_to_generate = [args.paper]

        for paper_id in papers_to_generate:
            try:
                generate_paper(paper_id, analysis_results, config, logger)
            except Exception as e:
                logger.error(f"Failed to generate {paper_id} paper: {e}", exc_info=True)

    # Pipeline complete
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)

    if not args.skip_paper:
        logger.info("\nGenerated outputs:")
        if args.paper == 'all' or args.paper == 'journal':
            logger.info(f"  - Journal:     papers/journal_collaboration/build/paper.pdf")
        if args.paper == 'all' or args.paper == 'conference':
            logger.info(f"  - Conference:  papers/conference_development/build/paper.pdf")
        if args.paper == 'all' or args.paper == 'course':
            logger.info(f"  - Course:      papers/liascript_course/build/course.md")

    logger.info(f"\nOther outputs:")
    logger.info(f"  - Analysis cache: {args.cache}")
    logger.info(f"  - Figures: papers/shared/figures/")
    logger.info(f"  - Logs: pipeline.log")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
