#!/usr/bin/env python3
"""
Quick Test Script
Tests the pipeline with a small subset of data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.data_loader import LiaScriptDataLoader
import yaml

def main():
    print("="*70)
    print("LiaScript Paper Pipeline - Quick Test")
    print("="*70)
    print()

    # Load config
    with open('config/paper_config.yaml') as f:
        config = yaml.safe_load(f)

    data_config = config['paper']['data']

    # Initialize loader
    print("1. Loading data...")
    loader = LiaScriptDataLoader(
        base_path=data_config['base_path'],
        raw_folder=data_config['raw_folder']
    )

    # Load datasets
    df = loader.load_all(validated_only=True)
    print(f"   ✓ Loaded {len(df)} courses")
    print(f"   ✓ Columns: {len(df.columns)}")

    # Apply transformations
    print("\n2. Applying transformations...")
    df = loader.categorize_licenses(df)
    df = loader.extract_dewey_categories(df)
    df = loader.extract_first_from_list_columns(df)
    df = loader.add_temporal_features(df)
    print("   ✓ Transformations complete")

    # Summary statistics
    print("\n3. Summary Statistics:")
    summary = loader.get_summary_statistics(df)

    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n   {key}:")
            for k, v in list(value.items())[:5]:
                print(f"     - {k}: {v}")
            if len(value) > 5:
                print(f"     ... ({len(value) - 5} more)")
        else:
            print(f"   {key}: {value}")

    # Test analyses (just descriptive_stats)
    print("\n4. Testing analysis module...")
    from analyses import descriptive_stats

    results = descriptive_stats.run_analysis(df, config['paper'])
    print(f"   ✓ Analysis complete: {len(results)} result categories")

    print("\n   Sample results:")
    if 'corpus' in results:
        print(f"     - Total courses: {results['corpus']['total_courses']}")
    if 'content_length' in results:
        print(f"     - Median words: {results['content_length']['median_words']:.0f}")
    if 'language_diversity' in results:
        print(f"     - Languages: {results['language_diversity']['unique_languages']}")

    print("\n" + "="*70)
    print("✓ Quick test PASSED")
    print("="*70)
    print("\nYou can now run the full pipeline:")
    print("  python run_pipeline.py")
    print()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
