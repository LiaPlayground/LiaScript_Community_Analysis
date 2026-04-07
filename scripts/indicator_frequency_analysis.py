#!/usr/bin/env python3
"""
Indicator Frequency Analysis for LiaScript Detection

Analyzes the distribution of heuristic indicators across all detected files
to quantify the false-positive rate and justify AI validation necessity.
"""

import pandas as pd
import pickle
import json
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path("/media/sz/Data/Connected_Lecturers/LiaScript/raw")
OUTPUT_DIR = Path("/home/sz/Desktop/Python/LiaScript_Paper/data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("INDICATOR FREQUENCY ANALYSIS")
print("=" * 80)

# Load data
print("\n[1/3] Loading data...")
with open(DATA_DIR / "LiaScript_files.p", "rb") as f:
    files_df = pickle.load(f)

print(f"  ✓ Total files detected: {len(files_df):,}")

# Load validated courses to calculate false-positive rate
try:
    with open(DATA_DIR / "LiaScript_commits.p", "rb") as f:
        commits_df = pickle.load(f)
    validated_courses_count = len(commits_df)
    print(f"  ✓ Validated courses: {validated_courses_count:,}")
except Exception as e:
    print(f"  ⚠ Could not load validated courses: {e}")
    validated_courses_count = None

# Define indicator columns and their metadata
INDICATORS = {
    'liaIndi_comment_in_beginning': {
        'name': 'comment_in_beginning',
        'description': 'HTML comment at file start (metadata)',
        'strength': 'Weak'
    },
    'liaIndi_version_statement': {
        'name': 'version_statement',
        'description': '`version:` statement',
        'strength': 'Weak'
    },
    'liaIndi_liascript_in_content': {
        'name': 'liascript_in_content',
        'description': '"liascript" mentioned in content',
        'strength': 'Weak'
    },
    'liaIndi_import_statement': {
        'name': 'import_statement',
        'description': '`import:` (Template system)',
        'strength': 'Very Strong'
    },
    'liaIndi_narrator_statement': {
        'name': 'narrator_statement',
        'description': '`narrator:` (Text-to-Speech)',
        'strength': 'Strong'
    },
    'liaIndi_liaTemplates_used': {
        'name': 'liaTemplates_used',
        'description': 'LiaScript templates used',
        'strength': 'Strong'
    },
    'liaIndi_video_syntax': {
        'name': 'video_syntax',
        'description': '`!?[` (Video embedding)',
        'strength': 'Very Strong'
    },
    'liaIndi_lia_button': {
        'name': 'lia_button',
        'description': 'LiaScript badge present',
        'strength': 'Strong'
    },
    'liaIndi_lia_in_h1': {
        'name': 'lia_in_h1',
        'description': '"liascript" in main heading',
        'strength': 'Moderate'
    },
    'liaIndi_Lia_in_filename': {
        'name': 'Lia_in_filename',
        'description': '"liascript" in filename',
        'strength': 'Weak'
    }
}

# Calculate frequencies
print("\n[2/3] Calculating indicator frequencies...")
results = []

for col, meta in INDICATORS.items():
    if col in files_df.columns:
        # Count TRUE values (indicator present)
        count = files_df[col].sum() if files_df[col].dtype == 'bool' else (files_df[col] == True).sum()
        pct = (count / len(files_df)) * 100

        results.append({
            'indicator': meta['name'],
            'description': meta['description'],
            'strength': meta['strength'],
            'files': int(count),
            'percentage': round(pct, 1)
        })
        print(f"  ✓ {meta['name']}: {count:,} ({pct:.1f}%)")
    else:
        print(f"  ⚠ Column '{col}' not found in dataset")

# Sort by frequency (descending)
results_sorted = sorted(results, key=lambda x: x['files'], reverse=True)

# Calculate validation statistics
if validated_courses_count is not None:
    validated_courses = validated_courses_count
    validation_rate = (validated_courses / len(files_df)) * 100
    false_positive_rate = 100 - validation_rate

    print(f"\n  Validation Statistics:")
    print(f"  ✓ Validated courses: {validated_courses:,} ({validation_rate:.1f}%)")
    print(f"  ✓ False positives: {len(files_df) - validated_courses:,} ({false_positive_rate:.1f}%)")
else:
    validated_courses = None
    validation_rate = None
    false_positive_rate = None
    print(f"\n  ⚠ Could not calculate validation rate")

# Create output structure
output = {
    'metadata': {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'total_files': int(len(files_df)),
        'validated_courses': int(validated_courses) if validated_courses else None,
        'validation_rate_pct': round(validation_rate, 1) if validation_rate else None,
        'false_positive_rate_pct': round(false_positive_rate, 1) if false_positive_rate else None
    },
    'indicator_frequencies': results_sorted
}

# Save results
print("\n[3/3] Saving results...")

# JSON output
json_path = OUTPUT_DIR / "indicator_frequency_analysis.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  ✓ JSON: {json_path}")

# CSV output (table for paper)
csv_path = OUTPUT_DIR / "indicator_frequency_table.csv"
pd.DataFrame(results_sorted).to_csv(csv_path, index=False)
print(f"  ✓ CSV: {csv_path}")

# Markdown table for easy copy-paste
md_path = OUTPUT_DIR / "indicator_frequency_table.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Indicator Frequency Distribution\n\n")
    f.write(f"**Total files detected:** {len(files_df):,}\n")
    if validated_courses:
        f.write(f"**Validated courses:** {validated_courses:,} ({validation_rate:.1f}%)\n")
        f.write(f"**False positives:** {len(files_df) - validated_courses:,} ({false_positive_rate:.1f}%)\n")
    f.write("\n| Indicator | Description | Strength | Files | % |\n")
    f.write("|-----------|-------------|----------|-------|---|\n")
    for r in results_sorted:
        f.write(f"| `{r['indicator']}` | {r['description']} | {r['strength']} | {r['files']:,} | {r['percentage']}% |\n")
print(f"  ✓ Markdown: {md_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
if false_positive_rate is not None:
    print(f"\n📊 Key Finding: {false_positive_rate:.1f}% false-positive rate demonstrates")
    print(f"   the critical necessity of AI validation for LiaScript detection.")
else:
    print(f"\n📊 Indicator frequencies calculated. Load validated courses data for FP rate.")
