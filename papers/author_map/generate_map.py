#!/usr/bin/env python3
"""
LiaScript Author Map Generator

Generates an interactive world map of LiaScript authors and their courses.

Usage:
    python generate_map.py              # Full run (geocode + build map)
    python generate_map.py --skip-geocoding  # Only rebuild map from cached data
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def run_step(script: str, description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate LiaScript Author Map")
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Skip geocoding step, use cached data",
    )
    args = parser.parse_args()

    if not args.skip_geocoding:
        run_step("geocode_users.py", "Step 1: Geocoding user locations")
    else:
        print("Skipping geocoding (using cached data)")

    run_step("build_map.py", "Step 2: Building interactive map")

    output = BASE_DIR / "build" / "author_map.html"
    print(f"\n{'='*60}")
    print(f"  Done! Open the map:")
    print(f"  file://{output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
