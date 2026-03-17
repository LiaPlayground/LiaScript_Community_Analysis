"""
Geocode GitHub committer locations to lat/lon coordinates.

Maps courses to their actual committers (from contributors_list) rather than
to repository accounts, which are often organisational containers.

Uses Nominatim (OpenStreetMap) with a JSON cache to avoid repeated lookups.
Outputs: data/geocoded_users.json
"""

import json
import os
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = Path("/home/sz/Desktop/Python/LiaScript_Paper/data_march_2026/liascript_march_2026/raw")
CACHE_FILE = DATA_DIR / "geocode_cache.json"
OUTPUT_FILE = DATA_DIR / "geocoded_users.json"

# Locations that are jokes or unresolvable
BLACKLIST = {
    "terra", "planet earth", "the world", "outer space", "the internet",
    "sol, orion arm", "remote", "maybe in the existence", "fysiweb",
    "the existence", "$ [  \"$home\" -eq \"wa\" ];", "ebarojas",
    "maybe in the existence",
}

# Manual overrides for ambiguous or company-only entries
MANUAL_COORDS = {
    "TU Bergakademie Freiberg": (50.9194, 13.3397),
    "Freiberg": (50.9194, 13.3397),
    "Freiberg (Germany)": (50.9194, 13.3397),
    "Freiberg, Saxony, Germany": (50.9194, 13.3397),
    "Freiberg (Sachsen)": (50.9194, 13.3397),
    "University of Hildesheim": (52.1345, 9.9501),
    "Hochschule Heilbronn": (49.1214, 9.2095),
    "TH Köln (University of Applied Sciences)": (50.9322, 6.9177),
}


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def geocode_location(geolocator, location_str: str, cache: dict) -> tuple | None:
    """Geocode a location string, using cache first."""
    if not location_str:
        return None

    normalized = location_str.strip()
    lower = normalized.lower()

    if lower in BLACKLIST:
        return None

    # Check manual overrides
    for key, coords in MANUAL_COORDS.items():
        if key.lower() == lower:
            return coords

    # Check cache
    if normalized in cache:
        val = cache[normalized]
        return tuple(val) if val else None

    # Query Nominatim
    try:
        result = geolocator.geocode(normalized, timeout=10)
        if result:
            coords = (result.latitude, result.longitude)
            cache[normalized] = list(coords)
            return coords
        else:
            cache[normalized] = None
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"  Geocoding error for '{normalized}': {e}")
        return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_profiles = pd.read_pickle(RAW_DIR / "LiaScript_user_profiles.p")
    df_consolidated = pd.read_pickle(RAW_DIR / "LiaScript_consolidated.p")

    # Build committer -> courses mapping using contributors_list
    committer_courses = {}
    for _, row in df_consolidated.iterrows():
        contributors = row.get("contributors_list", [])
        if not isinstance(contributors, list) or not contributors:
            continue
        # Deduplicate committers per course
        unique_committers = set(contributors)
        course_info = {
            "repo_name": row["repo_name"],
            "repo_user": row["repo_user"],
            "file_name": row["file_name"],
            "download_url": row.get("file_download_url", ""),
            "lia_url": f"https://liascript.github.io/course/?{row.get('file_download_url', '')}",
        }
        for committer in unique_committers:
            committer_courses.setdefault(committer, []).append(course_info)

    print(f"Loaded {len(df_profiles)} user profiles")
    print(f"Unique committers with courses: {len(committer_courses)}")

    # Geocode
    geolocator = Nominatim(user_agent="liascript_research_map_2026")
    cache = load_cache()
    results = []
    geocoded_count = 0
    skipped_count = 0

    # Only process committers who have a profile
    committer_set = set(committer_courses.keys())
    df_relevant = df_profiles[df_profiles["login"].isin(committer_set)]
    print(f"Committers with GitHub profile: {len(df_relevant)}")

    for _, profile in df_relevant.iterrows():
        login = profile["login"]
        location = profile.get("location")
        company = profile.get("company")
        name = profile.get("name") or login
        user_type = profile.get("type", "User")

        # Determine location string to geocode
        loc_str = None
        if pd.notna(location) and location.strip():
            loc_str = location.strip()
        elif pd.notna(company) and company.strip():
            # Try company as fallback
            loc_str = company.strip().lstrip("@")

        coords = geocode_location(geolocator, loc_str, cache)

        if coords:
            geocoded_count += 1
            courses = committer_courses.get(login, [])
            results.append({
                "login": login,
                "name": name,
                "type": user_type,
                "location_raw": location if pd.notna(location) else None,
                "company": company if pd.notna(company) else None,
                "lat": coords[0],
                "lon": coords[1],
                "course_count": len(courses),
                "courses": courses,
            })
        else:
            skipped_count += 1

        # Rate limiting for Nominatim (1 req/sec policy)
        time.sleep(1.1)

    save_cache(cache)

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGeocoding complete:")
    print(f"  Geocoded: {geocoded_count}")
    print(f"  Skipped:  {skipped_count}")
    print(f"  Output:   {OUTPUT_FILE}")
    print(f"  Cache:    {CACHE_FILE} ({len(cache)} entries)")


if __name__ == "__main__":
    main()
