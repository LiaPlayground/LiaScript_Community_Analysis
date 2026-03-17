"""
Build an interactive Folium map from geocoded committer data.

Reads: data/geocoded_users.json
Outputs: build/author_map.html
"""

import json
from html import escape
from pathlib import Path

import folium
from folium.plugins import MarkerCluster

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "geocoded_users.json"
BUILD_DIR = BASE_DIR / "build"
OUTPUT_FILE = BUILD_DIR / "author_map.html"

# Colors by user type
COLORS = {
    "User": "#2E86AB",
    "Organization": "#A23B72",
}


def make_popup_html(user: dict) -> str:
    """Create HTML popup content for a map marker."""
    raw_name = user.get("name")
    name = escape(str(raw_name) if raw_name and raw_name == raw_name else user["login"])
    login = escape(user["login"])
    location = escape(user.get("location_raw") or "")
    company = escape(user.get("company") or "")
    user_type = user["type"]
    courses = user.get("courses", [])

    header_color = COLORS.get(user_type, "#333")

    lines = [
        f'<div style="min-width:280px; max-width:400px; font-family: sans-serif; font-size:13px;">',
        f'<h4 style="margin:0 0 4px 0; color:{header_color};">{name}</h4>',
        f'<p style="margin:0 0 2px 0; color:#666;">',
        f'  <a href="https://github.com/{login}" target="_blank">@{login}</a>',
        f'  &middot; <em>{user_type}</em>',
    ]
    if location:
        lines.append(f'  <br/>{location}')
    if company:
        lines.append(f'  <br/>{company}')
    lines.append('</p>')

    if courses:
        lines.append(f'<p style="margin:8px 0 4px 0;"><strong>{len(courses)} course{"s" if len(courses) != 1 else ""}:</strong></p>')
        lines.append('<ul style="margin:0; padding-left:18px; max-height:200px; overflow-y:auto;">')
        for c in courses[:50]:  # Limit to 50 in popup
            repo = escape(c.get("repo_name", ""))
            fname = escape(c.get("file_name", ""))
            lia_url = escape(c.get("lia_url", ""))
            label = f"{repo}/{fname}" if repo else fname
            lines.append(
                f'<li style="margin-bottom:2px;">'
                f'<a href="{lia_url}" target="_blank" title="Open in LiaScript">{label}</a>'
                f'</li>'
            )
        if len(courses) > 50:
            lines.append(f'<li>... and {len(courses) - 50} more</li>')
        lines.append('</ul>')
    else:
        lines.append('<p style="color:#999;"><em>No validated courses found</em></p>')

    lines.append('</div>')
    return "\n".join(lines)


def build_map(users: list) -> folium.Map:
    """Build a Folium map with clustered markers."""
    # Center on Europe (most users are there)
    m = folium.Map(
        location=[48.0, 10.0],
        zoom_start=3,
        tiles="CartoDB positron",
    )

    # Separate clusters for Users and Organizations
    cluster_users = MarkerCluster(name="Users", show=True)
    cluster_orgs = MarkerCluster(name="Organizations", show=True)

    for user in users:
        lat, lon = user["lat"], user["lon"]
        popup_html = make_popup_html(user)
        user_type = user["type"]
        color = "blue" if user_type == "User" else "purple"
        icon_name = "user" if user_type == "User" else "building"

        display_name = user.get("name") or user["login"]
        if not isinstance(display_name, str):
            display_name = user["login"]

        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=f"{display_name} ({user['course_count']} courses)",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        )

        if user_type == "Organization":
            marker.add_to(cluster_orgs)
        else:
            marker.add_to(cluster_users)

    cluster_users.add_to(m)
    cluster_orgs.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Stats box
    total = len(users)
    with_courses = sum(1 for u in users if u["course_count"] > 0)
    total_courses = sum(u["course_count"] for u in users)
    n_users = sum(1 for u in users if u["type"] == "User")
    n_orgs = sum(1 for u in users if u["type"] == "Organization")

    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:12px 16px; border-radius:8px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family:sans-serif; font-size:13px;">
      <strong>LiaScript Committer Map</strong><br/>
      {total} geocoded committers<br/>
      {total_courses} course contributions<br/>
      <em style="color:#999; font-size:11px;">(307 unique committers, {total} with location)</em>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE) as f:
        users = json.load(f)

    print(f"Loaded {len(users)} geocoded users")
    print(f"  Users: {sum(1 for u in users if u['type'] == 'User')}")
    print(f"  Organizations: {sum(1 for u in users if u['type'] == 'Organization')}")
    print(f"  With courses: {sum(1 for u in users if u['course_count'] > 0)}")
    print(f"  Total courses: {sum(u['course_count'] for u in users)}")

    m = build_map(users)
    m.save(str(OUTPUT_FILE))
    print(f"\nMap saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
