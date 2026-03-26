"""
Contributor-Repository Graph

Builds an interactive bipartite graph (Users <-> Repositories) to analyze
whether LiaScript courses are isolated activities or connected through
"super users" who contribute across many repositories.

Outputs: build/contributor_graph.html (standalone, interactive vis.js graph)

Usage:
    pipenv run python build_graph.py
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
BUILD_DIR = BASE_DIR / "build"

# Load shared config for data paths and colors
with open(PROJECT_ROOT / "papers" / "shared" / "config.yaml") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["data"]["base_path"]) / config["data"]["raw_folder"]

# Colors from shared config
COLOR_USER = "#2E86AB"
COLOR_REPO = "#A23B72"
COLOR_EDGE = "#cccccc"
COLOR_BRIDGE = "#F18F01"  # highlight for super-users


def load_data():
    """Load consolidated course data (pickle preferred, CSV fallback)."""
    pickle_path = RAW_DIR / "LiaScript_consolidated.p"
    csv_path = RAW_DIR / "LiaScript_consolidated.csv"

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {RAW_DIR}\n"
            f"The external drive may not be mounted. Mount it and try again."
        )

    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
    elif csv_path.exists():
        print(f"Pickle not found, using CSV fallback: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        available = [f.name for f in RAW_DIR.glob("LiaScript_*")]
        raise FileNotFoundError(
            f"Neither LiaScript_consolidated.p nor .csv found in {RAW_DIR}\n"
            f"Available files: {available}"
        )

    print(f"Loaded {len(df)} courses from {df['repo_url'].nunique()} repositories")
    return df


DOC_AGGREGATE_THRESHOLD = 20  # repos with more docs get aggregated


def build_tripartite_graph(df):
    """
    Build a tripartite graph: User <-> Document <-> Repository.

    - Each LiaScript document is a node connected to its contributors and its repo.
    - For repos with > DOC_AGGREGATE_THRESHOLD documents, documents are aggregated
      into a single summary node to keep the graph manageable.
    """
    G = nx.Graph()

    # Pre-compute repo-level stats
    repo_doc_counts = df.groupby(["repo_user", "repo_name"]).size().to_dict()

    # Collect all valid contributors per user (for user node stats)
    user_docs = defaultdict(set)       # user -> set of doc_ids
    user_repos = defaultdict(set)      # user -> set of repo_keys
    repo_contributors = defaultdict(set)

    for _, row in df.iterrows():
        contributors = row.get("contributors_list", [])
        if not isinstance(contributors, list) or not contributors:
            continue

        repo_key = f"{row['repo_user']}/{row['repo_name']}"
        n_docs_in_repo = repo_doc_counts.get((row["repo_user"], row["repo_name"]), 1)
        doc_name = row.get("file_name", "unknown.md")
        doc_id = f"{repo_key}/{doc_name}"

        valid_contributors = set()
        for user in set(contributors):
            if not user or not isinstance(user, str):
                continue
            user = user.strip()
            if user.lower() in ("unknown", "nan", ""):
                continue
            valid_contributors.add(user)
            user_docs[user].add(doc_id)
            user_repos[user].add(repo_key)
            repo_contributors[repo_key].add(user)

        if not valid_contributors:
            continue

        # Add repo node (once per repo)
        repo_node = f"repo:{repo_key}"
        if repo_node not in G:
            G.add_node(
                repo_node,
                label=repo_key,
                node_type="repo",
                contributor_count=0,  # updated later
                course_count=n_docs_in_repo,
            )

        if n_docs_in_repo > DOC_AGGREGATE_THRESHOLD:
            # Aggregate: single summary doc node per repo
            agg_node = f"doc_agg:{repo_key}"
            if agg_node not in G:
                G.add_node(
                    agg_node,
                    label=f"{n_docs_in_repo} documents",
                    node_type="doc_agg",
                    repo_key=repo_key,
                    doc_count=n_docs_in_repo,
                )
                G.add_edge(agg_node, repo_node, edge_type="contains")
            # Connect users to the aggregate node
            for user in valid_contributors:
                user_node = f"user:{user}"
                if not G.has_edge(user_node, agg_node):
                    G.add_edge(user_node, agg_node, edge_type="contributes")
        else:
            # Individual document node
            doc_node = f"doc:{doc_id}"
            G.add_node(
                doc_node,
                label=doc_name.replace(".md", ""),
                node_type="doc",
                repo_key=repo_key,
            )
            G.add_edge(doc_node, repo_node, edge_type="contains")
            for user in valid_contributors:
                user_node = f"user:{user}"
                G.add_edge(user_node, doc_node, edge_type="contributes")

    # Add/update user nodes with stats
    for user, docs in user_docs.items():
        user_node = f"user:{user}"
        G.add_node(
            user_node,
            label=user,
            node_type="user",
            repo_count=len(user_repos[user]),
            course_count=len(docs),
        )

    # Update repo contributor counts
    for repo_key, contribs in repo_contributors.items():
        repo_node = f"repo:{repo_key}"
        if repo_node in G:
            G.nodes[repo_node]["contributor_count"] = len(contribs)

    n_users = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "user")
    n_docs = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "doc")
    n_agg = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "doc_agg")
    n_repos = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "repo")

    print(f"\nTripartite graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Users: {n_users}")
    print(f"  Documents: {n_docs} (+ {n_agg} aggregated nodes)")
    print(f"  Repos: {n_repos}")

    return G


def compute_metrics(G):
    """Compute graph metrics to identify super-users and isolated clusters."""
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)

    # User nodes only
    user_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "user"]
    repo_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "repo"]

    # Degree distribution for users (= number of repos they contribute to)
    user_degrees = {n: G.degree(n) for n in user_nodes}
    multi_repo_users = {n: d for n, d in user_degrees.items() if d > 1}

    # Betweenness centrality (identifies bridge users)
    betweenness = nx.betweenness_centrality(G)
    top_bridges = sorted(
        [(n, bc) for n, bc in betweenness.items() if G.nodes[n]["node_type"] == "user"],
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    # Component analysis
    isolated_users = sum(
        1 for comp in components
        if len(comp) <= 2  # just one user + one repo
        and any(G.nodes[n]["node_type"] == "user" for n in comp)
    )

    metrics = {
        "total_components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "largest_component_pct": len(components[0]) / G.number_of_nodes() * 100 if components else 0,
        "isolated_pairs": isolated_users,
        "multi_repo_users": len(multi_repo_users),
        "single_repo_users": len(user_nodes) - len(multi_repo_users),
        "top_bridge_users": [
            {
                "user": G.nodes[n]["label"],
                "betweenness": round(bc, 4),
                "repos": G.nodes[n]["repo_count"],
                "courses": G.nodes[n]["course_count"],
            }
            for n, bc in top_bridges
        ],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("GRAPH METRICS")
    print("=" * 60)
    print(f"Connected components:     {metrics['total_components']}")
    print(f"Largest component:        {metrics['largest_component_size']} nodes "
          f"({metrics['largest_component_pct']:.1f}%)")
    print(f"Isolated pairs (1 user):  {metrics['isolated_pairs']}")
    print(f"Multi-repo users:         {metrics['multi_repo_users']}")
    print(f"Single-repo users:        {metrics['single_repo_users']}")
    print(f"\nTop bridge users (betweenness centrality):")
    for entry in metrics["top_bridge_users"][:10]:
        print(f"  {entry['user']:30s}  BC={entry['betweenness']:.4f}  "
              f"repos={entry['repos']}  courses={entry['courses']}")

    return metrics


COLOR_DOC = "#66BB6A"       # green for documents
COLOR_DOC_AGG = "#2E7D32"   # dark green for aggregated doc nodes


def export_vis_js(G, metrics):
    """Export graph as standalone HTML with vis.js for interactive visualization."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare node data for vis.js
    nodes = []
    betweenness = nx.betweenness_centrality(G)

    for node_id, data in G.nodes(data=True):
        ntype = data["node_type"]
        degree = G.degree(node_id)
        bc = betweenness.get(node_id, 0)

        if ntype == "user":
            size = max(8, min(50, 8 + data["repo_count"] * 6))
            color = COLOR_BRIDGE if data["repo_count"] > 3 else COLOR_USER
            shape = "dot"
            title = (
                f"<b>{data['label']}</b><br>"
                f"Repos: {data['repo_count']}<br>"
                f"Documents: {data['course_count']}<br>"
                f"Betweenness: {bc:.4f}"
            )
        elif ntype == "doc":
            size = 5
            color = COLOR_DOC
            shape = "square"
            title = (
                f"<b>{data['label']}</b><br>"
                f"Repository: {data['repo_key']}"
            )
        elif ntype == "doc_agg":
            size = max(10, min(35, 10 + data["doc_count"] * 0.2))
            color = COLOR_DOC_AGG
            shape = "square"
            title = (
                f"<b>{data['doc_count']} documents</b> (aggregated)<br>"
                f"Repository: {data['repo_key']}"
            )
        elif ntype == "repo":
            n_docs = data["course_count"]
            size = max(8, min(45, 8 + n_docs * 0.5 + data["contributor_count"] * 3))
            if n_docs >= 5:
                color = "#6A1B5A"
            elif n_docs >= 2:
                color = COLOR_REPO
            else:
                color = "#D4A0C0"
            shape = "diamond"
            title = (
                f"<b>{data['label']}</b><br>"
                f"Documents: {n_docs}<br>"
                f"Contributors: {data['contributor_count']}"
            )
        else:
            continue

        # Labels: super-users always, repos with multiple docs, aggregated doc nodes
        repo_short = data["label"].split("/")[-1] if "/" in data.get("label", "") else data.get("label", "")
        is_super = ntype == "user" and bc > 0.003

        if is_super:
            label = data["label"]
            font = {"size": 14 + min(bc * 80, 20), "color": "#222", "strokeWidth": 3, "strokeColor": "#fff"}
        elif ntype == "doc_agg":
            label = f"{data['doc_count']} docs"
            font = {"size": 9, "color": "#1B5E20", "strokeWidth": 2, "strokeColor": "#fff"}
        elif ntype == "repo" and data["course_count"] >= 2:
            label = f"{repo_short} ({data['course_count']})"
            font = {"size": 9, "color": "#666", "strokeWidth": 2, "strokeColor": "#fff"}
        elif ntype == "user" and degree > 1:
            label = data["label"]
            font = {"size": 10, "color": "#333"}
        else:
            label = ""
            font = {"size": 8, "color": "#999"}

        nodes.append({
            "id": node_id,
            "label": label,
            "title": title,
            "size": size,
            "color": color,
            "shape": shape,
            "font": font,
            "borderWidth": 2 if bc > 0.01 else 1,
        })

    # Prepare edge data — different colors for contribution vs. containment
    edges = []
    for u, v, edata in G.edges(data=True):
        etype = edata.get("edge_type", "")
        if etype == "contains":
            edge_color = {"color": "#BBBBBB", "opacity": 0.4}
            width = 1
            dashes = True
        else:
            edge_color = {"color": COLOR_EDGE, "opacity": 0.5}
            width = 1
            dashes = False
        edges.append({
            "from": u,
            "to": v,
            "width": width,
            "color": edge_color,
            "dashes": dashes,
        })

    # Stats for the sidebar
    stats_html = f"""
    <h3>Graph Overview</h3>
    <table>
      <tr><td>Connected components</td><td><b>{metrics['total_components']}</b></td></tr>
      <tr><td>Largest component</td><td><b>{metrics['largest_component_size']}</b> nodes ({metrics['largest_component_pct']:.1f}%)</td></tr>
      <tr><td>Isolated pairs</td><td><b>{metrics['isolated_pairs']}</b></td></tr>
      <tr><td>Multi-repo users</td><td><b>{metrics['multi_repo_users']}</b></td></tr>
      <tr><td>Single-repo users</td><td><b>{metrics['single_repo_users']}</b></td></tr>
    </table>
    <h3>Top Bridge Users</h3>
    <table>
      <tr><th>User</th><th>Repos</th><th>Courses</th></tr>
      {"".join(
          f'<tr><td>{e["user"]}</td><td>{e["repos"]}</td><td>{e["courses"]}</td></tr>'
          for e in metrics["top_bridge_users"][:15]
      )}
    </table>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LiaScript Contributor-Repository Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; }}
  #sidebar {{
    width: 320px; padding: 16px; overflow-y: auto;
    background: #f8f9fa; border-right: 1px solid #ddd; font-size: 13px;
  }}
  #sidebar h2 {{ margin-bottom: 12px; color: #333; }}
  #sidebar h3 {{ margin: 16px 0 8px 0; color: #555; font-size: 14px; }}
  #sidebar table {{ width: 100%; border-collapse: collapse; }}
  #sidebar td, #sidebar th {{ padding: 3px 6px; text-align: left; border-bottom: 1px solid #eee; }}
  #sidebar th {{ font-weight: 600; }}
  #graph {{ flex: 1; }}
  .legend {{ margin-top: 20px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .legend-dot {{ width: 14px; height: 14px; border-radius: 50%; }}
  .legend-diamond {{ width: 14px; height: 14px; transform: rotate(45deg); }}
  #controls {{ margin-top: 16px; }}
  #controls label {{ display: block; margin: 4px 0; cursor: pointer; }}
  #controls button {{
    margin-top: 8px; padding: 6px 12px; cursor: pointer;
    background: {COLOR_USER}; color: white; border: none; border-radius: 4px;
  }}
  #controls button:hover {{ opacity: 0.85; }}
</style>
</head>
<body>
<div id="sidebar">
  <h2>LiaScript Contributor Graph</h2>
  <p style="color:#666; margin-bottom:12px;">
    Tripartite graph: <b>Users</b> (circles) contribute to <b>Documents</b> (squares),
    which belong to <b>Repositories</b> (diamonds).
    <span style="color:{COLOR_BRIDGE}; font-weight:bold;">Orange</span> = super-users (&gt;3 repos).
  </p>

  <div id="controls">
    <label><input type="checkbox" id="hideIsolated"> Hide isolated nodes (degree=1)</label>
    <label><input type="checkbox" id="physicsToggle" checked> Physics simulation</label>
    <button onclick="network.fit()">Reset view</button>
  </div>

  <div class="legend">
    <h3>Users</h3>
    <div class="legend-item"><div class="legend-dot" style="background:{COLOR_USER};"></div> User (1-3 repos)</div>
    <div class="legend-item"><div class="legend-dot" style="background:{COLOR_BRIDGE};"></div> Super-User (&gt;3 repos)</div>
    <h3>Documents</h3>
    <div class="legend-item"><div style="width:12px;height:12px;background:{COLOR_DOC};"></div> LiaScript document</div>
    <div class="legend-item"><div style="width:12px;height:12px;background:{COLOR_DOC_AGG};"></div> Aggregated ({DOC_AGGREGATE_THRESHOLD}+ docs)</div>
    <h3>Repositories</h3>
    <div class="legend-item"><div class="legend-diamond" style="background:#D4A0C0;"></div> Repository (1 doc)</div>
    <div class="legend-item"><div class="legend-diamond" style="background:{COLOR_REPO};"></div> Repository (2-4 docs)</div>
    <div class="legend-item"><div class="legend-diamond" style="background:#6A1B5A;"></div> Repository (5+ docs)</div>
    <h3>Edges</h3>
    <div class="legend-item"><div style="width:20px;border-top:2px solid #ccc;"></div> User &rarr; Document</div>
    <div class="legend-item"><div style="width:20px;border-top:2px dashed #bbb;"></div> Document &rarr; Repository</div>
  </div>

  {stats_html}
</div>
<div id="graph"></div>

<script>
const allNodes = {json.dumps(nodes)};
const allEdges = {json.dumps(edges)};

const nodesDataset = new vis.DataSet(allNodes);
const edgesDataset = new vis.DataSet(allEdges);

const container = document.getElementById('graph');
const data = {{ nodes: nodesDataset, edges: edgesDataset }};
const options = {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -40,
      centralGravity: 0.005,
      springLength: 100,
      springConstant: 0.02,
      damping: 0.4,
    }},
    stabilization: {{ iterations: 200 }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    navigationButtons: true,
    keyboard: true,
  }},
  nodes: {{
    borderWidth: 1,
    shadow: true,
  }},
  edges: {{
    smooth: {{ type: 'continuous' }},
  }},
}};

const network = new vis.Network(container, data, options);

// Hide isolated toggle
document.getElementById('hideIsolated').addEventListener('change', function() {{
  const hide = this.checked;
  const updates = allNodes
    .filter(n => {{
      const edges = edgesDataset.get({{ filter: e => e.from === n.id || e.to === n.id }});
      return edges.length <= 1;
    }})
    .map(n => ({{ id: n.id, hidden: hide }}));
  nodesDataset.update(updates);
}});

// Physics toggle
document.getElementById('physicsToggle').addEventListener('change', function() {{
  network.setOptions({{ physics: {{ enabled: this.checked }} }});
}});
</script>
</body>
</html>"""

    output_file = BUILD_DIR / "contributor_graph.html"
    output_file.write_text(html, encoding="utf-8")
    print(f"\nInteractive graph saved to: {output_file}")
    return output_file


def main():
    df = load_data()
    G = build_tripartite_graph(df)
    metrics = compute_metrics(G)
    export_vis_js(G, metrics)


if __name__ == "__main__":
    main()
