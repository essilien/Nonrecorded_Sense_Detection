#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
durel_style_visualize.py

Create DURel-style word usage graphs for all multi-cluster words.

Expected input:
- results_dir/overview.json
- results_dir/*.json   (one file per word)

Each per-word JSON should contain at least:
{
  "word": ...,
  "items": [...],
  "labels": [...],
  "embeddings": [...],
  "summary": {
      "anchor_clusters": [...],
      "candidate_clusters": [...],
      ...
  }
}

If `embeddings` are missing, this script will skip that word and report it.

Visualization style:
- node = usage
- edge weight = cosine similarity between XL-LEXEME usage embeddings
- node color = cluster id
- node shape = source type (anchor / corpus)
- candidate clusters = thicker black border
- anchor-supported clusters = shown in title/center labels

Example:
    python durel_style_visualize.py \
      --results_dir xllexeme_results \
      --output_dir durel_graphs

Recommended install:
    pip install matplotlib networkx numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing overview.json and per-word result json files")
    parser.add_argument("--output_dir", required=True, help="Directory to save graph PNGs")
    parser.add_argument("--min_similarity", type=float, default=0.45, help="Only draw edges with cosine similarity >= this value")
    parser.add_argument("--max_edges_per_node", type=int, default=6, help="Keep only top-k strongest edges per node above threshold")
    parser.add_argument("--figsize", type=float, nargs=2, default=[9, 7], help="Figure size, e.g. --figsize 9 7")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_color_map(unique_labels: List[int]) -> Dict[int, Any]:
    cmap = plt.get_cmap("tab10")
    color_map = {}
    for i, lab in enumerate(unique_labels):
        color_map[lab] = cmap(i % 10)
    return color_map


def build_sparse_graph(
    items: List[Dict[str, Any]],
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_similarity: float,
    max_edges_per_node: int,
) -> nx.Graph:
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, 0.0)

    G = nx.Graph()

    for idx, (item, lab) in enumerate(zip(items, labels)):
        G.add_node(
            idx,
            label=int(lab),
            source=item.get("source", "corpus"),
            text=item.get("text", ""),
            item_id=item.get("item_id", f"item_{idx}"),
        )

    n = len(items)
    for i in range(n):
        row = sims[i].copy()
        candidate_js = np.where(row >= min_similarity)[0]
        if len(candidate_js) == 0:
            continue

        # keep strongest top-k
        candidate_js = sorted(candidate_js, key=lambda j: row[j], reverse=True)[:max_edges_per_node]

        for j in candidate_js:
            if i >= j:
                continue
            G.add_edge(i, j, weight=float(row[j]))

    return G


def draw_word_graph(
    word_obj: Dict[str, Any],
    output_path: Path,
    min_similarity: float,
    max_edges_per_node: int,
    figsize: tuple[float, float],
    dpi: int,
) -> bool:
    word = word_obj["word"]
    items = word_obj.get("items")
    labels = word_obj.get("labels")
    embeddings = word_obj.get("embeddings")
    summary = word_obj.get("summary", {})
    anchor_clusters = set(summary.get("anchor_clusters", []))
    candidate_clusters = set(summary.get("candidate_clusters", []))

    if not items or labels is None or embeddings is None:
        return False

    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings, dtype=float)

    if len(items) != len(labels) or len(items) != len(embeddings):
        return False

    G = build_sparse_graph(
        items=items,
        labels=labels,
        embeddings=embeddings,
        min_similarity=min_similarity,
        max_edges_per_node=max_edges_per_node,
    )

    pos = nx.spring_layout(G, seed=42, weight="weight")

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    color_map = make_color_map(unique_labels)

    # put isolated nodes somewhere if needed
    for idx in range(len(items)):
        if idx not in pos:
            pos[idx] = np.random.RandomState(idx).rand(2)

    plt.figure(figsize=figsize)

    # edges
    if G.number_of_edges() > 0:
        for u, v, edata in G.edges(data=True):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=0.5 + 2.5 * edata["weight"],
                alpha=min(0.9, max(0.15, edata["weight"])),
                edge_color="gray",
            )

    # nodes by source and cluster
    markers = {"anchor": "o", "corpus": "^"}
    for cluster_id in unique_labels:
        for source, marker in markers.items():
            nodes = [
                n for n, data in G.nodes(data=True)
                if data["label"] == cluster_id and data["source"] == source
            ]
            if not nodes:
                continue

            edgecolors = []
            linewidths = []
            for _ in nodes:
                if cluster_id in candidate_clusters:
                    edgecolors.append("black")
                    linewidths.append(1.4)
                else:
                    edgecolors.append("white")
                    linewidths.append(0.8)

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_color=[color_map[cluster_id]] * len(nodes),
                node_shape=marker,
                node_size=150 if source == "anchor" else 120,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=0.92,
            )

    # cluster labels at cluster centers
    for cluster_id in unique_labels:
        cluster_nodes = [i for i, lab in enumerate(labels) if int(lab) == cluster_id]
        if not cluster_nodes:
            continue

        xy = np.array([pos[n] for n in cluster_nodes])
        center = xy.mean(axis=0)

        tag_parts = []
        if cluster_id in anchor_clusters:
            tag_parts.append("anchor")
        if cluster_id in candidate_clusters:
            tag_parts.append("candidate")
        tag = "/".join(tag_parts) if tag_parts else "other"

        plt.text(
            center[0],
            center[1],
            f"C{cluster_id}\n{tag}",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )

    n_clusters = summary.get("n_clusters", len(unique_labels))
    plt.title(
        f"{word} | DURel-style usage graph\n"
        f"clusters={n_clusters} | anchor_clusters={sorted(anchor_clusters)} | candidate_clusters={sorted(candidate_clusters)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = load_json(results_dir / "overview.json")
    multi_cluster_words = {
        item["word"]
        for item in overview
        if item.get("status") == "ok" and item.get("n_clusters", 0) > 1
    }

    generated = []
    skipped = []

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "overview.json":
            continue

        obj = load_json(json_file)
        word = obj.get("word")
        if word not in multi_cluster_words:
            continue

        ok = draw_word_graph(
            word_obj=obj,
            output_path=output_dir / f"{word}_durel_graph.png",
            min_similarity=args.min_similarity,
            max_edges_per_node=args.max_edges_per_node,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
        )

        if ok:
            generated.append(word)
        else:
            skipped.append(word)

    manifest = {
        "generated": generated,
        "skipped": skipped,
        "n_generated": len(generated),
        "n_skipped": len(skipped),
        "note": "Words are skipped if embeddings/items/labels are missing from per-word result JSON."
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(generated)} DURel-style graphs.")
    if skipped:
        print("Skipped due to missing embeddings/items/labels:", ", ".join(skipped))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()