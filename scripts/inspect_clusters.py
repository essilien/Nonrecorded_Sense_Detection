#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_clusters.py

Inspect XL-LEXEME clustering results and produce:
1. a ranked summary table of multi-cluster words
2. per-word compact inspection text files
3. a CSV for paper writing / error analysis

Input:
- xllexeme_results/overview.json
- xllexeme_results/*.json

Example:
    python inspect_clusters.py \
      --results_dir xllexeme_results \
      --output_dir xllexeme_inspection
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing overview.json and per-word result json files")
    parser.add_argument("--output_dir", required=True, help="Directory to save inspection outputs")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_candidate(item: Dict[str, Any]) -> float:
    if item.get("status") != "ok":
        return -1.0
    n_clusters = item.get("n_clusters", 0)
    candidate_clusters = item.get("candidate_clusters", []) or []
    n_items = item.get("n_total_items", 0)
    return 2.0 * n_clusters + 3.0 * len(candidate_clusters) + min(n_items / 20.0, 3.0)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_cluster_report(word_result: Dict[str, Any]) -> str:
    lines = []
    word = word_result["word"]
    meaning = word_result.get("meaning", "")
    prep = word_result.get("prep_stats", {})
    summary = word_result.get("summary", {})

    lines.append(f"WORD: {word}")
    lines.append(f"MEANING: {meaning}")
    lines.append(
        f"USABLE_ANCHORS={prep.get('n_anchor_after', 0)} | "
        f"USABLE_CORPUS={prep.get('n_corpus_after', 0)} | "
        f"N_CLUSTERS={summary.get('n_clusters', 0)}"
    )
    lines.append(f"ANCHOR_CLUSTERS={summary.get('anchor_clusters', [])}")
    lines.append(f"CANDIDATE_CLUSTERS={summary.get('candidate_clusters', [])}")
    lines.append("")

    for cluster in summary.get("clusters", []):
        cid = cluster["cluster_id"]
        lines.append(
            f"[CLUSTER {cid}] size={cluster['size']} "
            f"anchor={cluster['n_anchor']} corpus={cluster['n_corpus']}"
        )
        for ex in cluster.get("examples", []):
            lines.append(f"- {ex}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "inspection_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    overview = load_json(results_dir / "overview.json")

    all_rows = []
    multicluster_rows = []
    candidate_rows = []

    for item in overview:
        row = {
            "word": item.get("word"),
            "status": item.get("status"),
            "n_anchor_after": item.get("n_anchor_after"),
            "n_corpus_after": item.get("n_corpus_after"),
            "n_total_items": item.get("n_total_items", ""),
            "n_clusters": item.get("n_clusters", ""),
            "anchor_clusters": "|".join(map(str, item.get("anchor_clusters", []))) if item.get("anchor_clusters") else "",
            "candidate_clusters": "|".join(map(str, item.get("candidate_clusters", []))) if item.get("candidate_clusters") else "",
            "reason": item.get("reason", ""),
            "priority_score": f"{score_candidate(item):.2f}",
        }
        all_rows.append(row)

        if item.get("status") == "ok" and item.get("n_clusters", 0) > 1:
            multicluster_rows.append(row)

        if item.get("status") == "ok":
            cand = dict(row)
            cand["n_candidate_clusters"] = len(item.get("candidate_clusters", []) or [])
            candidate_rows.append(cand)

    candidate_rows.sort(key=lambda x: float(x["priority_score"]), reverse=True)
    multicluster_rows.sort(key=lambda x: float(x["priority_score"]), reverse=True)

    common_fields = [
        "word", "status", "n_anchor_after", "n_corpus_after", "n_total_items",
        "n_clusters", "anchor_clusters", "candidate_clusters", "reason", "priority_score"
    ]
    write_csv(output_dir / "summary_all_words.csv", all_rows, common_fields)
    write_csv(output_dir / "summary_multicluster_words.csv", multicluster_rows, common_fields)

    cand_fields = common_fields + ["n_candidate_clusters"]
    write_csv(output_dir / "candidate_priority.csv", candidate_rows, cand_fields)

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "overview.json":
            continue
        obj = load_json(json_file)
        report = compact_cluster_report(obj)
        with open(report_dir / f"{obj['word']}.txt", "w", encoding="utf-8") as f:
            f.write(report)

    multi_words = [r["word"] for r in multicluster_rows]
    skipped_words = [r["word"] for r in all_rows if r["status"] == "skipped"]
    one_cluster_words = [
        r["word"] for r in all_rows
        if r["status"] == "ok" and str(r["n_clusters"]) == "1"
    ]

    md = []
    md.append("# Cluster Inspection Summary")
    md.append("")
    md.append(f"- Total words in overview: {len(all_rows)}")
    md.append(f"- Successfully clustered: {sum(1 for r in all_rows if r['status']=='ok')}")
    md.append(f"- Skipped: {len(skipped_words)}")
    md.append(f"- Multi-cluster words: {len(multi_words)}")
    md.append("")
    md.append("## Multi-cluster words")
    md.append(", ".join(multi_words) if multi_words else "None")
    md.append("")
    md.append("## Single-cluster words")
    md.append(", ".join(one_cluster_words) if one_cluster_words else "None")
    md.append("")
    md.append("## Skipped words")
    md.append(", ".join(skipped_words) if skipped_words else "None")
    md.append("")
    md.append("## Highest-priority candidates")
    for row in candidate_rows[:10]:
        md.append(
            f"- {row['word']}: clusters={row['n_clusters']}, "
            f"candidate_clusters={row['candidate_clusters']}, "
            f"usable_corpus={row['n_corpus_after']}, score={row['priority_score']}"
        )

    with open(output_dir / "inspection_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved inspection outputs to {output_dir}")


if __name__ == "__main__":
    main()
