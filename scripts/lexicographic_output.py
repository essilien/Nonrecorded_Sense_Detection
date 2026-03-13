#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lexicographic_output.py

Convert XL-LEXEME clustering results into a lexicographic candidate draft file.

Example:
    python lexicographic_output.py \
      --results_dir xllexeme_results \
      --output_json lexicographic_candidates.json \
      --output_csv lexicographic_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def candidate_priority(cluster: Dict[str, Any]) -> float:
    return float(cluster.get("n_corpus", 0))


def make_candidate_entry(
    word_result: Dict[str, Any],
    cluster: Dict[str, Any],
    max_examples: int,
) -> Dict[str, Any]:
    return {
        "word": word_result["word"],
        "recorded_meaning": word_result.get("meaning", ""),
        "source_cluster_id": cluster["cluster_id"],
        "cluster_size": cluster.get("size", 0),
        "n_anchor": cluster.get("n_anchor", 0),
        "n_corpus": cluster.get("n_corpus", 0),
        "representative_examples": cluster.get("examples", [])[:max_examples],
        "cluster_label": "",
        "candidate_gloss": "",
        "lexicographic_status": "candidate_nonrecorded_sense",
        "confidence": "",
        "notes": "",
        "priority_score": candidate_priority(cluster),
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    overview = load_json(results_dir / "overview.json")
    status_by_word = {item["word"]: item for item in overview if item.get("status") == "ok"}

    candidates: List[Dict[str, Any]] = []

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "overview.json":
            continue

        word_result = load_json(json_file)
        word = word_result.get("word")
        if word not in status_by_word:
            continue

        summary = word_result.get("summary", {})
        candidate_cluster_ids = set(summary.get("candidate_clusters", []))

        for cluster in summary.get("clusters", []):
            cid = cluster["cluster_id"]
            if cid not in candidate_cluster_ids:
                continue
            if cluster.get("n_anchor", 0) != 0:
                continue
            if cluster.get("n_corpus", 0) < args.min_cluster_size:
                continue

            candidates.append(make_candidate_entry(word_result, cluster, args.max_examples))

    candidates.sort(key=lambda x: x["priority_score"], reverse=True)

    save_json(Path(args.output_json), candidates)

    csv_rows = []
    for item in candidates:
        csv_rows.append({
            "word": item["word"],
            "recorded_meaning": item["recorded_meaning"],
            "source_cluster_id": item["source_cluster_id"],
            "cluster_size": item["cluster_size"],
            "n_anchor": item["n_anchor"],
            "n_corpus": item["n_corpus"],
            "candidate_gloss": item["candidate_gloss"],
            "cluster_label": item["cluster_label"],
            "confidence": item["confidence"],
            "notes": item["notes"],
            "priority_score": item["priority_score"],
            "representative_examples": " || ".join(item["representative_examples"]),
        })

    write_csv(
        Path(args.output_csv),
        csv_rows,
        fieldnames=[
            "word",
            "recorded_meaning",
            "source_cluster_id",
            "cluster_size",
            "n_anchor",
            "n_corpus",
            "candidate_gloss",
            "cluster_label",
            "confidence",
            "notes",
            "priority_score",
            "representative_examples",
        ],
    )

    print(f"Saved {len(candidates)} candidate entries.")
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
