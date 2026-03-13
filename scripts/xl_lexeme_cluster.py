#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xl_lexeme_cluster.py

Cluster the first N target words with XL-LEXEME using:
- monosemy_corpus_with_anchors.json
- weibo_usage_corpus_clean.json

Google Colab setup:
    !pip install -q scikit-learn pandas numpy matplotlib
    !pip install -q git+https://github.com/pierluigic/xl-lexeme.git

Example:
    !python xl_lexeme_cluster.py \
        --monosemy monosemy_corpus_with_anchors.json \
        --corpus weibo_usage_corpus_clean.json \
        --output_dir xllexeme_results \
        --first_n 40
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monosemy", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--first_n", type=int, default=40)
    parser.add_argument("--min_corpus_usages", type=int, default=15)
    parser.add_argument("--min_anchor_usages", type=int, default=3)
    parser.add_argument("--distance_threshold", type=float, default=0.35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_items_per_word", type=int, default=60)
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BAD_SUBSTRINGS = [
    "超话", "展开 c", "微博正文", "发布了头条文章", "合集", "随机卡", "LOL", "凹3",
    "老师上车", "老师行不行", "见🍎", "前文指路", "专栏 ·", "VS", "yyds"
]
EMOJI_HEAVY_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
NON_TEXTY_RE = re.compile(r"[🚗🍗🧑🍳👯♀️🙋🏻♀️💁🏻♀️😱😥😩🙄🤯]+")
HASHTAG_LEFTOVER_RE = re.compile(r"[#@]|O\s+\S+")
TOPIC_MARK_RE = re.compile(r"^")
REPEATED_PUNCT_RE = re.compile(r"[!！?？~～.。]{4,}")


def extra_weibo_filter(text: str, target_word: str) -> bool:
    if not text or target_word not in text:
        return True
    if len(text.strip()) < 6:
        return True
    if TOPIC_MARK_RE.search(text):
        return True
    if any(s in text for s in BAD_SUBSTRINGS):
        return True
    if NON_TEXTY_RE.search(text):
        return True
    if REPEATED_PUNCT_RE.search(text):
        return True
    if len(EMOJI_HEAVY_RE.findall(text)) >= 3:
        return True
    if HASHTAG_LEFTOVER_RE.search(text) and len(text) < 12:
        return True
    return False

def load_xl_lexeme():
    import os
    import torch
    import transformers
    import huggingface_hub
    import sentence_transformers.util as st_util

    print("huggingface_hub:", getattr(huggingface_hub, "__version__", "unknown"))
    print("sentence_transformers:", getattr(__import__("sentence_transformers"), "__version__", "unknown"))
    print("transformers:", getattr(transformers, "__version__", "unknown"))

    # ---- compatibility patch: huggingface_hub ----
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return (
                    os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    or os.environ.get("HUGGINGFACE_TOKEN")
                )
        huggingface_hub.HfFolder = HfFolder

    if not hasattr(huggingface_hub, "Repository"):
        class Repository:
            def __init__(self, *args, **kwargs):
                pass
        huggingface_hub.Repository = Repository

    # ---- compatibility patch: sentence_transformers.util.snapshot_download ----
    if not hasattr(st_util, "snapshot_download"):
        from huggingface_hub import snapshot_download as hf_snapshot_download
        st_util.snapshot_download = hf_snapshot_download

    # ---- compatibility patch: transformers.AdamW ----
    transformers.AdamW = torch.optim.AdamW

    from WordTransformer import WordTransformer, InputExample  # type: ignore
    model = WordTransformer("pierluigic/xl-lexeme")
    return model, InputExample

def encode_occurrences(model, InputExample, items: List[Dict[str, Any]], batch_size: int = 16) -> np.ndarray:
    examples = []
    for item in items:
        examples.append(InputExample(texts=item["text"], positions=[item["target_start"], item["target_end"]]))
    emb = model.encode(examples, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(emb)


def prepare_word_items(
    word_entry: Dict[str, Any],
    corpus_records: List[Dict[str, Any]],
    min_corpus_usages: int,
    min_anchor_usages: int,
    max_items_per_word: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    word = word_entry["word"]
    anchors = word_entry.get("anchor_contexts", []) or []

    kept_anchors = []
    for i, sent in enumerate(anchors):
        if not sent or word not in sent:
            continue
        start = sent.find(word)
        kept_anchors.append({
            "item_id": f"{word}_anchor_{i+1}",
            "source": "anchor",
            "text": sent.strip(),
            "target_start": start,
            "target_end": start + len(word),
        })

    kept_corpus = []
    for rec in corpus_records:
        if rec.get("target_word") != word:
            continue
        if not rec.get("keep_for_analysis", False):
            continue
        text = rec.get("normalized_text", "") or rec.get("full_context", "")
        if extra_weibo_filter(text, word):
            continue
        kept_corpus.append({
            "item_id": rec.get("usage_id"),
            "source": "corpus",
            "text": text.strip(),
            "target_start": rec.get("target_start", text.find(word)),
            "target_end": rec.get("target_end", text.find(word) + len(word)),
            "post_id": rec.get("post_id"),
        })

    seen = set()
    deduped_corpus = []
    for item in kept_corpus:
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        deduped_corpus.append(item)
    kept_corpus = deduped_corpus

    stats = {
        "word": word,
        "meaning": word_entry.get("meaning", ""),
        "n_anchor_before": len(anchors),
        "n_anchor_after": len(kept_anchors),
        "n_corpus_after": len(kept_corpus),
    }

    if len(kept_anchors) < min_anchor_usages or len(kept_corpus) < min_corpus_usages:
        return [], stats

    total_items = kept_anchors + kept_corpus
    if len(total_items) > max_items_per_word:
        room_for_corpus = max_items_per_word - len(kept_anchors)
        kept_corpus = kept_corpus[:max(0, room_for_corpus)]
        total_items = kept_anchors + kept_corpus

    return total_items, stats


def run_agglomerative(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    return model.fit_predict(embeddings)


def summarize_clusters(items: List[Dict[str, Any]], labels: np.ndarray) -> Dict[str, Any]:
    clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item, lab in zip(items, labels):
        clusters[int(lab)].append(item)

    cluster_summaries = []
    anchor_clusters = set()

    for lab, members in sorted(clusters.items(), key=lambda x: x[0]):
        source_counts = Counter(m["source"] for m in members)
        if source_counts.get("anchor", 0) > 0:
            anchor_clusters.add(lab)
        cluster_summaries.append({
            "cluster_id": lab,
            "size": len(members),
            "n_anchor": source_counts.get("anchor", 0),
            "n_corpus": source_counts.get("corpus", 0),
            "examples": [m["text"] for m in members[:5]],
            "items": members,
        })

    candidate_clusters = [c["cluster_id"] for c in cluster_summaries if c["n_anchor"] == 0 and c["n_corpus"] > 0]

    return {
        "n_clusters": len(cluster_summaries),
        "anchor_clusters": sorted(anchor_clusters),
        "candidate_clusters": candidate_clusters,
        "clusters": cluster_summaries,
    }


def main() -> None:
    args = parse_args()

    monosemy = load_json(args.monosemy)
    corpus = load_json(args.corpus)

    target_entries = monosemy[: args.first_n]
    target_words = [x["word"] for x in target_entries]

    corpus_by_word = defaultdict(list)
    for rec in corpus:
        w = rec.get("target_word")
        if w in target_words:
            corpus_by_word[w].append(rec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, InputExample = load_xl_lexeme()

    overview = []

    for idx, entry in enumerate(target_entries, start=1):
        word = entry["word"]
        items, prep_stats = prepare_word_items(
            word_entry=entry,
            corpus_records=corpus_by_word[word],
            min_corpus_usages=args.min_corpus_usages,
            min_anchor_usages=args.min_anchor_usages,
            max_items_per_word=args.max_items_per_word,
        )

        if not items:
            overview.append({
                "word": word,
                "status": "skipped",
                **prep_stats,
                "reason": "too_few_usable_items",
            })
            print(f"[{idx}/{len(target_entries)}] SKIP {word}: too few usable items")
            continue

        print(f"[{idx}/{len(target_entries)}] ENCODE {word}: {len(items)} items", flush=True)
        embeddings = encode_occurrences(model, InputExample, items, batch_size=args.batch_size)
        labels = run_agglomerative(embeddings, distance_threshold=args.distance_threshold)
        summary = summarize_clusters(items, labels)

        result = {
            "word": word,
            "meaning": entry.get("meaning", ""),
            "prep_stats": prep_stats,
            "distance_threshold": args.distance_threshold,
            "summary": summary,
        }

        with open(output_dir / f"{idx:02d}_{word}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        overview.append({
            "word": word,
            "status": "ok",
            **prep_stats,
            "n_total_items": len(items),
            "n_clusters": summary["n_clusters"],
            "anchor_clusters": summary["anchor_clusters"],
            "candidate_clusters": summary["candidate_clusters"],
        })
        print(f"[{idx}/{len(target_entries)}] DONE {word}: {summary['n_clusters']} clusters", flush=True)

    with open(output_dir / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
