# Nonrecorded Sense Detection in Mandarin Chinese from Weibo Texts
This project aims to detect non-recored words senses automatically in Mandarin Chinese based on recent word usages collected from Weibo texts during January to March, 2026. The corpura used in this project, including a monosemy corpus and a word usage corpus collected from Weibo, can be found in [Monosemy-targeted Corpus](https://github.com/essilien/Monosemy-targeted_Corpus).

## Repository Structure

```text
.
├── data/
│   ├── clustering_data/
│   ├── durel_graphs/
│   └── lexicographic_candidates.csv
├── scripts/
│   ├── xl_lexeme_cluster.py
│   ├── lexicographic_output.py
│   ├── visualization.py
│   └── inspect_clusters.py
└── lexicographic_entries_result.json
```

## Data

The `data/` directory contains the intermediate data and supporting materials used in the analysis.

### `clustering_data/`

This folder contains the full set of clustering outputs generated with XL-LEXEME. It is the renamed version of the original `xl_lexeme_results` directory.

Its contents include:

- clustered contextual usages
- anchor-containing clusters
- candidate clusters without anchors
- overview files for inspection

### `durel_graphs/`

This folder contains graph-based visualization outputs used for semantic analysis and interpretation.

### `lexicographic_candidates.csv`

This file contains candidate items selected for lexicographic inspection and further analysis.

## Scripts

The `scripts/` directory contains the main scripts used in the workflow.

### `xl_lexeme_cluster.py`

Runs clustering over target-word usages using XL-LEXEME embeddings. It takes anchor data and corpus usages as input and produces cluster-level outputs.

### `lexicographic_output.py`

Generates lexicographic-style outputs from the clustering results. It is used to transform candidate clusters into a dictionary-oriented format.

### `visualization.py`

Produces visualization outputs for semantic analysis and interpretation.

### `inspect_clusters.py`

Supports manual inspection of clustering results, including the review of cluster contents and candidate senses.

## Final Output

### `lexicographic_entries_result.json`

This is the final output file of the repository. It contains the lexicographic-style entries derived from the clustering and candidate selection process.

## Workflow Overview

1. Cluster contextual usages with `xl_lexeme_cluster.py`
2. Inspect and analyze cluster outputs
3. Generate visualizations where needed
4. Produce lexicographic-style entries with `lexicographic_output.py`
5. Save the final results in `lexicographic_entries_result.json`

## Notes

- `clustering_data/` is the renamed version of the original `xl_lexeme_results/` folder.
- The repository separates intermediate data, executable scripts, and final outputs for clarity.
