# Nonrecorded Sense Detection in Mandarin Chinese from Weibo Texts
## Overview
The project applies clustering-based analysis to contextualized word usages in order to identify potential semantic extensions that are not yet recorded in standard lexicographic resources. The corpura used in this project, including a monosemy corpus and a word usage corpus collected from Weibo, can be found in [Monosemy-targeted Corpus](https://github.com/essilien/Monosemy-targeted_Corpus). The repository includes intermediate clustering outputs, visualization materials, candidate lists for inspection, processing scripts, and the final lexicographic-style result file.

## Data Basis

The analysis in this repository is based on Mandarin Chinese usage data collected from Weibo.

The workflow also relies on synthetic anchor contexts representing recorded dictionary meanings. These anchors serve as semantic reference points during clustering and help distinguish recorded-sense clusters from candidate non-recorded-sense clusters.

## Repository Structure

```text
.
├── data/
│   ├── clustering_data/
│   ├── durel_graphs/
│   └── lexicographic_candidates/
├── scripts/
│   ├── xl_lexeme_cluster.py
│   ├── lexicographic_output.py
│   ├── visualization.py
│   └── inspect_clusters.py
└── lexicographic_entries_result.json
```

## Data

The `data/` directory contains the intermediate data and supporting materials used in the detection workflow.

### `clustering_data/`

This folder contains the full set of clustering outputs generated with XL-LEXEME. It is the renamed version of the original `xl_lexeme_results` directory.

Its contents include word-level clustering results, anchor-associated clusters, candidate clusters without anchors, and overview files used for inspection and interpretation.

### `durel_graphs/`

This folder contains graph-based visualization outputs used for semantic analysis and interpretation.

### `lexicographic_candidates/`

The csv file contains candidate items selected for lexicographic inspection and further analysis. The raw data can be found in the json file.

## Scripts

The `scripts/` directory contains the main scripts used in the workflow.

### `xl_lexeme_cluster.py`

This script performs clustering over target-word usages using XL-LEXEME embeddings. It takes anchor data and corpus usages as input and produces cluster-level outputs.

### `lexicographic_output.py`

This script generates lexicographic-style outputs from clustering results. It is used to transform candidate clusters into a dictionary-oriented representation.

### `visualization.py`

This script produces visualization outputs for semantic analysis and interpretation.

### `inspect_clusters.py`

This script supports manual inspection of clustering results, including the review of cluster contents and candidate senses.

## Final Output

### `lexicographic_entries_result.json`

This is the final output file of the repository. It contains the lexicographic-style entries derived from the clustering and candidate selection process.

These entries represent the final candidate non-recorded senses identified through the pipeline.

## Workflow Overview

The repository is organized around the following workflow:

1. Cluster contextual usages with `xl_lexeme_cluster.py`
2. Inspect and review the clustering results
3. Generate visualizations for interpretation
4. Select candidate items for lexicographic analysis
5. Produce final lexicographic-style entries with `lexicographic_output.py`

## Limitations

Several limitations should be noted:

- Lexicographic monosemy does not always guarantee true semantic monosemy in actual language use.
- Social media texts are noisy, highly contextual, and often pragmatically underspecified.
- Candidate clusters do not automatically constitute validated new senses and may still require manual interpretation.
- Synthetic anchors are controlled semantic references rather than naturally occurring usage examples.

## License

This repository and its contents are intended for research use only.

Please ensure that any use of the data complies with applicable laws, institutional requirements, and the platform policies of the original data source.

## Contact

For questions about the scripts, data structure, or final outputs, please open an issue in this repository.
