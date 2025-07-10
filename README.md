**This repository hosts the codebase for GTAB-AC.**

To run the code:

1. Follow the steps under `preprocessing/` to collect:
   - OTU abundance matrices (csv)
   - Association matrices (for feature enhancement, csv)
   - The SILVA refseq fasta (fasta)
2. Place all collected data in the project’s `data/` directory.

<br><br>
**Training**:

Execute `pl_ptrain.py` to train the model.

<br><br>
**Prediction**:

Run `pl_predict.py` to generate simulated predictions for pre-processed, feature-aligned datasets.

<br><br>
**External dataset**:

SILVA:

https://www.arb-silva.de/no_cache/download/archive/release_138.2/Exports/

Chen's:

https://doi.org/10.1080/19490976.2021.2025016
<br><br>
**Revision History**:

June 10, 2025

Create repository

June 18, 2025

The core implementation of the GTAB-AC framework has been made available, and the corresponding datasets and preprocessing pipeline will be released after further curation.


