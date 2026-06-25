# OncoVNN

**OncoVNN: Interpretable Mapping of Gene–Drug and Pathway–Drug Interactions in Cancer**

---

## Overview

OncoVNN is a biologically structured neural network for cancer drug response prediction. It combines:

- **ChemBERTa transformer** — encodes drug molecular structure from SMILES strings (768-dim embeddings)
- **Visible Neural Network (VNN)** — encodes gene expression via a pruned Gene Ontology (GO) DAG, where every hidden unit corresponds to a specific GO biological process
- **Parallel MLP** — unconstrained data-driven complement to the VNN branch
- **Hierarchical VNN GradCAM** — attribution framework that propagates predictive importance through the GO hierarchy to produce gene-level and pathway-level relevance scores

The model achieves Pearson *r* = 0.9344 on GDSC2 data while remaining fully interpretable through the GO structure.

---

## Repository Structure

```
OncoVNN/
├── OncoVNN.py          # Model definition (VNN, MLP, ModularClassifier, OncoVNN)
├── data_loader.py      # GDSC data loading and train/test splitting
├── data_encoding.py    # Drug SMILES → ChemBERTa embedding; RNA preprocessing
├── drug_bert_model.py  # ChemBERTa tokenizer and mean-pooling encoder
├── util.py             # Ontology/mapping file parsers
├── attribution.py      # Hierarchical VNN GradCAM analysis and plots
├── requirements.txt
├── data/
│   ├── gene2id_mapping.txt      # Gene symbol → integer ID mapping (965 genes)
│   └── ontology_structure.txt   # GO DAG with gene–term and term–term edges
├── ESPF/
│   ├── drug_codes_chembl_freq_1500.txt
│   └── subword_units_map_chembl_freq_1500.csv
└── paper/
    ├── oncovnn_paper.tex    # LaTeX source for the manuscript
    └── architecture.tex     # Architecture figure (TikZ)
```

---

## Data Requirements

The following files are **not included** in this repository due to their size. Download them from the sources below and place them in a `GDSC_data/` directory:

| File | Source |
|------|--------|
| `GDSC2_fitted_dose_response_25Feb20.xlsx` | [GDSC Portal](https://www.cancerrxgene.org/downloads/bulk_download) |
| `filtered_gene_expression.txt` | Derived from GDSC2 cell line RNA-seq (DepMap/CCLE) |
| `smile_inchi.csv` | Drug SMILES strings (from PubChem via GDSC drug list) |
| `Drug_listTue_Aug10_2021.csv` | GDSC drug metadata |

---

## Installation

```bash
git clone https://github.com/ShahriyariLab/OncoVNN.git
cd OncoVNN
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python OncoVNN.py --train --use_transformer --use_vnn --use_mlp \
                  --epochs 20 --modeldir ./model_OncoVNN --random_seed 42
```

### Evaluation

```bash
python OncoVNN.py --test --use_transformer --use_vnn --use_mlp \
                  --model_path ./model_OncoVNN/model.pt
```

### Hierarchical VNN GradCAM Attribution

After training, run attribution to produce gene-level and pathway-level scores, ranked bar charts, clustered heatmaps, and a PCA biplot:

```bash
python attribution.py --model_path ./model_OncoVNN/model.pt \
                      --topk 10 --outdir ./results/attribution

# Optionally add STRING protein–protein interaction network (requires internet):
python attribution.py --model_path ./model_OncoVNN/model.pt --with_string
```

Output files in `./results/attribution/`:
- `drug_ig_attribution.csv` — full drugs × genes attribution matrix
- `drug_top_genes.csv` — top-K genes per drug with direction labels
- `drug_gene_heatmap.png` — clustered gene-drug attribution heatmap
- `drug_gene_barplots.png` — ranked bar charts per representative drug
- `drug_pathway_heatmap.png` — GO term relevance heatmap
- `drug_attribution_pca.png` — PCA biplot of drug attribution profiles
- `drug_gene_networks.png` — STRING PPI network per drug (with `--with_string`)
