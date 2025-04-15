# Residue-Residue Contact Prediction using ESM2 + Structural Priors

This repository provides a machine learning pipeline for predicting residue-residue contacts within protein sequences, leveraging transformer-based protein language models (ESM2) and additional structural priors from similar proteins.

---

## 🚀 Project Overview

- **Objective:** Predict binary contact maps from a single protein sequence.
- **Key Idea:** Use ESM2 token embeddings and enhance predictions by integrating structural contact maps of similar proteins.
- **Input:** A protein sequence (1-letter amino acid codes).
- **Output:** A binary contact map (NxN matrix) indicating contacts between residue pairs.

---

## 🧠 Methodology

### 🔬 Extension of ESM2

- ESM2 provides contextual embeddings per residue.
- A **pairwise classifier** uses ESM2 embeddings of residue pairs, their sequence distance (`|i - j|`), and optionally a structural **prior** (contact map of a similar protein).

### 💡 Structural Prior

- We pool ESM2 embeddings for each protein to find similar sequences using **cosine similarity**.
- The contact map of the top-1 (val) or top-2 (train) most similar protein is used as an additional feature.

---

## 🧰 Repository Structure

```
├── config.py                    # Global config values
├── preprocessing.py            # Data preprocessing classes
├── train.py                    # Model training loop
├── model.py                    # Model definition & evaluation
├── evaluate_test.py            # Evaluation on test set
├── data.py                     # Dataset loader
├── bucket_sampler.py           # Sequence length bucket batching
├── run_training_pipeline.py    # End-to-end run script
├── visualization.py            # Metric and distribution plots
└── requirements.txt
```

---

## 🔄 Pipeline Steps

### 1. **Preprocessing**

To run only preprocessing:
```bash
python preprocessing.py
```
This extracts:
- Protein sequences and coordinates
- Contact maps
- ESM2 token embeddings
- Structural priors from similar proteins

### 2. **Training**

To run only model training:
```bash
python train.py
```
This trains the model and logs:
- F1, Precision, ROC-AUC, PR-AUC
- Model saved based on **best PR-AUC**

### 3. **Full Pipeline**

To run preprocessing + training in one:
```bash
python run_training_pipeline.py
```

### 4. **Evaluation on Test Set**

```bash
python evaluate_test.py
```
- Evaluates raw `.pdb` files
- Saves metrics (global and per protein) to `test_set_evaluation.xlsx`

### 5. **Visualizations**
```bash
python visualization.py
```
- Generates plots for metrics, length distribution, label imbalance, etc.

---

## ⚙️ Key Hyperparameters

Defined in `config.py` (CFG):
- `esm_model_name`: ESM2 model name
- `train_batch_size`, `val_batch_size`
- `max_sequence_length`: default 750
- `atoms_distance_threshold`: contact threshold in angstroms
- `train_bucket_size`, `val_bucket_size`
- `epochs`, `lr`, etc.

---

## 📊 Output Artifacts

- `/models/`: Trained model checkpoints
- `/logs/<timestamp>/metrics.json`: F1, PR-AUC, etc. per epoch
- `/reporting/test_set_evaluation.xlsx`: Final test evaluation
- `/visualizations/`: PNG plots (metrics, density, etc.)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```
