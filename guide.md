# GNN Challenge Repository - Complete Guide

## Table of Contents

1. [What Is This Repo?](#1-what-is-this-repo)
2. [Repository Structure](#2-repository-structure)
3. [How Each File Works](#3-how-each-file-works)
4. [Issues Found in the Original Code](#4-issues-found-in-the-original-code)
5. [Step-by-Step Changes Made](#5-step-by-step-changes-made)
6. [How to Run Everything](#6-how-to-run-everything)

---

## 1. What Is This Repo?

This is a **GNN (Graph Neural Network) challenge platform** for **node classification** on a citation network.

### The Task

- You have a **graph** of scientific papers (nodes) connected by citation links (edges).
- Each paper has a **feature vector** (bag-of-words representation of its content).
- Each paper belongs to a **class** (disease category).
- The goal: **predict the class of test papers** using a Graph Neural Network.

### The Dataset: PubMed

The actual dataset in the repo is **PubMed** (not Cora as the original README claimed):

| Property | Value |
|----------|-------|
| Nodes | 19,717 papers |
| Edges | 88,648 citation links |
| Node Features | 500-dimensional vector per paper |
| Classes | 3 (Diabetes Mellitus Experimental, Type 1, Type 2) |
| Train/Val/Test Split | 60% / 20% / 20% |

### The Workflow

```
[prepare_dataset.py]  -->  data/raw/, data/splits/, data/processed/
                                  |
                                  v
[starter_code/baseline.py]  -->  Train a GCN model  -->  results/best_model.pt
                                                                |
                                                                v
[starter_code/generate_submission.py]  -->  submissions/submission.private.csv
                                                      |
                                                      v
[scoring_script.py]  -->  Compare predictions vs hidden test labels  -->  leaderboard.json
                                                                              |
                                                                              v
[update_leaderboard.py]  -->  leaderboard.md
                                    |
                                    v
[.github/workflows/score.yml]  -->  Automates scoring on PRs
```

---

## 2. Repository Structure

```
gnn/
├── data/
│   ├── raw/                          # Raw graph data
│   │   ├── features.pt               # Node feature matrix [19717, 500]
│   │   ├── edges.pt                  # Edge index [2, 88648]
│   │   ├── labels.pt                 # All node labels [19717]
│   │   └── metadata.json             # Dataset metadata (counts, class names, etc.)
│   │
│   ├── splits/                       # Train/val/test masks
│   │   ├── train_mask.pt             # Boolean mask: which nodes are training
│   │   ├── val_mask.pt               # Boolean mask: which nodes are validation
│   │   └── test_mask.pt              # Boolean mask: which nodes are test
│   │
│   ├── processed/                    # Processed label files
│   │   ├── train_labels.pt           # Labels with -1 for non-training nodes
│   │   ├── val_labels.pt             # Labels with -1 for non-validation nodes
│   │   └── test_labels_hidden.private.pt     # Labels with -1 for non-test nodes (for scoring)
│   │
│   └── README.md                     # Data documentation
│
├── starter_code/
│   ├── baseline.py                   # GCN training script (PyTorch Geometric)
│   ├── generate_submission.py        # Generate submission CSV from trained model
│   └── requirements.txt              # Python dependencies (NEW - was missing)
│
├── submissions/                      # Where submission CSV files go
│   ├── submission.private.csv        # A submission file (gitignored)
│   ├── sample_submission.private     # Sample submission (private)
│   └── sample_submission.private.csv # Sample submission CSV (private)
│
├── prepare_dataset.py                # Downloads PubMed and creates data/ structure
├── scoring_script.py                 # Scores a submission CSV against ground truth
├── update_leaderboard.py             # Generates leaderboard.md from leaderboard.json
├── leaderboard.json                  # Machine-readable leaderboard data
├── leaderboard.md                    # Human-readable leaderboard table
├── Leaderboard.html                  # HTML version of leaderboard
├── README.md                         # Project overview and instructions
└── .github/workflows/score.yml       # CI: auto-scores on PR submissions
```

---

## 3. How Each File Works

### `prepare_dataset.py` - Dataset Preparation

**Purpose**: Downloads the PubMed dataset and saves it in organized `.pt` files.

**What it does**:
1. Downloads PubMed via `torch_geometric.datasets.Planetoid`
2. Extracts features (500-dim vectors), labels (0/1/2), and edges (citation links)
3. Creates a stratified 60/20/20 train/val/test split using `sklearn.train_test_split`
4. Saves everything as PyTorch tensors:
   - `data/raw/` - raw features, edges, labels, and metadata JSON
   - `data/splits/` - boolean masks for each split
   - `data/processed/` - label tensors where non-split nodes are set to -1

**When to run**: Only if you need to regenerate the data from scratch. The `data/` folder already has the files.

---

### `starter_code/baseline.py` - Model Training

**Purpose**: Trains a 2-layer GCN (Graph Convolutional Network) baseline model.

**Key components**:

- **`GCN` class** (line 56): The model architecture using `torch_geometric.nn.GCNConv`
  - Input layer: 500 features -> 16 hidden units
  - Output layer: 16 hidden units -> 3 classes
  - ReLU activation + dropout between layers
  - Log-softmax output for classification

- **`PubMedDataLoader` class** (line 87): Loads all `.pt` files from the data directory

- **`train_epoch` function** (line 135): One training step - forward pass, NLL loss on training nodes, backprop

- **`evaluate` function** (line 157): Computes accuracy and F1 on a given split

- **Training loop** (line 297): Trains for N epochs with early stopping based on validation accuracy

**Outputs**:
- `results/best_model.pt` - saved model checkpoint
- `submissions/submission.private.csv` - test predictions
- `results/training_history.png` - loss/accuracy plots
- `results/confusion_matrix.png` - test set confusion matrix
- `results/results.json` - numeric results

**Usage** (from repo root):
```bash
python starter_code/baseline.py --epochs 200
```

**Important args**:
- `--data-dir` - path to data folder (default: `data`)
- `--epochs` - training epochs (default: 200)
- `--hidden-channels` - hidden layer size (default: 16)
- `--lr` - learning rate (default: 0.01)
- `--patience` - early stopping patience (default: 50)

---

### `starter_code/generate_submission.py` - Submission Generation

**Purpose**: Loads a trained model checkpoint and generates a submission CSV.

**What it does**:
1. Loads data from `data/raw/` and `data/splits/`
2. Loads model weights from a `.pt` checkpoint
3. Runs inference on all nodes
4. Extracts predictions for test nodes only
5. Saves as CSV with columns: `node_id`, `target`

**Usage** (from repo root):
```bash
python starter_code/generate_submission.py --model-path results/best_model.pt
```

---

### `scoring_script.py` - Submission Scoring

**Purpose**: Scores a submission CSV against the hidden ground truth labels.

**What it does**:
1. Reads the submission CSV
2. Loads ground truth (tries CSV first for CI, falls back to `.pt` for local)
3. Computes 3 metrics: Macro F1, Accuracy, Macro Precision
4. Updates `leaderboard.json` with the results

**Usage**:
```bash
python scoring_script.py submissions/submission.private.csv
```

**Ground truth loading order**:
1. `data/test_labels_hidden.private.private.csv` - used in CI (restored from GitHub secrets)
2. `data/processed/test_labels_hidden.private.pt` + `data/splits/test_mask.pt` - used locally

---

### `update_leaderboard.py` - Leaderboard Generator

**Purpose**: Reads `leaderboard.json` and generates a formatted `leaderboard.md` table.

**What it does**:
1. Reads all submissions from `leaderboard.json`
2. Sorts by F1 score (descending)
3. Writes a markdown table to `leaderboard.md`

---

### `.github/workflows/score.yml` - CI Pipeline

**Purpose**: Automatically scores submissions when a PR modifies files in `submissions/`.

**What it does**:
1. Triggers on PRs that change `submissions/*.csv` or manual dispatch
2. Sets up Python 3.11 and installs dependencies from `starter_code/requirements.txt`
3. Restores private files from GitHub secrets (base64 encoded):
   - `submissions/sample_submission.private.csv`
   - `data/test_labels_hidden.private.private.csv`
4. Runs `scoring_script.py` against the submission
5. Runs `update_leaderboard.py` to regenerate the leaderboard
6. Commits and pushes the updated `leaderboard.md`

---

## 4. Issues Found in the Original Code

### Issue 1: README described Cora, but the repo uses PubMed

The README said:
- Dataset: **Cora** Citation Network
- 2,708 nodes, 7 classes, 1,433 features
- Classes: Case-Based, Genetic Algorithms, Neural Networks, etc.

But the actual data (`data/raw/metadata.json`) is:
- Dataset: **PubMed** Citation Network
- 19,717 nodes, 3 classes, 500 features
- Classes: Diabetes Mellitus Experimental, Type 1, Type 2

The baseline code (`baseline.py` line 1, line 375) also explicitly says "PubMed" and uses DM class names.

**Impact**: Anyone reading the README would set up the wrong dataset and get confused.

---

### Issue 2: `prepare_dataset.py` didn't match the actual data

The original script:
- Used `dgl.data.CoraGraphDataset()` (Cora via DGL library)
- Output flat CSV files: `data/train.csv`, `data/test.csv`, `data/edges.csv`
- Did an 80/20 train/test split only (no validation)

But the actual data directory uses:
- PubMed dataset
- PyTorch `.pt` tensor files in `data/raw/`, `data/splits/`, `data/processed/`
- 60/20/20 train/val/test split

**Impact**: Running `prepare_dataset.py` would overwrite the data with the wrong dataset in the wrong format, breaking `baseline.py` and `generate_submission.py`.

---

### Issue 3: `scoring_script.py` couldn't work locally

The original script hardcoded:
```python
truth = pd.read_csv('data/test_labels_hidden.private.private.csv')
```

This CSV only exists in CI (restored from GitHub secrets). Locally, the ground truth is stored as `data/processed/test_labels_hidden.private.pt` (a PyTorch tensor).

**Impact**: Running scoring locally would always fail with `FileNotFoundError`.

---

### Issue 4: Submission column name was hardcoded

The original scoring script always read `submission['target']`, but didn't handle cases where the submission might use `label` as the column name instead.

**Impact**: Submissions with a `label` column instead of `target` would crash.

---

### Issue 5: `starter_code/requirements.txt` was deleted

The CI workflow (`score.yml` line 29) runs:
```yaml
pip install -r starter_code/requirements.txt
```

But this file was deleted from the repo (shown as `D starter_code/requirements.txt` in git status).

**Impact**: Every CI run would fail at the dependency installation step.

---

### Issue 6: `baseline.py` didn't create its output directory

The script tries to save the best model to `results/best_model.pt`, but never creates the `results/` directory:
```python
model_path = Path(args.output_dir) / 'best_model.pt'
torch.save({...}, model_path)  # Crashes if results/ doesn't exist
```

**Impact**: First-time training would crash with `RuntimeError: Parent directory results does not exist`.

---

### Issue 7: README had wrong Quick Start instructions

The original README said:
```
pip install dgl torch numpy
```
But the code uses `torch_geometric` (PyTorch Geometric), not `dgl` (DGL). The code snippet in the README also used `from dgl.nn import GraphConv` while the actual baseline uses `from torch_geometric.nn import GCNConv`.

**Impact**: Following the README instructions would install the wrong library, and the code example wouldn't match what `baseline.py` actually does.

---

## 5. Step-by-Step Changes Made

### Change 1: Created `starter_code/requirements.txt`

**File**: `starter_code/requirements.txt` (NEW)

**Why**: The CI workflow needs this file, and users need to know what to install.

**Content**:
```
torch
torch_geometric
pandas
scikit-learn
numpy
matplotlib
seaborn
```

---

### Change 2: Rewrote `README.md`

**File**: `README.md`

**What changed**:

| Section | Before | After |
|---------|--------|-------|
| Dataset name | Cora | PubMed |
| Node count | 2,708 | 19,717 |
| Edge count | 5,429 | 88,648 |
| Feature count | 1,433 | 500 |
| Class count | 7 | 3 |
| Class names | CS research areas | Diabetes Mellitus categories |
| Random guess baseline | ~14% (1/7) | ~33% (1/3) |
| Data loading example | DGL + CSV | PyTorch `.pt` tensors |
| Install command | `pip install dgl torch numpy` | `pip install -r starter_code/requirements.txt` |
| Code example | DGL `GraphConv` | PyTorch Geometric `GCNConv` |
| Repo structure tree | Outdated (train.py, model.py, utils.py) | Actual files (baseline.py, generate_submission.py, etc.) |
| Quick Start | No actual runnable commands | Step-by-step: install, train, generate, score |

---

### Change 3: Rewrote `prepare_dataset.py`

**File**: `prepare_dataset.py`

**Before**:
```python
import dgl
dataset = dgl.data.CoraGraphDataset()
# ... outputs CSV files to data/train.csv, data/test.csv, etc.
```

**After**:
```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
# ... outputs .pt files to data/raw/, data/splits/, data/processed/
```

**Key differences**:
- Uses PyTorch Geometric instead of DGL (matching what baseline.py actually imports)
- Downloads PubMed instead of Cora
- Outputs PyTorch `.pt` tensor files instead of CSV files
- Creates the correct directory structure: `data/raw/`, `data/splits/`, `data/processed/`
- Creates 3-way split (60/20/20) with validation set, instead of 2-way (80/20)
- Generates `metadata.json` with dataset statistics
- Creates masked label files (train_labels, val_labels, test_labels_hidden.private)

---

### Change 4: Fixed `scoring_script.py`

**File**: `scoring_script.py`

**What changed**:

1. **Added `import torch`** - needed to load `.pt` files

2. **Added dual ground-truth loading** - tries CSV first (for CI), falls back to `.pt` (for local):
   ```python
   if os.path.exists(csv_path):
       # Load from CSV (CI environment)
       truth = pd.read_csv(csv_path)
   elif os.path.exists(pt_path):
       # Load from .pt files (local environment)
       test_labels = torch.load(pt_path, weights_only=False)
       test_mask = torch.load('data/splits/test_mask.pt', weights_only=False)
       # ... extract test labels
   else:
       print("Error: No ground truth file found")
       sys.exit(1)
   ```

3. **Added flexible column detection for submissions** - handles both `target` and `label` column names:
   ```python
   if 'target' in submission.columns:
       pred_col = 'target'
   elif 'label' in submission.columns:
       pred_col = 'label'
   else:
       pred_col = submission.columns[-1]
   ```

4. **Fixed usage comment** - changed from `scoring.py` to `scoring_script.py` (matching actual filename)

---

### Change 5: Fixed `starter_code/baseline.py`

**File**: `starter_code/baseline.py`

**What changed**: Added one line to create the output directory before saving:
```python
# Ensure output directory exists
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
```

This was added at line 276, right before the training loop. Without it, the first `torch.save()` call would crash because `results/` doesn't exist.

---

## 6. How to Run Everything

### Prerequisites

```bash
pip install torch torch_geometric pandas scikit-learn numpy matplotlib seaborn
```

### Step 1: Data is Already Prepared

The `data/` directory already contains all the `.pt` files. You do NOT need to run `prepare_dataset.py` unless you want to regenerate from scratch.

### Step 2: Generate Ground Truth CSV (for local scoring)

The scoring script needs a ground truth file. In CI this is restored from secrets, but locally you need to create it:

```bash
python -c "
import torch, pandas as pd
test_labels = torch.load('data/processed/test_labels_hidden.private.pt', weights_only=False)
test_mask = torch.load('data/splits/test_mask.pt', weights_only=False)
test_indices = torch.where(test_mask)[0].numpy()
labels = test_labels[test_mask].numpy()
pd.DataFrame({'node_id': test_indices, 'label': labels}).to_csv('data/test_labels_hidden.private.private.csv', index=False)
print('Done')
"
```

Alternatively, the fixed `scoring_script.py` can load directly from the `.pt` files without needing this CSV.

### Step 3: Train the Baseline Model

```bash
python starter_code/baseline.py --epochs 200
```

This will:
- Load the PubMed data
- Train a 2-layer GCN for up to 200 epochs (with early stopping)
- Save the best model to `results/best_model.pt`
- Save test predictions to `submissions/submission.private.csv`
- Save training plots and confusion matrix

Expected output: ~84-85% test accuracy.

### Step 4: Generate a Submission

```bash
python starter_code/generate_submission.py --model-path results/best_model.pt
```

### Step 5: Score the Submission

```bash
python scoring_script.py submissions/submission.private.csv
```

This prints F1, Accuracy, and Precision scores, and updates `leaderboard.json`.

### Step 6: Update the Leaderboard

```bash
python update_leaderboard.py
```

This reads `leaderboard.json` and generates a formatted `leaderboard.md`.
