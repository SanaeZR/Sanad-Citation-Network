# GNN Challenge: Paper Topic Prediction in Citation Networks
## Challenge Overview

Welcome to **Sanad Citation Network Challenge!**
This challenge focuses on node classification in a citation network using Graph Neural Networks (GNNs).

Participants are asked to predict the category of scientific papers based solely on the citation graph structure and node features.

Dataset used: PubMed Citation Network (standard benchmark in GNN literature)

### Problem Description

Scientific papers cite other papers, forming a graph where:

|Element           |Description                                    |
|------------------|-----------------------------------------------|
|Nodes             |represent papers                               |
|Edges             |represent citation links                       |
|Node features     |represent paper content (bag-of-words)         |
|Node labels       |represent disease categories                   |

Your Task

Given a paper and its citation neighborhood, predict the disease category of the paper.

This is a semi-supervised node classification task:

Only a subset of nodes are labeled for training

The model must generalize to unseen nodes

### Classes (Disease Categories)

Each paper belongs to one of 3 categories:

| Label | Category |
|-------|----------|
| 0 | Diabetes Mellitus Experimental |
| 1 | Diabetes Mellitus Type 1 |
| 2 | Diabetes Mellitus Type 2 |

### What Makes This Challenging?

**Graph Dependency**:
Papers are not independent — predictions rely on citation neighborhoods.

**Homophily Bias**:
Papers often cite papers from the same domain, but not always.

**Limited Labels**:
Only a portion of nodes are labeled for training (60%).

**Over-smoothing Risk**:
Deeper GNNs may degrade performance if not carefully designed.

**No External Information**:
Only graph structure and node features are allowed.

## Dataset
PubMed Citation Network Statistics
|Property |Value|
|---------|------------|
|Nodes |19,717 papers|
|Edges |88,648 citation links|
|Node Features |500|
|Classes |3|
|Graph Type| Homogeneous|

### Data Format

The dataset is provided as preprocessed PyTorch tensors:

```python
import torch

# Load raw data
features = torch.load('data/raw/features.pt')    # [19717, 500]
edges = torch.load('data/raw/edges.pt')           # [2, 88648]
labels = torch.load('data/raw/labels.pt')         # [19717]

# Load masks
train_mask = torch.load('data/splits/train_mask.pt')
val_mask = torch.load('data/splits/val_mask.pt')
test_mask = torch.load('data/splits/test_mask.pt')
```
Using features.pt, edges.pt, labels.pt, and masks:

1.Makes your GNN easy to implement

2.Efficient in memory

3.Compatible with PyG functions

4.Handles semi-supervised masking cleanly

5.Supports batching / inductive learning

6.Improves reproducibility and standardization
### Split Information

| Split | Nodes | Ratio |
|-------|-------|-------|
| Training | 11,829 | 60% |
| Validation | 3,942 | 20% |
| Test | 3,946 | 20% |

# Dataset & Private Files

The challenge uses private files for evaluation:

Submission file: submission.private.csv

Test labels: test_labels_hidden.private.pt

These files are hosted on Hugging Face and linked to this repository.

Private files are used internally for scoring and leaderboard computation; they are not included in the repository.

##  Workflow

Data Processing: Load node features, adjacency information, and graph structure.

Model Training:

GCN model with hidden layers (ReLU + dropout)

Output layer computes log softmax for node classification

## Evaluation:

Submissions are scored using the private test labels

The scoring workflow uses Hugging Face to download the private files securely

Private File Handling (Hugging Face)

The private files are downloaded automatically using a Hugging Face token (HF_TOKEN) in the CI workflow:

submission.private.csv → private submission for evaluation

test_labels_hidden.private.pt → private labels for scoring

This ensures reproducible evaluation while keeping sensitive data private.

### Evaluation Metric
| Metric                | Split            | Purpose                             | Definition                                                                                  |
| --------------------- | ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| Accuracy              | train, val, test | Overall node prediction correctness | Fraction of correctly predicted nodes out of all nodes in the split: `Accuracy = #correct_predictions / total_nodes` |
| Weighted F1           | train, val, test | Class-balanced node performance     | Harmonic mean of precision and recall, averaged across classes weighted by class size: `F1 = 2 * (Precision * Recall) / (Precision + Recall)` |
| Loss (NLL)            | train            | Optimization, early stopping        | Negative log-likelihood loss for classification: `NLL Loss = - (1/N) * sum_i y_i * log(y_hat_i)`, applied only on training nodes |
| Classification report | test             | Detailed per-class performance      | Shows **precision, recall, F1-score, and support** for each class separately; provides fine-grained insight for imbalanced classes. |
                                                                                              
Baseline Performance

| Model | Accuracy |
|-------|----------|
| Random Guess | ~33% |
| 2-layer GCN | ~78% |
| Target | 80%+ |

## Constraints

**To ensure fairness and pedagogical value**:

**No External Data**:
Use only the PubMed dataset.

**Graph Features Only**:
No handcrafted or text-based features beyond provided node features.

**CPU Training Only**:
Models must be trainable on CPU.

**Must Use GNNs**
At least one message-passing layer (GCN, GraphSAGE, GAT, etc.).

## Quick Start

### Install Dependencies
```bash
pip install -r starter_code/requirements.txt
```

### Train the Baseline
```bash
python starter_code/baseline.py --epochs 200
```

This trains a 2-layer GCN model, saves the best checkpoint to `results/best_model.pt`, and generates `submissions/submission.private.csv`.

### Generate a Submission
```bash
python starter_code/generate_submission.py --model-path results/best_model.pt
```

### Score a Submission
```bash
python scoring_script.py submissions/submission.private.csv
```

## Starter Model (GCN Baseline)

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
```



### Repository Structure
```
gnn-challenge/
├── data/
│   ├── raw/              # features.pt, edges.pt, labels.pt, metadata.json
│   ├── splits/           # train_mask.pt, val_mask.pt, test_mask.pt
│   └── processed/        # train_labels.pt, val_labels.pt, test_labels_hidden.private.pt
├── starter_code/
│   ├── baseline.py       # Baseline GCN training script
│   ├── generate_submission.py  # Generate submission CSV from trained model
│   └── requirements.txt  # Python dependencies
├── submissions/          # Submission CSV files
├── scoring_script.py     # Score a submission
├── update_leaderboard.py # Update leaderboard.md from leaderboard.json
├── prepare_dataset.py    # Script to regenerate the dataset
├── leaderboard.json
├── leaderboard.md
└── README.md
```
# Baseline Model (GCN) — Details
Overview

The script implements a Graph Convolutional Network (GCN) baseline for node classification on a graph dataset (in your example, the PubMed dataset). It handles data loading, training, evaluation, and test predictions, producing a CSV submission file. The design follows standard geometric deep learning practices.

## 1. Data Input

Dataset: Nodes represent entities (e.g., PubMed papers), edges represent connections (e.g., citation links).

Node features: Continuous vectors for each node (e.g., 2D or more depending on dataset).

Masks: Boolean masks for training, validation, and test splits.

Data loader (PubMedDataLoader) loads all tensors (features, edges, labels) and moves them to GPU/CPU.

## 2. Model Architecture

GCN layers: Default 2 layers (num_layers=2), each using GCNConv.

Forward pass:

Hidden layers: ReLU activation + dropout.

Output layer: Log softmax over classes.

Output: Node-level log probabilities for classification.

Training settings:

Optimizer: Adam

Loss: Negative log likelihood (F.nll_loss) over training nodes only

Regularization: Dropout + weight decay

Early stopping monitored via validation accuracy.

## 3. Training & Evaluation

Epoch loop:

Forward + backward pass on training nodes.

Evaluate on training and validation masks.

Save best model based on validation accuracy.

Early stopping if validation accuracy does not improve.

Metrics:

Accuracy

Weighted F1

Classification report per class

## 4. Submission

After training, predictions on test nodes are saved in a CSV (submissions/submission.private.csv) with columns: node_id, target.

Helper function save_submission_csv handles formatting and preview.

## 5. Key Points / Features

Modular design: GCN class separated from data loading.

Device agnostic: Automatically uses GPU if available.

Configurable via CLI arguments (--hidden-channels, --num-layers, --dropout, --epochs, etc.).

Provides detailed logs of loss, train/val accuracy, best epoch, and final test results.

Ready for extension to other datasets or graph-level tasks.
## Mental map of the GCN code
features.pt  → initial embeddings (x)
edges.pt     → adjacency matrix (edge_index)

x, edge_index
     ↓
GCNConv layer 1
     ↓
hidden embedding
     ↓
GCNConv layer 2
     ↓
final embedding
     ↓
log_softmax → predictions

# 🔐 Secure File Encryption (Hybrid RSA + AES)

To ensure privacy and prevent unauthorized access to hidden labels and private submissions, this repository uses hybrid encryption combining AES and RSA.

Why Hybrid Encryption?

AES (Advanced Encryption Standard) → Fast and efficient for encrypting large files.

RSA (Asymmetric Encryption) → Securely encrypts the AES key.

Combining both provides:

High performance (AES)

Secure key exchange (RSA)

# 🔑 How the Encryption Works

When a file is encrypted:

A random AES key is generated.

The file is encrypted using AES (CBC mode).

The AES key is encrypted using RSA (OAEP + SHA256).

The final encrypted file structure is:

[4 bytes: RSA key length]
[RSA-encrypted AES key]
[16 bytes IV]
[AES-encrypted data]
# 🔓 How Decryption Works

During scoring, files are automatically decrypted using the private RSA key:

def decrypt_file_rsa_aes(input_file, output_file, private_key_file):
    """Decrypt a file that was encrypted with hybrid AES + RSA."""
    with open(private_key_file, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    with open(input_file, "rb") as f:
        # Read length of RSA-encrypted AES key
        key_len_bytes = f.read(4)
        key_len = int.from_bytes(key_len_bytes, "big")

        # Read RSA-encrypted AES key
        encrypted_aes_key = f.read(key_len)

        # Decrypt AES key using RSA
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Read IV and encrypted content
        iv = f.read(16)
        encrypted_data = f.read()

    # AES decryption
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove PKCS7 padding
    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

    with open(output_file, "wb") as f:
        f.write(decrypted_data)

    print(f"Decrypted {input_file} → {output_file}")
🧠 Automatic Decryption During Scoring

When running:

python scoring_script.py submissions/file.csv

The script automatically detects encrypted files (.enc) and decrypts them before evaluation:

submission_file = sys.argv[1]

if submission_file.endswith(".enc"):
    decrypted_submission = submission_file.replace(".enc", ".csv")
    decrypt_file_rsa_aes(submission_file, decrypted_submission, "private_key.pem")
    submission_file = decrypted_submission

Hidden test labels are also decrypted automatically if needed.

# 🔐 Secure Key Handling in GitHub Actions

The private RSA key is never stored in the repository.

Instead:

The private key is stored as a GitHub Secret (PRIVATE_KEY)

It is Base64-encoded

It is restored during CI execution

GitHub Actions Workflow
- name: Restore PEM key
  run: |
    echo "${{ secrets.PRIVATE_KEY }}" | base64 -d > private_key.pem
    chmod 600 private_key.pem

- name: Run scoring
  run: python scoring_script.py submissions/submission.private.enc
Why This Is Secure

The private key exists only during CI execution

It is never committed to GitHub

File permissions are restricted (chmod 600)

Encrypted files can safely be stored in the repository

## References

- Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)  
- Official GitHub repository for the paper: [https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)
- **GNNs**: [Basira Lab youtube](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
