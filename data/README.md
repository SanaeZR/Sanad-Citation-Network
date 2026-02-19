# PubMed Dataset - Processed

This directory contains the processed PubMed citation network dataset.

## Dataset Statistics
- **Total nodes**: 19,717
- **Total edges**: 88,648
- **Node features**: 500
- **Classes**: 3

## Split Information
- **Training nodes**: 11,829 (60.0%)
- **Validation nodes**: 3,942 (20.0%)
- **Test nodes**: 3,946 (20.0%)

## Directory Structure

```
data/
├── raw/
│   ├── features.pt          # Node features [19717, 500]
│   ├── edges.pt             # Edge index [2, 88648]
│   ├── labels.pt            # All node labels [19717]
│   └── metadata.json        # Dataset metadata
│
├── splits/
│   ├── train_mask.pt        # Training node mask
│   ├── val_mask.pt          # Validation node mask
│   └── test_mask.pt         # Test node mask
│
└── processed/
    ├── train_labels.pt      # Training labels (non-train nodes = -1)
    ├── val_labels.pt        # Validation labels (non-val nodes = -1)
    └── test_labels_hidden.private.pt # Test labels (ONLY for final evaluation!)
```

## Usage

### Loading Data in Training Script

```python
import torch

# Load raw data
features = torch.load('data/raw/features.pt')
edges = torch.load('data/raw/edges.pt')
labels = torch.load('data/raw/labels.pt')

# Load masks
train_mask = torch.load('data/splits/train_mask.pt')
val_mask = torch.load('data/splits/val_mask.pt')
test_mask = torch.load('data/splits/test_mask.pt')

# Load training labels (test labels hidden)
train_labels = torch.load('data/processed/train_labels.pt')

# During training, use:
# - features, edges (all nodes/edges visible)
# - train_mask to select training nodes
# - train_labels for supervision (only train nodes visible)

# For final evaluation only:
test_labels = torch.load('data/processed/test_labels_hidden.private.pt')
```

## Important Notes

⚠️ **Test labels are HIDDEN during training** - They are stored in `test_labels_hidden.private.pt` and should ONLY be loaded for final model evaluation.

✓ **All edges remain visible** - The graph structure (edges.pt) is used for all splits to enable semi-supervised learning.

✓ **Stratified splitting** - Class distribution is maintained across all splits.

## Generated
- Date: 2026-02-13
- Seed: 42
- Strategy: Stratified
