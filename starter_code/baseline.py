"""
Baseline GCN Model Training Script
===================================

This script trains a GCN model on the prepared PubMed dataset.
It reads data from the organized data/ folder structure.

Usage (from repo root):
    python starter_code/baseline.py --epochs 200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score
from datetime import datetime
import pandas as pd
import secrets
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding


#RSA encritption 
def generate_rsa_keys(private_key_path="private_key.pem", public_key_path="public_key.pem"):
    """Generate RSA private/public keys for encryption."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    with open(private_key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    public_key = private_key.public_key()
    with open(public_key_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print("RSA keys generated.")
#Hybrid AES+RSA encription 
def encrypt_file_rsa_aes(input_file, output_file, public_key_file):
    """Encrypt a file using hybrid AES + RSA encryption."""
    # Load RSA public key
    with open(public_key_file, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    # Generate random AES key + IV
    aes_key = secrets.token_bytes(32)  # AES-256
    iv = secrets.token_bytes(16)

    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Encrypt file content with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()

    with open(input_file, "rb") as f:
        data = f.read()

    padded_data = padder.update(data) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Write output file: length of RSA key + encrypted AES key + IV + AES-encrypted data
    with open(output_file, "wb") as f:
        f.write(len(encrypted_key).to_bytes(4, "big"))
        f.write(encrypted_key)
        f.write(iv)
        f.write(encrypted_data)

    print(f"Encrypted {input_file} → {output_file}")



def save_submission_csv(test_mask, predictions, save_path='submission.csv'):
    """
    Save test predictions to CSV file for submission.
   
    Args:
        test_mask: Boolean mask indicating test nodes
        predictions: Predicted labels for test nodes
        save_path: Path to save CSV file
    """
    # Get test node indices
    test_indices = torch.where(test_mask)[0].cpu().numpy()
   
    # Create DataFrame
    submission_df = pd.DataFrame({
        'node_id': test_indices,
        'target': predictions
    })
   
    # Save to CSV
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"✓ Test predictions saved to: {save_path}")
    print(f"  Total test nodes: {len(submission_df)}")
    print(f"  Columns: node_id, target")
   
    # Show preview
    print(f"\nPreview of submission file:")
    print(submission_df.head(10))


class GCN(nn.Module):
    """Graph Convolutional Network for node classification."""
   
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
       
        self.num_layers = num_layers
        self.dropout = dropout
       
        self.convs = nn.ModuleList()
       
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_channels))
       
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
       
        # Output layer
        self.convs.append(GCNConv(hidden_channels, num_classes))
   
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
       
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class PubMedDataLoader:
    """Load PubMed data from organized folder structure."""
   
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        print(f"Loading data from: {self.data_dir.absolute()}")
       
        # Load metadata
        with open(self.data_dir / 'raw' / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
       
        # Load raw data
        self.features = torch.load(self.data_dir / 'raw' / 'features.pt')
        self.edges = torch.load(self.data_dir / 'raw' / 'edges.pt')
        self.labels = torch.load(self.data_dir / 'raw' / 'labels.pt')
       
        # Load masks
        self.train_mask = torch.load(self.data_dir / 'splits' / 'train_mask.pt')
        self.val_mask = torch.load(self.data_dir / 'splits' / 'val_mask.pt')
        self.test_mask = torch.load(self.data_dir / 'splits' / 'test_mask.pt')
       
        # Load processed labels (with hidden test labels)
        self.train_labels = torch.load(self.data_dir / 'processed' / 'train_labels.pt')
       
        print(f"✓ Data loaded successfully!")
        print(f"  Nodes: {self.metadata['num_nodes']}")
        print(f"  Edges: {self.metadata['num_edges']}")
        print(f"  Features: {self.metadata['num_features']}")
        print(f"  Classes: {self.metadata['num_classes']}")
        print(f"  Train: {self.train_mask.sum().item()}, "
              f"Val: {self.val_mask.sum().item()}, "
              f"Test: {self.test_mask.sum().item()}")
   
    def to(self, device):
        """Move all data to device."""
        self.features = self.features.to(device)
        self.edges = self.edges.to(device)
        self.labels = self.labels.to(device)
        self.train_labels = self.train_labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.device = device
        return self


def train_epoch(model, data_loader, optimizer):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
   
    # Forward pass
    out = model(data_loader.features, data_loader.edges)
   
    # Compute loss on training nodes only
    # Use train_labels which has -1 for non-training nodes
    loss = F.nll_loss(
        out[data_loader.train_mask],
        data_loader.labels[data_loader.train_mask]
    )
   
    # Backward pass
    loss.backward()
    optimizer.step()
   
    return loss.item()


@torch.no_grad()
def evaluate(model, data_loader, mask, return_predictions=False):
    """Evaluate model on a specific split."""
    model.eval()
   
    # Forward pass
    out = model(data_loader.features, data_loader.edges)
    pred = out.argmax(dim=1)
   
    # Calculate metrics
    y_true = data_loader.labels[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
   
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
   
    results = {
        'accuracy': accuracy,
        'f1': f1
    }
   
    if return_predictions:
        results['predictions'] = y_pred
        results['true_labels'] = y_true
   
    return results



def train_baseline(args):
    """Main training function."""
   
    print("=" * 80)
    print("Baseline GCN Training on PubMed Dataset")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
   
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")
   
    # Load data
    print("=" * 80)
    print("Loading Data")
    print("=" * 80)
    data_loader = PubMedDataLoader(args.data_dir)
    data_loader.to(device)
   
    # Create model
    print(f"\n{'=' * 80}")
    print("Creating Model")
    print("=" * 80)
    model = GCN(
        num_features=data_loader.metadata['num_features'],
        hidden_channels=args.hidden_channels,
        num_classes=data_loader.metadata['num_classes'],
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
   
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
   
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
   
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'loss': [],
        'train_acc': [],
        'val_acc': []
    }
   
    best_val_acc = 0
    best_epoch = 0
    patience = 0
   
    # Training loop
    print(f"\n{'=' * 80}")
    print("Training")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 80 + "\n")
   
    for epoch in range(1, args.epochs + 1):
        # Train
        loss = train_epoch(model, data_loader, optimizer)
       
        # Evaluate
        train_results = evaluate(model, data_loader, data_loader.train_mask)
        val_results = evaluate(model, data_loader, data_loader.val_mask)
       
        # Save history
        history['loss'].append(loss)
        history['train_acc'].append(train_results['accuracy'])
        history['val_acc'].append(val_results['accuracy'])
       
        # Check for best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_epoch = epoch
            patience = 0
           
            # Save best model
            if args.save_model:
                model_path = Path(args.output_dir) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_results['accuracy'],
                    'args': vars(args)
                }, model_path)
        else:
            patience += 1
       
        # Print progress
        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Train Acc: {train_results['accuracy']:.4f} | "
                  f"Val Acc: {val_results['accuracy']:.4f} | "
                  f"Best Val: {best_val_acc:.4f} (epoch {best_epoch})")
       
        # Early stopping
        if patience >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break
   
    # Load best model for evaluation
    if args.save_model:
        print(f"\nLoading best model from epoch {best_epoch}...")
        checkpoint = torch.load(Path(args.output_dir) / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
   
    # Final evaluation
    print(f"\n{'=' * 80}")
    print("Final Evaluation")
    print("=" * 80)
   
    train_results = evaluate(model, data_loader, data_loader.train_mask)
    val_results = evaluate(model, data_loader, data_loader.val_mask)
    test_results = evaluate(model, data_loader, data_loader.test_mask, return_predictions=True)
   
    print(f"\nTraining Set:")
    print(f"  Accuracy: {train_results['accuracy']:.4f}")
    print(f"  F1 Score: {train_results['f1']:.4f}")
   
    print(f"\nValidation Set:")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  F1 Score: {val_results['f1']:.4f}")
   
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
   
    # Detailed test set analysis
    print(f"\n{'=' * 80}")
    print("Test Set Classification Report")
    print("=" * 80)
   
    class_names = ['DM Experimental', 'DM Type 1', 'DM Type 2']
    print(classification_report(
        test_results['true_labels'],
        test_results['predictions'],
        target_names=class_names,
        digits=4
    ))
   
    # Save test predictions to CSV
    submission_path = 'submissions/submission.private.csv'
    save_submission_csv(
        data_loader.test_mask,
        test_results['predictions'],
        submission_path
    )

    # Encrypt the CSV submission
    csv_path = Path(submission_path)
    enc_path = csv_path.with_suffix(".enc")
    encrypt_file_rsa_aes(csv_path, enc_path, "public_key.pem")
    csv_path.unlink()  # Remove the unencrypted CSV
    print(f"✓ Encrypted submission saved to: {enc_path}")

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Final test accuracy: {test_results['accuracy']:.4f}")
    print("=" * 80 + "\n")
   
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GCN baseline on PubMed')
   
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing prepared data (default: data)')
   
    # Model architecture
    parser.add_argument('--hidden-channels', type=int, default=16,
                        help='Hidden layer size (default: 16)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of GCN layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
   
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
   
    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save best model (default: True)')
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print progress every N epochs (default: 10)')
   
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
   
    args = parser.parse_args()
   
    # Train model
    train_baseline(args)
