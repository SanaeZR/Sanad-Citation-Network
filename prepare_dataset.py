"""
Prepare PubMed Dataset
======================

Downloads the PubMed citation network via PyTorch Geometric and saves it
as organized PyTorch tensors under data/raw, data/splits, and data/processed.

Usage:
    python prepare_dataset.py
"""

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import secrets
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding

#RSA function
def generate_rsa_keys(private_key_path="private_key.pem", public_key_path="public_key.pem"):
    """Generate RSA private/public keys for encryption."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
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
# Hybrid AES+RSA fuction for big files
def encrypt_file_rsa_aes(input_file, output_file, public_key_file):
    """Encrypt a file using hybrid AES + RSA encryption."""
    # Load public key
    with open(public_key_file, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    # Generate AES key and IV
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

    # Encrypt file with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()

    with open(input_file, "rb") as f:
        data = f.read()

    padded_data = padder.update(data) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    with open(output_file, "wb") as f:
        f.write(len(encrypted_key).to_bytes(4, "big"))
        f.write(encrypted_key)
        f.write(iv)
        f.write(encrypted_data)

    print(f"File {input_file} encrypted to {output_file}.")


def prepare_pubmed_challenge(data_dir='data', seed=42):
    """
    Prepare the PubMed dataset with stratified train/val/test splits.
    Saves PyTorch tensors to data/raw, data/splits, data/processed.
    """
    print("Downloading PubMed dataset...")
    generate_rsa_keys()
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    data = dataset[0]

    features = data.x          # [19717, 500]
    labels = data.y            # [19717]
    edges = data.edge_index    # [2, 88648]

    num_nodes = features.shape[0]
    num_features = features.shape[1]
    num_classes = int(labels.max().item()) + 1
    num_edges = edges.shape[1]

    # Stratified train/val/test split (60/20/20)
    np.random.seed(seed)
    indices = np.arange(num_nodes)
    labels_np = labels.numpy()

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels_np, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels_np[temp_idx], random_state=seed
    )

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Create label variants (hidden labels for non-split nodes)
    train_labels = labels.clone()
    train_labels[~train_mask] = -1

    val_labels = labels.clone()
    val_labels[~val_mask] = -1

    test_labels_hidden = labels.clone()
    test_labels_hidden[~test_mask] = -1

    # Create directories
    raw_dir = Path(data_dir) / 'raw'
    splits_dir = Path(data_dir) / 'splits'
    processed_dir = Path(data_dir) / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    torch.save(features, raw_dir / 'features.pt')
    torch.save(edges, raw_dir / 'edges.pt')
    torch.save(labels, raw_dir / 'labels.pt')

    # Save metadata
    metadata = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "num_classes": num_classes,
        "class_names": [
            "Diabetes Mellitus Experimental",
            "Diabetes Mellitus Type 1",
            "Diabetes Mellitus Type 2"
        ],
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "stratified": True,
        "seed": seed,
        "train_count": int(train_mask.sum().item()),
        "val_count": int(val_mask.sum().item()),
        "test_count": int(test_mask.sum().item())
    }
    with open(raw_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Save masks
    torch.save(train_mask, splits_dir / 'train_mask.pt')
    torch.save(val_mask, splits_dir / 'val_mask.pt')
    torch.save(test_mask, splits_dir / 'test_mask.pt')

    # Save processed labels
    torch.save(train_labels, processed_dir / 'train_labels.pt')
    torch.save(val_labels, processed_dir / 'val_labels.pt')
    torch.save(test_labels_hidden, processed_dir / 'test_labels_hidden.private.pt')
    #TEST LABElS ENCRIPTION
    tmp_file = processed_dir / "test_labels_hidden.private.pt"
    encrypt_file_rsa_aes(tmp_file, processed_dir / "test_labels_hidden.enc", "public_key.pem")
    tmp_file.unlink()  # remove the unencrypted file

    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {num_edges}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Training nodes: {train_mask.sum().item()} ({train_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"Validation nodes: {val_mask.sum().item()} ({val_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"Test nodes: {test_mask.sum().item()} ({test_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"\nFiles saved to: {Path(data_dir).absolute()}")
    print("Done!")


if __name__ == "__main__":
    prepare_pubmed_challenge()
