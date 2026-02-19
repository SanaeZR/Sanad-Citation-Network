import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score
from datetime import datetime
import json
import sys
import os
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

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
        # Decrypt AES key
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        # Read IV
        iv = f.read(16)
        # Read AES-encrypted data
        encrypted_data = f.read()

    # Decrypt file with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding
    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

    # Save decrypted file
    with open(output_file, "wb") as f:
        f.write(decrypted_data)

    print(f"Decrypted {input_file} → {output_file}")



# Usage : python scoring_script.py submissions/file.csv

submission_file = sys.argv[1]
# Decrypt submission file if it's encrypted
if submission_file.endswith(".enc"):
    decrypted_submission = submission_file.replace(".enc", ".csv")
    decrypt_file_rsa_aes(submission_file, decrypted_submission, "private_key.pem")
    submission_file = decrypted_submission  # use decrypted CSV in rest of code

# Decrypt test labels if encrypted
test_labels_enc = 'data/processed/test_labels_hidden.enc'
test_labels_pt = 'data/processed/test_labels_hidden.private.pt'
if os.path.exists(test_labels_enc):
    decrypt_file_rsa_aes(test_labels_enc, test_labels_pt, "private_key.pem")


# Load submission
submission = pd.read_csv(submission_file)
submission_name = os.path.basename(submission_file)

# Load ground truth - try CSV first (CI), then fall back to .pt (local)
csv_path = 'data/test_labels_hidden.private.csv'
pt_path = 'data/processed/test_labels_hidden.private.pt'

if os.path.exists(csv_path):
    truth = pd.read_csv(csv_path)
    # Detect target column in CSV
    if 'target' in truth.columns:
        target_col = 'target'
    elif 'label' in truth.columns:
        target_col = 'label'
    else:
        target_col = truth.columns[-1]
    y_true = truth[target_col].values
elif os.path.exists(pt_path):
    # Load from .pt file - contains all labels with -1 for non-test nodes
    test_labels = torch.load(pt_path, weights_only=False)
    test_mask = torch.load('data/splits/test_mask.pt', weights_only=False)
    test_indices = torch.where(test_mask)[0].numpy()
    labels = test_labels[test_mask].numpy()
    truth = pd.DataFrame({'node_id': test_indices, 'label': labels})
    target_col = 'label'
    y_true = truth[target_col].values
else:
    print(f"Error: No ground truth file found at '{csv_path}' or '{pt_path}'")
    sys.exit(1)

# Detect prediction column in submission
if 'target' in submission.columns:
    pred_col = 'target'
elif 'label' in submission.columns:
    pred_col = 'label'
else:
    pred_col = submission.columns[-1]

# Align predictions with ground truth by node_id
submission = submission.rename(columns={pred_col: 'predicted'})
truth = truth.rename(columns={target_col: 'label'})

merged = truth.merge(submission[['node_id', 'predicted']], on='node_id', how='inner')

if len(merged) != len(truth):
    print(f"Warning: {len(truth) - len(merged)} ground truth nodes missing from submission!")

y_true = merged['label'].values
y_pred = merged['predicted'].values

# Compute scores
score_F1 = f1_score(y_true, y_pred, average='macro')
print(f'Submission F1 Score : {score_F1:.4f}')
score_accuracy = accuracy_score(y_true, y_pred)
print(f'Submission Accuracy Score : {score_accuracy:.4f}')
score_precision = precision_score(y_true, y_pred, average='macro')
print(f'Submission precision Score : {score_precision:.4f}')


# Save to leaderboard.json
def update_leaderboard(submission_name, scores, leaderboard_path='leaderboard.json'):
    """
    Update leaderboard.json with scoring results.
    Creates the file if it doesn't exist.
    Updates existing entry if Submission matches, otherwise adds a new row.
    """
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"last_updated": None, "submissions": []}

    entry = {
        "Submission": scores["Submission"],
        "F1": scores["F1"],
        "Accuracy": scores["Accuracy"],
        "Precision": scores["Precision"],
        "timestamp": scores["timestamp"]
    }

    found = False
    for i, sub in enumerate(leaderboard['submissions']):
        if sub['Submission'] == submission_name:
            leaderboard['submissions'][i] = entry
            found = True
            print(f"Updated existing entry for: {submission_name}")
            break

    if not found:
        leaderboard['submissions'].append(entry)
        print(f"Added new entry for: {submission_name}")

    leaderboard['last_updated'] = datetime.now().isoformat()

    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"Leaderboard saved to: {leaderboard_path}")


scores = {
    "Submission": submission_name,
    "F1": score_F1,
    "Accuracy": score_accuracy,
    "Precision": score_precision,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
}

update_leaderboard(submission_name, scores)
