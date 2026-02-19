import os
import json


LEADERBOARD_FILE = "leaderboard.json"

# Read scores from leaderboard.json
if not os.path.exists(LEADERBOARD_FILE):
    print("No leaderboard.json found, nothing to update.")
    exit(0)

with open(LEADERBOARD_FILE, "r") as f:
    leaderboard = json.load(f)

submissions = leaderboard.get("submissions", [])

if not submissions:
    print("No submissions found in leaderboard.json")
    exit(0)

# Generate leaderboard.md from leaderboard.json
MD_FILE = "leaderboard.md"

lines = []
lines.append("# Leaderboard\n\n")
lines.append("| Submission                  |   Macro F1 | Accuracy   | Precision  | Timestamp          |\n")
lines.append("|-----------------------------|------------|------------|------------|--------------------|")

# Sort by F1 descending
submissions_sorted = sorted(submissions, key=lambda x: x.get("F1", 0), reverse=True)

for sub in submissions_sorted:
    name = sub.get("Submission", "unknown")
    f1 = sub.get("F1", 0)
    acc = sub.get("Accuracy", 0)
    prec = sub.get("Precision", 0)
    ts = sub.get("timestamp", "")
    lines.append(f"| {name:<27} | {f1:>10.4f} | {acc:>10.4f} | {prec:>10.4f} | {ts:<18} |")

with open(MD_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print("Leaderboard updated")