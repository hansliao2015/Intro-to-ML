import os
import re
import csv

ROOTS = ["results_nn_ablation", "results_cnn_ablation"]

rows = []
header = [
    "experiment",
    "use_bn",
    "use_dropout",
    "use_residual",
    "pooling",
    "params",
    "best_val_acc"
]

def extract_value(pattern, text, default=None):
    m = re.search(pattern, text)
    return m.group(1) if m else default

for root in ROOTS:
    if not os.path.isdir(root):
        continue

    for exp in sorted(os.listdir(root)):
        exp_path = os.path.join(root, exp)
        log_file = os.path.join(exp_path, "log.txt")

        if not os.path.exists(log_file):
            continue

        with open(log_file, "r") as f:
            log = f.read()

        # Extract fields
        use_bn = extract_value(r"use_bn=(\d)", log)
        use_dropout = extract_value(r"use_dropout=(\d)", log)
        use_residual = extract_value(r"use_residual=(\d)", log)
        pooling = extract_value(r"pooling='(\w+)'", log)
        params = extract_value(r"Trainable Parameters: (\d+)", log)
        best_val_acc = extract_value(r"Best Validation Accuracy: ([\d\.]+)", log)

        rows.append([
            exp,
            use_bn or "",
            use_dropout or "",
            use_residual or "",
            pooling or "",
            params or "",
            best_val_acc or "",
        ])

# Save CSV
with open("ablation_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Saved ablation_summary.csv")
