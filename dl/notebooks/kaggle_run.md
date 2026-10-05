# Kaggle Execution Guide: Uncertainty-Aware Graph Fraud Learning

This notebook guide provides exact, copy-pasteable Jupyter cells to run the entire Deep Learning fraud pipeline on **Kaggle (GPU P100/T4)** using the official IEEE-CIS Fraud Detection competition data at `/kaggle/input/ieee-fraud-detection/`.

---

## Cell 1: Environment Setup & GPU Check

```python
# Cell 1: Verify GPU & Install necessary libraries
import os
import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Ensure working directories exist
os.makedirs("/kaggle/working/dl/results", exist_ok=True)
os.makedirs("/kaggle/working/dl/checkpoints", exist_ok=True)
os.chdir("/kaggle/working")

# Verify IEEE-CIS dataset is attached
data_path = "/kaggle/input/ieee-fraud-detection"
print(f"Dataset directory contents: {os.listdir(data_path)}")
```

---

## Cell 2: Fast Smoke Test (100,000 Transactions Subsample)

*Run this cell first to verify the complete 8-stage pipeline executes without errors in ~2 minutes (~3,500 frauds at natural 3.5% prevalence).*

```python
# Cell 2: Fast Smoke-Test (100,000 samples, ~3,500 frauds)
!python -m dl.run_all \
    --data_dir /kaggle/input/ieee-fraud-detection \
    --out_dir /kaggle/working/dl/results/smoke_test \
    --ckpt_dir /kaggle/working/dl/checkpoints/smoke_test \
    --sample 100000 \
    --seeds 42 \
    --mc_samples 15
```

---

## Cell 3: Full Production Run (Full Dataset Across 3 Seeds)

*Executes all models (LightGBM, XGBoost, Tabular MLP, Autoencoder, GraphSAGE, MC-Dropout GNN, Decision Engine) across seeds 42, 123, 456.*

```python
# Cell 3: Full 3-Seed Benchmark Execution
!python -m dl.run_all \
    --data_dir /kaggle/input/ieee-fraud-detection \
    --out_dir /kaggle/working/dl/results/full_run \
    --ckpt_dir /kaggle/working/dl/checkpoints/full_run \
    --seeds 42 123 456 \
    --mc_samples 30
```

---

## Cell 4: Display Summary Tables & Statistical Comparison

```python
# Cell 4: Display Benchmark Results Table
import pandas as pd
from IPython.display import display

summary_csv = "/kaggle/working/dl/results/full_run/multi_seed_summary.csv"
if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    print("=================== 3-SEED AGGREGATE RESULTS (MEAN ± STD) ===================")
    display(df_summary)
else:
    # Single seed fallback
    df_single = pd.read_csv("/kaggle/working/dl/results/full_run/seed_42/metrics_summary.csv")
    display(df_single)
```

---

## Cell 5: Display Decision Matrix & Evaluation Figures

```python
# Cell 5: Render Publication Plots & Decision Breakdown
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Display Evaluation Curves (PR, ROC, Calibration, Risk-Coverage)
plot_path = "/kaggle/working/dl/results/full_run/seed_42/evaluation_curves.png"
if os.path.exists(plot_path):
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(18, 14))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Deep Learning Model Performance & Uncertainty Analysis", fontsize=16, pad=20)
    plt.show()

# 2. Display Policy Decision Summary
policy_json = "/kaggle/working/dl/results/full_run/seed_42/decision_policy_summary.json"
if os.path.exists(policy_json):
    with open(policy_json) as f:
        policy_data = json.load(f)
    print("\n--- NEXT-BEST-ACTION DECISION POLICY SUMMARY ---")
    print(json.dumps(policy_data, indent=2))
```
