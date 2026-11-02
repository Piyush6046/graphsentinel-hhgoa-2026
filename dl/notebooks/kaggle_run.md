# Kaggle Execution Guide: Uncertainty-Aware Graph Fraud Learning

Pipeline: **Uncertainty-Aware Graph Learning for Card-Fraud Investigation and Next-Best-Action**  
Dataset: IEEE-CIS Fraud Detection — `/kaggle/input/ieee-fraud-detection/`  
(Note: the competition folder is `ieee-fraud-detection`, not `ieee-fraud-detection/`)

---

## Cell 1: Environment Setup & GPU Check

```python
# Cell 1: Verify GPU & Install necessary libraries
import os
import sys
import torch
import psutil

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"CPU RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

# Ensure working directories exist
os.makedirs("/kaggle/working/dl/results", exist_ok=True)
os.makedirs("/kaggle/working/dl/checkpoints", exist_ok=True)
os.chdir("/kaggle/working")

# Verify IEEE-CIS dataset is attached
data_path = "/kaggle/input/ieee-fraud-detection"
print(f"Dataset directory contents: {os.listdir(data_path)}")
# Expected: ['train_identity.csv', 'train_transaction.csv', 'test_identity.csv', ...]
```

---

## Cell 2: Fast Smoke Test (100,000 Transactions Subsample)

*Run first to verify the complete 9-stage pipeline executes without errors in ~3-4 minutes.*  
*Expected: ~3,500 frauds at natural 3.5% prevalence. All PASS/FAIL status shown for val and test.*

```python
# Cell 2: Fast Smoke-Test (100,000 samples, seeds 42)
!python -m dl.run_all \
    --data_dir /kaggle/input/ieee-fraud-detection \
    --out_dir /kaggle/working/dl/results/smoke_test \
    --ckpt_dir /kaggle/working/dl/checkpoints/smoke_test \
    --sample 100000 \
    --seeds 42 \
    --mc_samples 15 \
    --prob_source ensemble
```

---

## Cell 3: Resume from a Specific Stage

*Use `--start_stage` to skip completed stages. The pipeline will verify required artifacts exist.*

```python
# Resume from calibration onwards (skip data/graph/baselines/autoencoder/gnn/mc_dropout)
!python -m dl.run_all \
    --data_dir /kaggle/input/ieee-fraud-detection \
    --out_dir /kaggle/working/dl/results/smoke_test \
    --ckpt_dir /kaggle/working/dl/checkpoints/smoke_test \
    --sample 100000 \
    --seeds 42 \
    --mc_samples 15 \
    --start_stage calibration \
    --end_stage evaluate \
    --prob_source ensemble
```

Available stage names (in order):
`data`, `graph`, `baselines`, `autoencoder`, `gnn`, `mc_dropout`, `calibration`, `decision`, `evaluate`

---

## Cell 4: Run Decision Policy with Different Probability Sources

```python
# Compare probability sources for the next-best-action policy
for source in ["gnn", "lgb_graph", "ensemble"]:
    print(f"\n{'='*60}")
    print(f"Running policy with prob_source={source}")
    os.system(
        f"python -m dl.run_all "
        f"--data_dir /kaggle/input/ieee-fraud-detection "
        f"--out_dir /kaggle/working/dl/results/smoke_test "
        f"--ckpt_dir /kaggle/working/dl/checkpoints/smoke_test "
        f"--sample 100000 --seeds 42 --mc_samples 15 "
        f"--start_stage decision --end_stage evaluate "
        f"--prob_source {source}"
    )
```

---

## Cell 5: Full Production Run (Full Dataset, 3 Seeds)

*Executes all models across seeds 42, 123, 456. Saves `multi_seed_summary.csv` and `.md`.*

```python
# Cell 5: Full 3-Seed Benchmark Execution (~25-35 min on T4)
!python -m dl.run_all \
    --data_dir /kaggle/input/ieee-fraud-detection \
    --out_dir /kaggle/working/dl/results/full_run \
    --ckpt_dir /kaggle/working/dl/checkpoints/full_run \
    --seeds 42 123 456 \
    --mc_samples 30 \
    --prob_source ensemble
```

---

## Cell 6: Display Summary Tables & Statistical Comparison

```python
# Cell 6: Display Benchmark Results
import pandas as pd, json
from IPython.display import display, Markdown

# Multi-seed summary
summary_csv = "/kaggle/working/dl/results/full_run/multi_seed_summary.csv"
if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    print("========= 3-SEED AGGREGATE RESULTS (MEAN +/- STD) =========")
    display(df_summary)
else:
    # Single-seed fallback
    df_single = pd.read_csv("/kaggle/working/dl/results/smoke_test/metrics_summary.csv")
    print("========= SINGLE-SEED METRICS =========")
    display(df_single)

# Stage timings
timings_path = "/kaggle/working/dl/results/smoke_test/stage_timings.json"
if os.path.exists(timings_path):
    with open(timings_path) as f:
        timings = json.load(f)
    print("\n=== PER-STAGE TIMINGS ===")
    for stage, secs in timings.items():
        print(f"  {stage:20s}: {secs:.1f}s ({secs/60:.2f} min)")

# Calibration ECE summary
cal_path = "/kaggle/working/dl/results/smoke_test/calibration_summary.json"
if os.path.exists(cal_path):
    with open(cal_path) as f:
        cal_data = json.load(f)
    print("\n=== CALIBRATION ECE (before -> after) ===")
    for model, ece_info in cal_data["models"].items():
        t = ece_info.get("temperature", "n/a")
        print(f"  {model:25s}: {ece_info['test_ece_raw']:.4f} -> {ece_info['test_ece_calibrated']:.4f}  T={t}")

# Uncertainty evaluation
uncert_path = "/kaggle/working/dl/results/smoke_test/uncertainty_evaluation.json"
if os.path.exists(uncert_path):
    with open(uncert_path) as f:
        uncert = json.load(f)
    print("\n=== UNCERTAINTY QUALITY ===")
    print("AUROC of uncertainty signals for detecting errors:")
    for sig, val in uncert["auroc_by_signal"].items():
        print(f"  {sig:28s}: {val:.4f}")
    print(f"\nVerdict: {uncert['verdict']}")

# Decision policy summary
policy_path = "/kaggle/working/dl/results/smoke_test/decision_policy_summary.json"
if os.path.exists(policy_path):
    with open(policy_path) as f:
        policy = json.load(f)
    print("\n=== DECISION POLICY ===")
    print(f"Source: {policy['prob_source']}")
    caps = policy["operational_caps"]
    print(f"Non-ALLOW: Val [{caps['non_allow_val_status']}] {caps['non_allow_val_pct']:.2f}% | Test [{caps['non_allow_test_status']}] {caps['non_allow_test_pct']:.2f}%")
    print(f"ESCALATE : Val [{caps['escalate_val_status']}] {caps['escalate_val_pct']:.2f}% | Test [{caps['escalate_test_status']}] {caps['escalate_test_pct']:.2f}%")
    print(f"EVIDENCE : Val [{caps['evidence_val_status']}] {caps['evidence_val_pct']:.2f}% | Test [{caps['evidence_test_status']}] {caps['evidence_test_pct']:.2f}%")
```

---

## Cell 7: Display Evaluation Figures

```python
# Cell 7: Render evaluation plots
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plot_path = "/kaggle/working/dl/results/smoke_test/evaluation_curves.png"
if os.path.exists(plot_path):
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(18, 14))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Deep Learning Model Performance & Uncertainty Analysis", fontsize=16, pad=20)
    plt.show()
```

---

## Cell 8: Ring Metrics

```python
# Cell 8: Display ring-level fraud syndicate detection metrics
ring_path = "/kaggle/working/dl/results/smoke_test/ring_metrics.json"
if os.path.exists(ring_path):
    with open(ring_path) as f:
        rings = json.load(f)
    print("=== RING-LEVEL METRICS (connected components, size 3-50) ===")
    print(f"Number of rings:                  {rings['n_rings']}")
    if rings.get('size_distribution'):
        sd = rings['size_distribution']
        print(f"Size range:                       {sd.get('min')} - {sd.get('max')} (median {sd.get('median'):.1f})")
        print(f"Total transactions in rings:      {sd.get('total_txns')}")
    print(f"Ring-level PR-AUC:                {rings['ring_pr_auc']:.4f}")
    print(f"Ring-level ROC-AUC:               {rings['ring_roc_auc']:.4f}")
    print(f"Frauds inside rings:              {rings['ring_frauds']}")
    print(f"Frauds in rings missed by model:  {rings['frauds_in_rings_missed_by_model']}")
```

---

## Notes on Expected Results

| Finding | Explanation |
|---------|-------------|
| GNN < LightGBM on IEEE-CIS | Consistent with published literature; tabular features dominate. |
| MLP val PR-AUC > test PR-AUC | Temporal distribution shift: test is ~30 days later. Honest finding. |
| MC-Dropout variance ~0.001 | Low because model is confident; MC uncertainty may not beat simple baseline. |
| Non-ALLOW FAIL on test | Thresholds tuned on val; test may differ. PASS on val is what's guaranteed. |
| One giant ring before fix | Generic device keys (D_GENERIC) merged all cards. Now properly excluded. |
| log_past_card_fraud_count | Uses ONLY training-period labels. Val/test labels are never accessed (asserted). |
