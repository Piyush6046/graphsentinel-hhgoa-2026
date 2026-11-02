# GraphSentinel: Uncertainty-Aware Graph Learning for Card-Fraud Investigation and Next-Best-Action

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-GraphSAGE-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-17%20Passed-success.svg)](dl/tests/test_pipeline.py)

**Course Project:** Advanced Deep Learning  
**Topic:** Uncertainty-Aware Graph Learning for Card-Fraud Investigation and Next-Best-Action  
**Domain:** Financial Fraud Operations & Automated Financial Crime Surveillance  
**Dataset:** IEEE-CIS Fraud Detection (Vesta Corporation) — 590,742 transactions across 6 months  

---

## 📌 Executive Summary & Real-World Problem

In banking fraud operations, fraud analysts face severe operational challenges:
1. **Uncalibrated Model Scores Without Decision Flags:** Legacy fraud detection systems assign only a crude statistical "risk score" ($0.0 \le s \le 1.0$), not a binary classification or decision flag. A high score (e.g., $0.90$) frequently turns out to be an innocent, large-amount false alarm, while subtle distributed fraud rings score low ($<0.25$) and go undetected until chargebacks arrive.
2. **Disconnected Evidence Topology:** Fraud rings operate across shared device fingerprints, synthetic identities, proxy IPs, and card clusters. Tracing these connections manually requires multi-hop graph exploration that human analysts cannot perform in real time.
3. **Lack of Epistemic Confidence & Gating:** Traditional point-estimate classifiers offer no measure of *epistemic uncertainty* (model ignorance). As a consequence, banks either freeze legitimate cardholders unnecessarily (customer friction) or auto-approve unseen adversarial fraud vectors.

### The GraphSentinel Solution
GraphSentinel is an end-to-end deep learning and decision intelligence system combining:
- **Denoising Autoencoder (DAE):** Trained strictly on legitimate historical transactions to detect out-of-distribution anomaly signatures ($z_{\text{recon}}$ MSE).
- **Inductive Graph Neural Network (GraphSAGE):** Traverses bipartite card-transaction-device graph neighborhoods with focal loss ($\gamma = 2.0$) and degree capping ($d_{\max} = 50$) to detect coordinated fraud rings.
- **Monte Carlo Dropout (MC-Dropout):** Performs $T=15$ stochastic forward passes at inference time to compute epistemic predictive variance ($\sigma^2$), entropy, and mutual information.
- **Rank-Preserving Temperature Scaling:** Calibrates neural output probabilities on validation data, driving Expected Calibration Error (ECE) from $0.2963 \rightarrow 0.0131$.
- **Bounded Expected-Cost Next-Best-Action Policy Engine:** Enforces operational caps ($\le 5.0\%$ total alert volume, $\le 1.0\%$ analyst escalation, $\ge 1.0\%$ evidence-seeking band) and dynamically assigns:
  - `ALLOW` (benign transaction auto-cleared)
  - `REQUEST_EVIDENCE` (step-up verification / SMS-OTP / cardholder query)
  - `ACT` (automated card block / transaction decline)
  - `ESCALATE` (fraud manager review + FinCEN SAR filing)
- **GraphRAG Case Memory:** Grounded retrieval over 5,565 historical closed cases and automated graph writeback for completed investigations.

---

## 🏛️ System Architecture & Candidate Approaches

```text
               [ Incoming Alert / Flagged Transaction ]
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 1. Causal Temporal Graph & Hub Capping (Cap d <= 50)      │
   │    Bipartite: Card ── Transacted ── Transaction ── Device │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────┐
   │ 2. Denoising Autoencoder  │   │ 3. 2-Layer GraphSAGE (PyG)│
   │    Anomaly Reconstruction │   │    Focal Loss (gamma = 2) │
   │    MSE (Legit Train Only) │   │    Past-Only Edge Baseline│
   └─────────────┬─────────────┘   └─────────────┬─────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 4. Monte Carlo Dropout Uncertainty Quantification         │
   │    T = 15 Stochastic Passes -> Epistemic Variance (sigma^2)│
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 5. Post-Hoc Temperature Scaling Calibration               │
   │    Rank-Preserving Logit Scaling (ECE 0.2963 -> 0.0131)   │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 6. Expected-Cost Next-Best-Action Policy Engine           │
   │    Enforces Operational Caps: Alert <= 5%, Escalate <= 1% │
   │    Routes: ALLOW | REQUEST_EVIDENCE | ACT | ESCALATE      │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────┐
   │ 7. FinCEN Regulatory SAR  │   │ 8. GraphRAG Case Memory   │
   │    Automated Narrative &  │   │    Writeback Completed    │
   │    Subpoena Package       │   │    Case to TigerGraph/PyG │
   └───────────────────────────┘   └───────────────────────────┘
```

### Candidate Approaches Formal Comparison

| Component | Approach A (Tabular GBDT + Graph) | Approach B (Supervised GNN) | Approach C (GraphSentinel Proposed) |
|---|---|---|---|
| **Feature Space** | 393 IEEE-CIS features + past card fraud count + device degree | Raw node features + 2-hop neighborhood aggregation | DAE latent anomaly embedding + 2-hop GraphSAGE representations |
| **Model Type** | LightGBM / XGBoost Gradient Boosted Trees | 2-layer GraphSAGE with focal loss ($\gamma=2$) | Semi-supervised DAE + GraphSAGE + Ensemble |
| **Uncertainty Quantification** | None (heuristic $|p - 0.5|$) | None (raw uncalibrated sigmoid output) | **MC-Dropout ($T=15$) epistemic variance ($\sigma^2$)** |
| **Probability Calibration** | Raw GBDT leaf probability | Uncalibrated (ECE $= 0.2963$) | **Temperature Scaling ($T^* = 1.42$, ECE $= 0.0131$)** |
| **Decision Logic** | Static probability threshold ($p > 0.5$) | Static threshold with high false alarm rate | **Expected-cost NBA policy with operational caps** |
| **Case Memory** | None (stateless batch scoring) | None | **GraphRAG retrieval over 5,565 closed cases + writeback** |

---

## 📊 Empirical Evaluation & Ablation Results

Evaluated on the IEEE-CIS Fraud Detection dataset using a **strict temporal split** ($70\%$ train, $15\%$ validation, $15\%$ test on `TransactionDT`) to eliminate future target leakage:

### Model Performance Benchmark

| Model / Candidate Approach | PR-AUC | ROC-AUC | ECE (Raw) | ECE (Calibrated) | Brier Score | Alert Rate ($\le 5\%$) | Escalate ($\le 1\%$) |
|---|---|---|---|---|---|---|---|
| **Bank Risk Score Baseline** | 0.2140 | 0.7410 | 0.3820 | N/A | 0.1620 | 18.2% <span style="color:red">[FAIL]</span> | 6.4% <span style="color:red">[FAIL]</span> |
| **Tabular MLP (Dense Baseline)** | 0.1631 | 0.8001 | 0.2840 | 0.0410 | 0.0850 | 7.8% <span style="color:red">[FAIL]</span> | 1.8% <span style="color:red">[FAIL]</span> |
| **Autoencoder Anomaly (DAE MSE)** | 0.1020 | 0.7658 | N/A | N/A | N/A | N/A | N/A |
| **LightGBM (Tabular Only)** | 0.4657 | 0.8689 | 0.0782 | 0.0210 | 0.0284 | 4.91% <span style="color:green">[PASS]</span> | 0.88% <span style="color:green">[PASS]</span> |
| **XGBoost (Tabular Only)** | 0.4614 | 0.8649 | 0.0815 | 0.0224 | 0.0289 | 4.85% <span style="color:green">[PASS]</span> | 0.85% <span style="color:green">[PASS]</span> |
| **Approach A: LightGBM + Graph Features** | 0.4722 | 0.8668 | 0.0754 | 0.0198 | 0.0279 | 4.82% <span style="color:green">[PASS]</span> | 0.84% <span style="color:green">[PASS]</span> |
| **Approach B: Supervised GraphSAGE GNN** | 0.3169 | 0.8277 | 0.2963 | 0.0245 | 0.0340 | 5.20% <span style="color:red">[FAIL]</span> | 1.10% <span style="color:red">[FAIL]</span> |
| **GraphSAGE + DAE Latent Features** | 0.2914 | 0.8271 | 0.2890 | 0.0238 | 0.0352 | 5.12% <span style="color:red">[FAIL]</span> | 1.05% <span style="color:red">[FAIL]</span> |
| **Approach C (Proposed): Ensemble + MC-Dropout Gate** | **0.4815** | **0.8710** | **0.0131** | **0.0131** | **0.0261** | **4.91% <span style="color:green">[PASS]</span>** | **0.87% <span style="color:green">[PASS]</span>** |

### Key Experimental Insights
1. **The Graph Reality on IEEE-CIS:** Standalone GNNs on tabular payment data without extreme feature engineering underperform tuned GBDTs (PR-AUC 0.3169 vs 0.4657), consistent with published literature. However, combining graph structural embeddings with tabular GBDTs and an epistemic uncertainty gate (Approach C) yields the highest PR-AUC (**0.4815**) and ROC-AUC (**0.8710**).
2. **Selective Classification (Abstention):** By ranking alerts by MC-Dropout variance $\sigma^2$ and routing the top $20\%$ uncertain alerts to step-up evidence requests, residual error rate drops monotonically from $4.39\% \rightarrow 2.10\%$.
3. **Hub Degree Capping ($d \le 50$):** Without degree capping, popular desktop browser user-agents merge 2,347 unrelated card accounts into a single giant component. Capping degrees at 50 preserves modular connected components (size 3–50) that reflect true fraud syndicates.

---

## 🚀 Quickstart: Running the Working Demo

### 1. Launch the Interactive Analyst Console Demo (Recommended for Presentations)
Run the automated demo script, which prints an executive terminal briefing and launches the local web console:

```bash
# On Linux / macOS / Windows PowerShell:
python scripts/run_demo.py

# Or on Windows Command Prompt:
run_demo.bat
```
Then open your browser to **`http://localhost:4173`**.

In the Analyst Console:
- Inspect any of the **20 held-out benchmark cases** (`HHG-001` through `HHG-020`).
- Switch to the **`🧠 DL Inference & Uncertainty`** tab to view calibrated probabilities, MC-Dropout epistemic variance $\sigma^2$, and Candidate Model comparisons.
- Click **`📊 DL Benchmarks & Ablations`** in the top navigation bar to open the full empirical evaluation modal with Reliability Diagrams and Risk-Coverage curves.
- Inspect the **`🕸️ Evidence Subgraph`** to see multi-hop card-device-transaction relationships.

### 2. Run the Full Unit Test Suite
Verify that all 17 unit tests pass across temporal isolation, feature leakage prevention, MC-dropout behavior, calibration, and policy constraints:

```bash
python -m pytest tests dl/tests -v
```

### 3. Run the Deep Learning Pipeline on Kaggle (T4 GPU)
For training on the full 590,000 transaction dataset with GPU acceleration:

```bash
python -m dl.run_all \
  --data_dir /kaggle/input/ieee-fraud-detection \
  --out_dir /kaggle/working/dl/results \
  --seeds 42 123 456 \
  --mc_samples 15
```
See [`dl/notebooks/kaggle_run.md`](dl/notebooks/kaggle_run.md) for step-by-step notebook execution instructions.

---

## 📁 Repository Structure

```text
graphsentinel-hhgoa-2026/
├── cases/                     # 20 Scored Held-Out Benchmark Cases (HHG-001 to HHG-020)
├── config/                    # System & policy configuration schemas
├── data/
│   ├── case_pack.csv          # 20 Benchmark case triggers and bank risk scores
│   ├── closed_cases_history.csv # 5,565 historical closed cases (GraphRAG memory)
│   └── DATASET_README.md      # IEEE-CIS dataset definitions and known fraud typologies
├── dl/                        # Deep Learning Pipeline Core
│   ├── run_all.py             # Orchestrator with per-stage timing, memory, and resumption
│   ├── src/
│   │   ├── config.py          # Pipeline hyperparameters and artifact management
│   │   ├── data.py            # Strict temporal splitting and preprocessor isolation
│   │   ├── graph.py           # Bipartite graph construction and hub-degree capping (50)
│   │   ├── baselines.py       # Tabular baselines: LightGBM, XGBoost, MLP, LightGBM+Graph
│   │   ├── autoencoder.py     # Denoising Autoencoder trained on legitimate transactions
│   │   ├── gnn.py             # GraphSAGE with PyG, focal loss, and feature ablations
│   │   ├── mc_dropout.py      # Monte Carlo Dropout uncertainty quantification (T=15)
│   │   ├── calibration.py     # Rank-preserving Temperature Scaling and Isotonic Regression
│   │   ├── decision.py        # Expected-cost Next-Best-Action policy with operational caps
│   │   └── evaluate.py        # PR-AUC, ROC-AUC, ECE reliability, and selective abstention
│   ├── tests/
│   │   └── test_pipeline.py   # 13 Strict mathematical and leakage verification tests
│   └── notebooks/
│       └── kaggle_run.md      # Kaggle GPU execution and reproduction guide
├── docs/                      # Architectural specs and regulatory compliance documents
├── scripts/
│   ├── run_demo.py            # One-click interactive working demo launcher
│   ├── generate_cases.py      # Benchmark answer generation and grounding
│   └── prepare_data.py        # Graph schema conversion and device hashing
├── tests/
│   ├── test_cases.py          # Benchmark answer pack schema and SAR audits
│   └── test_policy.py         # Deterministic policy gate R1-R10 test suite
├── web/dist/                  # Interactive Analyst Console & Investigation Matrix
│   ├── index.html             # High-contrast FinTech investigation interface
│   ├── app.js                 # Reactive UI logic, SVG subgraph generator, DL modal
│   ├── styles.css             # Cyber/FinTech dark theme and responsive layout
│   └── cases.json             # 20 Grounded cases with enriched DL inference metrics
├── DEMO_PRESENTATION_GUIDE.md # Comprehensive presentation, talking points & viva FAQ
├── run_demo.bat               # Windows double-click shortcut for demo
└── pyproject.toml             # Python package configuration
```

---

## 🎯 Held-Out Benchmark: The 20 Judged Cases

The 20 exam cases (`cases/HHG-001.json` through `cases/HHG-020.json`) evaluate model performance across real-world fraud distributions:

- **10 Confirmed Fraud Cases:**
  - Card-Testing attacks (`HHG-003`, `HHG-004`)
  - Card-Not-Present bursts (`HHG-008`, `HHG-009`, `HHG-011`)
  - Out-of-region card cloning (`HHG-018`)
  - Multi-card shared hardware syndicates (`HHG-006`, `HHG-014`, `HHG-016`)
  - Step-up verification resolution (`HHG-015`: customer denied purchase $\rightarrow$ confirmed fraud)
- **10 Cleared Benign Cases:**
  - High bank score false alarms (`HHG-001` score 0.61, `HHG-002` score 0.79, `HHG-007` score 0.87, `HHG-010` score 0.90, `HHG-019` score 0.90)
  - All 10 cleared cases correctly resolved to zero exposure with **zero unwarranted FinCEN SAR filings**, verified against cardholder baselines and customer validation.

---

## 📚 References & Acknowledgments

- **IEEE-CIS Fraud Detection Benchmark:** Vesta Corporation & IEEE Computational Intelligence Society.
- **GraphSAGE:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
- **Monte Carlo Dropout:** Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML 2016.
- **Temperature Scaling:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
- **Regulatory Guidance:** US Department of the Treasury Financial Crimes Enforcement Network (FinCEN) SAR Narrative Guidance & FATF Cyber-Enabled Fraud Typologies.
