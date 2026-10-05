# Deep Learning Pipeline References and Citations

This document lists foundational research papers, architectural inspirations, and open-source software libraries referenced in this pipeline. All implementations in `dl/` are original custom implementations built specifically for this project.

---

## 1. Research Papers & Theoretical Foundations

1. **Graph Neural Networks (GraphSAGE)**
   - **Paper:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. In Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 1024–1034.
   - **Application:** Inductive neighborhood aggregation and feature pooling across heterogeneous transaction, card, and device relationship graphs.

2. **Monte Carlo Dropout for Uncertainty Estimation**
   - **Paper:** Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. In International Conference on Machine Learning (ICML 2016), pp. 1050–1059.
   - **Application:** Epistemic uncertainty quantification via stochastic forward passes at inference time, computing predictive variance, Shannon entropy, and mutual information.

3. **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**
   - **Paper:** Kendall, A., & Gal, Y. (2017). In Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 5574–5584.
   - **Application:** Distinction between aleatoric (data/observation noise) and epistemic (model parameter uncertainty) components for risk-sensitive decision routing.

4. **Focal Loss for Dense Object Detection & Imbalanced Classification**
   - **Paper:** Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. In IEEE International Conference on Computer Vision (ICCV 2017), pp. 2980–2988.
   - **Application:** Mitigating severe fraud class imbalance (~3.5% positive fraud rate) by dynamically down-weighting well-classified easy negative transactions.

5. **On Calibration of Modern Neural Networks**
   - **Paper:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). In International Conference on Machine Learning (ICML 2017), pp. 1321–1330.
   - **Application:** Expected Calibration Error (ECE), reliability diagrams, and confidence-score temperature scaling.

6. **Selective Classification (Risk-Coverage Analysis)**
   - **Paper:** Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. In Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 4878–4887.
   - **Application:** Abstention curves, error vs. coverage trade-offs, and automated escalation to human fraud analysts.

---

## 2. Open-Source Repositories & Implementations Referenced

1. **PyTorch Geometric (PyG)**
   - **Repository:** `pyg-team/pytorch_geometric` (https://github.com/pyg-team/pytorch_geometric)
   - **License:** MIT License
   - **Role:** Graph neural network message-passing primitives and standard benchmark conventions.

2. **Graph Fraud Detection Baselines (waittim/graph-fraud-detection)**
   - **Repository:** `waittim/graph-fraud-detection` (https://github.com/waittim/graph-fraud-detection)
   - **License:** MIT License
   - **Role:** Graph construction concepts on transaction datasets (connecting transactions via shared identity/card keys and hub degree capping).

3. **ApexFi / Financial Graph Learning Benchmarks**
   - **Repository:** Open benchmarks for heterogeneous graph fraud detection on transaction networks.
   - **License:** Apache 2.0 / MIT
   - **Role:** Bipartite graph projection representations and temporal evaluation design principles.

4. **LightGBM & XGBoost**
   - **Repositories:** `microsoft/LightGBM` (MIT License) and `dmlc/xgboost` (Apache 2.0 License)
   - **Role:** Gradient-boosted decision tree baseline implementations for tabular fraud detection.

---

## 3. Dataset Attribution

- **Dataset:** IEEE-CIS Fraud Detection Benchmark (Vesta Corporation & IEEE Computational Intelligence Society).
- **Kaggle Source:** https://www.kaggle.com/c/ieee-fraud-detection
- **Task:** Predict the probability that an online transaction is fraudulent (`isFraud` target).
