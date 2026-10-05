# NexusWatch AI — 3–5 Minute Video Demo Script

Follow these visual cues and timestamps when recording your screen demonstration:

| Timestamp | Topic & Focus | On-Screen Action | Key Talking Points |
|---|---|---|---|
| **0:00 – 0:35** | **Problem Statement** | Open `http://localhost:4173` in full screen; show the executive KPI banner and case queue. | *“Fraud alerts are invitations to investigate, not verdicts.”* Explain the 10 Fraud / 10 Legitimate split and zero-hallucination policy gating. |
| **0:35 – 1:15** | **Architecture & Pipeline** | Scroll down to the **Pipeline Blueprint** (Steps 01 to 05). | Explain TigerGraph multi-hop GSQL traversals, 5,565 closed case memory retrieval, deterministic R1–R10 policy gate, and graph writeback. |
| **1:15 – 2:10** | **Hidden Syndicate Ring (HHG-014)** | Select case **HHG-014** from the queue. Show **Overview & Subgraph** and **Policy Execution** tabs. | Low risk score (0.05) deceived tabular models, but TigerGraph exposed a shared Samsung mobile fingerprint connected to 4 prior fraud cases. Show auto-freeze and L2 SAR routing. |
| **2:10 – 2:50** | **Undocumented Burst Exposure (HHG-006)** | Select case **HHG-006**. Click the **Regulatory SAR Preview** tab. | 4 rapid high-value transactions ($1,906 exposure). Explain why it is an undocumented burst rather than low-value card testing. Show auto-generated FinCEN SAR report. |
| **2:50 – 3:35** | **False Positive & Restraint (HHG-010)** | Select case **HHG-010**. Click **Evidence Ledger** tab. | High 0.90 risk score, but customer verification confirmed legitimacy. Under Rule R3, exposure resets to $0.00 and no SAR is filed. Show GSQL query provenance. |
| **3:35 – 4:15** | **Reproducibility & Test Suite** | Switch to Terminal. Run `python -m pytest -q` and list `cases/`. | Show all 4 test suites passing (`4 passed`), schema compliance, and deterministic writeback to TigerGraph. |
| **4:15 – 4:35** | **Conclusion & Outro** | Switch back to browser dashboard. | Wrap up: *“Graph first, policy last, evidence throughout.”* Thank TigerGraph & HHGoa 2026. |
