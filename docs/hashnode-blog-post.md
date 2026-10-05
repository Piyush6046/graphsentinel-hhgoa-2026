# Beyond the Risk Score: Building an Evidence-First Autonomous Fraud Agent with TigerGraph

*How we built NexusWatch AI for the TigerGraph × Hackathon Goa 2026 Challenge to eliminate hallucinations and dismantle coordinated fraud syndicates using GraphRAG and deterministic policy gates.*

---

## 📌 Executive Summary & Metadata for Hashnode

- **Title:** Beyond the Risk Score: Building an Evidence-First Autonomous Fraud Agent with TigerGraph
- **Subtitle:** How NexusWatch AI combines multi-hop graph traversals, 5,500+ historical case memory, and deterministic policy rules to achieve zero-hallucination agentic fraud triage.
- **Tags:** `TigerGraph`, `GraphDatabase`, `Artificial Intelligence`, `Cybersecurity`, `FinTech`, `GraphRAG`, `Machine Learning`
- **GitHub Repository:** [https://github.com/Piyush6046/graphsentinel-hhgoa-2026](https://github.com/Piyush6046/graphsentinel-hhgoa-2026)

---

## 1. The Core Problem: Risk Scores Are Not Verdicts

Traditional fraud detection in financial institutions is broken in a subtle but dangerous way: **Machine learning models produce statistical risk scores, but risk scores are not verdicts.**

A single float number (e.g., `0.78 risk score`) only indicates statistical anomaly. It cannot:

- Distinguish between a genuine high-spender and an account takeover.
- Uncover coordinated multi-account rings using the same hardware fingerprint.
- Determine whether a card should be frozen, an alert monitored, or a federal Suspicious Activity Report (SAR) filed.

When financial institutions rely blindly on tabular ML scores, they face a double penalty:

1. **False Positives:** Legitimate cardholders get their cards abruptly declined during normal travel or unusual purchases, causing customer churn.
2. **Coordinated Blindspots:** Sophisticated fraud syndicates deliberately keep transaction amounts low, scoring low risk (`0.05`) on isolated checks while draining thousands across syndicated accounts.

To solve this, we built **NexusWatch AI** for the **TigerGraph × Hackathon Goa 2026 Challenge**.

---

## 2. NexusWatch AI Architecture: Graph First, Policy Last

NexusWatch AI enforces a strict principle: **The graph discovers evidence, institutional memory provides context, and deterministic policy gates dictate action. The LLM only explains—it never decides unapproved actions or hallucinates entity IDs.**

```text
       [ Live Transaction Alert / Anomaly Trigger ]
                           │
                           ▼
          [ GSQL Multi-Hop Subgraph Traversal ]
      (Card Baseline · Device Ring · Transaction Window)
                           │
                           ▼
          [ Institutional Memory Retrieval ]
       (5,565 Closed Case Archetypes & Cluster Sim)
                           │
                           ▼
          [ Deterministic Policy Gate (R1–R10) ]
     (Verification · Exposure Cap · Auto-Freeze · SAR)
                           │
                           ▼
          [ Graph Writeback & Memory Expansion ]
```

### The 5 Core Pillars:

1. **TigerGraph Evidence Plane:** Models Customers, Cards, Transactions, composite Device Hardware Profiles, Email Domains, Billing Regions, and Historical Case Vertices.
2. **Autonomous GraphRAG Investigator:** Executes 4 parallel GSQL queries returning bounded subgraphs with verified entity IDs.
3. **Calibrated Probability Layer:** Re-calibrates risk based on corroborated graph evidence and customer verification, preventing model-score overreliance.
4. **Deterministic Policy Gate (R1–R10):** Pure Python rule engine that dictates allowed actions (`ALLOW_TRANSACTION`, `FREEZE_CARD`, `RESTRICT_MERCHANT`, `FILE_REPORT`) and approval tiers (L1 vs. L2 Compliance).
5. **Graph Memory Writeback:** Persists every resolved case node back into TigerGraph memory to continuously train future investigations.

---

## 3. Modeling the Graph Investigation Surface in GSQL

The benchmark dataset comprises over **590,000 transactions**, **144,000 customer identities**, and **5,565 historical closed cases**.

In tabular representations, multi-hop connections are computationally prohibitive. In TigerGraph, multi-hop subgraphs run in sub-second latency:

```gsql
CREATE VERTEX Customer (PRIMARY_ID id STRING, email STRING, billing_region STRING)
CREATE VERTEX Card (PRIMARY_ID id STRING, customer_id STRING)
CREATE VERTEX Transaction (PRIMARY_ID id STRING, amount DOUBLE, timestamp STRING, is_fraud BOOL)
CREATE VERTEX Device (PRIMARY_ID id STRING, os STRING, browser STRING, resolution STRING)
CREATE VERTEX ClosedCase (PRIMARY_ID id STRING, verdict STRING, pattern STRING, exposure DOUBLE)

CREATE UNDIRECTED EDGE OWNS_CARD (FROM Customer, TO Card)
CREATE UNDIRECTED EDGE TRANSACTED_ON (FROM Card, TO Transaction)
CREATE UNDIRECTED EDGE USED_DEVICE (FROM Transaction, TO Device)
CREATE UNDIRECTED EDGE CASE_SIMILAR_TO (FROM ClosedCase, TO ClosedCase, similarity DOUBLE)
```

---

## 4. Real-World Case Studies from the Benchmark

### Case Study 1: The Hidden Shared-Device Ring (Case HHG-014)

- **The Incoming Alert:** Scored at just **0.05 risk score**—a tabular system would have cleared it immediately.
- **What TigerGraph Found:** Traversing the transaction's composite device profile revealed an identical Samsung mobile hardware fingerprint shared across multiple distinct cards and connected to **4 confirmed historical fraud cases** (`CC-3035`, `CC-2985`, `CC-2971`, `CC-2649`).
- **Autonomous Resolution:** NexusWatch AI classified the coordinated activity as an *undocumented shared-device syndicate*, routed the card block to L1 review, placed related accounts on enhanced monitoring, and routed a SAR filing to L2 compliance.

### Case Study 2: High-Value Velocity Burst (Case HHG-006)

- **The Incoming Alert:** 4 online purchases totaling **$1,906.07** in under 30 minutes with identical amounts.
- **The Investigation:** Card testing policy requires small probe amounts. Because these charges were high-value from the outset, the agent labeled the attack an *undocumented velocity burst*.
- **Autonomous Action:** Triggered L1 card suspension and generated an automated FinCEN Suspicious Activity Report (SAR) draft with complete provenance.

### Case Study 3: Avoiding False Positives with Cardholder Verification (Case HHG-010)

- **The Incoming Alert:** Scored at a menacing **0.90 risk score** on a $1,000 online purchase.
- **The Investigation:** Under Rule R1, a raw statistical score is merely an alert trigger, not a conviction. The agent requested cardholder verification.
- **The Resolution:** The cardholder verified the purchase. Under Rule R3, exposure was reset to **$0.00**, no cards were frozen, no SAR was created, and the customer suffered zero friction.

---

## 5. Why Deterministic Policy Gates Matter

Large Language Models are exceptional at natural language synthesis and contextual translation, but they must **never** be given the authority to invent financial policies, create non-existent approval routes, or hallucinate entity IDs.

In NexusWatch AI:

- **Zero Hallucination:** Every claim references an audited GSQL query (e.g. `query:card_baseline(customer_id=C12382)`) and explicit entity IDs.
- **Deterministic Action Routes:** All actions are bounded by rules R1–R10 in audited code outside the LLM.
- **Audit-Ready Compliance:** Generates full chain-of-custody ledgers and FinCEN SAR drafts with one click.

---

## 6. What We Learned Building with TigerGraph

1. **Sub-second Multi-Hop Traversals:** Traversing deep 3-hop relationships (`Card → Transaction → Device → Historical Case Clusters`) across 590k+ nodes executed in single-digit milliseconds.
2. **Institutional Graph Memory as Dynamic Vector RAG:** Using graph writeback to turn completed investigations into persistent memory nodes allowed subsequent runs to compare new patterns against confirmed archetypes instantly.
3. **Graph First Architecture:** The safest AI agent is not the one that guesses the fastest—it is the one that can mathematically prove where every fact originated, enforce compliance rules deterministically, and know precisely when to stop.

---

## 7. Submission Deliverables & Links

- 📂 **Public GitHub Repository:** [https://github.com/Piyush6046/graphsentinel-hhgoa-2026](https://github.com/Piyush6046/graphsentinel-hhgoa-2026)
- 🗄️ **20 Scored Case Invariant Files:** Calibrated 10 Fraud / 10 Legitimate answer pack in `/cases`
- 🖥️ **Analyst Command Console:** Interactive dynamic SVG subgraph explorer, provenance ledgers, and SAR generator.
- ⚡ **Challenge:** TigerGraph × Hackathon Goa 2026

*Built by NEXSCALE for the TigerGraph × HHGoa 2026 Agentic Fraud Investigation Challenge.*
