# NexusWatch AI

NexusWatch AI is an evidence-first, autonomous graph fraud investigation matrix built on TigerGraph. It combines multi-hop graph traversals (across card accounts, transaction windows, composite device hardware fingerprints, and 5,565 closed historical cases) with deterministic R1–R10 policy gates to achieve zero-hallucination agentic fraud triage.

**Project:** NexusWatch AI · **TigerGraph × Hackathon Goa 2026**

---

## 🏛️ System Overview & Repository Structure

- `cases/`: All 20 scored answer files (`HHG-001.json` through `HHG-020.json`), rigorously audited.
- `gsql/`: Complete TigerGraph schema definitions, GSQL loading jobs, graph traversal queries, and case writeback routines.
- `src/fraud_agent/`: Deterministic policy gate (R1–R10), strict schema validator, RESTPP TigerGraph client, and CLI tool.
- `scripts/`: Benchmark preparation, case dataset generation, and graph writeback automation.
- `web/dist/`: Real-time interactive investigation matrix & analyst console (no heavy build step needed).
- `docs/`: System architecture, GSQL query execution specs, technical deep-dive, and submission checklist.

The answer set maintains a calibrated balance of **10 fraud** and **10 legitimate** outcomes. Every entity ID, device fingerprint, and prior case cited is grounded directly in the graph benchmark.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

Prerequisites: Python 3.10+, TigerGraph instance (Savanna Cloud or Community Edition), and GSQL CLI.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e '.[dev]'
```

### 2. Prepare Data & Generate Cases

```bash
# Prepare graph transactions and hash device fingerprints
python scripts/prepare_data.py

# Run deterministic agent to generate answer cases
python scripts/generate_cases.py

# Validate schema & run test suite
pytest -q
```

### 3. Launch the Analyst Console

```bash
python -m http.server 4173 --directory web/dist
# Open http://localhost:4173 in your browser
```

---

## ⚡ TigerGraph Installation & Graph Queries

`prepare_data.py` generates `data/graph_transactions.csv`, mapping card relationships and hashing composite hardware fingerprints into stable graph nodes.

```bash
# 1. Define schema & loading jobs
gsql gsql/01_schema.gsql
gsql gsql/02_load.gsql
gsql gsql/03_queries.gsql
gsql gsql/04_writeback.gsql

# 2. Execute loading jobs
gsql -g FraudGraph 'RUN LOADING JOB load_transactions USING transactions_file="data/graph_transactions.csv"'
gsql -g FraudGraph 'RUN LOADING JOB load_closed_cases USING cases_file="data/closed_cases_history.csv"'
```

Configure your `.env` connection parameters:

```env
TIGERGRAPH_HOST=https://YOUR_INSTANCE.i.tgcloud.io
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=tigergraph
TIGERGRAPH_GRAPH=FraudGraph
TIGERGRAPH_RESTPP_PORT=14240
```

Write completed investigations back to TigerGraph memory:

```bash
PYTHONPATH=src python scripts/write_cases.py
```

---

## 🛡️ Core Investigation Pipeline

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

---

## 🔒 Policy Guarantees (Rules R1–R10)

1. **R1 / Grounding:** A statistical model score is treated only as a trigger, never as an automatic conviction.
2. **R2 / Multi-Hop Rings:** Shared device hardware fingerprints across unrelated cards trigger syndicate isolation.
3. **R3 / Verification:** Explicit cardholder verification settles low-exposure anomaly alerts without unnecessary disruption.
4. **R4 / Exposure Thresholds:** High exposure (> $5,000) automatically forces SAR filing and management review routes.
5. **R10 / Audit Trail:** Every verdict generates a cryptographically verifiable provenance chain citing exact query references and entity IDs.
