# Beyond the Risk Score: Building an Evidence-First Fraud Agent with TigerGraph

Fraud detection systems are notoriously good at sounding alarms and remarkably bad at explaining what should happen next. A statistical risk score tells an analyst where to look; it cannot tell them whether a cardholder should be frozen, whether connected accounts are in jeopardy, or whether a regulatory Suspicious Activity Report (SAR) is required under federal policy.

For the **TigerGraph × Hackathon Goa 2026 Challenge**, we built **NexusWatch AI**—an autonomous, evidence-first graph fraud investigation matrix that treats evidence gathering, risk calibration, and policy execution as distinct, auditable stages.

---

## 1. The Graph is the Ultimate Investigation Surface

Tabular models evaluate transactions in silos. Real financial syndicates, however, exploit the complex web of relationships across multi-account rings, shared device hardware, and temporal clusters.

Our TigerGraph schema models:
- **Entities:** `Customer`, `Card`, `Transaction`, composite `Device` hardware profiles, `Email` domains, billing `Region`, and historical `ClosedCase` vertices.
- **Relationships:** `OWNS_CARD`, `TRANSACTED_ON`, `USED_DEVICE`, `TRANSACTED_IN_REGION`, and multi-hop graph neighbor edges.

This relationship-first architecture pays off when individual transactions appear benign. In **Case HHG-014**, an alert arrived with a model risk score of only **0.05**. In a tabular system, this would pass unnoticed. In TigerGraph, multi-hop traversals instantly revealed that the transaction's exact mobile hardware fingerprint was reused across multiple customer cards and connected to **4 confirmed historical fraud cases**.

---

## 2. Autonomous GraphRAG Without Hallucinations

For every incoming alert, NexusWatch AI executes four bounded GSQL subgraphs in parallel:
1. **Customer Baseline:** Calculates historical percentile distributions, median amounts, and velocity.
2. **Transaction Window:** Bounded temporal traversal capturing burst activity.
3. **Device / Entity Neighborhood:** Multi-hop traversal uncovering shared hardware syndicates.
4. **Institutional Case Memory:** Graph similarity matching against 5,565 historical closed cases.

The LLM is strictly used for synthesis and human-readable summarization. All entity IDs, transaction hashes, and query references are deterministically grounded.

---

## 3. Deterministic Policy Gate (Rules R1–R10)

Compliance policies distinguish automatic actions from tiered L1/L2 approvals, and separate internal fraud cases from regulatory SAR filings. We encoded these boundaries as deterministic code outside the LLM:
- **Card Blocks (< $2,500):** Routed to L1 Analyst Review.
- **Regulatory SAR Filings:** Always routed to L2 Compliance with FinCEN documentation drafts.
- **Case Creation & Customer Verification:** Handled automatically.

In **Case HHG-006**, four near-identical online purchases totaled $1,906.07 within 30 minutes. NexusWatch AI classified this as an *undocumented high-value burst pattern*, generated an internal case, recommended card isolation, and drafted a SAR filing under policy thresholds.

---

## 4. Calibrated Restraint & Zero False Positives

Half of the challenge benchmark cases are legitimate alerts. NexusWatch AI avoids customer-damaging false positives by enforcing Rule R1 and R3 verification gates.

In **Case HHG-010**, an alert started with a high **0.90 risk score** on a $1,000 transaction. Instead of instantly locking the account, the agent requested simulated cardholder verification. Once confirmed, Rule R3 settled the investigation as legitimate: exposure reset to $0.00, no card was blocked, and no SAR was generated.

---

## 5. Summary & Key Results

- **20/20 Scored Cases:** Calibrated 10 Fraud / 10 Legitimate answer pack with 100% schema invariant compliance.
- **Real-Time Analyst Console:** Responsive dashboard featuring dynamic SVG subgraph visualization, chain-of-custody ledgers, and SAR document generators.
- **Graph Writeback:** Every resolved case is stored back into TigerGraph memory for continuous institutional learning.

**NexusWatch AI Principle:** *Graph first, policy last, and evidence throughout.*
