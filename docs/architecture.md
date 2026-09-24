# NexusWatch AI Architecture

NexusWatch AI enforces strict separation of concerns across evidence retrieval, policy gating, and autonomous action execution.

1. **TigerGraph Evidence Plane:** `Customer → Card → Transaction` establishes ownership and chronological transaction baselines. Hardware device fingerprints, IP regions, and merchant relationships expose synthetic syndicates and fraud rings.
2. **GraphRAG Autonomous Investigator:** Traverses customer baselines, bounded transaction windows, multi-hop device neighbors, and similar closed cases in parallel. Emits structured claims with exact GSQL query references and entity IDs.
3. **Evidence Calibration Layer:** Final probability is mathematically derived from corroborated graph evidence strength and customer verification state, preventing risk-score hallucinations.
4. **Deterministic Policy Gate (R1–R10):** Enforces SAR filing conditions, exposure thresholds, and exact multi-tier approval routing in deterministic code outside the LLM.
5. **Graph Memory Writeback:** Invokes `write_investigation_case` to persist audited findings into TigerGraph memory for continuous real-time learning.
6. **Analyst Command Console:** Interactive cyber-intelligence console featuring dynamic SVG subgraph visualization, provenance ledgers, policy transition matrices, and FinCEN SAR compliance previews.

## Core Guardrails

- All entity IDs are strictly grounded in graph database queries.
- Legitimate verdicts cannot contain non-zero financial exposure or affected transaction IDs.
- `SAR.file` state strictly aligns with the deterministic `FILE_REPORT` policy action.
- Action routes (`auto`, `analyst_review`, `l2_compliance`) are strictly bound by policy rules R1–R10.
- Investigation halts deterministically once customer verification or stopping thresholds are achieved.
