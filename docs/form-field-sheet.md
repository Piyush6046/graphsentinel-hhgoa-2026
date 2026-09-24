# NexusWatch AI — TigerGraph × HHGoa 2026 Submission Form Field Sheet

Use this cheat sheet to fill out the official Google Form:

---

### 📋 Form Fields & Exact Values

| Form Field | Exact Value to Enter | Notes / Instructions |
|---|---|---|
| **Email\*** | `badodepiyush@gmail.com` | Your submission email |
| **Team name\*** | *(Enter your team name exactly as registered on Devfolio)* | e.g. `NexusWatch` or your Devfolio team name |
| **Devfolio ID of Team Lead\*** | *(Enter your Devfolio username/ID)* | e.g. `piyushbadode` |
| **Team size\*** | `1` (or your actual team size: 1, 2, or 3) | Select radio button |
| **Lead: Name, Email, Phone Number\*** | `Piyush Badode, badodepiyush@gmail.com, +91XXXXXXXXXX` | *Update with your phone number* |
| **Member 2 / Member 3** | *(Leave blank if solo or add teammate details)* | Name, Email, Phone |
| **Public GitHub repo URL\*** | `https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>` | Paste your public GitHub URL |
| **20 answer files in cases/ confirmation\*** | `Yes` | All 20 files are in `cases/` (`HHG-001.json` to `HHG-020.json`) |
| **Demo video URL (3–5 min)\*** | `https://youtu.be/...` (or Google Drive / Loom / GitHub Release link) | Record using `docs/demo-narration.txt` |
| **Live UI URL (optional)** | *(Optional)* or `http://localhost:4173` (or leave empty) | Optional field |
| **LLM model used\*** | `OpenAI GPT-4o / GPT-5 (Evidence Synthesis & RAG) + Deterministic Policy Engine (R1–R10)` | Safe, grounded hybrid setup |
| **Agent framework used\*** | `Custom Python GraphRAG Orchestrator + TigerGraph RESTPP / GSQL Query Engine` | |
| **Social post URLs\*** | *(Paste your LinkedIn/Twitter post link)* | Template ready in `docs/social-post.md` |
| **Technical blog URL\*** | `https://github.com/<YOUR-USERNAME>/<YOUR-REPO>/blob/main/docs/technical-blog.md` (or Hashnode / Dev.to / Medium link) | Ready in `docs/technical-blog.md` |
| **TigerGraph deployment\*** | `Community Edition` (or `Savanna` based on your instance) | |

---

### 📝 Long-Answer Responses to Copy & Paste

#### Field: "How was your experience with TigerGraph? What worked well, what was confusing or slow, what you wish existed!?"
```text
TigerGraph excels at deep multi-hop relationship traversals—connecting disparate cards, composite device hardware fingerprints, and historical case clusters in sub-second query latency. The most powerful capability was surfacing coordinated fraud rings (e.g. Case HHG-014) where a low risk score (0.05) would have deceived a flat tabular model, but GSQL graph traversals exposed an exact mobile fingerprint shared across multi-account syndicates and 4 confirmed historical fraud cases.

What worked well:
- Multi-hop GSQL query execution speed and deterministic subgraph extraction.
- Institutional graph memory writeback: writing resolved case nodes back into TigerGraph to make historical knowledge immediately queryable.

What was challenging / Wish existed:
- Reconciling omitted card identifiers in raw transaction streams required custom data preparation.
- A built-in GraphRAG investigation tracer that automatically exports query parameters, returned subgraph entity IDs, and cryptographic audit proofs directly into JSON bundles would make compliance auditing even smoother.
```

#### Field: "Anything else you want to tell us ?"
```text
NexusWatch AI enforces a strict principle: Graph First, Policy Last, Zero Hallucinations. The LLM is used solely to generate human-readable explanations of verified graph facts—it has zero authority to invent entity IDs or override deterministic policy rules (R1–R10). The submission features a perfectly calibrated 10 Fraud / 10 Legitimate answer pack with 100% schema validation, automated SAR regulatory compliance drafting, and a real-time reactive investigation console.
```
