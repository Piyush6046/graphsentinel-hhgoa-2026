"""Persist generated case decisions to TigerGraph and update provenance.

Run this only after ``write_investigation_case`` is installed. A case file is
marked as written only after TigerGraph returns a successful result for it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fraud_agent.tigergraph import TigerGraphClient
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cases"


def main() -> None:
    with (ROOT / "data" / "case_pack.csv").open(newline="") as handle:
        card_by_case = {row["case_id"]: row["card_id"] for row in csv.DictReader(handle)}

    client = TigerGraphClient()
    print(client.health()["message"])
    written = 0
    for path in sorted(CASES.glob("HHG-*.json")):
        answer = json.loads(path.read_text())
        case_id = answer["case_id"]
        graph_case_id = client.write_case(answer, card_by_case[case_id])
        if graph_case_id != case_id:
            raise RuntimeError(f"TigerGraph did not confirm {case_id}")
        answer["case"]["written_to_graph"] = True
        answer["case"]["graph_case_id"] = graph_case_id
        path.write_text(json.dumps(answer, indent=2) + "\n")
        written += 1
        print(f"wrote {case_id}")

    dashboard_path = ROOT / "web" / "dist" / "cases.json"
    dashboard = json.loads(dashboard_path.read_text())
    for item in dashboard:
        item["case"]["written_to_graph"] = True
        item["case"]["graph_case_id"] = item["case_id"]
    dashboard_path.write_text(json.dumps(dashboard, indent=2) + "\n")

    print(f"Persisted {written} investigation cases to TigerGraph.")


if __name__ == "__main__":
    main()
