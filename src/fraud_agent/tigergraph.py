"""Small dependency-free client for installed GraphSentinel GSQL queries."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


class TigerGraphClient:
    def __init__(self, host: str | None = None, token: str | None = None, graph: str = "FraudGraph"):
        self.host = (host or os.environ.get("TIGERGRAPH_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("TIGERGRAPH_TOKEN", "")
        self.graph = os.environ.get("TIGERGRAPH_GRAPH", graph)
        if not self.host:
            raise ValueError("Set TIGERGRAPH_HOST to a Savanna or Community Edition endpoint")

    def _request(self, path: str, *, method: str = "GET", body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.host}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.load(response)
        if payload.get("error"):
            raise RuntimeError(payload.get("message", "TigerGraph request failed"))
        return payload

    def query(self, name: str, **params) -> dict:
        # RESTPP treats a literal "+" as data in query parameters, so encode
        # spaces as %20 instead of HTML form-style plus signs.
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        return self._request(f"/restpp/query/{self.graph}/{name}?{query}")

    def health(self) -> dict:
        return self._request("/restpp/echo")

    def write_case(self, answer: dict, card_id: str) -> str:
        case = answer["case"]
        case_id = answer["case_id"]
        try:
            result = self.query(
                "write_investigation_case", case_id=case_id, card_id=card_id,
                status=case["status"], verdict=case["verdict"],
                fraud_probability=case["fraud_probability"], pattern=case["pattern"],
                exposure_usd=case["exposure_usd"], summary=case["summary"],
                action_json=json.dumps(answer["next_best_actions"]["final"]),
                updated_at="2016-12-31 23:59:59",
            )
            if result.get("results") is not None:
                return case_id
        except Exception:
            pass

        # Direct RESTPP upsert fallback to FraudCase vertex and FC_ON_CARD edge
        payload = {
            "vertices": {
                "FraudCase": {
                    case_id: {
                        "case_id": {"value": case_id},
                        "card_id": {"value": card_id},
                        "status": {"value": case["status"]},
                        "verdict": {"value": case["verdict"]},
                        "fraud_probability": {"value": float(case["fraud_probability"])},
                        "pattern": {"value": case["pattern"]},
                        "pattern_description": {"value": case.get("pattern_description", "")},
                        "affected_txn_ids": {"value": json.dumps(case.get("affected_txn_ids", []))},
                        "first_suspicious_txn_id": {"value": case.get("first_suspicious_txn_id", "")},
                        "connected_card_ids": {"value": json.dumps(case.get("connected_card_ids", []))},
                        "connected_device_profiles": {"value": json.dumps(case.get("connected_device_profiles", []))},
                        "exposure_usd": {"value": float(case.get("exposure_usd", 0))},
                        "evidence_json": {"value": json.dumps(case.get("evidence", []))},
                        "similar_prior_cases": {"value": json.dumps(case.get("similar_prior_cases", []))},
                        "summary": {"value": case.get("summary", "")},
                        "stop_reason": {"value": answer.get("stop_reason", "")},
                        "next_best_actions_json": {"value": json.dumps(answer.get("next_best_actions", {}))},
                        "sar_json": {"value": json.dumps(answer.get("sar", {}))},
                    }
                }
            },
            "edges": {
                "FraudCase": {
                    case_id: {
                        "FC_ON_CARD": {
                            "Card": {
                                card_id: {}
                            }
                        }
                    }
                }
            }
        }
        res = self._request(f"/restpp/graph/{self.graph}", method="POST", body=payload)
        return case_id if not res.get("error") else ""

