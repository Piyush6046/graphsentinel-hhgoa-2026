"""Validation helpers for benchmark answer files."""

from __future__ import annotations

from typing import Any

PATTERNS = {
    "card_testing",
    "card_not_present_fraud",
    "card_not_present_new_device",
    "out_of_region_use",
    "account_takeover",
    "undocumented",
    "none",
}
STATUSES = {"open", "closed_fraud", "closed_legitimate", "escalated"}
VERDICTS = {"fraud", "legitimate", "uncertain"}
ACTIONS = {
    "ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS", "GENERATE_REPORT",
    "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}


def validate_answer(answer: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = {
        "case_id", "case", "evidence_requests", "next_best_actions", "sar",
        "stop_reason", "tool_calls", "tokens", "latency_s",
    }
    missing = required_top - answer.keys()
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
        return errors

    case = answer["case"]
    required_case = {
        "status", "verdict", "fraud_probability", "pattern",
        "pattern_description", "affected_txn_ids", "first_suspicious_txn_id",
        "connected_card_ids", "connected_device_profiles", "exposure_usd",
        "evidence", "similar_prior_cases", "summary", "written_to_graph",
        "graph_case_id",
    }
    missing_case = required_case - case.keys()
    if missing_case:
        errors.append(f"missing case fields: {sorted(missing_case)}")
    if case.get("status") not in STATUSES:
        errors.append("invalid case.status")
    if case.get("verdict") not in VERDICTS:
        errors.append("invalid case.verdict")
    if case.get("pattern") not in PATTERNS:
        errors.append("invalid case.pattern")
    probability = case.get("fraud_probability")
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        errors.append("case.fraud_probability must be between 0 and 1")
    if case.get("verdict") == "legitimate":
        if case.get("affected_txn_ids"):
            errors.append("legitimate cases must have no affected transactions")
        if case.get("exposure_usd") != 0:
            errors.append("legitimate cases must have zero exposure")

    actions = answer["next_best_actions"]
    for stage in ("initial", "final"):
        for item in actions.get(stage, []):
            if item.get("action") not in ACTIONS:
                errors.append(f"invalid action in {stage}: {item.get('action')}")
            if item.get("route") not in {"auto", "L1", "L2"}:
                errors.append(f"invalid route in {stage}: {item.get('route')}")
    final_names = {item.get("action") for item in actions.get("final", [])}
    if answer["sar"].get("file") != ("FILE_REPORT" in final_names):
        errors.append("sar.file must agree with final FILE_REPORT action")
    return errors

