"""Executable version of the benchmark fraud policy.

The policy module is intentionally deterministic. Graph evidence determines facts;
the language model may summarize them, but it cannot invent actions or approvals.
"""

from __future__ import annotations

AUTO_ACTIONS = {
    "ALLOW_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
}


def approval_route(action: str, exposure_usd: float = 0.0) -> str:
    """Return the exact policy approval route for an action."""
    if action in AUTO_ACTIONS:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure_usd <= 2500 else "L2"
    if action in {"BLOCK_ALL_CARDS", "FILE_REPORT"}:
        return "L2"
    raise ValueError(f"Unknown policy action: {action}")


def case_required(probability: float, evidence_requested: bool, disputed: bool) -> bool:
    return probability >= 0.30 or evidence_requested or disputed


def report_required(
    *,
    fraud_probability: float,
    exposure_usd: float,
    shared_origin: bool,
    coordinated: bool,
    undocumented: bool,
) -> bool:
    """Policy 3a: strong suspicion plus one filing condition."""
    strongly_suspected = fraud_probability >= 0.70
    filing_condition = (
        exposure_usd > 1000 or shared_origin or coordinated or undocumented
    )
    return strongly_suspected and filing_condition


def action(action_name: str, reason: str, exposure_usd: float = 0.0) -> dict:
    return {
        "action": action_name,
        "route": approval_route(action_name, exposure_usd),
        "reason": reason,
    }

