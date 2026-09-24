import json
from pathlib import Path

from fraud_agent.schemas import validate_answer


ROOT = Path(__file__).resolve().parents[1]


def answers():
    return [json.loads(p.read_text()) for p in sorted((ROOT / "cases").glob("HHG-*.json"))]


def test_exact_answer_pack():
    docs = answers()
    assert len(docs) == 20
    assert [d["case_id"] for d in docs] == [f"HHG-{n:03d}" for n in range(1, 21)]
    assert all(not validate_answer(d) for d in docs)


def test_balanced_outcomes_and_sar_consistency():
    docs = answers()
    assert sum(d["case"]["verdict"] == "fraud" for d in docs) == 10
    assert sum(d["case"]["verdict"] == "legitimate" for d in docs) == 10
    for d in docs:
        final = {a["action"] for a in d["next_best_actions"]["final"]}
        assert d["sar"]["file"] == ("FILE_REPORT" in final)
        if d["case"]["verdict"] == "legitimate":
            assert d["case"]["affected_txn_ids"] == []
            assert d["case"]["exposure_usd"] == 0
