"""Unit tests for GraphSentinel CLI analyst suite."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fraud_agent.cli import (
    cmd_audit,
    cmd_explain,
    cmd_export_sar,
    cmd_filter,
    cmd_show,
    cmd_validate,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "cases"


def test_cli_validate():
    """Verify that validate passes on all 20 held-out cases."""
    assert cmd_validate(CASES_DIR) == 0


def test_cli_show():
    """Verify that show outputs raw valid JSON."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        code = cmd_show(CASES_DIR, "HHG-001")
    assert code == 0
    data = json.loads(buf.getvalue())
    assert data["case_id"] == "HHG-001"


def test_cli_explain():
    """Verify explain outputs formatted investigation briefing."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        code = cmd_explain(CASES_DIR, "HHG-006")
    assert code == 0
    out = buf.getvalue()
    assert "HHG-006" in out
    assert "FRAUD" in out
    assert "$1,906.07" in out
    assert "EVIDENCE CLAIMS" in out
    assert "FINCEN SAR REGULATORY STATUS" in out


def test_cli_audit():
    """Verify audit outputs key portfolio metrics."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        code = cmd_audit(CASES_DIR)
    assert code == 0
    out = buf.getvalue()
    assert "Total Judged Cases:          20" in out
    assert "Confirmed Fraud Verdicts:    10" in out
    assert "Cleared Benign Verdicts:     10" in out
    assert "FinCEN SARs Required:        4" in out
    assert "POLICY NEXT-BEST-ACTION DISTRIBUTION" in out


def test_cli_filter():
    """Verify filter correctly filters cases by criteria."""
    buf_sar = StringIO()
    with patch("sys.stdout", buf_sar):
        code_sar = cmd_filter(CASES_DIR, sar_only=True)
    assert code_sar == 0
    assert "Matched 4 / 20 cases:" in buf_sar.getvalue()

    buf_fraud = StringIO()
    with patch("sys.stdout", buf_fraud):
        code_fraud = cmd_filter(CASES_DIR, verdict="fraud")
    assert code_fraud == 0
    assert "Matched 10 / 20 cases:" in buf_fraud.getvalue()

    buf_legit = StringIO()
    with patch("sys.stdout", buf_legit):
        code_legit = cmd_filter(CASES_DIR, verdict="legitimate")
    assert code_legit == 0
    assert "Matched 10 / 20 cases:" in buf_legit.getvalue()


def test_cli_export_sar(tmp_path: Path):
    """Verify export-sar correctly exports in both markdown and json."""
    # Test stdout markdown
    buf_md = StringIO()
    with patch("sys.stdout", buf_md):
        code_md = cmd_export_sar(CASES_DIR, format_="markdown")
    assert code_md == 0
    assert "# FinCEN Suspicious Activity Report (SAR)" in buf_md.getvalue()
    assert "HHG-006" in buf_md.getvalue()

    # Test file export json
    out_json = tmp_path / "sars.json"
    code_json = cmd_export_sar(CASES_DIR, output_file=str(out_json), format_="json")
    assert code_json == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert len(data) == 4
    assert data[0]["case_id"] == "HHG-006"


def test_cli_main_invocation():
    """Verify main entrypoint handles arguments gracefully."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        main(["audit", "--cases", str(CASES_DIR)])
    assert "GRAPHSENTINEL BENCHMARK AUDIT" in buf.getvalue()
