from fraud_agent.policy import approval_route, case_required, report_required


def test_approval_routes():
    assert approval_route("ALLOW_TRANSACTION") == "auto"
    assert approval_route("DECLINE_TRANSACTION") == "L1"
    assert approval_route("BLOCK_CARD", 2500) == "L1"
    assert approval_route("BLOCK_CARD", 2500.01) == "L2"
    assert approval_route("FILE_REPORT") == "L2"


def test_case_and_report_gates():
    assert case_required(0.3, False, False)
    assert case_required(0.1, True, False)
    assert not case_required(0.1, False, False)
    assert report_required(
        fraud_probability=0.86,
        exposure_usd=200,
        shared_origin=True,
        coordinated=False,
        undocumented=False,
    )
    assert not report_required(
        fraud_probability=0.60,
        exposure_usd=5000,
        shared_origin=True,
        coordinated=True,
        undocumented=True,
    )

