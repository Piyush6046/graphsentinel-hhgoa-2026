"""Create compact analysis tables from the full benchmark CSV files.

The original 393 transaction features remain untouched in ``data/``. This script
selects the explainable columns the investigation agent uses and materializes a
device-profile key for fast graph-neighborhood analysis.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

BASE_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    *[f"M{i}" for i in range(1, 10)],
    "customer_id", "ts", "channel", "risk_score",
]

IDENTITY_COLUMNS = [
    "TransactionID", "id_01", "id_02", "id_05", "id_06", "id_09", "id_10",
    "id_11", "id_12", "id_15", "id_16", "id_23", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def clean_part(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def main() -> None:
    print("Reading explainable transaction columns…")
    tx = pd.read_csv(DATA / "transactions.csv", usecols=BASE_COLUMNS, low_memory=False)
    tx["TransactionID"] = tx["TransactionID"].astype(str)
    tx["ts"] = pd.to_datetime(tx["ts"])
    tx.to_parquet(DATA / "transactions_compact.parquet", index=False)

    print("Reading identity signals and constructing device profiles…")
    identity = pd.read_csv(DATA / "identity.csv", usecols=IDENTITY_COLUMNS, low_memory=False)
    identity["TransactionID"] = identity["TransactionID"].astype(str)
    profile_cols = ["DeviceInfo", "id_30", "id_31", "id_33"]
    identity["device_profile"] = identity[profile_cols].apply(
        lambda row: " | ".join(clean_part(v) for v in row), axis=1
    )
    identity.to_parquet(DATA / "identity_compact.parquet", index=False)

    # Materialize a TigerGraph-ready table. The source deliberately omits card
    # IDs, so benchmark/closed-case mappings are used where known and a stable
    # K1 identifier is used only for non-benchmark background customers.
    pack = pd.read_csv(DATA / "case_pack.csv", usecols=["customer_id", "card_id"])
    closed = pd.read_csv(DATA / "closed_cases_history.csv", usecols=["customer_id", "card_id"])
    cards = pd.concat([pack, closed], ignore_index=True).drop_duplicates("customer_id")
    card_map = dict(zip(cards.customer_id, cards.card_id))
    graph = tx[["TransactionID", "customer_id", "ts", "TransactionAmt", "ProductCD",
                "channel", "risk_score", "addr1", "P_emaildomain", "card4", "card6"]].copy()
    graph["card_id"] = graph.customer_id.map(card_map).fillna(graph.customer_id + "-K1")
    ident_graph = identity[["TransactionID", "device_profile", "DeviceInfo", "DeviceType",
                            "id_30", "id_31", "id_33", "id_15", "id_23"]].copy()
    ident_graph["device_id"] = ident_graph.device_profile.map(
        lambda value: "D-" + hashlib.sha1(value.encode()).hexdigest()[:16]
    )
    graph = graph.merge(ident_graph, on="TransactionID", how="left")
    graph.to_csv(DATA / "graph_transactions.csv", index=False)
    print(f"Prepared {len(tx):,} transactions and {len(identity):,} identity records.")


if __name__ == "__main__":
    main()
