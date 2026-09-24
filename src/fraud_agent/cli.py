"""Command-line entrypoint for validating and inspecting generated cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import validate_answer


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphsentinel")
    parser.add_argument("command", choices=["validate", "show"])
    parser.add_argument("--cases", default="cases")
    parser.add_argument("--case-id", default="HHG-001")
    args = parser.parse_args()

    cases_dir = Path(args.cases)
    if args.command == "show":
        print((cases_dir / f"{args.case_id}.json").read_text())
        return

    failures = 0
    for path in sorted(cases_dir.glob("HHG-*.json")):
        errors = validate_answer(json.loads(path.read_text()))
        if errors:
            failures += 1
            print(f"{path.name}: {'; '.join(errors)}")
        else:
            print(f"{path.name}: valid")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

