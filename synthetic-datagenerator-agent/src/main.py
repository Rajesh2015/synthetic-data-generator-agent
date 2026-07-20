import json
from dotenv import load_dotenv
from src.crew import build_crew

load_dotenv()


def main():
    print("\n" + "=" * 60)
    print(" Data Contract → Fake Data Generator (CrewAI)")
    print("=" * 60 + "\n")

    crew = build_crew()
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print(" VALIDATION REPORT")
    print("=" * 60)

    try:
        report = json.loads(str(result))
        summary = report.get("summary", {})
        print(f"  Total checks : {summary.get('total_checks')}")
        print(f"  Passed       : {summary.get('passed')}")
        print(f"  Failed       : {summary.get('failed')}")
        print(f"  Pass rate    : {summary.get('pass_rate')}")
        print()

        for table, tdata in report.get("tables", {}).items():
            print(f"  [{table}]  {tdata.get('total_rows')} rows total")
            for check in tdata.get("checks", []):
                icon = "✓" if check["result"] == "PASS" else "✗"
                print(f"    {icon} {check['rule']} — {check['detail']}")
    except (json.JSONDecodeError, TypeError):
        print(result)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
