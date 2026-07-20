import json
import duckdb
from crewai.tools import tool
from src.config import DB_PATH


@tool("Validate Generated Data")
def validate_data(contract_path: str) -> str:
    """
    Validate the data in DuckDB against the ODCS contract quality rules.
    Checks: PK uniqueness per batch, FK referential integrity, NOT NULL on
    required fields, enum constraint adherence, and numeric range bounds.
    Returns a structured validation report with pass/fail per rule.
    """
    import yaml
    from src.models.contract_schema import ParsedContract, TableSchema, ContractInfo

    with open(contract_path) as f:
        raw = yaml.safe_load(f)

    contract = ParsedContract(
        id=raw.get("id", "unknown"),
        info=ContractInfo(**raw.get("info", {"title": "Unknown", "version": "1.0"})),
        models={
            tname: TableSchema(**tdata)
            for tname, tdata in raw.get("models", {}).items()
        },
    )

    conn = duckdb.connect(DB_PATH)
    report = {"contract": contract.id, "tables": {}, "summary": {}}
    total_checks = 0
    total_passed = 0

    for table_name, table in contract.models.items():
        table_report = {"checks": [], "passed": 0, "failed": 0}

        try:
            total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            batch_1_rows = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE _batch_id = 1"
            ).fetchone()[0]
        except Exception as e:
            table_report["error"] = str(e)
            report["tables"][table_name] = table_report
            continue

        # --- PK uniqueness within each batch ---
        pk = contract.get_primary_key(table_name)
        if pk:
            dupes = conn.execute(
                f"SELECT _batch_id, COUNT(*) as cnt FROM {table_name} "
                f"GROUP BY _batch_id, {pk} HAVING cnt > 1"
            ).fetchall()
            result = "PASS" if not dupes else "FAIL"
            table_report["checks"].append({
                "rule": f"PK uniqueness ({pk})",
                "result": result,
                "detail": f"{len(dupes)} duplicate PK(s) found" if dupes else "All PKs unique per batch",
            })
            if result == "PASS":
                table_report["passed"] += 1
            else:
                table_report["failed"] += 1

        # --- NOT NULL on required fields ---
        for fname, field in table.fields.items():
            if field.required:
                nulls = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {fname} IS NULL"
                ).fetchone()[0]
                result = "PASS" if nulls == 0 else "FAIL"
                table_report["checks"].append({
                    "rule": f"NOT NULL ({fname})",
                    "result": result,
                    "detail": f"{nulls} NULL value(s)" if nulls else "No NULLs",
                })
                if result == "PASS":
                    table_report["passed"] += 1
                else:
                    table_report["failed"] += 1

        # --- Enum constraints ---
        for fname, field in table.fields.items():
            if field.enum:
                enum_list = ", ".join(f"'{v}'" for v in field.enum)
                violations = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE {fname} NOT IN ({enum_list})"
                ).fetchone()[0]
                result = "PASS" if violations == 0 else "FAIL"
                table_report["checks"].append({
                    "rule": f"Enum constraint ({fname})",
                    "result": result,
                    "detail": f"{violations} out-of-enum value(s)" if violations else "All values in enum",
                })
                if result == "PASS":
                    table_report["passed"] += 1
                else:
                    table_report["failed"] += 1

        # --- FK referential integrity (batch 1 only) ---
        for fname, ref in contract.get_foreign_keys(table_name).items():
            parent_table, parent_field = ref.split(".")
            orphans = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} t "
                f"WHERE t._batch_id = 1 AND t.{fname} NOT IN "
                f"(SELECT {parent_field} FROM {parent_table} WHERE _batch_id = 1)"
            ).fetchone()[0]
            result = "PASS" if orphans == 0 else "FAIL"
            table_report["checks"].append({
                "rule": f"FK integrity ({fname} → {ref})",
                "result": result,
                "detail": f"{orphans} orphaned FK(s)" if orphans else "All FKs resolve",
            })
            if result == "PASS":
                table_report["passed"] += 1
            else:
                table_report["failed"] += 1

        # --- Numeric range checks ---
        for fname, field in table.fields.items():
            if field.minimum is not None:
                violations = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {fname} < {field.minimum}"
                ).fetchone()[0]
                result = "PASS" if violations == 0 else "FAIL"
                table_report["checks"].append({
                    "rule": f"Min range ({fname} >= {field.minimum})",
                    "result": result,
                    "detail": f"{violations} value(s) below minimum" if violations else "All values above minimum",
                })
                if result == "PASS":
                    table_report["passed"] += 1
                else:
                    table_report["failed"] += 1

        table_report["total_rows"] = total_rows
        table_report["batch_1_rows"] = batch_1_rows
        total_checks += table_report["passed"] + table_report["failed"]
        total_passed += table_report["passed"]
        report["tables"][table_name] = table_report

    conn.close()

    report["summary"] = {
        "total_checks": total_checks,
        "passed": total_passed,
        "failed": total_checks - total_passed,
        "pass_rate": f"{(total_passed / total_checks * 100):.1f}%" if total_checks else "N/A",
    }

    return json.dumps(report, indent=2)
