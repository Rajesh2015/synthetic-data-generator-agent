import json
import duckdb
from pathlib import Path
from crewai.tools import tool


@tool("Profile Source Data Distributions")
def profile_source_data(source_db_path: str, contract_path: str) -> str:
    """
    Connect to an existing DuckDB containing anonymized reference/warehouse data.
    For each table in the ODCS contract, compute:
      - Total row count and year-over-year record growth
      - Enum fields: value distribution as percentages
      - Numeric fields: min, max, avg, p25, p50, p75
      - All fields: NULL rate
    Returns a JSON stats object the LLM uses to derive realistic generation weights.
    Falls back gracefully when the reference database does not exist.
    """
    import yaml
    from src.models.contract_schema import ParsedContract, TableSchema, ContractInfo

    if not Path(source_db_path).exists():
        return json.dumps({
            "status": "no_reference_data",
            "message": (
                "Reference database not found. "
                "Run scripts/seed_reference_data.py to create sample data, "
                "or point SOURCE_DB_PATH at your anonymized warehouse export."
            ),
        })

    with open(contract_path) as f:
        raw = yaml.safe_load(f)
    contract = ParsedContract(
        id=raw.get("id", "unknown"),
        info=ContractInfo(**raw.get("info", {"title": "Unknown", "version": "1.0"})),
        models={t: TableSchema(**d) for t, d in raw.get("models", {}).items()},
    )

    conn = duckdb.connect(source_db_path, read_only=True)
    stats = {}

    for table_name, table in contract.models.items():
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception:
            stats[table_name] = {"error": "table not found in reference DB"}
            continue

        table_stats: dict = {"row_count": row_count, "fields": {}}

        # Year-over-year growth (if created_at exists)
        if "created_at" in table.fields:
            try:
                rows = conn.execute(
                    f"SELECT YEAR(created_at) AS yr, COUNT(*) AS cnt "
                    f"FROM {table_name} WHERE created_at IS NOT NULL "
                    f"GROUP BY yr ORDER BY yr"
                ).fetchall()
                table_stats["year_distribution"] = {str(r[0]): r[1] for r in rows}
            except Exception:
                pass

        for fname, field in table.fields.items():
            fstats: dict = {}

            # NULL rate
            try:
                nulls = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {fname} IS NULL"
                ).fetchone()[0]
                fstats["null_rate"] = round(nulls / row_count, 4) if row_count else 0.0
            except Exception:
                fstats["null_rate"] = None

            # Enum distribution
            if field.enum:
                try:
                    rows = conn.execute(
                        f"SELECT {fname}, COUNT(*) AS cnt FROM {table_name} "
                        f"WHERE {fname} IS NOT NULL GROUP BY {fname} ORDER BY cnt DESC"
                    ).fetchall()
                    total = sum(r[1] for r in rows)
                    fstats["type"] = "enum"
                    fstats["distribution"] = {
                        str(r[0]): round(r[1] / total, 4) for r in rows
                    }
                except Exception:
                    pass

            # Numeric stats — covers both ODCS v3.1.0 and legacy type names
            elif field.type in ("integer", "long", "float", "double", "number", "decimal"):
                try:
                    row = conn.execute(
                        f"SELECT MIN({fname}), MAX({fname}), AVG({fname}), "
                        f"PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {fname}), "
                        f"PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {fname}), "
                        f"PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {fname}) "
                        f"FROM {table_name} WHERE {fname} IS NOT NULL"
                    ).fetchone()
                    fstats["type"] = "numeric"
                    fstats["min"]  = round(float(row[0]), 2) if row[0] is not None else None
                    fstats["max"]  = round(float(row[1]), 2) if row[1] is not None else None
                    fstats["avg"]  = round(float(row[2]), 2) if row[2] is not None else None
                    fstats["p25"]  = round(float(row[3]), 2) if row[3] is not None else None
                    fstats["p50"]  = round(float(row[4]), 2) if row[4] is not None else None
                    fstats["p75"]  = round(float(row[5]), 2) if row[5] is not None else None
                except Exception:
                    pass

            table_stats["fields"][fname] = fstats

        stats[table_name] = table_stats

    conn.close()
    return json.dumps({"status": "ok", "stats": stats}, indent=2)
