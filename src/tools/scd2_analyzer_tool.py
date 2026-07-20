import json
import duckdb
from pathlib import Path
from crewai.tools import tool


@tool("Analyze SCD2 Change Patterns")
def analyze_scd2_patterns(scd2_db_path: str, contract_path: str) -> str:
    """
    Connect to an existing DuckDB SCD2 cleansed layer
    (tables must have effective_date DATE, end_date DATE, is_current BOOLEAN columns).
    For each table that has SCD2-tracked fields in the contract, compute:
      - Total records and unique natural keys
      - Average and max number of versions per natural key
      - Per-field change frequency: what proportion of version transitions
        involved a change to that specific field
      - Top co-change patterns: which fields tend to change together
    Returns a JSON pattern summary the LLM uses to build realistic change scenarios.
    Falls back gracefully if the SCD2 database does not exist.
    """
    import yaml
    from src.models.contract_schema import ParsedContract, TableSchema, ContractInfo

    if not Path(scd2_db_path).exists():
        return json.dumps({
            "status": "no_scd2_data",
            "message": (
                "SCD2 reference database not found. "
                "Run scripts/seed_reference_data.py to create sample data, "
                "or point SCD2_DB_PATH at your existing cleansed layer export."
            ),
        })

    with open(contract_path) as f:
        raw = yaml.safe_load(f)
    contract = ParsedContract(
        id=raw.get("id", "unknown"),
        info=ContractInfo(**raw.get("info", {"title": "Unknown", "version": "1.0"})),
        models={t: TableSchema(**d) for t, d in raw.get("models", {}).items()},
    )

    conn = duckdb.connect(scd2_db_path, read_only=True)
    patterns = {}

    for table_name in contract.models:
        # Without x-change-tracking annotations, treat all non-PK, non-unique,
        # non-timestamp fields as candidates for change analysis.
        table_fields = contract.models[table_name].fields
        tracked_fields = [
            fname for fname, f in table_fields.items()
            if not f.primaryKey and not f.unique
            and f.type not in ("timestamp", "date")
            and "created" not in fname.lower()
        ]
        if not tracked_fields:
            continue

        # Natural key: unique non-PK field (e.g. customer_code, product_code)
        natural_key = next(
            (fname for fname, f in table_fields.items()
             if f.unique and not f.primaryKey),
            None,
        )
        if not natural_key:
            continue

        try:
            conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        except Exception:
            patterns[table_name] = {"error": "table not found in SCD2 DB"}
            continue

        table_patterns: dict = {}

        # Version counts per natural key
        try:
            r = conn.execute(
                f"SELECT AVG(cnt), MAX(cnt), COUNT(*) FROM "
                f"(SELECT {natural_key}, COUNT(*) AS cnt FROM {table_name} GROUP BY {natural_key})"
            ).fetchone()
            table_patterns["unique_keys"]           = int(r[2]) if r[2] else 0
            table_patterns["avg_versions_per_key"]  = round(float(r[0]), 2) if r[0] else 1.0
            table_patterns["max_versions"]          = int(r[1]) if r[1] else 1
        except Exception:
            table_patterns["avg_versions_per_key"] = 1.0

        # Per-field change frequency using window function
        field_freq: dict = {}
        for field in tracked_fields:
            try:
                r = conn.execute(f"""
                    WITH consecutive AS (
                        SELECT
                            {natural_key},
                            {field},
                            LAG({field}) OVER (
                                PARTITION BY {natural_key} ORDER BY effective_date
                            ) AS prev_val
                        FROM {table_name}
                    )
                    SELECT
                        COUNT(CASE WHEN {field} IS DISTINCT FROM prev_val
                                    AND prev_val IS NOT NULL THEN 1 END) AS changes,
                        COUNT(CASE WHEN prev_val IS NOT NULL THEN 1 END)  AS transitions
                    FROM consecutive
                """).fetchone()
                transitions = r[1] or 0
                field_freq[field] = round(r[0] / transitions, 4) if transitions > 0 else 0.0
            except Exception:
                field_freq[field] = 0.0

        table_patterns["field_change_frequency"] = field_freq

        # Co-change pattern fingerprints (which fields change together)
        if len(tracked_fields) > 1:
            try:
                lag_exprs = ", ".join(
                    f"LAG({f}) OVER (PARTITION BY {natural_key} ORDER BY effective_date) AS prev_{f}"
                    for f in tracked_fields
                )
                changed_exprs = " || ',' || ".join(
                    f"CASE WHEN {f} IS DISTINCT FROM prev_{f} THEN '{f}' ELSE '' END"
                    for f in tracked_fields
                )
                rows = conn.execute(f"""
                    WITH consecutive AS (
                        SELECT {natural_key}, {', '.join(tracked_fields)}, {lag_exprs}
                        FROM {table_name}
                    ),
                    fingerprints AS (
                        SELECT {changed_exprs} AS pattern
                        FROM consecutive
                        WHERE prev_{tracked_fields[0]} IS NOT NULL
                    )
                    SELECT pattern, COUNT(*) AS cnt
                    FROM fingerprints
                    WHERE pattern != '{','.join([''] * len(tracked_fields))}'
                    GROUP BY pattern
                    ORDER BY cnt DESC
                    LIMIT 10
                """).fetchall()
                total = sum(r[1] for r in rows)
                table_patterns["co_change_patterns"] = [
                    {
                        "fields": [f for f in r[0].split(",") if f],
                        "frequency": round(r[1] / total, 4),
                        "count": r[1],
                    }
                    for r in rows
                ] if total > 0 else []
            except Exception:
                pass

        patterns[table_name] = table_patterns

    conn.close()
    return json.dumps({"status": "ok", "patterns": patterns}, indent=2)
