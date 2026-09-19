import json
import random
import duckdb
from datetime import datetime, timedelta
from faker import Faker
from crewai.tools import tool
from src.config import DB_PATH, NUM_CHANGE_BATCHES, CHANGE_RATE

fake = Faker()

# Maps SCD2 field names to default regeneration strategies
_FIELD_MUTATORS = {
    "email": lambda _: fake.email(),
    "phone": lambda _: f"+{fake.numerify(text='##########')}",
    "address_line1": lambda _: fake.street_address(),
    "city": lambda _: fake.city(),
    "country_code": lambda old: random.choice(["US", "GB", "DE", "FR", "AU", "IN", "CA", "SG"]),
    "category": lambda old: random.choice(["Electronics", "Clothing", "Home", "Sports", "Beauty", "Books", "Toys", "Food"]),
    "unit_price": lambda old: round(float(old or 10.0) * random.uniform(0.85, 1.25), 2),
    "is_available": lambda old: not old,
    "budget": lambda old: round(float(old or 5000.0) * random.uniform(0.9, 1.3), 2),
    "channel": lambda old: random.choice(["Search", "Social", "Email", "Display", "Video", "Affiliate"]),
    "lead_score": lambda old: min(100, max(0, int((old or 50) + random.randint(-15, 15)))),
    "company": lambda _: fake.company(),
    "lead_source": lambda old: random.choice(["organic_search", "paid_search", "social_media", "referral", "email_campaign", "webinar", "direct"]),
    "first_name": lambda _: fake.first_name(),
    "last_name": lambda _: fake.last_name(),
}


def _mutate_field(field_name: str, old_value, enum_list: list = None):
    if enum_list:
        choices = [v for v in enum_list if v != old_value]
        if choices:
            return random.choice(choices)
        elif enum_list:
            return random.choice(enum_list)
    mutator = _FIELD_MUTATORS.get(field_name)
    if mutator:
        return mutator(old_value)
    return old_value


def build_scd2_cleansed_tables(conn, contract) -> dict:
    """
    Construct cleansed SCD2 dimension tables (<table_name>_scd2) from raw batch snapshots.
    Computes effective_date, end_date (or 9999-12-31), is_current, and version columns.
    """
    scd2_summary = {}
    for table_name, table in contract.models.items():
        natural_key = next(
            (fname for fname, f in table.fields.items() if f.unique and not f.primaryKey),
            None
        )
        if not natural_key:
            natural_key = contract.get_primary_key(table_name)
        if not natural_key:
            continue

        try:
            raw_cols = [
                r[0] for r in conn.execute(f'DESCRIBE SELECT * FROM "{table_name}"').fetchall()
                if r[0] not in ("_batch_id", "_snapshot_date")
            ]
            
            # Prevent collision if domain table already has 'end_date' or 'effective_date'
            cols_sql_list = []
            for c in raw_cols:
                if c == "end_date":
                    cols_sql_list.append(f'"{c}" AS "scheduled_end_date"')
                elif c == "effective_date":
                    cols_sql_list.append(f'"{c}" AS "source_effective_date"')
                else:
                    cols_sql_list.append(f'"{c}"')
                    
            col_sql = ", ".join(cols_sql_list)
            scd2_table_name = f"{table_name}_scd2"

            
            ddl = f"""
            CREATE OR REPLACE TABLE "{scd2_table_name}" AS
            WITH ordered_snapshots AS (
                SELECT 
                    {col_sql},
                    _snapshot_date AS effective_date,
                    LEAD(_snapshot_date) OVER (PARTITION BY "{natural_key}" ORDER BY _snapshot_date, _batch_id) AS next_effective_date,
                    ROW_NUMBER() OVER (PARTITION BY "{natural_key}" ORDER BY _snapshot_date, _batch_id) AS version,
                    ROW_NUMBER() OVER (PARTITION BY "{natural_key}" ORDER BY _snapshot_date DESC, _batch_id DESC) AS rev_rank
                FROM "{table_name}"
            )
            SELECT 
                ROW_NUMBER() OVER (ORDER BY "{natural_key}", version) AS scd_id,
                {col_sql},
                effective_date,
                CASE 
                    WHEN next_effective_date IS NOT NULL THEN GREATEST(effective_date, (next_effective_date - INTERVAL '1 day')::DATE)
                    ELSE '9999-12-31'::DATE
                END AS end_date,
                (rev_rank = 1) AS is_current,
                version
            FROM ordered_snapshots
            ORDER BY "{natural_key}", version;
            """
            conn.execute(ddl)
            count = conn.execute(f'SELECT COUNT(*) FROM "{scd2_table_name}"').fetchone()[0]
            scd2_summary[scd2_table_name] = {"rows": count, "natural_key": natural_key}
        except Exception as e:
            scd2_summary[f"{table_name}_scd2"] = {"error": str(e)}

    return scd2_summary


@tool("Simulate SCD2 Change Batches")
def simulate_changes(
    contract_path: str,
    num_batches: int = NUM_CHANGE_BATCHES,
    change_rate: float = CHANGE_RATE,
    change_patterns_json: str = "{}",
) -> str:
    """
    Generate change batches for SCD2 simulation. For each batch, selects a
    percentage of records from SCD2-tracked tables and mutates their tracked fields.
    New rows are inserted with an incremented batch_id, simulating how a source
    system emits changed records over time (Type-2 versioning) and in-place updates (Type-1).
    Generates cleansed <table_name>_scd2 dimensional tables with effective_date,
    end_date, and is_current fields for testing downstream ETL pipelines.
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

    try:
        enrichment: dict = json.loads(change_patterns_json)
    except (json.JSONDecodeError, TypeError):
        enrichment = {}

    change_tracking_map = enrichment.get("change_tracking", {})
    per_table_patterns = enrichment.get("change_patterns", enrichment)

    conn = duckdb.connect(DB_PATH)
    change_summary = {}
    skipped = {}

    for batch_num in range(2, num_batches + 2):
        batch_changes = {}

        for table_name, table_schema in contract.models.items():
            pk = contract.get_primary_key(table_name)
            
            # Fetch baseline snapshot date from batch 1
            try:
                base_date_row = conn.execute(
                    f'SELECT MIN(_snapshot_date) FROM "{table_name}"'
                ).fetchone()
                base_date = base_date_row[0] if base_date_row and base_date_row[0] else (datetime.now().date() - timedelta(days=180))
                if isinstance(base_date, str):
                    base_date = datetime.strptime(base_date, "%Y-%m-%d").date()
            except Exception:
                base_date = datetime.now().date() - timedelta(days=180)

            snapshot_date = base_date + timedelta(days=(batch_num - 1) * 30)

            tracked_fields = change_tracking_map.get(table_name, [])
            if not tracked_fields:
                tracked_fields = contract.get_change_tracked_fields(table_name)
            if not tracked_fields:
                tracked_fields = [
                    fname for fname, f in table_schema.fields.items()
                    if not f.primaryKey and not f.unique
                    and f.type not in ("timestamp", "date")
                    and "created" not in fname.lower()
                ]

            if not tracked_fields:
                skipped[table_name] = "no changeable fields identified"
                continue

            try:
                prev_batch = batch_num - 1
                all_rows = conn.execute(
                    f'SELECT * FROM "{table_name}" WHERE _batch_id = ?', [prev_batch]
                ).fetchdf()
                if all_rows.empty:
                    all_rows = conn.execute(
                        f'SELECT * FROM "{table_name}" WHERE _batch_id = 1'
                    ).fetchdf()
            except Exception as e:
                skipped[table_name] = f"query failed: {type(e).__name__}: {e}"
                continue

            if all_rows.empty:
                skipped[table_name] = "no baseline rows to mutate"
                continue

            if pk and pk in all_rows.columns:
                all_rows = all_rows.drop_duplicates(subset=[pk])

            n_to_change = max(1, int(len(all_rows) * change_rate))
            changed_rows = all_rows.sample(n=min(n_to_change, len(all_rows)), random_state=batch_num)
            changed_rows = changed_rows.copy()

            table_freq = per_table_patterns.get(table_name, {}).get("field_change_frequency", {})
            co_patterns = per_table_patterns.get(table_name, {}).get("co_change_patterns", [])

            def _pick_fields_for_record():
                if co_patterns:
                    patterns  = [p["fields"]     for p in co_patterns]
                    weights   = [p["frequency"]  for p in co_patterns]
                    if patterns and len(patterns) == len(weights):
                        try:
                            return random.choices(patterns, weights=weights, k=1)[0]
                        except Exception:
                            return random.choice(patterns)
                    elif patterns:
                        return random.choice(patterns)
                elif table_freq:
                    available = [f for f in tracked_fields if f in changed_rows.columns]
                    weights   = [table_freq.get(f, 0.1) for f in available]
                    if available and len(available) == len(weights):
                        n_fields  = random.randint(1, min(3, len(available)))
                        try:
                            return list(set(random.choices(available, weights=weights, k=n_fields)))
                        except Exception:
                            return [random.choice(available)]
                    elif available:
                        return [random.choice(available)]
                return tracked_fields


            # Apply per-row field selection
            for idx in changed_rows.index:
                fields_to_change = _pick_fields_for_record()
                for field in fields_to_change:
                    if field in changed_rows.columns:
                        field_def = table_schema.fields.get(field)
                        enum_list = field_def.enum if field_def else None
                        changed_rows.at[idx, field] = _mutate_field(
                            field, changed_rows.at[idx, field], enum_list=enum_list
                        )

            changed_rows["_batch_id"] = batch_num
            changed_rows["_snapshot_date"] = snapshot_date

            col_names = list(changed_rows.columns)
            col_list = ", ".join(f'"{c}"' for c in col_names)
            placeholders = ", ".join(["?" for _ in col_names])

            for _, row in changed_rows.iterrows():
                vals = [row[c] for c in col_names]
                conn.execute(
                    f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})', vals
                )

            batch_changes[table_name] = {
                "records_changed": len(changed_rows),
                "fields_mutated": tracked_fields,
            }

        change_summary[f"batch_{batch_num}"] = {
            "snapshot_date": str(snapshot_date),
            "changes": batch_changes,
        }

    # Build cleansed SCD2 dimensional tables (<table_name>_scd2)
    scd2_tables = build_scd2_cleansed_tables(conn, contract)

    conn.close()

    return json.dumps(
        {
            "status": "success",
            "batches_generated": num_batches,
            "change_rate": change_rate,
            "summary": change_summary,
            "scd2_tables_created": scd2_tables,
            "skipped_tables": skipped,
            "db_path": DB_PATH,
        },
        indent=2,
    )
