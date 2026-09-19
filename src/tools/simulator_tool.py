import json
import random
import duckdb
from datetime import datetime, timedelta
from faker import Faker
from crewai.tools import tool
from src.config import DB_PATH, NUM_CHANGE_BATCHES, CHANGE_RATE

fake = Faker()

# Maps SCD2 field names to a regeneration strategy
_FIELD_MUTATORS = {
    "email": lambda _: fake.email(),
    "phone": lambda _: f"+{fake.numerify(text='##########')}",
    "address_line1": lambda _: fake.street_address(),
    "city": lambda _: fake.city(),
    "country_code": lambda old: random.choice(["US", "GB", "DE", "FR", "AU", "IN", "CA", "SG"]),
    "status": lambda old: random.choice(["active", "inactive", "suspended", "contacted", "qualified"]),
    "category": lambda old: random.choice(["Electronics", "Clothing", "Home", "Sports", "Beauty", "Books", "Toys", "Food"]),
    "unit_price": lambda old: round(float(old) * random.uniform(0.85, 1.25), 2),
    "is_available": lambda old: not old,
    "budget": lambda old: round(float(old or 5000.0) * random.uniform(0.9, 1.3), 2),
    "channel": lambda old: random.choice(["Search", "Social", "Email", "Display", "Video", "Affiliate"]),
    "lead_score": lambda old: min(100, max(0, int((old or 50) + random.randint(-15, 15)))),
    "company": lambda _: fake.company(),
    "lead_source": lambda old: random.choice(["organic_search", "paid_search", "social_media", "referral", "email_campaign", "webinar", "direct"]),
}



def _mutate_field(field_name: str, old_value):
    mutator = _FIELD_MUTATORS.get(field_name)
    if mutator:
        return mutator(old_value)
    return old_value


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
    system emits changed records over time.
    Only tables with x-change-tracking fields receive change batches.
    Orders table is append-only and is skipped.
    change_patterns_json: optional JSON from the Distribution Analyst containing
    per-table field_change_frequency weights derived from real SCD2 history.
    When provided, field selection per record mirrors real production change patterns
    (e.g. address_line1 + city change together 22% of the time).
    """
    import yaml
    from src.models.contract_schema import ParsedContract, TableSchema, ContractInfo

    with open(contract_path) as f:
        import yaml as _yaml
        raw = _yaml.safe_load(f)

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

    # Accept either the full enrichment object (with 'change_tracking' and
    # 'change_patterns' keys) or a pre-extracted change_patterns section.
    change_tracking_map = enrichment.get("change_tracking", {})
    # Per-table field frequencies / co-change patterns live under 'change_patterns'
    # in the full enrichment object; fall back to the top level for a bare section.
    per_table_patterns = enrichment.get("change_patterns", enrichment)

    conn = duckdb.connect(DB_PATH)
    change_summary = {}
    skipped = {}

    for batch_num in range(2, num_batches + 2):
        snapshot_date = datetime.now().date() + timedelta(days=batch_num - 1)
        batch_changes = {}

        for table_name in contract.models:
            # SCD2-tracked fields come from the LLM enrichment (change_tracking key).
            # No longer read from x-change-tracking annotations in the contract.
            tracked_fields = change_tracking_map.get(table_name, [])
            if not tracked_fields:
                skipped[table_name] = "no SCD2-tracked fields in change_tracking"
                continue

            try:
                all_rows = conn.execute(
                    f"SELECT * FROM {table_name} WHERE _batch_id = 1"
                ).fetchdf()
            except Exception as e:
                # Surface the reason instead of silently skipping — a missing
                # pandas/fetchdf dependency used to hide here and yield zero batches.
                skipped[table_name] = f"query failed: {type(e).__name__}: {e}"
                continue

            if all_rows.empty:
                skipped[table_name] = "no batch-1 rows to mutate"
                continue

            n_to_change = max(1, int(len(all_rows) * change_rate))
            changed_rows = all_rows.sample(n=n_to_change, random_state=batch_num)
            changed_rows = changed_rows.copy()

            # Use LLM-derived field frequencies if available; else mutate all tracked fields.
            table_freq = per_table_patterns.get(table_name, {}).get("field_change_frequency", {})
            co_patterns = per_table_patterns.get(table_name, {}).get("co_change_patterns", [])

            def _pick_fields_for_record():
                """Select which fields to mutate for one record."""
                if co_patterns:
                    # Use co-change patterns from real SCD2 history
                    patterns  = [p["fields"]     for p in co_patterns]
                    weights   = [p["frequency"]  for p in co_patterns]
                    return random.choices(patterns, weights=weights, k=1)[0]
                elif table_freq:
                    # Use per-field frequencies as independent weights
                    available = [f for f in tracked_fields if f in changed_rows.columns]
                    weights   = [table_freq.get(f, 0.1) for f in available]
                    n_fields  = random.randint(1, min(3, len(available)))
                    return list(set(random.choices(available, weights=weights, k=n_fields)))
                else:
                    return tracked_fields

            # Apply per-row field selection
            for idx in changed_rows.index:
                fields_to_change = _pick_fields_for_record()
                for field in fields_to_change:
                    if field in changed_rows.columns:
                        changed_rows.at[idx, field] = _mutate_field(field, changed_rows.at[idx, field])

            changed_rows["_batch_id"] = batch_num
            changed_rows["_snapshot_date"] = snapshot_date

            col_names = list(changed_rows.columns)
            col_list = ", ".join(col_names)
            placeholders = ", ".join(["?" for _ in col_names])

            for _, row in changed_rows.iterrows():
                vals = [row[c] for c in col_names]
                conn.execute(
                    f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})", vals
                )

            batch_changes[table_name] = {
                "records_changed": n_to_change,
                "fields_mutated": tracked_fields,
            }

        change_summary[f"batch_{batch_num}"] = {
            "snapshot_date": str(snapshot_date),
            "changes": batch_changes,
        }

    conn.close()

    return json.dumps(
        {
            "status": "success",
            "batches_generated": num_batches,
            "change_rate": change_rate,
            "summary": change_summary,
            "skipped_tables": skipped,
            "db_path": DB_PATH,
        },
        indent=2,
    )
