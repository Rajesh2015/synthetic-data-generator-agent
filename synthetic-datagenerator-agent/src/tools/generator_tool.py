import json
import random
import string
import duckdb
from datetime import datetime, timedelta
from faker import Faker
from crewai.tools import tool
from src.config import DB_PATH, NUM_RECORDS

fake = Faker()
Faker.seed(42)
random.seed(42)

# ODCS v3.1.0 type names → DuckDB types
ODCS_TO_DUCKDB = {
    # v3.1.0 types
    "string":       "VARCHAR",
    "integer":      "INTEGER",
    "long":         "BIGINT",
    "float":        "FLOAT",
    "double":       "DOUBLE",
    "number":       "DOUBLE",
    "boolean":      "BOOLEAN",
    "date":         "DATE",
    "timestamp":    "TIMESTAMP",
    "timestamp_tz": "TIMESTAMPTZ",
    # legacy aliases (v2.x / datacontract.com format) kept for backward compat
    "varchar":      "VARCHAR",
    "decimal":      "DOUBLE",
}


def _pattern_to_value(pattern: str, used: set) -> str:
    for _ in range(200):
        if pattern == r"CUST-[0-9]{6}":
            val = f"CUST-{random.randint(100000, 999999)}"
        elif pattern == r"SKU-[A-Z]{3}-[0-9]{4}":
            letters = "".join(random.choices(string.ascii_uppercase, k=3))
            val = f"SKU-{letters}-{random.randint(1000, 9999)}"
        elif pattern == r"ORD-[0-9]{8}":
            val = f"ORD-{random.randint(10000000, 99999999)}"
        else:
            val = pattern
        if val not in used:
            used.add(val)
            return val
    return val


def _by_faker_strategy(strategy: str, field_info: dict, field_name: str, uniq: dict, conn) -> any:
    """Dispatch to the correct Faker call given a strategy name."""
    if strategy == "sequence":
        return None  # handled by caller via seq counter

    if strategy == "regex":
        uniq.setdefault(field_name, set())
        return _pattern_to_value(field_info.get("pattern", ""), uniq[field_name])

    if strategy == "first_name":  return fake.first_name()
    if strategy == "last_name":   return fake.last_name()
    if strategy == "city":        return fake.city()
    if strategy == "street_address": return fake.street_address()
    if strategy == "product_name":   return fake.catch_phrase()
    if strategy == "company":        return fake.company()
    if strategy == "phone_number":
        return f"+{fake.numerify(text='##########')}"

    if strategy == "email":
        uniq.setdefault(field_name, set())
        for _ in range(200):
            val = fake.email()
            if val not in uniq[field_name]:
                uniq[field_name].add(val)
                return val
        return fake.email()

    if strategy == "enum":
        enums = field_info.get("enum") or []
        weights = field_info.get("enum_weights")
        if enums:
            return random.choices(enums, weights=weights, k=1)[0] if weights else random.choice(enums)
        return None

    if strategy == "boolean":
        return random.random() > 0.3

    if strategy == "price":
        lo = float(field_info.get("minimum") or 0.01)
        hi = float(field_info.get("maximum") or 999.99)
        return round(random.uniform(lo, hi), 2)

    if strategy == "integer":
        lo = int(field_info.get("minimum") or 1)
        hi = int(field_info.get("maximum") or 100)
        return random.randint(lo, hi)

    if strategy == "past_date":
        return (datetime.now() - timedelta(days=random.randint(30, 730))).date()

    if strategy == "past_datetime":
        return datetime.now() - timedelta(
            days=random.randint(30, 730), seconds=random.randint(0, 86400)
        )

    if strategy == "foreign_key":
        ref = field_info.get("references", "")
        if ref and conn:
            parent_table, parent_field = ref.split(".")
            try:
                rows = conn.execute(
                    f"SELECT {parent_field} FROM {parent_table} WHERE _batch_id = 1"
                ).fetchall()
                ids = [r[0] for r in rows]
                return random.choice(ids) if ids else 1
            except Exception:
                return 1

    if strategy in ("derived_from_parent", "computed"):
        return None  # post-processed

    return None


def _fallback_by_type(field_name: str, field_info: dict, uniq: dict, conn) -> any:
    """
    Type-based generation when the LLM provides no inferred_faker hint.
    Covers any standard ODCS contract without custom annotations.
    """
    ftype = field_info.get("type", "string")

    # PK → sequence
    if field_info.get("primaryKey"):
        return None  # handled by seq counter in caller

    # FK → lookup
    if field_info.get("references"):
        return _by_faker_strategy("foreign_key", field_info, field_name, uniq, conn)

    # Enum → random choice
    if field_info.get("enum"):
        return _by_faker_strategy("enum", field_info, field_name, uniq, conn)

    # Pattern → regex
    if field_info.get("pattern"):
        return _by_faker_strategy("regex", field_info, field_name, uniq, conn)

    # Format hint
    if field_info.get("format") == "email":
        return _by_faker_strategy("email", field_info, field_name, uniq, conn)

    # Type-based fallback
    if ftype == "boolean":
        return random.choice([True, False])
    if ftype == "integer":
        lo = int(field_info.get("minimum") or 1)
        hi = int(field_info.get("maximum") or 1000)
        return random.randint(lo, hi)
    if ftype in ("number", "double", "float", "decimal"):
        lo = float(field_info.get("minimum") or 0.01)
        hi = float(field_info.get("maximum") or 9999.99)
        return round(random.uniform(lo, hi), 2)
    if ftype == "date":
        return _by_faker_strategy("past_date", field_info, field_name, uniq, conn)
    if ftype == "timestamp":
        return _by_faker_strategy("past_datetime", field_info, field_name, uniq, conn)

    # string fallback — name-based heuristics
    name = field_name.lower()
    if "email" in name:
        return _by_faker_strategy("email", field_info, field_name, uniq, conn)
    if "first" in name and "name" in name:
        return fake.first_name()
    if "last" in name and "name" in name:
        return fake.last_name()
    if "name" in name:
        return fake.name()
    if "phone" in name:
        return _by_faker_strategy("phone_number", field_info, field_name, uniq, conn)
    if "address" in name:
        return fake.street_address()
    if "city" in name:
        return fake.city()
    if "country" in name:
        return fake.country_code()
    if "status" in name or "state" in name:
        return random.choice(["active", "inactive"])
    if "price" in name or "amount" in name or "cost" in name:
        return round(random.uniform(0.99, 999.99), 2)
    return fake.word()


def _get_value(field_name: str, field_info: dict, seq: dict, uniq: dict, conn) -> any:
    """
    Resolve a field value. Priority order:
      1. Sequence counter (PK fields)
      2. LLM-inferred strategy from enrichment (inferred_faker key)
      3. Type-based + name-based fallback (works on any plain ODCS contract)
    """
    # 1. PK → always a sequence
    if field_info.get("primaryKey"):
        seq[field_name] = seq.get(field_name, 0) + 1
        return seq[field_name]

    # 2. LLM-inferred strategy (set by Distribution Analyst via enrichment JSON)
    strategy = field_info.get("inferred_faker")
    if strategy:
        return _by_faker_strategy(strategy, field_info, field_name, uniq, conn)

    # 3. Type + name heuristics — works on any bare ODCS contract
    return _fallback_by_type(field_name, field_info, uniq, conn)


def _post_process(table_name: str, record: dict, conn) -> dict:
    if table_name != "orders":
        return record

    product_id = record.get("product_id")
    if product_id and conn:
        try:
            row = conn.execute(
                "SELECT unit_price FROM products WHERE product_id = ? AND _batch_id = 1",
                [product_id],
            ).fetchone()
            record["unit_price_at_order"] = row[0] if row else round(random.uniform(0.01, 999.99), 2)
        except Exception:
            record["unit_price_at_order"] = round(random.uniform(0.01, 999.99), 2)
    else:
        record["unit_price_at_order"] = round(random.uniform(0.01, 999.99), 2)

    record["total_amount"] = round(
        record.get("quantity", 1) * record.get("unit_price_at_order", 0.0), 2
    )
    return record


def _create_table(conn, table_name: str, fields: dict):
    col_defs = []
    for fname, finfo in fields.items():
        db_type = ODCS_TO_DUCKDB.get(finfo["type"], "VARCHAR")
        col_defs.append(f"{fname} {db_type}")
    col_defs.append("_batch_id INTEGER")
    col_defs.append("_snapshot_date DATE")
    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
    conn.execute(ddl)


@tool("Generate Initial Data Batch")
def generate_initial_batch(
    contract_path: str,
    num_records: int = NUM_RECORDS,
    enrichment_json: str = "{}",
) -> str:
    """
    Generate the first batch of fake data for all tables in the ODCS contract,
    respecting referential integrity and field constraints. Stores data in DuckDB.
    Tables are populated in dependency order (parents before children).
    enrichment_json: optional JSON produced by the Schema Enricher agent containing
    per-field enum_weights and inferred_faker overrides.
    """
    import yaml
    from pathlib import Path
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
        enrichment: dict = json.loads(enrichment_json)
    except (json.JSONDecodeError, TypeError):
        enrichment = {}

    conn = duckdb.connect(DB_PATH)
    seq: dict = {}
    uniq: dict = {}
    table_counts = {}
    snapshot_date = datetime.now().date()

    for table_name in contract.get_dependency_order():
        table = contract.models[table_name]
        table_enrichment = enrichment.get(table_name, {})
        fields_info = {}
        for fname, f in table.fields.items():
            field_enrichment = table_enrichment.get(fname, {})
            fields_info[fname] = {
                "type": f.type,
                "required": f.required,
                "primaryKey": f.primaryKey,
                "unique": f.unique,
                "enum": f.enum,
                "pattern": f.pattern,
                "minimum": f.minimum,
                "maximum": f.maximum,
                "references": f.references,
                # LLM-inferred Faker strategy (from Distribution Analyst enrichment)
                "inferred_faker": field_enrichment.get("inferred_faker"),
                # LLM-supplied realistic weights for enum fields
                "enum_weights": field_enrichment.get("enum_weights"),
                # format hint for fallback resolution (e.g. format: email)
                "format": f.format if hasattr(f, "format") else None,
            }

        _create_table(conn, table_name, fields_info)

        records = []
        for _ in range(num_records):
            record = {
                fname: _get_value(fname, finfo, seq, uniq, conn)
                for fname, finfo in fields_info.items()
            }
            record = _post_process(table_name, record, conn)
            record["_batch_id"] = 1
            record["_snapshot_date"] = snapshot_date
            records.append(record)

        col_names = list(fields_info.keys()) + ["_batch_id", "_snapshot_date"]
        placeholders = ", ".join(["?" for _ in col_names])
        col_list = ", ".join(col_names)
        for rec in records:
            vals = [rec.get(c) for c in col_names]
            conn.execute(f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})", vals)

        table_counts[table_name] = num_records

    conn.close()

    return json.dumps({
        "status": "success",
        "batch_id": 1,
        "snapshot_date": str(snapshot_date),
        "records_generated": table_counts,
        "db_path": DB_PATH,
    }, indent=2)
