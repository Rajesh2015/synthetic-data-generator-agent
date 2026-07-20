"""
One-time script to create two reference DuckDB files:

  data/reference.duckdb   — anonymized source-layer snapshot (~500 rows per table)
                            Reflects realistic enum distributions and numeric ranges.

  data/cleansed_scd2.duckdb — SCD2 cleansed layer for customers and products
                               Each natural key has 1–3 versions with realistic
                               field-change patterns.

Run once before the main pipeline:
  python -m scripts.seed_reference_data
"""

import random
import string
import duckdb
from datetime import date, timedelta
from faker import Faker
from pathlib import Path

fake = Faker()
Faker.seed(0)
random.seed(0)

ROOT = Path(__file__).parent.parent
SOURCE_DB  = str(ROOT / "data" / "reference.duckdb")
SCD2_DB    = str(ROOT / "data" / "cleansed_scd2.duckdb")

# ── realistic distributions derived from e-commerce industry benchmarks ──────
CUSTOMER_STATUS_DIST  = [("active", 0.72), ("inactive", 0.21), ("suspended", 0.07)]
COUNTRY_DIST          = [("US", 0.45), ("GB", 0.15), ("DE", 0.12), ("FR", 0.10),
                          ("AU", 0.08), ("IN", 0.05), ("CA", 0.03), ("SG", 0.02)]
CATEGORY_DIST         = [("Electronics", 0.28), ("Clothing", 0.25), ("Home", 0.16),
                          ("Sports", 0.10), ("Beauty", 0.08), ("Books", 0.07),
                          ("Toys", 0.04), ("Food", 0.02)]
ORDER_STATUS_DIST     = [("delivered", 0.55), ("shipped", 0.18), ("confirmed", 0.12),
                          ("pending", 0.08), ("cancelled", 0.05), ("returned", 0.02)]

# SCD2 change-pattern weights: which field(s) change together per transition
# (mirrors real e-commerce change events)
CUSTOMER_CHANGE_PATTERNS = [
    (["email"],                              0.28),  # email-only update
    (["address_line1", "city"],              0.22),  # address move
    (["phone"],                              0.15),  # phone-only
    (["status"],                             0.14),  # status flip
    (["email", "phone"],                     0.10),  # full contact update
    (["address_line1", "city", "country_code"], 0.07), # international move
    (["country_code"],                       0.04),
]
PRODUCT_CHANGE_PATTERNS = [
    (["unit_price"],                         0.55),  # repricing
    (["is_available"],                       0.20),  # stock toggle
    (["category", "unit_price"],             0.15),  # reclassify + reprice
    (["unit_price", "is_available"],         0.10),
]


def _weighted_choice(pairs):
    choices, weights = zip(*pairs)
    return random.choices(choices, weights=weights, k=1)[0]


def _pattern_choice(patterns):
    fields_list, weights = zip(*patterns)
    return list(random.choices(fields_list, weights=weights, k=1)[0])


def _code(prefix, n):
    return f"{prefix}-{random.randint(10**(n-1), 10**n - 1)}"


def _sku():
    return f"SKU-{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(1000,9999)}"


def _past_date(start_year=2020):
    start = date(start_year, 1, 1)
    span  = (date.today() - start).days
    return start + timedelta(days=random.randint(0, span))


# ── Source layer (reference.duckdb) ──────────────────────────────────────────

def seed_source(n_customers=500, n_products=200, n_orders=2000):
    conn = duckdb.connect(SOURCE_DB)

    conn.execute("""
        CREATE OR REPLACE TABLE customers (
            customer_id   INTEGER PRIMARY KEY,
            customer_code VARCHAR UNIQUE,
            first_name    VARCHAR, last_name VARCHAR,
            email         VARCHAR UNIQUE,
            phone         VARCHAR,
            address_line1 VARCHAR, city VARCHAR, country_code VARCHAR,
            status        VARCHAR,
            created_at    TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            product_code VARCHAR UNIQUE,
            product_name VARCHAR,
            category     VARCHAR,
            unit_price   DOUBLE,
            currency     VARCHAR,
            is_available BOOLEAN,
            created_at   TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE orders (
            order_id             INTEGER PRIMARY KEY,
            order_code           VARCHAR UNIQUE,
            customer_id          INTEGER,
            product_id           INTEGER,
            quantity             INTEGER,
            unit_price_at_order  DOUBLE,
            total_amount         DOUBLE,
            order_status         VARCHAR,
            order_date           DATE,
            created_at           TIMESTAMP
        )
    """)

    # customers
    used_emails = set()
    for i in range(1, n_customers + 1):
        email = fake.email()
        while email in used_emails:
            email = fake.email()
        used_emails.add(email)
        conn.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                i, _code("CUST", 6),
                fake.first_name(), fake.last_name(),
                email,
                f"+{fake.numerify('##########')}",
                fake.street_address(), fake.city(),
                _weighted_choice(COUNTRY_DIST),
                _weighted_choice(CUSTOMER_STATUS_DIST),
                fake.date_time_between(start_date="-4y", end_date="now"),
            ],
        )

    # products
    for i in range(1, n_products + 1):
        cat = _weighted_choice(CATEGORY_DIST)
        # realistic price ranges per category
        price_range = {
            "Electronics": (29.99, 2999.99), "Clothing": (9.99, 299.99),
            "Home": (14.99, 599.99),         "Sports": (9.99, 499.99),
            "Beauty": (4.99, 149.99),        "Books": (4.99, 59.99),
            "Toys": (4.99, 199.99),          "Food": (1.99, 49.99),
        }.get(cat, (9.99, 499.99))
        price = round(random.uniform(*price_range), 2)
        conn.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
            [
                i, _sku(), fake.catch_phrase(), cat, price, "USD",
                random.random() > 0.15,   # 85% available
                fake.date_time_between(start_date="-4y", end_date="now"),
            ],
        )

    # orders
    customer_ids = [r[0] for r in conn.execute("SELECT customer_id FROM customers").fetchall()]
    product_rows = {
        r[0]: r[1]
        for r in conn.execute("SELECT product_id, unit_price FROM products").fetchall()
    }
    for i in range(1, n_orders + 1):
        cid = random.choice(customer_ids)
        pid = random.choice(list(product_rows.keys()))
        qty = random.randint(1, 5)
        price = product_rows[pid]
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                i, _code("ORD", 8),
                cid, pid, qty, price,
                round(qty * price, 2),
                _weighted_choice(ORDER_STATUS_DIST),
                _past_date(),
                fake.date_time_between(start_date="-4y", end_date="now"),
            ],
        )

    conn.close()
    print(f"  reference.duckdb  — {n_customers} customers, {n_products} products, {n_orders} orders")


# ── SCD2 cleansed layer (cleansed_scd2.duckdb) ───────────────────────────────

def _apply_change(record: dict, fields: list, table: str) -> dict:
    updated = dict(record)
    for field in fields:
        if field == "email":
            updated["email"] = fake.email()
        elif field == "phone":
            updated["phone"] = f"+{fake.numerify('##########')}"
        elif field == "address_line1":
            updated["address_line1"] = fake.street_address()
        elif field == "city":
            updated["city"] = fake.city()
        elif field == "country_code":
            updated["country_code"] = _weighted_choice(COUNTRY_DIST)
        elif field == "status":
            current = updated.get("status", "active")
            choices = [s for s, _ in CUSTOMER_STATUS_DIST if s != current]
            updated["status"] = random.choice(choices)
        elif field == "unit_price":
            updated["unit_price"] = round(updated["unit_price"] * random.uniform(0.85, 1.20), 2)
        elif field == "is_available":
            updated["is_available"] = not updated.get("is_available", True)
        elif field == "category":
            current = updated.get("category")
            choices = [c for c, _ in CATEGORY_DIST if c != current]
            updated["category"] = random.choice(choices)
    return updated


def seed_scd2(n_customers=300, n_products=150):
    conn = duckdb.connect(SCD2_DB)

    conn.execute("""
        CREATE OR REPLACE TABLE customers (
            surrogate_key INTEGER,
            customer_code VARCHAR,
            first_name    VARCHAR, last_name VARCHAR,
            email         VARCHAR,
            phone         VARCHAR,
            address_line1 VARCHAR, city VARCHAR, country_code VARCHAR,
            status        VARCHAR,
            effective_date DATE,
            end_date       DATE,
            is_current     BOOLEAN
        )
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE products (
            surrogate_key INTEGER,
            product_code  VARCHAR,
            product_name  VARCHAR,
            category      VARCHAR,
            unit_price    DOUBLE,
            is_available  BOOLEAN,
            effective_date DATE,
            end_date       DATE,
            is_current     BOOLEAN
        )
    """)

    surrogate = 0

    # customers SCD2
    used_emails = set()
    for i in range(1, n_customers + 1):
        email = fake.email()
        while email in used_emails:
            email = fake.email()
        used_emails.add(email)

        record = {
            "customer_code": _code("CUST", 6),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": email,
            "phone": f"+{fake.numerify('##########')}",
            "address_line1": fake.street_address(),
            "city": fake.city(),
            "country_code": _weighted_choice(COUNTRY_DIST),
            "status": _weighted_choice(CUSTOMER_STATUS_DIST),
        }

        n_versions = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
        eff = _past_date(2021)
        versions = []

        for v in range(n_versions):
            end = eff + timedelta(days=random.randint(90, 400)) if v < n_versions - 1 else None
            is_current = v == n_versions - 1
            surrogate += 1
            versions.append((surrogate, record.copy(), eff, end, is_current))
            if end:
                record = _apply_change(record, _pattern_choice(CUSTOMER_CHANGE_PATTERNS), "customers")
                eff = end

        for sk, rec, e, en, ic in versions:
            conn.execute(
                "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [sk, rec["customer_code"], rec["first_name"], rec["last_name"],
                 rec["email"], rec["phone"], rec["address_line1"], rec["city"],
                 rec["country_code"], rec["status"], e, en, ic],
            )

    # products SCD2
    surrogate = 0
    for i in range(1, n_products + 1):
        cat = _weighted_choice(CATEGORY_DIST)
        price_range = {
            "Electronics": (29.99, 2999.99), "Clothing": (9.99, 299.99),
            "Home": (14.99, 599.99),         "Sports": (9.99, 499.99),
            "Beauty": (4.99, 149.99),        "Books": (4.99, 59.99),
            "Toys": (4.99, 199.99),          "Food": (1.99, 49.99),
        }.get(cat, (9.99, 499.99))

        record = {
            "product_code": _sku(),
            "product_name": fake.catch_phrase(),
            "category": cat,
            "unit_price": round(random.uniform(*price_range), 2),
            "is_available": random.random() > 0.15,
        }

        n_versions = random.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.15, 0.10])[0]
        eff = _past_date(2021)

        for v in range(n_versions):
            end = eff + timedelta(days=random.randint(60, 300)) if v < n_versions - 1 else None
            is_current = v == n_versions - 1
            surrogate += 1
            conn.execute(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
                [surrogate, record["product_code"], record["product_name"],
                 record["category"], record["unit_price"], record["is_available"],
                 eff, end, is_current],
            )
            if end:
                record = _apply_change(record, _pattern_choice(PRODUCT_CHANGE_PATTERNS), "products")
                eff = end

    conn.close()
    print(f"  cleansed_scd2.duckdb — {n_customers} customer keys, {n_products} product keys (with SCD2 history)")


if __name__ == "__main__":
    print("Seeding reference databases...")
    seed_source()
    seed_scd2()
    print("Done.")
