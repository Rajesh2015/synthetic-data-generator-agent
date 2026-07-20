from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CONTRACT_PATH = str(ROOT_DIR / "contracts" / "ecommerce_contract.yaml")
DB_PATH       = str(ROOT_DIR / "data" / "dev.duckdb")

# Existing anonymized data — profiler reads distributions from here.
# Run scripts/seed_reference_data.py once to create these if you don't have real data.
SOURCE_DB_PATH = str(ROOT_DIR / "data" / "reference.duckdb")

# Existing SCD2 cleansed layer — analyzer reads change patterns from here.
SCD2_DB_PATH   = str(ROOT_DIR / "data" / "cleansed_scd2.duckdb")

NUM_RECORDS        = 50
NUM_CHANGE_BATCHES = 2
CHANGE_RATE        = 0.3
