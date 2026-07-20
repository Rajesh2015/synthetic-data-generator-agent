# Data Contract → Fake Data Generator (CrewAI POC)

A CrewAI pipeline that reads an [ODCS v3.1.0](https://bitol-io.github.io/open-data-contract-standard/) data
contract and produces realistic, referentially-consistent synthetic data in DuckDB — including SCD2
(slowly-changing-dimension) change history — without any hand-written Faker mappings in the contract itself.

Seven agents run in sequence:

1. **Contract Analyst** — parses the ODCS YAML (`contracts/ecommerce_contract.yaml`) into a schema summary.
2. **Source Data Profiler** — profiles an existing "production" DuckDB (`data/reference.duckdb`) for real
   enum distributions, numeric ranges, and null rates.
3. **SCD2 Pattern Analyst** — mines an existing SCD2 cleansed layer (`data/cleansed_scd2.duckdb`) for which
   fields change, how often, and which change together.
4. **Distribution & Change Analyst** — an LLM reasoning step that turns the above into concrete generation
   hints (Faker strategy per field, enum weights, SCD2 change patterns) purely by inferring from field
   names/types/constraints, since the contract has no custom `x-fake` annotations.
5. **Synthetic Data Engineer** — generates the first batch of fake rows into `data/dev.duckdb`.
6. **Change Data Specialist** — simulates further SCD2 change batches against that data.
7. **Data Quality Analyst** — validates everything (PK/FK integrity, enums, ranges, required fields) and
   writes an SCD2-readiness report.

## Prerequisites

- Python 3.11+ (native to your CPU architecture — see the Apple Silicon note below)
- An [Anthropic API key](https://console.anthropic.com/) with access to Claude, since every agent is backed
  by a Claude model via CrewAI's `LLM` class

### Apple Silicon (M1/M2/M3/M4) note

If your `python3` resolves to an x86_64 build running under Rosetta (common with some Anaconda installs —
check with `python3 -c "import platform; print(platform.machine())"`), installing dependencies will fail
building the `cryptography` wheel from source (it needs a Rust target that isn't installed). Use a native
arm64 interpreter instead, e.g. Homebrew's:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 4. Seed the reference DuckDB files the pipeline reads from
#    (creates data/reference.duckdb and data/cleansed_scd2.duckdb)
python -m scripts.seed_reference_data
```

Step 4 is optional but recommended — without it, the profiler and SCD2 analyzer tools fall back to
domain-default assumptions instead of stats derived from data.

## Running the pipeline

```bash
python -m src.main
```

This kicks off the full crew and prints a validation report, e.g.:

```
============================================================
 VALIDATION REPORT
============================================================
  Total checks : 42
  Passed       : 42
  Failed       : 0
  Pass rate    : 100.0%

  [customers]  50 rows total
    ✓ PK uniqueness (customer_id) — All PKs unique per batch
    ✓ NOT NULL (email) — No NULLs
    ...
```

Output data lands in `data/dev.duckdb`, with `_batch_id` / `_snapshot_date` columns distinguishing the
initial batch from later SCD2 change batches.

## Configuration

Tunable knobs live in [src/config.py](src/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `NUM_RECORDS` | 50 | Rows generated per table in the initial batch |
| `NUM_CHANGE_BATCHES` | 2 | Number of SCD2 change batches to simulate |
| `CHANGE_RATE` | 0.3 | Fraction of records mutated per change batch |

To generate data for a different domain, point `CONTRACT_PATH` at a different ODCS YAML file — the pipeline
infers Faker strategies and SCD2-tracked fields from field names/types/constraints, so no custom
annotations are required in the contract.

## Project layout

```
contracts/                  ODCS data contract(s)
data/                       DuckDB files (generated — not committed)
scripts/seed_reference_data.py   One-time script to create reference/SCD2 sample DBs
src/
  config.py                 Paths and pipeline tuning knobs
  crew.py                    Agent/Task/Crew definitions
  main.py                    Entry point
  models/contract_schema.py  Pydantic models for the parsed ODCS contract
  tools/                     CrewAI tools: parse, profile, analyze SCD2, generate, simulate, validate
```

## Troubleshooting

- **`ImportError: Anthropic native provider not available`** — `requirements.txt` installs
  `crewai[anthropic]`; if you installed plain `crewai` some other way, run
  `pip install "crewai[anthropic]"`.
- **Failed to build wheel for `cryptography`** — you're likely on a non-native Python interpreter; see the
  Apple Silicon note above.
- **Profiler/SCD2 analyzer return fallback data** — run `python -m scripts.seed_reference_data` first (see
  Setup step 4).
