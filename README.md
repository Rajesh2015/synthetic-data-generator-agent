# Data Contract → Fake Data Generator (CrewAI POC)

A CrewAI pipeline that reads an [ODCS v3.1.0](https://bitol-io.github.io/open-data-contract-standard/) data
contract and produces realistic, referentially-consistent synthetic data in DuckDB — including SCD2
(slowly-changing-dimension) change history — without any hand-written Faker mappings in the contract itself.

Eight agents run in sequence across nine tasks:

1. **Contract Analyst** — parses the ODCS YAML (`contracts/ecommerce_contract.yaml`) into a schema summary.
2. **Source Data Profiler** — profiles an existing "production" DuckDB (`data/reference.duckdb`) for real
   enum distributions, numeric ranges, and null rates.
3. **SCD2 Pattern Analyst** — mines an existing SCD2 cleansed layer (`data/cleansed_scd2.duckdb`) for which
   fields change, how often, and which change together.
4. **Distribution & Change Analyst** — drafts concrete generation hints (Faker strategy per field, enum
   weights, SCD2 change patterns) by inferring from field names, types, constraints, and observed data.
5. **Synthetic Data Plan Critic** — runs one cheap, bounded reflection pass over that draft, checking schema
   coverage, constraints, enum weights, tracked fields, and change-pattern consistency.
6. **Distribution & Change Analyst** — revises the draft once using the critic's findings. Only this final
   plan is passed downstream.
7. **Synthetic Data Engineer** — generates the first batch of fake rows into `data/dev.duckdb`.
8. **Change Data Specialist** — simulates further SCD2 change batches against that data.
9. **Data Quality Analyst** — validates everything (PK/FK integrity, enums, ranges, required fields) and
   writes an SCD2-readiness report.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11+ (uv can install a compatible version automatically)
- An API key for your chosen LLM provider (every agent is backed by an LLM via CrewAI's `LLM` class):
  - **Anthropic** (default) — an [Anthropic API key](https://console.anthropic.com/) with access to Claude
  - **Gemini** — a [Google AI Studio API key](https://aistudio.google.com/apikey)
  - **OpenAI** — an [OpenAI API key](https://platform.openai.com/) with access to GPT models

### Apple Silicon (M1/M2/M3/M4) note

If your existing Python resolves to an x86_64 build running under Rosetta (common with some Anaconda
installs), let uv install and manage a native arm64 interpreter:

```bash
uv python install 3.11
```

## Setup

```bash
# 1. Install the locked dependencies into a local .venv
uv sync

# 2. Configure your API key
cp .env.example .env
# then edit .env:
#   - Anthropic (default): set ANTHROPIC_API_KEY=sk-ant-...
#   - Gemini             : set LLM_PROVIDER=gemini and GEMINI_API_KEY=...
#   - OpenAI             : set LLM_PROVIDER=openai and OPENAI_API_KEY=...

# 3. Seed the reference DuckDB files the pipeline reads from
#    (creates data/reference.duckdb and data/cleansed_scd2.duckdb)
uv run python -m scripts.seed_reference_data
```

Step 3 is optional but recommended — without it, the profiler and SCD2 analyzer tools fall back to
domain-default assumptions instead of stats derived from data.

## Running the pipeline

```bash
uv run python -m src.main
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

## Running the UI

A [Streamlit](https://streamlit.io/) web UI wraps the same pipeline for non-CLI use:

```bash
pip install -r requirements.txt      # installs streamlit
streamlit run app.py                 # opens http://localhost:8501 in your browser
```

In the UI you can:

- **Pick or upload** an ODCS contract (uploads are saved under `contracts/_uploaded/`).
- **Run the pipeline** and watch the 7 agents progress live.
- **View** the validation report and a preview of every generated table (including the SCD2
  change batches), then **download** the resulting `dev.duckdb`.
- Optionally **seed reference data** from the sidebar (same as `python -m scripts.seed_reference_data`).

Provider and API key are read from `.env` (shown read-only in the sidebar) — the same
`LLM_PROVIDER` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` settings as the CLI. To change them,
edit `.env` and restart the app. Each run resets `data/dev.duckdb` first, so row counts don't
accumulate across runs.

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

### Choosing an LLM provider

Every agent runs on one of two model tiers — a cheap `fast` model for deterministic tool-calling agents
and the reflection critic, and a stronger `smart` model for the analytical agents. Both tiers are provider-agnostic and
selected with environment variables (see [.env.example](.env.example)):

| Env var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Provider for all agents. One of `anthropic`, `gemini`, `openai`. |
| `FAST_MODEL` | provider default | Overrides the fast-tier model id. |
| `SMART_MODEL` | provider default | Overrides the smart-tier model id. |

Default models per provider:

| Provider | `fast` | `smart` | API key |
|---|---|---|---|
| `anthropic` | `claude-haiku-4-5-20251001` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini/gemini-flash-latest` | `gemini/gemini-pro-latest` | `GEMINI_API_KEY` |
| `openai` | `gpt-5-nano` | `gpt-5-mini` | `OPENAI_API_KEY` |

To run on Gemini, set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in your `.env` — no code changes needed.
To run on OpenAI, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in your `.env` — no code changes needed.
The locked dependencies include CrewAI's Anthropic and Google GenAI integrations plus the OpenAI SDK, so
any provider works out of the box. After changing dependencies in `pyproject.toml`, run `uv lock` to update
`uv.lock` and `uv sync` to update the environment.

For an OpenAI-only setup using the cost-efficient defaults:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
FAST_MODEL=gpt-5-nano
SMART_MODEL=gpt-5-mini
```

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

- **`ImportError: Anthropic native provider not available`** — ensure the environment matches the lockfile
  by running `uv sync`.
- **Failed to build wheel for `cryptography`** — you're likely on a non-native Python interpreter; see the
  Apple Silicon note above.
- **Profiler/SCD2 analyzer return fallback data** — run `uv run python -m scripts.seed_reference_data` first (see
  Setup step 3).
