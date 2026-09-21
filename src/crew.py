from crewai import Agent, Task, Crew, Process, LLM
from src.tools import (
    parse_odcs_contract,
    profile_source_data,
    analyze_scd2_patterns,
    generate_initial_batch,
    simulate_changes,
    validate_data,
)
from src.config import (
    CONTRACT_PATH, SOURCE_DB_PATH, SCD2_DB_PATH,
    NUM_RECORDS, NUM_CHANGE_BATCHES, CHANGE_RATE,
    FAST_MODEL, SMART_MODEL,
)

# Provider-agnostic (Anthropic / Gemini / …) — see src/config.py.
# _fast : cheap tool-calling agents; _smart : stronger analytical agents.
_fast  = LLM(model=FAST_MODEL)
_smart = LLM(model=SMART_MODEL)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

contract_analyst = Agent(
    role="Data Contract Analyst",
    goal="Call parse_odcs_contract and return its JSON output.",
    backstory="You parse ODCS YAML contracts and return the structured schema summary.",
    tools=[parse_odcs_contract],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

data_profiler = Agent(
    role="Source Data Profiler",
    goal="Call profile_source_data on the reference DB and return its distribution stats JSON.",
    backstory="You run SQL distribution queries against existing anonymized warehouse data.",
    tools=[profile_source_data],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

scd2_analyst = Agent(
    role="SCD2 Pattern Analyst",
    goal="Call analyze_scd2_patterns on the SCD2 cleansed layer and return its change-pattern JSON.",
    backstory="You mine existing SCD2 history to find which fields change, how often, and together.",
    tools=[analyze_scd2_patterns],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

distribution_analyst = Agent(
    role="Distribution & Change Analyst",
    goal=(
        "Combine the source data profile and SCD2 change patterns into concrete generation "
        "hints: realistic enum weights, numeric ranges, and change scenarios grounded in "
        "real data — not guesses."
    ),
    backstory=(
        "You are a data scientist who bridges observed data patterns and synthetic data generation. "
        "When you see that 72% of customers are active in production, you ensure the generated "
        "dataset reflects that — not a uniform 33/33/33 split. "
        "When you see that in prod address_line1 and city change together 22% of the time, "
        "you encode that as a co-change pattern for the simulator. "
        "You produce a precise JSON enrichment spec — no prose."
    ),
    tools=[],           # pure LLM reasoning over the stats from prior tasks
    llm=_smart,
    allow_delegation=False,
    verbose=True,
)

enrichment_reflector = Agent(
    role="Synthetic Data Plan Critic",
    goal=(
        "Critically inspect the draft enrichment JSON against the parsed contract and return "
        "a precise, machine-readable list of omissions, contradictions, and unsafe assumptions."
    ),
    backstory=(
        "You are a meticulous reviewer of synthetic-data plans. You do not rewrite the plan or "
        "add prose. You verify schema coverage, constraint-aware Faker strategies, enum weights, "
        "SCD2 field eligibility, and change-pattern consistency, then provide targeted corrections."
    ),
    tools=[],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

data_generator = Agent(
    role="Synthetic Data Engineer",
    goal="Call generate_initial_batch with the contract path and enrichment hints. Return its JSON.",
    backstory="You invoke the Faker-based generator. All field values come from Faker, not from you.",
    tools=[generate_initial_batch],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

change_simulator = Agent(
    role="Change Data Specialist",
    goal="Call simulate_changes with the contract path and change patterns. Return its JSON.",
    backstory="You invoke the change simulation tool using the patterns derived from real SCD2 history.",
    tools=[simulate_changes],
    llm=_fast,
    allow_delegation=False,
    verbose=True,
)

validation_analyst = Agent(
    role="Data Quality & SCD2 Readiness Analyst",
    goal=(
        "Run the automated validator, then reason over the results to produce an actionable "
        "quality report with SCD2-readiness assessment."
    ),
    backstory=(
        "You run the rule-based validator first, then apply your expertise to interpret results: "
        "root causes for failures, whether the change history realistically exercises SCD2 merge "
        "logic (type-1 vs type-2 field handling, surrogate key generation, is_current flag), "
        "and concrete recommendations."
    ),
    tools=[validate_data],
    llm=_smart,
    allow_delegation=False,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

def build_crew(
    contract_path: str = CONTRACT_PATH,
    step_callback=None,
    task_callback=None,
) -> Crew:
    """Assemble the synthetic-data crew dynamically per contract path.

    Tasks are built here (not at import time) so callers — e.g. the Streamlit UI —
    can point the pipeline at an uploaded/selected contract and receive live
    progress via the crew callbacks.

    contract_path : ODCS YAML the whole pipeline reads (defaults to config value).
    step_callback : called after each agent step with an AgentAction/AgentFinish.
    task_callback : called after each task with a TaskOutput (has .agent, .raw).
    """
    task_parse = Task(
        description=f"Parse the ODCS contract at '{contract_path}' and return the full JSON schema summary.",
        expected_output="JSON with table schemas, dependency_order, foreign_keys, scd2_tracked_fields.",
        agent=contract_analyst,
    )

    task_profile = Task(
        description=(
            f"Profile the source reference database at '{SOURCE_DB_PATH}' using contract '{contract_path}'. "
            "Return the distribution stats JSON. "
            "If the DB does not exist the tool returns a graceful fallback — pass that through as-is."
        ),
        expected_output="JSON with enum distributions, numeric stats, null rates, and row counts per table.",
        agent=data_profiler,
        context=[task_parse],
    )

    task_analyze_scd2 = Task(
        description=(
            f"Analyze SCD2 change patterns in '{SCD2_DB_PATH}' using contract '{contract_path}'. "
            "Return the change-pattern JSON. "
            "If the DB does not exist the tool returns a graceful fallback — pass that through as-is."
        ),
        expected_output=(
            "JSON with per-table: avg_versions_per_key, field_change_frequency, co_change_patterns."
        ),
        agent=scd2_analyst,
        context=[task_parse],
    )

    task_enrich = Task(
        description=(
            "You have three inputs in your context:\n"
            "  1. Parsed ODCS schema (from task_parse)\n"
            "  2. Source data distribution stats (from task_profile)\n"
            "  3. SCD2 change patterns (from task_analyze_scd2)\n\n"
            "The contract is an ODCS v3.1.0 specification. Respect any explicit x-change-tracking annotations "
            "if present in task_parse. Otherwise, infer field strategies, faker types, and change-tracked fields "
            "from field names, types, descriptions, and constraints.\n\n"


            "Produce a JSON object with THREE top-level keys:\n\n"
            "A) 'generation_hints' — per table → per field:\n"
            "   - inferred_faker: which Faker strategy to use. Rules:\n"
            "     * primaryKey:true → 'sequence'\n"
            "     * references set → 'foreign_key'\n"
            "     * enum set → 'enum'\n"
            "     * pattern set → 'regex'\n"
            "     * format:email → 'email'\n"
            "     * field name contains 'first'+'name' → 'first_name'\n"
            "     * field name contains 'last'+'name' → 'last_name'\n"
            "     * field name contains 'phone' → 'phone_number'\n"
            "     * field name contains 'address' → 'street_address'\n"
            "     * field name = 'city' → 'city'\n"
            "     * field name contains 'price' or 'amount' → 'price'\n"
            "     * field name contains 'product'+'name' → 'product_name'\n"
            "     * type:boolean → 'boolean'\n"
            "     * type:integer → 'integer'\n"
            "     * type:date → 'past_date'\n"
            "     * type:timestamp → 'past_datetime'\n"
            "     * derived total fields (total_amount) → 'computed'\n"
            "     * snapshot price fields (unit_price_at_order) → 'derived_from_parent'\n"
            "   - enum_weights: list of floats (same order as enum values) derived from "
            "     REAL source data distribution. Use sensible domain defaults if no reference data.\n\n"
            "B) 'change_tracking' — per table, list of fields that should be SCD2-tracked:\n"
            "   Rules: PKs never change. Natural keys (unique non-PK) never change. "
            "   created_at/created_by never change. FK reference fields never change. "
            "   Fields like email, phone, address, city, country, status, price, category, "
            "   is_available DO change. Omit append-only tables (orders).\n\n"
            "C) 'change_patterns' — per table (SCD2-tracked tables only):\n"
            "   - field_change_frequency: {field: float} derived from SCD2 history "
            "     (or domain defaults if no SCD2 data)\n"
            "   - co_change_patterns: [{fields:[...], frequency:float}] — "
            "     which fields typically change TOGETHER\n\n"
            "Return ONLY the JSON object — no markdown, no prose."
        ),
        expected_output=(
            "A JSON object with keys 'generation_hints', 'change_tracking', and 'change_patterns'. "
            "generation_hints covers every field in every table with inferred_faker set. "
            "change_tracking lists SCD2-trackable fields per table."
        ),
        agent=distribution_analyst,
        context=[task_parse, task_profile, task_analyze_scd2],
    )

    task_reflect_enrichment = Task(
        description=(
            "Review the draft enrichment JSON against the parsed ODCS schema. This is a bounded "
            "reflection pass: identify concrete defects for the Distribution & Change Analyst to fix.\n\n"
            "Check all of the following:\n"
            "1. The JSON has exactly the required top-level sections: generation_hints, "
            "change_tracking, and change_patterns.\n"
            "2. Every table and field in the contract appears under generation_hints and has a valid "
            "inferred_faker strategy consistent with its PK, FK, enum, pattern, format, type, and name.\n"
            "3. Every enum_weights list has the same length as its enum, contains only non-negative "
            "numbers, and has a positive total.\n"
            "4. change_tracking excludes primary keys, unique natural keys, foreign keys, audit fields, "
            "and append-only tables. Every tracked field actually exists in its table.\n"
            "5. change_patterns only reference tracked fields; all frequencies are between 0 and 1; "
            "and every co-change pattern contains existing tracked fields.\n"
            "6. Recommendations are grounded in the source profile and SCD2 analysis when those inputs "
            "contain real observations.\n\n"
            "Return ONLY JSON with this shape:\n"
            "{\n"
            "  \"verdict\": \"PASS\" or \"REVISE\",\n"
            "  \"issues\": [\n"
            "    {\"path\": \"JSON path\", \"problem\": \"specific defect\", "
            "\"fix\": \"specific correction\"}\n"
            "  ],\n"
            "  \"checks_passed\": [\"short check name\"]\n"
            "}\n"
            "Use PASS only when issues is empty. Do not produce a replacement enrichment document."
        ),
        expected_output=(
            "JSON critique with verdict, issues, and checks_passed. Each issue identifies an exact "
            "JSON path and an actionable correction."
        ),
        agent=enrichment_reflector,
        context=[task_parse, task_profile, task_analyze_scd2, task_enrich],
    )

    task_refine_enrichment = Task(
        description=(
            "Produce the final enrichment JSON by revising the draft using every valid issue from the "
            "reflection critique. Re-check the result yourself before returning it. If the critique "
            "verdict is PASS, preserve the draft content.\n\n"
            "Return ONLY the complete corrected JSON object with the same three top-level keys: "
            "generation_hints, change_tracking, and change_patterns. Do not include the critique, "
            "markdown fences, commentary, or additional top-level keys."
        ),
        expected_output=(
            "Final corrected JSON with complete generation_hints, valid change_tracking, and internally "
            "consistent change_patterns, ready for the generator and simulator tools."
        ),
        agent=distribution_analyst,
        context=[
            task_parse,
            task_profile,
            task_analyze_scd2,
            task_enrich,
            task_reflect_enrichment,
        ],
    )

    task_generate = Task(
        description=(
            f"Generate {NUM_RECORDS} records per table using contract '{contract_path}'. "
            "Use the FINAL REVISED enrichment JSON in your context. Extract its 'generation_hints' section "
            "and pass it as the enrichment_json parameter to generate_initial_batch."
        ),
        expected_output="JSON summary: status, batch_id=1, records_generated per table, db_path.",
        agent=data_generator,
        context=[task_parse, task_refine_enrichment],
    )

    task_simulate = Task(
        description=(
            f"Generate {NUM_CHANGE_BATCHES} SCD2 change batches using contract '{contract_path}'. "
            "Pass the ENTIRE FINAL REVISED enrichment JSON from your context (the object with "
            "'generation_hints', 'change_tracking', and 'change_patterns' keys) as the "
            "change_patterns_json parameter to simulate_changes — do NOT extract a sub-section. "
            "The tool needs 'change_tracking' to know which fields to mutate and "
            "'change_patterns' for realistic field-change frequencies. "
            "This ensures mutations mirror real production change behaviour — "
            "same natural keys get the same kinds of changes seen in your existing SCD2 data."
        ),
        expected_output="JSON summary: batches_generated, change_rate, per-batch field mutation details.",
        agent=change_simulator,
        context=[task_parse, task_refine_enrichment, task_generate],
    )

    task_validate = Task(
        description=(
            f"Validate all DuckDB data against contract '{contract_path}' by calling validate_data. "
            "Then reason over the raw JSON report and produce a structured analysis:\n"
            "1. SUMMARY — plain-English pass/fail overview\n"
            "2. ROOT_CAUSES — for any failure, explain why it happened\n"
            "3. SCD2_READINESS — does the generated change history adequately stress-test:\n"
            "   - Type-2 inserts (new rows for changed records)?\n"
            "   - Type-1 overwrites (non-tracked fields)?\n"
            "   - Correct is_current / effective_date / end_date logic?\n"
            "   - Enough variety in which fields changed?\n"
            "4. RECOMMENDATIONS — top 3 actionable improvements"
        ),
        expected_output=(
            "Structured report with sections: RAW_VALIDATION, SUMMARY, "
            "ROOT_CAUSES, SCD2_READINESS, RECOMMENDATIONS."
        ),
        agent=validation_analyst,
        context=[task_parse, task_refine_enrichment, task_generate, task_simulate],
    )

    return Crew(
        agents=[
            contract_analyst,
            data_profiler,
            scd2_analyst,
            distribution_analyst,
            enrichment_reflector,
            data_generator,
            change_simulator,
            validation_analyst,
        ],
        tasks=[
            task_parse,
            task_profile,
            task_analyze_scd2,
            task_enrich,
            task_reflect_enrichment,
            task_refine_enrichment,
            task_generate,
            task_simulate,
            task_validate,
        ],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
        task_callback=task_callback,
    )
