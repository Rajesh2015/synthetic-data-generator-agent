"""Streamlit UI for the Data Contract → Synthetic Data Generator.

Run with:  streamlit run app.py

Lets a user pick/upload an ODCS contract, run the 7-agent CrewAI pipeline with
live per-agent progress, and view + download the generated DuckDB data and the
validation report. Provider + API key are taken from .env (never typed here).
"""

import os
import sys
import glob
import queue
import threading
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing src.config / src.crew — those read the provider and
# model ids (and instantiate LLM objects) from the environment at import time.
load_dotenv()

import streamlit as st
import duckdb

from src.config import (
    ROOT_DIR, CONTRACT_PATH, DB_PATH,
    LLM_PROVIDER, FAST_MODEL, SMART_MODEL,
)
from src.crew import build_crew
from src.tools.validator_tool import validate_data

# Which env var holds the key for the configured provider.
_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

CONTRACTS_DIR = ROOT_DIR / "contracts"
UPLOAD_DIR = CONTRACTS_DIR / "_uploaded"


# ---------------------------------------------------------------------------
# Pipeline execution (background thread + queue for live progress)
# ---------------------------------------------------------------------------

def _run_crew(crew, q: "queue.Queue"):
    """Run the crew in a worker thread. Only touches the queue — never st.*
    (worker threads have no Streamlit ScriptRunContext)."""
    try:
        result = crew.kickoff()
        q.put(("done", result))
    except Exception as e:  # surface API errors (429/503/auth) to the UI
        q.put(("error", e))


def run_pipeline(contract_path: str):
    """Reset output, kick off the crew, and render live progress until done.
    Persists the outcome to st.session_state so results survive reruns."""
    # The generator APPENDS (CREATE TABLE IF NOT EXISTS, hardcoded _batch_id=1),
    # so wipe the output DB first or rows accumulate across runs.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    q: queue.Queue = queue.Queue()

    def task_cb(output):
        q.put((
            "task",
            getattr(output, "agent", None),
            getattr(output, "raw", "") or "",
        ))

    crew = build_crew(contract_path=contract_path, task_callback=task_cb)
    steps = [t.agent.role for t in crew.tasks]

    thread = threading.Thread(target=_run_crew, args=(crew, q), daemon=True)
    thread.start()

    st.info(f"Running {len(steps)} agents on `{Path(contract_path).name}` …")
    bar = st.progress(0.0)
    checklist = st.empty()

    def render(done: int):
        lines = []
        for i, role in enumerate(steps):
            icon = "✅" if i < done else ("⏳" if i == done else "⬜")
            lines.append(f"{icon} {role}")
        checklist.markdown("\n\n".join(lines))

    render(0)
    done = 0
    result = None
    error = None
    outputs = []  # (role, raw) per completed task

    while thread.is_alive() or not q.empty():
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            continue
        kind = item[0]
        if kind == "task":
            done += 1
            outputs.append((item[1], item[2]))
            bar.progress(min(done / len(steps), 1.0))
            render(done)
        elif kind == "done":
            result = item[1]
        elif kind == "error":
            error = item[1]
    thread.join()

    # Persist so the Results section survives the rerun triggered by widgets
    # (e.g. the download button).
    st.session_state["ran"] = True
    st.session_state["contract_path"] = contract_path
    st.session_state["error"] = None if error is None else f"{type(error).__name__}: {error}"
    st.session_state["report"] = "" if result is None else (getattr(result, "raw", None) or str(result))
    st.session_state["step_outputs"] = outputs


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_results():
    if st.session_state.get("error"):
        st.error(f"Pipeline failed: {st.session_state['error']}")
        st.caption(
            "If this is a Gemini `429 RESOURCE_EXHAUSTED` / `503`, it's a free-tier "
            "quota/availability limit — retry later or use a paid key / Anthropic."
        )

    contract_path = st.session_state.get("contract_path", CONTRACT_PATH)

    # --- Validation (structured, from the tool directly for reliability) ---
    st.subheader("Validation")
    try:
        import json
        report = json.loads(validate_data.func(contract_path))
        summ = report.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total checks", summ.get("total_checks", "—"))
        c2.metric("Passed", summ.get("passed", "—"))
        c3.metric("Failed", summ.get("failed", "—"))
        c4.metric("Pass rate", summ.get("pass_rate", "—"))
        for table, tdata in report.get("tables", {}).items():
            with st.expander(f"{table} — {tdata.get('total_rows', 0)} rows"):
                rows = [
                    {"rule": c["rule"], "result": c["result"], "detail": c["detail"]}
                    for c in tdata.get("checks", [])
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not compute validation report: {e}")

    # --- Data preview ---
    st.subheader("Generated data")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        tables = [
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]
        for t in tables:
            dist = con.execute(
                f'SELECT _batch_id, count(*) AS rows FROM "{t}" GROUP BY 1 ORDER BY 1'
            ).df()
            sample = con.execute(f'SELECT * FROM "{t}" LIMIT 100').df()
            with st.expander(f"{t}  ·  batches: {dist['_batch_id'].tolist()}"):
                st.caption("Rows per batch (batch 1 = initial, 2+ = SCD2 change batches)")
                st.dataframe(dist, use_container_width=True, hide_index=True)
                st.caption("Sample (first 100 rows)")
                st.dataframe(sample, use_container_width=True, hide_index=True)
        con.close()
    except Exception as e:
        st.warning(f"Could not read generated data: {e}")

    # --- Analyst narrative + download ---
    if st.session_state.get("report"):
        st.subheader("Data Quality analyst report")
        st.markdown(st.session_state["report"])

    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            st.download_button(
                "⬇ Download dev.duckdb",
                data=f.read(),
                file_name="dev.duckdb",
                mime="application/octet-stream",
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Synthetic Data Generator", page_icon="🧪", layout="wide")
st.title("🧪 Data Contract → Synthetic Data Generator")
st.caption("ODCS contract → referentially-consistent fake data + SCD2 change history in DuckDB.")

# --- Environment status (read-only; provider/key come from .env) ---
with st.sidebar:
    st.header("Environment")
    key_env = _KEY_ENV.get(LLM_PROVIDER, "?")
    key_present = bool(os.getenv(key_env))
    st.write(f"**Provider:** `{LLM_PROVIDER}`")
    st.write(f"**Fast model:** `{FAST_MODEL}`")
    st.write(f"**Smart model:** `{SMART_MODEL}`")
    if key_present:
        st.success(f"{key_env} detected")
    else:
        st.error(f"{key_env} not set in .env — runs will fail auth.")
    st.caption("Change provider/key by editing `.env`, then restart the app.")

    st.divider()
    st.header("Reference data (optional)")
    st.caption("Seeds sample production DBs so the profiler/SCD2 analyzer use real stats.")
    if st.button("Seed reference data"):
        with st.spinner("Seeding reference DuckDB files…"):
            proc = subprocess.run(
                [sys.executable, "-m", "scripts.seed_reference_data"],
                cwd=str(ROOT_DIR), capture_output=True, text=True,
            )
        if proc.returncode == 0:
            st.success("Seeded.")
            st.code((proc.stdout or "").strip() or "done")
        else:
            st.error("Seeding failed.")
            st.code((proc.stderr or proc.stdout or "").strip())

# --- Contract picker ---
st.subheader("1 · Choose a data contract")
existing = sorted(glob.glob(str(CONTRACTS_DIR / "*.yaml")))
labels = [Path(p).name for p in existing]
col_pick, col_up = st.columns(2)
with col_pick:
    default_idx = labels.index(Path(CONTRACT_PATH).name) if Path(CONTRACT_PATH).name in labels else 0
    chosen = st.selectbox("Existing contracts", labels, index=default_idx if labels else 0) if labels else None
with col_up:
    uploaded = st.file_uploader("…or upload an ODCS YAML", type=["yaml", "yml"])

if uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    contract_path = str(dest)
    st.success(f"Using uploaded contract: `{uploaded.name}`")
elif chosen:
    contract_path = str(CONTRACTS_DIR / chosen)
else:
    contract_path = CONTRACT_PATH

# --- Run ---
st.subheader("2 · Run the pipeline")
if not bool(os.getenv(_KEY_ENV.get(LLM_PROVIDER, ""))):
    st.warning(f"No {_KEY_ENV.get(LLM_PROVIDER, 'API key')} in .env — the run will fail authentication.")

if st.button("▶ Run pipeline", type="primary"):
    run_pipeline(contract_path)

# --- Results (persist across reruns once a run has happened) ---
if st.session_state.get("ran"):
    st.divider()
    st.subheader("3 · Results")
    render_results()
