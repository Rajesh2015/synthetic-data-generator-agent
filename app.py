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


def _safe_remove_db(path_str: str):

    """Safely remove a DuckDB file on Windows, handling locks gracefully."""
    if not os.path.exists(path_str):
        return
    import gc
    import time
    gc.collect()
    for _ in range(5):
        try:
            os.remove(path_str)
            return
        except PermissionError:
            time.sleep(0.2)


def _safe_read_bytes(path_str: str) -> bytes | None:
    """Safely read database bytes for download, handling Windows file locks gracefully."""
    if not os.path.exists(path_str):
        return None
    import gc
    import time
    gc.collect()
    for _ in range(5):
        try:
            with open(path_str, "rb") as f:
                return f.read()
        except PermissionError:
            time.sleep(0.2)
    return None


def run_pipeline(contract_path: str):
    """Reset output, kick off the crew, and render live progress until done.
    Persists the outcome to st.session_state so results survive reruns."""
    # The generator APPENDS (CREATE TABLE IF NOT EXISTS, hardcoded _batch_id=1),
    # so wipe the output DB first or rows accumulate across runs.
    _safe_remove_db(DB_PATH)

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
    st.subheader("Validation & SCD2 Integrity")
    try:
        import json
        report = json.loads(validate_data.func(contract_path))
        summ = report.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total checks", summ.get("total_checks", "—"))
        c2.metric("Passed", summ.get("passed", "—"))
        c3.metric("Failed", summ.get("failed", "—"))
        c4.metric("Pass rate", summ.get("pass_rate", "—"))

        st.caption("📋 Standard Contract Quality Rules")
        for table, tdata in report.get("tables", {}).items():
            with st.expander(f"{table} — {tdata.get('total_rows', 0)} rows"):
                rows = [
                    {"rule": c["rule"], "result": c["result"], "detail": c["detail"]}
                    for c in tdata.get("checks", [])
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

        scd2_val = report.get("scd2_validation", {})
        if scd2_val:
            st.caption("⏳ SCD2 Temporal Logic Validation (effective_date, end_date, is_current)")
            for scd2_table, sdata in scd2_val.items():
                with st.expander(f"⚡ {scd2_table} — Temporal Integrity"):
                    srows = [
                        {"rule": c["rule"], "result": c["result"], "detail": c["detail"]}
                        for c in sdata.get("checks", [])
                    ]
                    if srows:
                        st.dataframe(srows, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not compute validation report: {e}")

    # --- Data preview ---
    st.subheader("Generated Data & SCD2 Cleansed Dimension Layer")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            all_tables = [
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            ]
            
            raw_tables = [t for t in all_tables if not t.endswith("_scd2")]
            scd2_tables = [t for t in all_tables if t.endswith("_scd2")]

            tab_raw, tab_scd2 = st.tabs(["📦 Raw Batch Snapshots", "⏳ Cleansed SCD2 Dimensions"])

            with tab_raw:
                for t in raw_tables:
                    dist = con.execute(
                        f'SELECT _batch_id, count(*) AS rows FROM "{t}" GROUP BY 1 ORDER BY 1'
                    ).df()
                    sample = con.execute(f'SELECT * FROM "{t}" LIMIT 100').df()
                    with st.expander(f"{t}  ·  batches: {dist['_batch_id'].tolist()}"):
                        st.caption("Rows per batch (batch 1 = initial, 2+ = SCD2 change batches)")
                        st.dataframe(dist, use_container_width=True, hide_index=True)
                        st.caption("Sample (first 100 rows)")
                        st.dataframe(sample, use_container_width=True, hide_index=True)

            with tab_scd2:
                if not scd2_tables:
                    st.info("No SCD2 dimension tables generated yet. Run the pipeline to build SCD2 historical tables.")
                for t in scd2_tables:
                    sample_scd2 = con.execute(f'SELECT * FROM "{t}" ORDER BY scd_id LIMIT 100').df()
                    cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    active_cnt = con.execute(f'SELECT COUNT(*) FROM "{t}" WHERE is_current = TRUE').fetchone()[0]
                    with st.expander(f"✨ {t}  ·  {cnt} total versions ({active_cnt} active current)"):
                        st.caption("Transformed SCD2 Dimension with effective_date, end_date, is_current, and version")
                        st.dataframe(sample_scd2, use_container_width=True, hide_index=True)
        finally:
            con.close()
    except Exception as e:
        st.warning(f"Could not read generated data: {e}")

    # --- Analyst narrative + download ---
    if st.session_state.get("report"):
        st.subheader("Data Quality analyst report")
        st.markdown(st.session_state["report"])

    db_bytes = _safe_read_bytes(DB_PATH)
    if db_bytes:
        st.download_button(
            "⬇ Download dev.duckdb",
            data=db_bytes,
            file_name="dev.duckdb",
            mime="application/octet-stream",
        )



import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page Header Animation: Oil Rig & Tagline
# ---------------------------------------------------------------------------

def render_oil_mining_animation():
    html_code = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background: transparent;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        overflow: hidden;
    }
    .oil-mining-card {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e293b 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 14px 18px 12px 18px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.7);
        text-align: center;
    }
    .oil-tagline-text {
        margin-top: 6px;
        font-size: 1.45rem;
        font-weight: 900;
        letter-spacing: 3px;
        background: linear-gradient(90deg, #38bdf8, #fbbf24, #34d399, #38bdf8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-transform: uppercase;
    }
    .oil-sub-text {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 2px;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    @keyframes beam-nod {
        0%, 100% { transform: rotate(-11deg); }
        50% { transform: rotate(11deg); }
    }
    @keyframes rod-move {
        0%, 100% { transform: translateY(-12px); }
        50% { transform: translateY(12px); }
    }
    @keyframes pulse-flow {
        0% { stroke-dashoffset: 40; opacity: 0.4; }
        50% { opacity: 1; }
        100% { stroke-dashoffset: 0; opacity: 0.4; }
    }
    @keyframes float-up-1 {
        0% { transform: translateY(10px); opacity: 0; }
        40% { opacity: 1; }
        100% { transform: translateY(-90px); opacity: 0; }
    }
    @keyframes float-up-2 {
        0% { transform: translateY(15px); opacity: 0; }
        50% { opacity: 0.9; }
        100% { transform: translateY(-110px); opacity: 0; }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .pump-walking-beam {
        transform-origin: 250px 60px;
        animation: beam-nod 2.6s ease-in-out infinite;
    }
    .polished-rod {
        animation: rod-move 2.6s ease-in-out infinite;
    }
    .data-pulse-stream {
        animation: pulse-flow 1.2s linear infinite;
    }
    .p1 { animation: float-up-1 2.2s ease-out infinite; }
    .p2 { animation: float-up-2 2.8s ease-out infinite 0.4s; }
    .p3 { animation: float-up-1 2.5s ease-out infinite 1.1s; }
    .p-text1 { animation: float-up-2 3.0s ease-out infinite 0.2s; }
    .p-text2 { animation: float-up-1 2.7s ease-out infinite 0.8s; }
    .p-text3 { animation: float-up-2 3.2s ease-out infinite 1.4s; }
    .p-text4 { animation: float-up-1 2.4s ease-out infinite 1.8s; }
</style>
</head>
<body>
<div class="oil-mining-card">
    <svg viewBox="0 0 500 205" style="width: 100%; max-width: 500px; height: 170px; margin: 0 auto; display: block;" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#0f172a"/>
                <stop offset="100%" stop-color="#1e293b"/>
            </linearGradient>
            <linearGradient id="groundGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#334155"/>
                <stop offset="40%" stop-color="#1e293b"/>
                <stop offset="100%" stop-color="#0f172a"/>
            </linearGradient>
            <linearGradient id="oilPoolGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#020617"/>
                <stop offset="50%" stop-color="#0f172a"/>
                <stop offset="100%" stop-color="#020617"/>
            </linearGradient>
            <linearGradient id="pipeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#475569"/>
                <stop offset="50%" stop-color="#94a3b8"/>
                <stop offset="100%" stop-color="#334155"/>
            </linearGradient>
            <linearGradient id="beamGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#38bdf8"/>
                <stop offset="100%" stop-color="#0284c7"/>
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>

        <rect x="0" y="0" width="500" height="135" fill="url(#skyGrad)" rx="8"/>
        <rect x="0" y="135" width="500" height="70" fill="url(#groundGrad)"/>
        <line x1="0" y1="135" x2="500" y2="135" stroke="#475569" stroke-width="2"/>
        
        <path d="M 0 155 Q 120 150 250 158 T 500 152" fill="none" stroke="#1e293b" stroke-width="2" opacity="0.6"/>
        <path d="M 0 175 Q 150 180 300 172 T 500 178" fill="none" stroke="#0f172a" stroke-width="3" opacity="0.8"/>

        <ellipse cx="250" cy="190" rx="210" ry="12" fill="url(#oilPoolGrad)" stroke="#38bdf8" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.8"/>
        <path d="M 50 190 Q 150 185 250 190 T 450 190 Q 350 197 250 195 T 50 190 Z" fill="#020617" opacity="0.95"/>

        <rect x="244" y="125" width="12" height="65" fill="url(#pipeGrad)" opacity="0.9"/>
        
        <line x1="250" y1="188" x2="250" y2="125" stroke="#38bdf8" stroke-width="4" stroke-linecap="round"/>
        <line x1="250" y1="188" x2="250" y2="40" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-dasharray="6 8" class="data-pulse-stream"/>

        <g>
            <circle cx="250" cy="175" r="3.5" fill="#f59e0b" class="p1" filter="url(#glow)"/>
            <circle cx="250" cy="165" r="4.5" fill="#38bdf8" class="p2" filter="url(#glow)"/>
            <circle cx="250" cy="155" r="3.0" fill="#38bdf8" class="p3"/>
            <text x="254" y="140" fill="#38bdf8" font-size="9" font-weight="bold" class="p-text1">01</text>
            <text x="238" y="120" fill="#f59e0b" font-size="9" font-weight="bold" class="p-text2">DATA</text>
            <text x="254" y="95" fill="#38bdf8" font-size="9" font-weight="bold" class="p-text3">101</text>
            <text x="236" y="70" fill="#10b981" font-size="9" font-weight="bold" class="p-text4">OIL</text>
        </g>

        <line x1="175" y1="135" x2="235" y2="60" stroke="#64748b" stroke-width="3.5"/>
        <line x1="325" y1="135" x2="265" y2="60" stroke="#64748b" stroke-width="3.5"/>
        <line x1="250" y1="135" x2="250" y2="60" stroke="#475569" stroke-width="2" stroke-dasharray="3 3"/>
        
        <line x1="190" y1="115" x2="310" y2="115" stroke="#475569" stroke-width="2"/>
        <line x1="205" y1="95" x2="295" y2="95" stroke="#475569" stroke-width="2"/>
        <line x1="220" y1="75" x2="280" y2="75" stroke="#475569" stroke-width="2"/>

        <polygon points="240,135 260,135 254,60 246,60" fill="#475569"/>

        <g class="pump-walking-beam">
            <polygon points="170,55 320,55 315,65 175,65" fill="url(#beamGrad)" filter="url(#glow)"/>
            <path d="M 170 42 C 158 48 156 65 172 72 Z" fill="#0284c7"/>
            <line x1="162" y1="60" x2="162" y2="128" stroke="#cbd5e1" stroke-width="2" class="polished-rod"/>
        </g>

        <g>
            <circle cx="305" cy="90" r="14" fill="#0f172a" stroke="#38bdf8" stroke-width="2.5"/>
            <line x1="305" y1="90" x2="315" y2="60" stroke="#f59e0b" stroke-width="3"/>
        </g>

        <rect x="238" y="125" width="24" height="10" fill="#334155" rx="2"/>
        <rect x="242" y="121" width="16" height="5" fill="#64748b"/>

        <g opacity="0.85">
            <circle cx="90" cy="40" r="18" fill="#0284c7" opacity="0.15"/>
            <text x="90" y="44" fill="#38bdf8" font-size="10" font-weight="bold" text-anchor="middle">DATAFRAME</text>
            
            <circle cx="410" cy="40" r="18" fill="#f59e0b" opacity="0.15"/>
            <text x="410" y="44" fill="#f59e0b" font-size="10" font-weight="bold" text-anchor="middle">DUCKDB</text>
        </g>
    </svg>

    <div class="oil-tagline-text">"DATA IS THE NEW OIL"</div>
    <div class="oil-sub-text">Mining raw schemas into high-value synthetic data & SCD2 intelligence</div>
</div>
</body>
</html>"""
    components.html(html_code, height=270, scrolling=False)



# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Synthetic Data Generator", page_icon="🧪", layout="wide")
st.title("🧪 Data Contract → Synthetic Data Generator")
st.caption("ODCS contract → referentially-consistent fake data + SCD2 change history in DuckDB.")

render_oil_mining_animation()


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
