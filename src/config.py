import os
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

# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
# The pipeline uses two tiers of model:
#   - "fast"  : cheap, high-throughput model for the deterministic tool-calling
#               agents (parse, profile, analyze, generate, simulate).
#   - "smart" : stronger reasoning model for the analytical agents
#               (distribution/change analyst, validation analyst).
#
# Switch providers with the LLM_PROVIDER env var ("anthropic", "gemini", or "openai").
# CrewAI's LLM class is backed by LiteLLM, so provider-prefixed model ids
# ("gemini/...") work out of the box given the right API key.
#
# Required API keys:
#   - anthropic : ANTHROPIC_API_KEY
#   - gemini    : GEMINI_API_KEY
#   - openai    : OPENAI_API_KEY
#
# You can override the individual model ids per tier with FAST_MODEL /
# SMART_MODEL without touching this file.

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

# Friendly aliases → canonical provider key.
_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-genai": "gemini",
    "claude": "anthropic",
    "gpt": "openai",
    "chatgpt": "openai",
}
LLM_PROVIDER = _PROVIDER_ALIASES.get(LLM_PROVIDER, LLM_PROVIDER)

_PROVIDER_MODELS = {
    "anthropic": {
        "fast":  "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-5",
    },
    "gemini": {
        # "latest" aliases always resolve to a currently-available model —
        # pinned 2.5-* ids get gated ("no longer available to new users").
        "fast":  "gemini/gemini-flash-latest",
        "smart": "gemini/gemini-pro-latest",
    },
    "openai": {
        "fast":  "gpt-5-nano",
        "smart": "gpt-5-mini",
    },
}

if LLM_PROVIDER not in _PROVIDER_MODELS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
        f"Choose one of: {', '.join(_PROVIDER_MODELS)}."
    )

FAST_MODEL  = os.getenv("FAST_MODEL",  _PROVIDER_MODELS[LLM_PROVIDER]["fast"])
SMART_MODEL = os.getenv("SMART_MODEL", _PROVIDER_MODELS[LLM_PROVIDER]["smart"])
