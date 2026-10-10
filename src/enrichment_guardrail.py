"""Deterministic and optional Jev validation for enrichment plans.

The deterministic checks protect generator invariants.  Jev is deliberately
limited to ambiguous semantic mappings and is never trusted with structural
constraints such as primary keys, foreign keys, enums, or data types.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import yaml

from src.tools.generator_tool import SUPPORTED_FAKER_STRATEGIES


REQUIRED_SECTIONS = {"generation_hints", "change_tracking", "change_patterns"}
AUDIT_FIELDS = {"created_at", "created_by", "updated_at", "updated_by"}
JEV_API_URL = "https://api.typesafe.ai/v1/systemone"

_JEV_STRING_CHOICES = {
    "word": "A short generic word or label.",
    "name": "A full person's name.",
    "first_name": "A person's given name.",
    "last_name": "A person's family name.",
    "email": "An email address.",
    "phone_number": "A telephone number.",
    "street_address": "A street or postal address.",
    "city": "A city name.",
    "country_code": "An ISO-style country code.",
    "company": "A company or organization name.",
    "product_name": "A product, campaign, or commercial display name.",
}


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a single Markdown JSON fence."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("enrichment output must be a JSON object")
    return value


def _load_contract(contract_path: str) -> dict[str, Any]:
    with Path(contract_path).open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict) or not isinstance(contract.get("models"), dict):
        raise ValueError("contract must contain an ODCS 'models' object")
    return contract


def _required_strategy(field_name: str, field: dict[str, Any]) -> str | None:
    """Return a non-negotiable strategy when the contract determines it."""
    name = field_name.lower()
    if field.get("primaryKey"):
        return "sequence"
    if field.get("references"):
        return "foreign_key"
    if field.get("enum"):
        return "enum"
    if field.get("pattern"):
        return "regex"
    if field.get("format") == "email":
        return "email"
    if name == "total_amount":
        return "computed"
    if name == "unit_price_at_order":
        return "derived_from_parent"

    field_type = field.get("type")
    if field_type == "boolean":
        return "boolean"
    if field_type in {"integer", "long"}:
        return "integer"
    if field_type == "date":
        return "past_date"
    if field_type in {"timestamp", "timestamp_tz"}:
        return "past_datetime"
    return None


def validate_enrichment_plan(
    plan: dict[str, Any], contract: dict[str, Any]
) -> list[str]:
    """Return precise validation errors; an empty list means the plan is safe."""
    errors: list[str] = []
    actual_sections = set(plan)
    if actual_sections != REQUIRED_SECTIONS:
        missing = sorted(REQUIRED_SECTIONS - actual_sections)
        extra = sorted(actual_sections - REQUIRED_SECTIONS)
        if missing:
            errors.append(f"missing top-level sections: {missing}")
        if extra:
            errors.append(f"unexpected top-level sections: {extra}")

    generation = plan.get("generation_hints")
    tracking = plan.get("change_tracking")
    patterns = plan.get("change_patterns")
    if not isinstance(generation, dict):
        errors.append("generation_hints must be an object")
        generation = {}
    if not isinstance(tracking, dict):
        errors.append("change_tracking must be an object")
        tracking = {}
    if not isinstance(patterns, dict):
        errors.append("change_patterns must be an object")
        patterns = {}

    models = contract.get("models", {})
    for table_name, table in models.items():
        table_hints = generation.get(table_name)
        if not isinstance(table_hints, dict):
            errors.append(f"generation_hints.{table_name} must be an object")
            table_hints = {}

        fields = table.get("fields", {})
        unknown_fields = sorted(set(table_hints) - set(fields))
        if unknown_fields:
            errors.append(
                f"generation_hints.{table_name} contains unknown fields: {unknown_fields}"
            )
        for field_name, field in fields.items():
            path = f"generation_hints.{table_name}.{field_name}"
            hint = table_hints.get(field_name)
            if not isinstance(hint, dict):
                errors.append(f"{path} is missing or is not an object")
                continue

            strategy = hint.get("inferred_faker")
            if strategy not in SUPPORTED_FAKER_STRATEGIES:
                errors.append(
                    f"{path}.inferred_faker '{strategy}' is unsupported; choose one of "
                    f"{sorted(SUPPORTED_FAKER_STRATEGIES)}"
                )
            required = _required_strategy(field_name, field)
            if required and strategy != required:
                errors.append(
                    f"{path}.inferred_faker must be '{required}' because of the field constraints"
                )

            weights = hint.get("enum_weights")
            enum = field.get("enum")
            if weights is not None:
                if not enum:
                    errors.append(f"{path}.enum_weights is set on a non-enum field")
                elif not isinstance(weights, list) or len(weights) != len(enum):
                    errors.append(
                        f"{path}.enum_weights must contain exactly {len(enum)} values"
                    )
                elif (
                    any(not isinstance(weight, (int, float)) or weight < 0 for weight in weights)
                    or sum(weights) <= 0
                ):
                    errors.append(
                        f"{path}.enum_weights must be non-negative numbers with a positive total"
                    )

        tracked = tracking.get(table_name, [])
        if not isinstance(tracked, list):
            errors.append(f"change_tracking.{table_name} must be a list")
            tracked = []
        append_only = "append-only" in str(table.get("description", "")).lower()
        if append_only and tracked:
            errors.append(f"change_tracking.{table_name} must be empty for an append-only table")
        for field_name in tracked:
            field = fields.get(field_name)
            path = f"change_tracking.{table_name}"
            if field is None:
                errors.append(f"{path} references unknown field '{field_name}'")
            elif (
                field.get("primaryKey")
                or field.get("unique")
                or field.get("references")
                or field_name.lower() in AUDIT_FIELDS
            ):
                errors.append(f"{path} contains ineligible field '{field_name}'")

        table_patterns = patterns.get(table_name, {})
        if table_patterns and not isinstance(table_patterns, dict):
            errors.append(f"change_patterns.{table_name} must be an object")
            continue
        if not isinstance(table_patterns, dict):
            continue
        field_frequencies = table_patterns.get("field_change_frequency", {})
        if not isinstance(field_frequencies, dict):
            errors.append(
                f"change_patterns.{table_name}.field_change_frequency must be an object"
            )
            field_frequencies = {}
        for field_name, frequency in field_frequencies.items():
            if field_name not in tracked:
                errors.append(
                    f"change_patterns.{table_name}.field_change_frequency references "
                    f"untracked field '{field_name}'"
                )
            if not isinstance(frequency, (int, float)) or not 0 <= frequency <= 1:
                errors.append(
                    f"change_patterns.{table_name}.field_change_frequency.{field_name} "
                    "must be between 0 and 1"
                )
        co_changes = table_patterns.get("co_change_patterns", [])
        if not isinstance(co_changes, list):
            errors.append(f"change_patterns.{table_name}.co_change_patterns must be a list")
            co_changes = []
        for index, co_change in enumerate(co_changes):
            co_fields = co_change.get("fields", []) if isinstance(co_change, dict) else []
            frequency = co_change.get("frequency") if isinstance(co_change, dict) else None
            unknown = [field for field in co_fields if field not in tracked]
            if unknown:
                errors.append(
                    f"change_patterns.{table_name}.co_change_patterns[{index}] references "
                    f"untracked fields {unknown}"
                )
            if not isinstance(frequency, (int, float)) or not 0 <= frequency <= 1:
                errors.append(
                    f"change_patterns.{table_name}.co_change_patterns[{index}].frequency "
                    "must be between 0 and 1"
                )

    unknown_tables = sorted(set(generation) - set(models))
    if unknown_tables:
        errors.append(f"generation_hints contains unknown tables: {unknown_tables}")
    unknown_tracking_tables = sorted(set(tracking) - set(models))
    if unknown_tracking_tables:
        errors.append(f"change_tracking contains unknown tables: {unknown_tracking_tables}")
    unknown_pattern_tables = sorted(set(patterns) - set(models))
    if unknown_pattern_tables:
        errors.append(f"change_patterns contains unknown tables: {unknown_pattern_tables}")
    return errors


def _ambiguous_string_fields(
    plan: dict[str, Any], contract: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    ambiguous = []
    for table_name, table in contract.get("models", {}).items():
        for field_name, field in table.get("fields", {}).items():
            if field.get("type") != "string" or _required_strategy(field_name, field):
                continue
            ambiguous.append((table_name, field_name, field))
    return ambiguous


def apply_jev_strategy_choices(
    plan: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ambiguous string strategies with one optional, batched Jev call.

    With no JEV_API_KEY, or on a transient Jev error, the already validated plan
    is returned unchanged. Low-confidence choices are likewise ignored.
    """
    api_key = os.getenv("JEV_API_KEY", "").strip()
    if not api_key:
        return plan

    ambiguous = _ambiguous_string_fields(plan, contract)
    if not ambiguous:
        return plan

    state_fields: dict[str, Any] = {}
    questions: dict[str, Any] = {}
    question_paths: dict[str, tuple[str, str]] = {}
    for index, (table_name, field_name, field) in enumerate(ambiguous):
        question_id = f"field_{index}"
        question_paths[question_id] = (table_name, field_name)
        state_fields[question_id] = {
            "table": table_name,
            "field": field_name,
            "description": field.get("description"),
            "current_strategy": plan["generation_hints"][table_name][field_name].get(
                "inferred_faker"
            ),
        }
        questions[question_id] = {
            "type": "choice",
            "instructions": (
                f"Choose the Faker strategy that best represents `fields.{question_id}`. "
                "Use only the field name and description; prefer word when no semantic "
                "provider clearly applies."
            ),
            "criteria": _JEV_STRING_CHOICES,
        }

    body = json.dumps({
        "state": {"fields": state_fields},
        "model": os.getenv("JEV_MODEL", "jev-latest"),
        "questions": questions,
    }).encode("utf-8")
    request = urllib.request.Request(
        JEV_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        timeout = float(os.getenv("JEV_TIMEOUT_SECONDS", "15"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        warnings.warn(
            f"Jev strategy resolution failed; preserving the validated draft: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return plan

    try:
        minimum_confidence = float(os.getenv("JEV_MIN_CONFIDENCE", "0.75"))
    except ValueError:
        minimum_confidence = 0.75
    for question_id, answer in result.get("answers", {}).items():
        path = question_paths.get(question_id)
        if not path or not isinstance(answer, dict):
            continue
        choice = answer.get("choice")
        confidence = answer.get("confidence", 0)
        if (
            choice in _JEV_STRING_CHOICES
            and isinstance(confidence, (int, float))
            and confidence >= minimum_confidence
        ):
            table_name, field_name = path
            plan["generation_hints"][table_name][field_name]["inferred_faker"] = choice
    return plan


def make_enrichment_guardrail(contract_path: str, *, use_jev: bool = True):
    """Build a CrewAI-compatible guardrail bound to one contract path.

    ``use_jev`` is enabled for the initial draft and disabled for the final
    refinement so one pipeline run makes at most one batched Jev request.
    """
    contract = _load_contract(contract_path)

    # CrewAI validates this callable dynamically.  Leave its return annotation
    # unspecified for compatibility across CrewAI/Pydantic releases.
    def guardrail(output):
        try:
            plan = _extract_json(output.raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return False, f"Return only valid enrichment JSON: {exc}"

        errors = validate_enrichment_plan(plan, contract)
        if errors:
            feedback = "Enrichment validation failed:\n- " + "\n- ".join(errors[:30])
            return False, feedback

        if use_jev:
            plan = apply_jev_strategy_choices(plan, contract)
        return True, json.dumps(plan, indent=2)

    return guardrail
