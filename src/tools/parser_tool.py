import json
import yaml
from pathlib import Path
from crewai.tools import tool


@tool("Parse ODCS Contract")
def parse_odcs_contract(contract_path: str) -> str:
    """
    Parse an Open Data Contract Standard (ODCS) v3.1.0 contract YAML file using
    the datacontract-cli Python library for validation, then extract the full
    schema summary (tables, fields, types, constraints, FK relationships,
    dependency order, SCD2 change-tracked fields).
    Returns a JSON summary. Lint warnings are included but do not block parsing.
    """
    from datacontract.data_contract import DataContract

    path = Path(contract_path)
    if not path.exists():
        return json.dumps({"error": f"Contract not found: {contract_path}"})

    # --- datacontract-cli: validate the ODCS v3.1.0 contract ---
    dc = DataContract(data_contract_file=str(path))
    lint_result = dc.lint()
    lint_warnings = [
        str(c) for c in lint_result.checks if not c.passed
    ] if not lint_result.passed else []

    # --- raw YAML: access full field data including x- custom extensions ---
    with open(path) as f:
        raw = yaml.safe_load(f)

    api_version = raw.get("apiVersion", "unknown")
    if not api_version.startswith("v3"):
        lint_warnings.append(
            f"Expected apiVersion v3.1.0, found '{api_version}'. "
            "Some features may not work correctly."
        )

    models_raw: dict = raw.get("models", {})

    # Build dependency order from FK references
    dependency_order = _topological_sort(models_raw)

    summary: dict = {
        "id": raw.get("id", "unknown"),
        "apiVersion": api_version,
        "title": raw.get("info", {}).get("title", ""),
        "lint_passed": lint_result.passed,
        "lint_warnings": lint_warnings,
        "dependency_order": dependency_order,
        "tables": {},
    }

    for table_name, table_raw in models_raw.items():
        fields_raw: dict = table_raw.get("fields", {})
        pk     = _find_primary_key(fields_raw)
        fks    = _find_foreign_keys(fields_raw)
        quality = table_raw.get("quality", [])

        summary["tables"][table_name] = {
            "primary_key": pk,
            "foreign_keys": fks,
            # scd2_tracked_fields is now inferred by the Distribution Analyst LLM
            # from field names, descriptions, and types — not from custom annotations
            "quality_rules": quality,
            "fields": {
                fname: {
                    "type":        fdata.get("type"),
                    "description": fdata.get("description"),
                    "required":    fdata.get("required", False),
                    "primaryKey":  fdata.get("primaryKey", False),
                    "unique":      fdata.get("unique", False),
                    "enum":        fdata.get("enum"),
                    "pattern":     fdata.get("pattern"),
                    "format":      fdata.get("format"),
                    "minimum":     fdata.get("minimum") or fdata.get("exclusiveMinimum"),
                    "maximum":     fdata.get("maximum") or fdata.get("exclusiveMaximum"),
                    "references":  fdata.get("references"),
                    "default":     fdata.get("default"),
                }
                for fname, fdata in fields_raw.items()
            },
        }

    return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# Helpers — operate on the raw YAML dict so they work with any ODCS version
# ---------------------------------------------------------------------------

def _find_primary_key(fields: dict) -> str | None:
    for fname, fdata in fields.items():
        if fdata.get("primaryKey"):
            return fname
    return None


def _find_foreign_keys(fields: dict) -> dict:
    return {
        fname: fdata["references"]
        for fname, fdata in fields.items()
        if fdata.get("references")
    }


def _topological_sort(models: dict) -> list:
    deps: dict = {t: set() for t in models}
    for table_name, table_raw in models.items():
        for fdata in table_raw.get("fields", {}).values():
            ref = fdata.get("references", "")
            if ref:
                parent = ref.split(".")[0]
                if parent in deps:
                    deps[table_name].add(parent)

    ordered, visited = [], set()

    def visit(t):
        if t in visited:
            return
        visited.add(t)
        for dep in deps.get(t, set()):
            visit(dep)
        ordered.append(t)

    for t in models:
        visit(t)
    return ordered
