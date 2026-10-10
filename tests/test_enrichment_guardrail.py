import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from src.enrichment_guardrail import (
    _load_contract,
    _required_strategy,
    apply_jev_strategy_choices,
    validate_enrichment_plan,
)


def _valid_plan(contract):
    generation = {}
    tracking = {}
    patterns = {}
    for table_name, table in contract["models"].items():
        generation[table_name] = {}
        append_only = "append-only" in table.get("description", "").lower()
        tracked = []
        for field_name, field in table["fields"].items():
            strategy = _required_strategy(field_name, field)
            if strategy is None:
                strategy = (
                    "price"
                    if field.get("type") in {"number", "float", "double", "decimal"}
                    else "word"
                )
            hint = {"inferred_faker": strategy}
            if field.get("enum"):
                hint["enum_weights"] = [1] * len(field["enum"])
            generation[table_name][field_name] = hint
            if (
                field.get("x-change-tracking")
                and not append_only
                and not field.get("primaryKey")
                and not field.get("unique")
                and not field.get("references")
            ):
                tracked.append(field_name)
        tracking[table_name] = tracked
        if tracked:
            patterns[table_name] = {
                "field_change_frequency": {name: 0.2 for name in tracked},
                "co_change_patterns": [],
            }
    return {
        "generation_hints": generation,
        "change_tracking": tracking,
        "change_patterns": patterns,
    }


class EnrichmentGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.contract = _load_contract("contracts/ecommerce_contract.yaml")
        self.plan = _valid_plan(self.contract)

    def test_valid_plan_passes(self):
        self.assertEqual(validate_enrichment_plan(self.plan, self.contract), [])

    def test_unsupported_strategy_fails(self):
        plan = copy.deepcopy(self.plan)
        plan["generation_hints"]["customers"]["first_name"][
            "inferred_faker"
        ] = "invented_provider"
        errors = validate_enrichment_plan(plan, self.contract)
        self.assertTrue(any("unsupported" in error for error in errors))

    def test_constraint_strategy_mismatch_fails(self):
        plan = copy.deepcopy(self.plan)
        plan["generation_hints"]["customers"]["customer_id"][
            "inferred_faker"
        ] = "integer"
        errors = validate_enrichment_plan(plan, self.contract)
        self.assertTrue(any("must be 'sequence'" in error for error in errors))

    def test_invalid_change_frequency_fails(self):
        plan = copy.deepcopy(self.plan)
        plan["change_patterns"]["customers"]["field_change_frequency"]["phone"] = 1.2
        errors = validate_enrichment_plan(plan, self.contract)
        self.assertTrue(any("must be between 0 and 1" in error for error in errors))

    @patch("src.enrichment_guardrail.urllib.request.urlopen")
    def test_high_confidence_jev_choice_updates_ambiguous_field(self, urlopen):
        response = MagicMock()
        response.read.return_value = json.dumps({
            "answers": {
                "field_0": {
                    "type": "choice",
                    "choice": "first_name",
                    "confidence": 0.95,
                }
            }
        }).encode()
        urlopen.return_value.__enter__.return_value = response

        with patch.dict("os.environ", {"JEV_API_KEY": "test-key"}):
            result = apply_jev_strategy_choices(self.plan, self.contract)

        self.assertEqual(
            result["generation_hints"]["customers"]["first_name"]["inferred_faker"],
            "first_name",
        )


if __name__ == "__main__":
    unittest.main()
