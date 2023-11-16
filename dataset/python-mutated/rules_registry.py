"""Registry for rules and their related specification files."""
from __future__ import annotations
import json
import os
from core import constants
from core import feconf
from typing import Dict, List, Optional, TypedDict

class RuleSpecsExtensionDict(TypedDict):
    """Dictionary representation of rule specs of an extension."""
    interactionId: str
    format: str
    ruleTypes: Dict[str, Dict[str, List[str]]]

class Registry:
    """Registry of rules."""
    _state_schema_version_to_html_field_types_to_rule_specs: Dict[Optional[int], Dict[str, RuleSpecsExtensionDict]] = {}

    @classmethod
    def get_html_field_types_to_rule_specs(cls, state_schema_version: Optional[int]=None) -> Dict[str, RuleSpecsExtensionDict]:
        if False:
            print('Hello World!')
        "Returns a dict containing a html_field_types_to_rule_specs dict of\n        the specified state schema version, if available.\n\n        Args:\n            state_schema_version: int|None. The state schema version to retrieve\n                the html_field_types_to_rule_specs for. If None, the current\n                state schema version's html_field_types_to_rule_specs will be\n                returned.\n\n        Returns:\n            dict. The html_field_types_to_rule_specs specs for the given state\n            schema version.\n\n        Raises:\n            Exception. No html_field_types_to_rule_specs json file found for the\n                given state schema version.\n        "
        specs_from_json: Dict[str, RuleSpecsExtensionDict] = {}
        cached = state_schema_version in cls._state_schema_version_to_html_field_types_to_rule_specs
        if not cached:
            if state_schema_version is None:
                specs_from_json = json.loads(constants.get_package_file_contents('extensions', feconf.HTML_FIELD_TYPES_TO_RULE_SPECS_EXTENSIONS_MODULE_PATH))
                cls._state_schema_version_to_html_field_types_to_rule_specs[state_schema_version] = specs_from_json
            else:
                file_name = 'html_field_types_to_rule_specs_state_v%i.json' % state_schema_version
                spec_file = os.path.join(feconf.LEGACY_HTML_FIELD_TYPES_TO_RULE_SPECS_EXTENSIONS_MODULE_DIR, file_name)
                try:
                    specs_from_json = json.loads(constants.get_package_file_contents('extensions', spec_file))
                except Exception as e:
                    raise Exception('No specs json file found for state schema v%i' % state_schema_version) from e
                cls._state_schema_version_to_html_field_types_to_rule_specs[state_schema_version] = specs_from_json
        return cls._state_schema_version_to_html_field_types_to_rule_specs[state_schema_version]