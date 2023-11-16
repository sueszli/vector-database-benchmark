"""Utility methods for customization args of interactions."""
from __future__ import annotations
from core import schema_utils
from core import utils
from core.domain import state_domain
from extensions import domain
from typing import Dict, List, Mapping, Optional, Union
MYPY = False
if MYPY:
    AcceptableCustomizationArgsTypes = Union[str, int, bool, List[str], domain.GraphDict, Dict[str, Optional[str]], List[state_domain.SubtitledHtml], List[state_domain.SubtitledHtmlDict], state_domain.SubtitledHtmlDict, state_domain.SubtitledUnicodeDict, state_domain.SubtitledUnicode, domain.ImageAndRegionDict, domain.CustomizationArgSubtitledUnicodeDefaultDict, List[domain.CustomizationArgSubtitledUnicodeDefaultDict], None]
    CustomizationArgsDictType = Mapping[str, Mapping[str, AcceptableCustomizationArgsTypes]]

def validate_customization_args_and_values(item_name: str, item_type: str, customization_args: CustomizationArgsDictType, ca_specs_to_validate_against: List[domain.CustomizationArgSpec], fail_on_validation_errors: bool=False) -> None:
    if False:
        while True:
            i = 10
    "Validates the given `customization_args` dict against the specs set\n    out in 'ca_specs_to_validate_against'. 'item_name' and 'item_type' are\n    used to populate any error messages that arise during validation.\n    Note that this may modify the given customization_args dict, if it has\n    extra keys. It also normalizes any HTML in the customization_args dict.\n\n    Args:\n        item_name: str. This is always 'interaction'.\n        item_type: str. The item_type is the ID of the interaction.\n        customization_args: dict. The customization dict. The keys are names\n            of customization_args and the values are dicts with a\n            single key, 'value', whose corresponding value is the value of\n            the customization arg.\n        ca_specs_to_validate_against: list(dict). List of spec dictionaries.\n            Is used to check if some keys are missing in customization_args.\n            Dicts have the following structure:\n                - name: str. The customization variable name.\n                - description: str. The customization variable description.\n                - default_value: *. The default value of the customization\n                    variable.\n        fail_on_validation_errors: bool. Whether to raise errors if\n            validation fails for customization args.\n\n    Raises:\n        ValidationError. The given 'customization_args' is not valid.\n        ValidationError. The given 'customization_args' is missing at least one\n            key.\n    "
    ca_spec_names = [ca_spec.name for ca_spec in ca_specs_to_validate_against]
    if not isinstance(customization_args, dict):
        raise utils.ValidationError('Expected customization args to be a dict, received %s' % customization_args)
    for arg_name in customization_args.keys():
        if not isinstance(arg_name, str):
            raise utils.ValidationError('Invalid customization arg name: %s' % arg_name)
        if arg_name not in ca_spec_names:
            raise utils.ValidationError('%s %s does not support customization arg %s.' % (item_name.capitalize(), item_type, arg_name))
    for ca_spec in ca_specs_to_validate_against:
        if ca_spec.name not in customization_args:
            raise utils.ValidationError('Customization argument is missing key: %s' % ca_spec.name)
        try:
            customization_args[ca_spec.name]['value'] = schema_utils.normalize_against_schema(customization_args[ca_spec.name]['value'], ca_spec.schema)
        except Exception as e:
            if fail_on_validation_errors:
                raise utils.ValidationError(e)