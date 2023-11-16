import copy
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping
import dpath.util
import pendulum
from airbyte_protocol.models import AirbyteRecordMessage, ConfiguredAirbyteCatalog
from jsonschema import Draft7Validator, FormatChecker, FormatError, ValidationError, validators
timestamp_regex = re.compile('^\\d{4}-\\d?\\d-\\d?\\d(\\s|T)\\d?\\d:\\d?\\d:\\d?\\d(.\\d+)?.*$')
strict_integer_type_checker = Draft7Validator.TYPE_CHECKER.redefine('integer', lambda _, value: isinstance(value, int))
Draft7ValidatorWithStrictInteger = validators.extend(Draft7Validator, type_checker=strict_integer_type_checker)

class CustomFormatChecker(FormatChecker):

    @staticmethod
    def check_datetime(value: str) -> bool:
        if False:
            i = 10
            return i + 15
        valid_format = timestamp_regex.match(value)
        try:
            pendulum.parse(value, strict=False)
        except ValueError:
            valid_time = False
        else:
            valid_time = True
        return valid_format and valid_time

    def check(self, instance, format):
        if False:
            i = 10
            return i + 15
        if instance is not None and format == 'date-time':
            if not self.check_datetime(instance):
                raise FormatError(f'{instance} has invalid datetime format')
        else:
            return super().check(instance, format)

def _enforce_no_additional_top_level_properties(json_schema: Dict[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    'Create a copy of the schema in which `additionalProperties` is set to False for the dict of top-level properties.\n\n    This method will override the value of `additionalProperties` if it is set,\n    or will create the property and set it to False if it does not exist.\n    '
    enforced_schema = copy.deepcopy(json_schema)
    dpath.util.new(enforced_schema, 'additionalProperties', False)
    return enforced_schema

def verify_records_schema(records: List[AirbyteRecordMessage], catalog: ConfiguredAirbyteCatalog, fail_on_extra_columns: bool) -> Mapping[str, Mapping[str, ValidationError]]:
    if False:
        while True:
            i = 10
    'Check records against their schemas from the catalog, yield error messages.\n    Only first record with error will be yielded for each stream.\n    '
    stream_validators = {}
    for stream in catalog.streams:
        schema_to_validate_against = stream.stream.json_schema
        if fail_on_extra_columns:
            schema_to_validate_against = _enforce_no_additional_top_level_properties(schema_to_validate_against)
        stream_validators[stream.stream.name] = Draft7ValidatorWithStrictInteger(schema_to_validate_against, format_checker=CustomFormatChecker())
    stream_errors = defaultdict(dict)
    for record in records:
        validator = stream_validators.get(record.stream)
        if not validator:
            logging.error(f'Received record from the `{record.stream}` stream, which is not in the catalog.')
            continue
        errors = list(validator.iter_errors(record.data))
        for error in errors:
            stream_errors[record.stream][str(error.schema_path)] = error
    return stream_errors