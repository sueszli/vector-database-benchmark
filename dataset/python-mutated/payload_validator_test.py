"""Tests for various errors raised by validate method of payload validator."""
from __future__ import annotations
from core.controllers import payload_validator
from core.tests import test_utils
from typing import Any, Dict, List, Tuple

class PayloadValidationUnitTests(test_utils.GenericTestBase):

    def test_invalid_args_raises_exceptions(self) -> None:
        if False:
            return 10
        list_of_invalid_args_with_schema_and_errors: List[Tuple[Dict[str, Any], Dict[str, Any], List[str]]] = [({'exploration_id': 2}, {'exploration_id': {'schema': {'type': 'basestring'}}}, ["Schema validation for 'exploration_id' failed: Expected string, received 2"]), ({'version': 'random_string'}, {'version': {'schema': {'type': 'int'}}}, ["Schema validation for 'version' failed: Could not convert str to int: random_string"]), ({'exploration_id': 'any_exp_id'}, {}, ["Found extra args: ['exploration_id']."]), ({}, {'exploration_id': {'schema': {'type': 'basestring'}}}, ['Missing key in handler args: exploration_id.'])]
        for (handler_args, handler_args_schema, error_msg) in list_of_invalid_args_with_schema_and_errors:
            (normalized_value, errors) = payload_validator.validate_arguments_against_schema(handler_args, handler_args_schema, allowed_extra_args=False, allow_string_to_bool_conversion=False)
            self.assertEqual(normalized_value, {})
            self.assertEqual(error_msg, errors)

    def test_valid_args_do_not_raises_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        list_of_valid_args_with_schema: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = [({}, {'exploration_id': {'schema': {'type': 'basestring'}, 'default_value': None}}, {}), ({}, {'exploration_id': {'schema': {'type': 'basestring'}, 'default_value': 'default_exp_id'}}, {'exploration_id': 'default_exp_id'}), ({'exploration_id': 'any_exp_id'}, {'exploration_id': {'schema': {'type': 'basestring'}}}, {'exploration_id': 'any_exp_id'}), ({'apply_draft': 'true'}, {'apply_draft': {'schema': {'type': 'bool', 'new_key_for_argument': 'new_key_for_apply_draft'}}}, {'new_key_for_apply_draft': True})]
        for (handler_args, handler_args_schema, normalized_value_for_args) in list_of_valid_args_with_schema:
            (normalized_value, errors) = payload_validator.validate_arguments_against_schema(handler_args, handler_args_schema, allowed_extra_args=False, allow_string_to_bool_conversion=True)
            self.assertEqual(normalized_value, normalized_value_for_args)
            self.assertEqual(errors, [])

class CheckConversionOfStringToBool(test_utils.GenericTestBase):
    """Test class to check behaviour of convert_string_to_bool method."""

    def test_convert_string_to_bool(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test case to check behaviour of convert_string_to_bool method.'
        self.assertTrue(payload_validator.convert_string_to_bool('true'))
        self.assertFalse(payload_validator.convert_string_to_bool('false'))
        self.assertEqual(payload_validator.convert_string_to_bool('any_other_value'), 'any_other_value')

class CheckGetCorrespondingKeyForObjectMethod(test_utils.GenericTestBase):
    """Test class to check behaviour of get_corresponding_key_for_object
    method."""

    def test_get_new_arg_key_from_schema(self) -> None:
        if False:
            while True:
                i = 10
        'Test case to check behaviour of new arg key name.'
        sample_arg_schema = {'schema': {'new_key_for_argument': 'sample_new_arg_name'}}
        new_key_name = payload_validator.get_corresponding_key_for_object(sample_arg_schema)
        self.assertEqual(new_key_name, 'sample_new_arg_name')

class CheckGetSchemaTypeMethod(test_utils.GenericTestBase):
    """Test class to check behaviour of get_schema_type method."""

    def test_get_schema_type_from_schema(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test case to check behaviour of get_schema_type method.'
        sample_arg_schema = {'schema': {'type': 'bool'}}
        schema_type = payload_validator.get_schema_type(sample_arg_schema)
        self.assertEqual(schema_type, 'bool')