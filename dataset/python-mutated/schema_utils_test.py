"""Tests for object schema definitions."""
from __future__ import annotations
import inspect
import re
from core import schema_utils
from core.tests import test_utils
from typing import Any, Dict, List, Tuple
SCHEMA_KEY_ITEMS = schema_utils.SCHEMA_KEY_ITEMS
SCHEMA_KEY_LEN = schema_utils.SCHEMA_KEY_LEN
SCHEMA_KEY_PROPERTIES = schema_utils.SCHEMA_KEY_PROPERTIES
SCHEMA_KEY_TYPE = schema_utils.SCHEMA_KEY_TYPE
SCHEMA_KEY_KEYS = schema_utils.SCHEMA_KEY_KEYS
SCHEMA_KEY_VALUES = schema_utils.SCHEMA_KEY_VALUES
SCHEMA_KEY_POST_NORMALIZERS = schema_utils.SCHEMA_KEY_POST_NORMALIZERS
SCHEMA_KEY_CHOICES = schema_utils.SCHEMA_KEY_CHOICES
SCHEMA_KEY_NAME = schema_utils.SCHEMA_KEY_NAME
SCHEMA_KEY_SCHEMA = schema_utils.SCHEMA_KEY_SCHEMA
SCHEMA_KEY_OBJ_TYPE = schema_utils.SCHEMA_KEY_OBJ_TYPE
SCHEMA_KEY_VALIDATORS = schema_utils.SCHEMA_KEY_VALIDATORS
SCHEMA_KEY_DEFAULT_VALUE = schema_utils.SCHEMA_KEY_DEFAULT_VALUE
SCHEMA_KEY_VALIDATION_METHOD = schema_utils.SCHEMA_KEY_VALIDATION_METHOD
SCHEMA_KEY_OBJECT_CLASS = schema_utils.SCHEMA_KEY_OBJECT_CLASS
SCHEMA_KEY_DESCRIPTION = 'description'
SCHEMA_KEY_UI_CONFIG = 'ui_config'
SCHEMA_KEY_REPLACEMENT_UI_CONFIG = 'replacement_ui_config'
OPTIONAL_SCHEMA_KEYS = [SCHEMA_KEY_CHOICES, SCHEMA_KEY_POST_NORMALIZERS, SCHEMA_KEY_UI_CONFIG, SCHEMA_KEY_VALIDATORS, SCHEMA_KEY_DEFAULT_VALUE]
SCHEMA_TYPE_BOOL = schema_utils.SCHEMA_TYPE_BOOL
SCHEMA_TYPE_CUSTOM = schema_utils.SCHEMA_TYPE_CUSTOM
SCHEMA_TYPE_DICT = schema_utils.SCHEMA_TYPE_DICT
SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS = schema_utils.SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS
SCHEMA_TYPE_FLOAT = schema_utils.SCHEMA_TYPE_FLOAT
SCHEMA_TYPE_HTML = schema_utils.SCHEMA_TYPE_HTML
SCHEMA_TYPE_INT = schema_utils.SCHEMA_TYPE_INT
SCHEMA_TYPE_LIST = schema_utils.SCHEMA_TYPE_LIST
SCHEMA_TYPE_BASESTRING = schema_utils.SCHEMA_TYPE_BASESTRING
SCHEMA_TYPE_UNICODE = schema_utils.SCHEMA_TYPE_UNICODE
SCHEMA_TYPE_UNICODE_OR_NONE = schema_utils.SCHEMA_TYPE_UNICODE_OR_NONE
SCHEMA_TYPE_OBJECT_DICT = schema_utils.SCHEMA_TYPE_OBJECT_DICT
ALLOWED_SCHEMA_TYPES = [SCHEMA_TYPE_BOOL, SCHEMA_TYPE_CUSTOM, SCHEMA_TYPE_DICT, SCHEMA_TYPE_FLOAT, SCHEMA_TYPE_HTML, SCHEMA_TYPE_INT, SCHEMA_TYPE_LIST, SCHEMA_TYPE_BASESTRING, SCHEMA_TYPE_UNICODE, SCHEMA_TYPE_UNICODE_OR_NONE, SCHEMA_TYPE_OBJECT_DICT, SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS]
ALLOWED_CUSTOM_OBJ_TYPES = ['Filepath', 'MathExpressionContent', 'MusicPhrase', 'ParameterName', 'SanitizedUrl', 'Graph', 'ImageWithRegions', 'ListOfTabs', 'SkillSelector', 'SubtitledHtml', 'SubtitledUnicode', 'SvgFilename', 'AllowedVariables', 'PositiveInt']
UI_CONFIG_SPECS: Dict[str, Dict[str, Any]] = {SCHEMA_TYPE_BOOL: {}, SCHEMA_TYPE_DICT: {}, SCHEMA_TYPE_FLOAT: {}, SCHEMA_TYPE_HTML: {'hide_complex_extensions': {'type': SCHEMA_TYPE_BOOL}, 'placeholder': {'type': SCHEMA_TYPE_UNICODE}}, SCHEMA_TYPE_INT: {}, SCHEMA_TYPE_LIST: {'add_element_text': {'type': SCHEMA_TYPE_UNICODE}}, SCHEMA_TYPE_UNICODE: {'rows': {'type': SCHEMA_TYPE_INT, 'validators': [{'id': 'is_at_least', 'min_value': 1}]}, 'coding_mode': {'type': SCHEMA_TYPE_UNICODE, 'choices': ['none', 'python', 'coffeescript']}, 'placeholder': {'type': SCHEMA_TYPE_UNICODE}}}
VALIDATOR_SPECS: Dict[str, Dict[str, Any]] = {SCHEMA_TYPE_BOOL: {}, SCHEMA_TYPE_DICT: {}, SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS: {}, SCHEMA_TYPE_FLOAT: {'is_at_least': {'min_value': {'type': SCHEMA_TYPE_FLOAT}}, 'is_at_most': {'max_value': {'type': SCHEMA_TYPE_FLOAT}}}, SCHEMA_TYPE_HTML: {}, SCHEMA_TYPE_INT: {'is_at_least': {'min_value': {'type': SCHEMA_TYPE_INT}}, 'is_at_most': {'max_value': {'type': SCHEMA_TYPE_INT}}}, SCHEMA_TYPE_LIST: {'has_length_at_least': {'min_value': {'type': SCHEMA_TYPE_INT, 'validators': [{'id': 'is_at_least', 'min_value': 1}]}}, 'has_length_at_most': {'max_value': {'type': SCHEMA_TYPE_INT, 'validators': [{'id': 'is_at_least', 'min_value': 1}]}}, 'is_uniquified': {}, 'has_unique_subtitled_contents': {}}, SCHEMA_TYPE_UNICODE: {'matches_regex': {'regex': {'type': SCHEMA_TYPE_UNICODE, 'validators': [{'id': 'is_regex'}]}}, 'is_nonempty': {}, 'is_regex': {}, 'is_valid_email': {}, 'is_valid_user_id': {}, 'is_valid_math_expression': {'algebraic': {'type': SCHEMA_TYPE_BOOL}}, 'is_valid_algebraic_expression': {}, 'is_valid_numeric_expression': {}, 'is_valid_math_equation': {}, 'is_supported_audio_language_code': {}, 'is_url_fragment': {}, 'has_length_at_least': {'min_value': {'type': SCHEMA_TYPE_INT}}, 'has_length_at_most': {'max_value': {'type': SCHEMA_TYPE_INT}}}, SCHEMA_TYPE_CUSTOM: {'has_subtitled_html_non_empty': {}, 'has_expected_subtitled_content_length': {'max_value': {'type': SCHEMA_TYPE_INT}}}}

def _validate_ui_config(obj_type: str, ui_config: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    'Validates the value of a UI configuration.\n\n        Args:\n            obj_type: str. UI config spec type.\n            ui_config: dict. The UI config that needs to be validated.\n\n        Raises:\n            AssertionError. The object fails to validate against the schema.\n    '
    reference_dict = UI_CONFIG_SPECS[obj_type]
    assert set(ui_config.keys()) <= set(reference_dict.keys()), 'Missing keys: %s, Extra keys: %s' % (list(set(reference_dict.keys()) - set(ui_config.keys())), list(set(ui_config.keys()) - set(reference_dict.keys())))
    for (key, value) in ui_config.items():
        schema_utils.normalize_against_schema(value, reference_dict[key])

def _validate_validator(obj_type: str, validator: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    "Validates the value of a 'validator' field.\n\n    Args:\n        obj_type: str. The type of the object.\n        validator: dict. The Specs that needs to be validated.\n\n    Raises:\n        AssertionError. The object fails to validate against the schema.\n    "
    reference_dict = VALIDATOR_SPECS[obj_type]
    assert 'id' in validator, 'id is not present in validator'
    assert validator['id'] in reference_dict, '%s is not present in reference_dict' % validator['id']
    customization_keys = list(validator.keys())
    customization_keys.remove('id')
    assert set(customization_keys) == set(reference_dict[validator['id']].keys()), 'Missing keys: %s, Extra keys: %s' % (list(set(reference_dict[validator['id']].keys()) - set(customization_keys)), list(set(customization_keys) - set(reference_dict[validator['id']].keys())))
    for key in customization_keys:
        value = validator[key]
        schema = reference_dict[validator['id']][key]
        try:
            schema_utils.normalize_against_schema(value, schema)
        except Exception as e:
            raise AssertionError(e) from e
    validator_fn = schema_utils.get_validator(validator['id'])
    assert set(inspect.getfullargspec(validator_fn).args) == set(customization_keys + ['obj']), 'Missing keys: %s, Extra keys: %s' % (list(set(customization_keys + ['obj']) - set(inspect.getfullargspec(validator_fn).args)), list(set(inspect.getfullargspec(validator_fn).args) - set(customization_keys + ['obj'])))

def _validate_dict_keys(dict_to_check: Dict[str, Any], required_keys: List[str], optional_keys: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Checks that all of the required keys, and possibly some of the optional\n    keys, are in the given dict.\n\n    Args:\n        dict_to_check: dict. The dict which needs to be validated.\n        required_keys: list. Keys which are required to be in the dictionary.\n        optional_keys: list. Keys which are optional in the dictionary.\n\n    Raises:\n        AssertionError. The dict is missing required keys.\n        AssertionError. The dict contains extra keys.\n    '
    assert set(required_keys) <= set(dict_to_check.keys()), 'Missing keys: %s' % dict_to_check
    assert set(dict_to_check.keys()) <= set(required_keys + optional_keys), 'Extra keys: %s' % dict_to_check

def validate_schema(schema: Dict[str, Any]) -> None:
    if False:
        return 10
    "Validates a schema.\n\n    This is meant to be a utility function that should be used by tests to\n    ensure that all schema definitions in the codebase are valid.\n\n    Each schema is a dict with at least a key called 'type'. The 'type' can\n    take one of the SCHEMA_TYPE_* values declared above. In addition, there\n    may be additional keys for specific types:\n    - 'list' requires an additional 'items' property, which specifies the type\n      of the elements in the list. It also allows for an optional 'len'\n      property which specifies the len of the list.\n    - 'dict' requires an additional 'properties' property, which specifies the\n      names of the keys in the dict, and schema definitions for their values.\n    - 'object_dict' requires any of the one additional schema keys either\n      'validation_method' or 'object_class'.\n      validation_method, takes the method which is written in\n      domain_obejcts_vaildator.\n      object_class, takes a domain class as its value.\n    There may also be an optional 'post_normalizers' key whose value is a list\n    of normalizers.\n\n    Args:\n        schema: dict. The schema that needs to be validated.\n\n    Raises:\n        AssertionError. The schema is not valid.\n    "
    assert isinstance(schema, dict), 'Expected dict, got %s' % schema
    assert SCHEMA_KEY_TYPE in schema, '%s is not present in schema key types' % SCHEMA_KEY_TYPE
    assert schema[SCHEMA_KEY_TYPE] in ALLOWED_SCHEMA_TYPES, '%s is not an allowed schema type' % schema[SCHEMA_KEY_TYPE]
    if schema[SCHEMA_KEY_TYPE] == SCHEMA_TYPE_CUSTOM:
        _validate_dict_keys(schema, [SCHEMA_KEY_TYPE, SCHEMA_KEY_OBJ_TYPE], [SCHEMA_KEY_REPLACEMENT_UI_CONFIG, SCHEMA_KEY_VALIDATORS])
        assert schema[SCHEMA_KEY_OBJ_TYPE] in ALLOWED_CUSTOM_OBJ_TYPES, schema
    elif schema[SCHEMA_KEY_TYPE] == SCHEMA_TYPE_LIST:
        _validate_dict_keys(schema, [SCHEMA_KEY_ITEMS, SCHEMA_KEY_TYPE], OPTIONAL_SCHEMA_KEYS + [SCHEMA_KEY_LEN])
        validate_schema(schema[SCHEMA_KEY_ITEMS])
        if SCHEMA_KEY_LEN in schema:
            assert isinstance(schema[SCHEMA_KEY_LEN], int), 'Expected int, got %s' % schema[SCHEMA_KEY_LEN]
            assert schema[SCHEMA_KEY_LEN] > 0, 'Expected length greater than 0, got %s' % schema[SCHEMA_KEY_LEN]
    elif schema[SCHEMA_KEY_TYPE] == SCHEMA_TYPE_DICT:
        _validate_dict_keys(schema, [SCHEMA_KEY_PROPERTIES, SCHEMA_KEY_TYPE], OPTIONAL_SCHEMA_KEYS)
        assert isinstance(schema[SCHEMA_KEY_PROPERTIES], list), 'Expected list, got %s' % schema[SCHEMA_KEY_LEN]
        for prop in schema[SCHEMA_KEY_PROPERTIES]:
            _validate_dict_keys(prop, [SCHEMA_KEY_NAME, SCHEMA_KEY_SCHEMA], [SCHEMA_KEY_DESCRIPTION])
            assert isinstance(prop[SCHEMA_KEY_NAME], str), 'Expected %s, got %s' % (str, prop[SCHEMA_KEY_NAME])
            validate_schema(prop[SCHEMA_KEY_SCHEMA])
            if SCHEMA_KEY_DESCRIPTION in prop:
                assert isinstance(prop[SCHEMA_KEY_DESCRIPTION], str), 'Expected %s, got %s' % (str, prop[SCHEMA_KEY_DESCRIPTION])
    elif schema[SCHEMA_KEY_TYPE] == SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS:
        _validate_dict_keys(schema, [SCHEMA_KEY_TYPE, SCHEMA_KEY_KEYS, SCHEMA_KEY_VALUES], OPTIONAL_SCHEMA_KEYS)
        items = [SCHEMA_KEY_VALUES, SCHEMA_KEY_KEYS]
        for item in items:
            assert isinstance(schema[item], dict), 'Expected dict, got %s' % schema[item]
            _validate_dict_keys(schema[item], [SCHEMA_KEY_SCHEMA], OPTIONAL_SCHEMA_KEYS)
            schema_item = schema[item]
            validate_schema(schema_item[SCHEMA_KEY_SCHEMA])
    elif schema[SCHEMA_KEY_TYPE] == SCHEMA_TYPE_OBJECT_DICT:
        _validate_dict_keys(schema, [SCHEMA_KEY_TYPE], [SCHEMA_KEY_VALIDATION_METHOD, SCHEMA_KEY_OBJECT_CLASS] + OPTIONAL_SCHEMA_KEYS)
    else:
        _validate_dict_keys(schema, [SCHEMA_KEY_TYPE], OPTIONAL_SCHEMA_KEYS)
    if SCHEMA_KEY_UI_CONFIG in schema:
        _validate_ui_config(schema[SCHEMA_KEY_TYPE], schema[SCHEMA_KEY_UI_CONFIG])
    if SCHEMA_KEY_POST_NORMALIZERS in schema:
        assert isinstance(schema[SCHEMA_KEY_POST_NORMALIZERS], list), 'Expected list, got %s' % schema[SCHEMA_KEY_POST_NORMALIZERS]
        for post_normalizer in schema[SCHEMA_KEY_POST_NORMALIZERS]:
            assert isinstance(post_normalizer, dict), 'Expected dict, got %s' % post_normalizer
            assert 'id' in post_normalizer, 'id is not present in %s' % post_normalizer
            schema_utils.Normalizers.get(post_normalizer['id'])
    if SCHEMA_KEY_VALIDATORS in schema:
        assert isinstance(schema[SCHEMA_KEY_VALIDATORS], list), 'Expected list, got %s' % schema[SCHEMA_KEY_VALIDATORS]
        for validator in schema[SCHEMA_KEY_VALIDATORS]:
            assert isinstance(validator, dict), 'Expected dict, got %s' % schema[SCHEMA_KEY_VALIDATORS]
            assert 'id' in validator, 'id is not present in %s' % validator
            _validate_validator(schema[SCHEMA_KEY_TYPE], validator)

class SchemaValidationUnitTests(test_utils.GenericTestBase):
    """Test validation of schemas."""
    GLOBAL_VALIDATORS_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_DICT, 'properties': [{'name': 'unicodeListProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}}, {'name': 'unicodeProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}]}
    GLOBAL_VALIDATORS: List[Dict[str, str]] = [{'id': 'does_not_contain_email'}]

    def arbitary_method(self, obj: Dict[str, str]) -> None:
        if False:
            while True:
                i = 10
        "Only required for testing.\n\n        Args:\n            obj: dict. Argument which needs to be validated.\n\n        Raises:\n            Exception. Given argument is missing 'any_arg'.\n        "
        if 'any_arg' not in obj:
            raise Exception("Missing 'any_arg'.")

    def test_schemas_are_correctly_validated(self) -> None:
        if False:
            while True:
                i = 10
        'Test validation of schemas.'
        invalid_schemas_with_error_messages = [(['type'], re.escape("Expected dict, got ['type']")), ({'type': 'invalid'}, 'invalid is not an allowed schema type'), ({'type': 'dict'}, "Missing keys: {'type': 'dict'}"), ({'type': 'list', 'items': {}}, 'type is not present in schema key types'), ({'type': 'list', 'items': {'type': 'unicode'}, 'len': -1}, 'Expected length greater than 0, got -1'), ({'type': 'list', 'items': {'type': 'unicode'}, 'len': 0}, 'Expected length greater than 0, got 0'), ({'type': 'list', 'items': {'type': 'unicode'}, 'validators': [{'id': 'has_length_at_most', 'max_value': 0}]}, re.escape("Validation failed: is_at_least ({'min_value': 1}) for object 0")), ({'type': 'dict', 'items': {'type': 'float'}}, re.escape("Missing keys: {'type': 'dict', 'items': {'type': 'float'}}")), ({'type': 'dict', 'properties': {123: {'type': 'unicode'}}}, re.escape("'len'")), ({'type': 'variable_keys_dict', 'keys': 1, 'values': {'schema': {'type': 'basestring'}}}, 'Expected dict, got 1'), ({'type': 'variable_keys_dict', 'fake_arg': 'value', 'values': {'schema': {'type': 'basestring'}}}, "Missing keys: {'type': 'variable_keys_dict', 'fake_arg': 'value', 'values': {'schema': {'type': 'basestring'}}}"), ({'type': 'unicode', 'validators': [{'id': 'fake_validator'}]}, 'fake_validator is not present in reference_dict'), ({'type': 'unicode', 'validators': [{'id': 'is_nonempty', 'fake_arg': 'unused_value'}]}, re.escape("Missing keys: [], Extra keys: ['fake_arg']")), ({'type': 'unicode', 'validators': [{'id': 'matches_regex'}]}, re.escape("Missing keys: ['regex'], Extra keys: []")), ({'type': 'float', 'validators': [{'id': 'is_at_least', 'min_value': 'value_of_wrong_type'}]}, 'Could not convert str to float: value_of_wrong_type'), ({'type': 'unicode', 'ui_config': {'rows': -1}}, re.escape("Validation failed: is_at_least ({'min_value': 1}) for object -1")), ({'type': 'unicode', 'ui_config': {'coding_mode': 'invalid_mode'}}, re.escape("Received invalid_mode which is not in the allowed range of choices: ['none', 'python', 'coffeescript']"))]
        valid_schemas: List[Dict[str, Any]] = [{'type': 'float'}, {'type': 'bool'}, {'type': 'dict', 'properties': [{'name': 'str_property', 'schema': {'type': 'unicode'}}]}, {'type': 'variable_keys_dict', 'keys': {'schema': {'type': 'basestring'}}, 'values': {'schema': {'type': 'float'}}}, {'type': 'list', 'items': {'type': 'list', 'items': {'type': 'list', 'items': {'type': 'bool'}, 'len': 100}}}, {'type': 'list', 'items': {'type': 'unicode'}, 'validators': [{'id': 'has_length_at_most', 'max_value': 3}]}, {'type': 'float', 'validators': [{'id': 'is_at_least', 'min_value': 3.0}]}, {'type': 'unicode', 'ui_config': {'rows': 5}}, {'type': 'unicode', 'ui_config': {'coding_mode': 'python'}}, {'type': 'object_dict', 'validation_method': self.arbitary_method}]
        for schema in valid_schemas:
            validate_schema(schema)
        for (schemas, error_msg) in invalid_schemas_with_error_messages:
            with self.assertRaisesRegex((AssertionError, KeyError), error_msg):
                validate_schema(schemas)

    def test_normalize_against_schema_raises_exception(self) -> None:
        if False:
            return 10
        'Tests if normalize against schema raises exception\n        for invalid key.\n        '
        with self.assertRaisesRegex(Exception, 'Invalid schema type: invalid'):
            schema = {SCHEMA_KEY_TYPE: 'invalid'}
            schema_utils.normalize_against_schema('obj', schema)

    def test_is_nonempty_validator(self) -> None:
        if False:
            while True:
                i = 10
        'Tests if static method is_nonempty returns true iff obj\n        is not an empty str.\n        '
        is_nonempty = schema_utils.get_validator('is_nonempty')
        self.assertTrue(is_nonempty('non-empty string'))
        self.assertTrue(is_nonempty(' '))
        self.assertTrue(is_nonempty('    '))
        self.assertFalse(is_nonempty(''))

    def test_is_at_most_validator(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests if static method is_at_most returns true iff obj\n        is at most a value.\n        '
        is_at_most = schema_utils.get_validator('is_at_most')
        self.assertTrue(is_at_most(2, 3))
        self.assertTrue(is_at_most(2, 2))
        self.assertFalse(is_at_most(2, 1))

    def test_has_length_at_least_validator(self) -> None:
        if False:
            print('Hello World!')
        'Tests if static method has_length_at_least returns true iff\n        given list has length of at least the given value.\n        '
        has_len_at_least = schema_utils.get_validator('has_length_at_least')
        self.assertTrue(has_len_at_least(['elem'], 0))
        self.assertTrue(has_len_at_least(['elem'], 1))
        self.assertFalse(has_len_at_least(['elem'], 2))

    def test_get_raises_invalid_validator_id(self) -> None:
        if False:
            print('Hello World!')
        "Tests if class method 'get' in _Validator raises exception\n        for invalid validator id.\n        "
        with self.assertRaisesRegex(Exception, 'Invalid validator id: some invalid validator method name'):
            schema_utils.get_validator('some invalid validator method name')

    def test_is_valid_algebraic_expression_validator(self) -> None:
        if False:
            print('Hello World!')
        'Tests for the is_valid_algebraic_expression static method with\n        algebraic type.\n        '
        is_valid_algebraic_expression = schema_utils.get_validator('is_valid_algebraic_expression')
        self.assertTrue(is_valid_algebraic_expression('a+b*2'))
        self.assertTrue(is_valid_algebraic_expression('3+4/2'))
        self.assertFalse(is_valid_algebraic_expression('3+4/a*'))

    def test_is_valid_numeric_expression_validator(self) -> None:
        if False:
            return 10
        'Tests for the is_valid_numeric_expression static method with\n        numeric type.\n        '
        is_valid_numeric_expression = schema_utils.get_validator('is_valid_numeric_expression')
        self.assertFalse(is_valid_numeric_expression('a+b*2'))
        self.assertTrue(is_valid_numeric_expression('3+4/2'))

    def test_is_valid_math_equation_validator(self) -> None:
        if False:
            return 10
        'Tests for the is_valid_math_equation static method.'
        is_valid_math_equation = schema_utils.get_validator('is_valid_math_equation')
        self.assertTrue(is_valid_math_equation('a+b=c'))
        self.assertTrue(is_valid_math_equation('x^2+y^2=z^2'))
        self.assertTrue(is_valid_math_equation('y = m*x + b'))
        self.assertTrue(is_valid_math_equation('alpha^a + beta^b = gamma^(-c)'))
        self.assertTrue(is_valid_math_equation('a+b=0'))
        self.assertTrue(is_valid_math_equation('0=a+b'))
        self.assertTrue(is_valid_math_equation('(a/b)+c=(4^3)*a'))
        self.assertTrue(is_valid_math_equation('2^alpha-(-3) = 3'))
        self.assertTrue(is_valid_math_equation('(a+b)^2 = a^2 + b^2 + 2*a*b'))
        self.assertTrue(is_valid_math_equation('(a+b)^2 = a^2 + b^2 + 2ab'))
        self.assertTrue(is_valid_math_equation('x/a + y/b = 1'))
        self.assertTrue(is_valid_math_equation('3 = -5 + pi^x'))
        self.assertTrue(is_valid_math_equation('0.4 + 0.5 = alpha * 4'))
        self.assertTrue(is_valid_math_equation('sqrt(a+b)=c - gamma/2.4'))
        self.assertTrue(is_valid_math_equation('abs(35 - x) = 22.3'))
        self.assertFalse(is_valid_math_equation('3 -= 2/a'))
        self.assertFalse(is_valid_math_equation('3 == 2/a'))
        self.assertFalse(is_valid_math_equation('x + y = '))
        self.assertFalse(is_valid_math_equation('(a+b = 0)'))
        self.assertFalse(is_valid_math_equation('pi = 3.1415'))
        self.assertFalse(is_valid_math_equation('a+b=0=a-b'))
        self.assertFalse(is_valid_math_equation('alpha - beta/c'))
        self.assertFalse(is_valid_math_equation('2^alpha-(-3*) = 3'))
        self.assertFalse(is_valid_math_equation('a~b = 0'))
        self.assertFalse(is_valid_math_equation('a+b<=0'))
        self.assertFalse(is_valid_math_equation('a+b>=0'))
        self.assertFalse(is_valid_math_equation('a+b<0'))
        self.assertFalse(is_valid_math_equation('a+b>0'))
        self.assertFalse(is_valid_math_equation('5+3=8'))
        self.assertFalse(is_valid_math_equation('(a+(b)=0'))
        self.assertFalse(is_valid_math_equation('a+b=c:)'))

    def test_is_supported_audio_language_code(self) -> None:
        if False:
            return 10
        is_supported_audio_language_code = schema_utils.get_validator('is_supported_audio_language_code')
        self.assertTrue(is_supported_audio_language_code('en'))
        self.assertTrue(is_supported_audio_language_code('fr'))
        self.assertTrue(is_supported_audio_language_code('de'))
        self.assertFalse(is_supported_audio_language_code(''))
        self.assertFalse(is_supported_audio_language_code('zz'))
        self.assertFalse(is_supported_audio_language_code('test'))

    def test_is_url_fragment(self) -> None:
        if False:
            return 10
        validate_url_fragment = schema_utils.get_validator('is_url_fragment')
        self.assertTrue(validate_url_fragment('math'))
        self.assertTrue(validate_url_fragment('computer-science'))
        self.assertTrue(validate_url_fragment('bio-tech'))
        self.assertFalse(validate_url_fragment(''))
        self.assertFalse(validate_url_fragment('Abc'))
        self.assertFalse(validate_url_fragment('!@#$%^&*()_+='))

    def test_global_validators_raise_exception_when_error_in_dict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(AssertionError, '^Validation failed: does_not_contain_email .* email@email.com$'):
            obj = {'unicodeListProp': ['not email', 'not email 2'], 'unicodeProp': 'email@email.com'}
            schema_utils.normalize_against_schema(obj, self.GLOBAL_VALIDATORS_SCHEMA, global_validators=self.GLOBAL_VALIDATORS)

    def test_global_validators_raise_exception_when_error_in_list(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(AssertionError, '^Validation failed: does_not_contain_email .* email2@email.com$'):
            obj = {'unicodeListProp': ['email2@email.com', 'not email 2'], 'unicodeProp': 'not email'}
            schema_utils.normalize_against_schema(obj, self.GLOBAL_VALIDATORS_SCHEMA, global_validators=self.GLOBAL_VALIDATORS)

    def test_global_validators_pass_when_no_error(self) -> None:
        if False:
            print('Hello World!')
        obj = {'unicodeListProp': ['not email', 'not email 2'], 'unicodeProp': 'not email'}
        normalized_obj = schema_utils.normalize_against_schema(obj, self.GLOBAL_VALIDATORS_SCHEMA, global_validators=self.GLOBAL_VALIDATORS)
        self.assertEqual(obj, normalized_obj)

    def test_is_regex_matched(self) -> None:
        if False:
            print('Hello World!')
        is_regex_matched = schema_utils.get_validator('is_regex_matched')
        self.assertTrue(is_regex_matched('exploration.EXP_ID_1.WzEuNjI2NTgxNDQwOTVlKzEyXQ==WzE3NThd', '(exploration|collection)\\.\\w+\\.\\w+'))
        self.assertFalse(is_regex_matched('WzEuNjI2NTgxNDQwOTVlKzEyXQ==WzE3NThd', '(exploration|collection)\\.\\w+\\.\\w+'))

    def test_is_search_query_string(self) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether a given string is contained within parenthesis and\n        double quotes.\n\n        Returns:\n            bool. A boolean value representing whether a given string is\n            contained within parenthesis and double quotes.\n        '
        is_search_query_string = schema_utils.get_validator('is_search_query_string')
        self.assertTrue(is_search_query_string('("A category")'))
        self.assertFalse(is_search_query_string('(missing-inner-quotes)'))
        self.assertFalse(is_search_query_string('missing-outer-parens'))

    def test_is_valid_username_string(self) -> None:
        if False:
            return 10
        'Checks whether given username string is valid.\n\n        Returns:\n            bool. A boolean value representing whether given username is\n            valid or not.\n        '
        is_valid_username_string = schema_utils.get_validator('is_valid_username_string')
        self.assertTrue(is_valid_username_string('alphabetic'))
        self.assertTrue(is_valid_username_string('alpha1234'))
        self.assertTrue(is_valid_username_string('un'))
        self.assertTrue(is_valid_username_string('username'))
        self.assertFalse(is_valid_username_string('invalidch@r'))
        self.assertFalse(is_valid_username_string('invalidch@434##@r'))
        self.assertFalse(is_valid_username_string('long' * 10))
        self.assertFalse(is_valid_username_string('admin'))
        self.assertFalse(is_valid_username_string('oppia'))
        self.assertFalse(is_valid_username_string(''))

    def test_has_expected_subtitled_content_length(self) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the given subtitled content does not exceed the\n        given length.\n\n        Returns:\n            bool. A boolean value representing whether the content matches\n            the given max length.\n        '
        has_expected_subtitled_content_length = schema_utils.get_validator('has_expected_subtitled_content_length')
        obj = {'content_id': 'id', 'unicode_str': 'Continueeeeeeeeeeeeeeee'}
        self.assertFalse(has_expected_subtitled_content_length(obj, 20))
        obj['unicode_str'] = 'Continue'
        self.assertTrue(has_expected_subtitled_content_length(obj, 20))

    def test_has_unique_subtitled_contents(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the subtitled html content has unique value or not.\n\n        Returns:\n            bool. A boolean value representing whether the content has unique\n            value.\n        '
        has_unique_subtitled_contents = schema_utils.get_validator('has_unique_subtitled_contents')
        obj_list = [{'content_id': 'id_1', 'html': '<p>1</p>'}, {'content_id': 'id_2', 'html': '<p>1</p>'}]
        self.assertFalse(has_unique_subtitled_contents(obj_list))
        obj_list[1]['html'] = '<p>2</p>'
        self.assertTrue(has_unique_subtitled_contents(obj_list))

    def test_has_length(self) -> None:
        if False:
            while True:
                i = 10
        'Tests if static method has_length returns true iff\n        given list has length of the given value.\n        '
        has_length = schema_utils.get_validator('has_length')
        self.assertTrue(has_length(['abcd', 'ab'], 2))
        self.assertFalse(has_length(['efg'], 2))
        self.assertFalse(has_length(['efg', 'ghj', 'huij'], 2))

class SchemaNormalizationUnitTests(test_utils.GenericTestBase):
    """Test schema-based normalization of objects."""

    def check_normalization(self, schema: Dict[str, Any], mappings: List[Tuple[Any, Any]], invalid_items_with_error_messages: List[Tuple[Any, str]]) -> None:
        if False:
            print('Hello World!')
        "Validates the schema and tests that values are normalized correctly.\n\n        Args:\n            schema: dict. The schema to normalize the value\n                against. Each schema is a dict with at least a key called\n                'type'. The 'type' can take one of the SCHEMA_TYPE_* values\n                declared above.\n            mappings: list(tuple). A list of 2-element tuples.\n                The first element of each item is expected to be normalized to\n                the second.\n            invalid_items_with_error_messages: list(tuple(*, str)). A list of\n                values with their corresponding messages. Each value is expected\n                to raise an AssertionError when normalized.\n        "
        validate_schema(schema)
        for (raw_value, expected_value) in mappings:
            self.assertEqual(schema_utils.normalize_against_schema(raw_value, schema), expected_value)
        for (value, error_msg) in invalid_items_with_error_messages:
            with self.assertRaisesRegex(Exception, error_msg):
                schema_utils.normalize_against_schema(value, schema)

    def test_float_schema(self) -> None:
        if False:
            i = 10
            return i + 15
        schema = {'type': schema_utils.SCHEMA_TYPE_FLOAT}
        mappings = [(1.2, 1.2), (3, 3.0), (-1, -1.0), ('1', 1.0)]
        invalid_values_with_error_messages = [([13], re.escape('Could not convert list to float: [13]')), ('abc', 'Could not convert str to float: abc'), (None, 'Could not convert NoneType to float: None')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_int_schema(self) -> None:
        if False:
            return 10
        schema = {'type': schema_utils.SCHEMA_TYPE_INT}
        mappings = [(1.2, 1), (3.7, 3), (-1, -1), ('1', 1)]
        invalid_values_with_error_messages = [([13], re.escape('Could not convert list to int: [13]')), ('abc', 'Could not convert str to int: abc'), (None, 'Could not convert NoneType to int: None')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_unicode_or_none_schema(self) -> None:
        if False:
            print('Hello World!')
        schema = {'type': schema_utils.SCHEMA_TYPE_UNICODE_OR_NONE}
        mappings = [('a', 'a'), ('', ''), (b'bytes', 'bytes'), (None, None)]
        invalid_values_with_error_messages: List[Tuple[List[str], str]] = [([], 'Expected unicode string or None, received')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_list_schema_with_len(self) -> None:
        if False:
            while True:
                i = 10
        schema = {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}, 'len': 2}
        mappings = [(['a', 'b'], ['a', 'b']), (['abc', ''], ['abc', '']), (['adaA13', '13'], ['adaA13', '13'])]
        invalid_values_with_error_messages = [(['1', 13], 'Expected unicode string, received 13'), ({'a': 'b'}, "Expected list, received {'a': 'b'}"), ({}, 'Expected list, received {}'), (None, 'Expected list, received None'), (123, 'Expected list, received 123'), ('abc', 'Expected list, received abc'), (['c'], 'Expected length of 2 got 1'), ([], 'Expected length of 2 got 0')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_html_schema(self) -> None:
        if False:
            while True:
                i = 10
        'Tests for valid html schema, an html string. Note that\n        html.cleaner() is called in normalize_against_schema.\n        '
        schema = {'type': schema_utils.SCHEMA_TYPE_HTML}
        mappings = [('<script></script>', ''), (b'<script></script>', ''), ('<a class="webLink" href="https://www.oppia.com/"><img src="images/oppia.png"></a>', '<a href="https://www.oppia.com/"></a>')]
        invalid_values_with_error_messages = [(['<script></script>', '<script></script>'], re.escape("Expected unicode HTML string, received ['<script></script>', '<script></script>']"))]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_schema_key_post_normalizers(self) -> None:
        if False:
            while True:
                i = 10
        'Test post normalizers in schema using basic html schema.'
        schema_1 = {'type': schema_utils.SCHEMA_TYPE_HTML, 'post_normalizers': [{'id': 'normalize_spaces'}]}
        obj_1 = 'a     a'
        normalize_obj_1 = schema_utils.normalize_against_schema(obj_1, schema_1)
        self.assertEqual(u'a a', normalize_obj_1)
        schema_2 = {'type': schema_utils.SCHEMA_TYPE_HTML, 'post_normalizers': [{'id': 'sanitize_url'}]}
        obj_2 = 'http://www.oppia.org/splash/<script>'
        normalize_obj_2 = schema_utils.normalize_against_schema(obj_2, schema_2)
        self.assertEqual(u'http://www.oppia.org/splash/', normalize_obj_2)

    def test_normalize_against_schema_for_bytes_unicode_works_fine(self) -> None:
        if False:
            i = 10
            return i + 15
        schema = {'type': schema_utils.SCHEMA_TYPE_UNICODE}
        obj = bytes('random string', 'utf-8')
        normalized_obj = schema_utils.normalize_against_schema(obj, schema)
        self.assertEqual('random string', normalized_obj)

    def test_list_schema(self) -> None:
        if False:
            print('Hello World!')
        schema = {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}
        mappings = [(['a', 'b'], ['a', 'b']), (['c'], ['c']), (['abc', ''], ['abc', '']), ([], []), (['adaA13', '13'], ['adaA13', '13'])]
        invalid_values_with_error_messages = [(['1', 13], 'Expected unicode string, received 13'), ({'a': 'b'}, re.escape("Expected list, received {'a': 'b'}")), ({}, 'Expected list, received {}'), (None, 'Expected list, received None'), (123, 'Expected list, received 123'), ('abc', 'Expected list, received abc')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_dict_schema(self) -> None:
        if False:
            print('Hello World!')
        schema = {'type': schema_utils.SCHEMA_TYPE_DICT, 'properties': [{'name': 'unicodeListProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}}, {'name': 'intProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_INT}}, {'name': 'dictProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_DICT, 'properties': [{'name': 'floatProp', 'schema': {'type': schema_utils.SCHEMA_TYPE_FLOAT}}]}}]}
        mappings = [({'unicodeListProp': [], 'intProp': 1, 'dictProp': {'floatProp': 3}}, {'unicodeListProp': [], 'intProp': 1, 'dictProp': {'floatProp': 3.0}}), ({'intProp': 10, 'unicodeListProp': ['abc', 'def'], 'dictProp': {'floatProp': -1.0}}, {'intProp': 10, 'unicodeListProp': ['abc', 'def'], 'dictProp': {'floatProp': -1.0}})]
        invalid_values_with_error_messages = [({'unicodeListProp': [], 'intPROP': 1, 'dictProp': {'floatProp': 3.0}}, re.escape("Missing keys: ['intProp'], Extra keys: ['intPROP']")), ({'unicodeListProp': ['aaa'], 'intProp': 1}, re.escape("Missing keys: ['dictProp'], Extra keys: []")), ({'unicodeListProp': [], 'intProp': 3, 'dictProp': {}}, re.escape("Missing keys: ['floatProp'], Extra keys: []")), (['unicodeListProp', 'intProp', 'dictProp'], re.escape("Expected dict, received ['unicodeListProp', 'intProp', 'dictProp']")), (None, 'Expected dict, received None'), (123, 'Expected dict, received 123'), ('abc', 'Expected dict, received abc')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_dict_with_variable_key_schema(self) -> None:
        if False:
            i = 10
            return i + 15
        schema = {'type': schema_utils.SCHEMA_TYPE_DICT_WITH_VARIABLE_NO_OF_KEYS, 'keys': {'schema': {'type': 'basestring'}}, 'values': {'schema': {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_INT}, 'len': 2}}}
        mappings = [({'skills_id1': [1.2, 3], 'skills_id2': [2.0, 0]}, {'skills_id1': [1, 3], 'skills_id2': [2, 0]}), ({'skills_id1': ['45', 2], 'skills_id2': [23, 3]}, {'skills_id1': [45, 2], 'skills_id2': [23, 3]}), ({'skills_id1': [1, 2], 'skills_id2': [2, 3]}, {'skills_id1': [1, 2], 'skills_id2': [2, 3]})]
        invalid_values_with_error_messages = [([1, 2], re.escape('Expected dict, received [1, 2]')), ({1: 2, 'topic_id1': 3}, 'Expected string, received 1'), ({'topics_id1': 1}, 'Expected list, received 1'), (None, 'Expected dict, received None'), ({'skill_id1': [45, 2, 34]}, 'Expected length of 2 got 3')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_string_schema(self) -> None:
        if False:
            print('Hello World!')
        schema = {'type': schema_utils.SCHEMA_TYPE_BASESTRING}
        mappings = [('test1', 'test1'), ('test2', 'test2')]
        invalid_values_with_error_messages = [(12, 'Expected string, received 12'), (None, 'Expected string, received None')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_object_dict_schema_with_validation_method_key(self) -> None:
        if False:
            i = 10
            return i + 15
        schema = {'type': SCHEMA_TYPE_OBJECT_DICT, 'validation_method': validation_method_for_testing}
        mappings = [({'arg_a': 'arbitary_argument_a', 'arg_b': 'arbitary_argument_b'}, {'arg_a': 'arbitary_argument_a', 'arg_b': 'arbitary_argument_b'})]
        invalid_values_with_error_messages = [({'arg_a': 'arbitary_argument_a'}, 'Missing arg_b in argument.')]
        self.check_normalization(schema, mappings, invalid_values_with_error_messages)

    def test_normalize_spaces(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test static method normalize_spaces; should collapse multiple\n        spaces.\n        '
        normalize_spaces = schema_utils.Normalizers.get('normalize_spaces')
        self.assertEqual('dog cat', normalize_spaces('dog     cat'))
        self.assertEqual('dog cat', normalize_spaces('  dog cat'))
        self.assertEqual('dog cat', normalize_spaces(' dog   cat   '))
        self.assertNotEqual('dog cat', normalize_spaces('dogcat'))

    def test_normalizer_get(self) -> None:
        if False:
            print('Hello World!')
        "Tests the class method 'get' of Normalizers, should return the\n        normalizer method corresponding to the given normalizer id.\n        "
        normalize_spaces = schema_utils.Normalizers.get('normalize_spaces')
        self.assertEqual('normalize_spaces', normalize_spaces.__name__)

    def test_normalizer_get_raises_exception_for_invalid_id(self) -> None:
        if False:
            while True:
                i = 10
        'Tests if class method get of Normalizers raises exception when given\n        an invalid normalizer id.\n        '
        with self.assertRaisesRegex(Exception, 'Invalid normalizer id: some invalid normalizer method name'):
            schema_utils.Normalizers.get('some invalid normalizer method name')
        with self.assertRaisesRegex(Exception, 'Invalid normalizer id: normalize_space'):
            schema_utils.Normalizers.get('normalize_space')

    def test_normalizer_sanitize_url(self) -> None:
        if False:
            while True:
                i = 10
        'Tests if static method sanitize_url of Normalizers correctly\n        sanitizes a URL when given its string representation and raises\n        error for invalid URLs.\n        '
        sanitize_url = schema_utils.Normalizers.get('sanitize_url')
        self.assertEqual('https://www.oppia.org/splash/', sanitize_url('https://www.oppia.org/splash/'))
        self.assertEqual('http://www.oppia.org/splash/', sanitize_url('http://www.oppia.org/splash/'))
        self.assertEqual(sanitize_url('http://example.com/~path;parameters?q=arg#fragment'), 'http://example.com/~path%3Bparameters?q%3Darg#fragment')
        self.assertEqual('https://www.web.com/%3Cscript%20type%3D%22text/javascript%22%3Ealert%28%27rm%20-rf%27%29%3B%3C/script%3E', sanitize_url('https://www.web.com/<script type="text/javascript">alert(\'rm -rf\');</script>'))
        self.assertEqual('', sanitize_url(''))
        with self.assertRaisesRegex(AssertionError, "Invalid URL: Sanitized URL should start with 'http://' or 'https://'; received oppia.org"):
            sanitize_url('oppia.org')
        with self.assertRaisesRegex(AssertionError, "Invalid URL: Sanitized URL should start with 'http://' or 'https://'; received www.oppia.org"):
            sanitize_url('www.oppia.org')

class ValidateArgumentHavingSpecificClass(test_utils.GenericTestBase):
    """Test class is to validate the arguments which have a corresponding domain
    class representation in the codebase. This test class is written uniquely
    because it returns an object which is different from all other cases.
    """

    def test_object_dict_schema_with_object_class_key(self) -> None:
        if False:
            return 10
        schema = {'type': SCHEMA_TYPE_OBJECT_DICT, 'object_class': ValidateClassForTesting}
        sample_dict = {'arg_a': 'arbitary_argument_a', 'arg_b': 'arbitary_argument_b'}
        arg1 = schema_utils.normalize_against_schema(sample_dict, schema)
        arg2 = ValidateClassForTesting.from_dict(sample_dict)
        self.assertEqual(arg1.arg_a, arg2.arg_a)

def validation_method_for_testing(obj: Dict[str, str]) -> Dict[str, str]:
    if False:
        return 10
    "Method to test 'validation_method' key of schema.\n\n    Args:\n        obj: dict. Dictionary form of the argument.\n\n    Returns:\n        dict(str, str). Returns a dict value after validation.\n\n    Raises:\n        Exception. If any one argument is missing.\n    "
    if 'arg_a' not in obj:
        raise Exception('Missing arg_a in argument.')
    if 'arg_b' not in obj:
        raise Exception('Missing arg_b in argument.')
    return obj

class ValidateClassForTesting:
    """Class to test 'object_class' key of schema."""

    def __init__(self, arg_a: str, arg_b: str) -> None:
        if False:
            return 10
        'Initializes the object.\n\n        Args:\n            arg_a: str. Random first argument for testing.\n            arg_b: str. Random second argument for testing.\n        '
        self.arg_a = arg_a
        self.arg_b = arg_b

    @classmethod
    def from_dict(cls, obj: Dict[str, str]) -> ValidateClassForTesting:
        if False:
            print('Hello World!')
        'Return the ValidateClassForTesting object from a dict.\n\n        Args:\n            obj: dict. Dictionary representation of the object.\n\n        Returns:\n            ValidateClassForTesting. The corresponding test object.\n        '
        return cls(obj['arg_a'], obj['arg_b'])

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Method to validate the test object.'
        if not isinstance(self.arg_a, str):
            raise Exception('Invalid type arg_a.')
        if not isinstance(self.arg_b, str):
            raise Exception('Invalid type arg_b.')