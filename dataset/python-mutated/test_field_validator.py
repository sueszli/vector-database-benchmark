from __future__ import annotations
import pytest
from airflow.providers.google.cloud.utils.field_validator import GcpBodyFieldValidator, GcpFieldValidationException, GcpValidationSpecificationException

class TestGcpBodyFieldValidator:

    def test_validate_should_not_raise_exception_if_field_and_body_are_both_empty(self):
        if False:
            return 10
        specification = []
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_body_is_none(self):
        if False:
            for i in range(10):
                print('nop')
        specification = []
        body = None
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(RuntimeError, match='The body to validate is `None`. Please provide a dictionary to validate.'):
            validator.validate(body)

    def test_validate_should_fail_if_specification_is_none(self):
        if False:
            while True:
                i = 10
        specification = None
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(TypeError):
            validator.validate(body)

    def test_validate_should_raise_exception_name_attribute_is_missing_from_specs(self):
        if False:
            return 10
        specification = [dict(allow_empty=False)]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(KeyError):
            validator.validate(body)

    def test_validate_should_raise_exception_if_field_is_not_present(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='name', allow_empty=False)]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_validate_a_single_field(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='name', allow_empty=False)]
        body = {'name': 'bigquery'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_body_is_not_a_dict(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='name', allow_empty=False)]
        body = [{'name': 'bigquery'}]
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(AttributeError):
            validator.validate(body)

    def test_validate_should_fail_for_set_allow_empty_when_field_is_none(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='name', allow_empty=True)]
        body = {'name': None}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_interpret_allow_empty_clause(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='name', allow_empty=True)]
        body = {'name': ''}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_raise_if_empty_clause_is_false(self):
        if False:
            for i in range(10):
                print('nop')
        specification = [dict(name='name', allow_empty=False)]
        body = {'name': None}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_raise_if_version_mismatch_is_found(self):
        if False:
            for i in range(10):
                print('nop')
        specification = [dict(name='name', allow_empty=False, api_version='v2')]
        body = {'name': 'value'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        validator.validate(body)

    def test_validate_should_interpret_optional_irrespective_of_allow_empty(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='name', allow_empty=False, optional=True)]
        body = {'name': None}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_interpret_optional_clause(self):
        if False:
            print('Hello World!')
        specification = [dict(name='name', allow_empty=False, optional=True)]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_raise_exception_if_optional_clause_is_false_and_field_not_present(self):
        if False:
            return 10
        specification = [dict(name='name', allow_empty=False, optional=False)]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_interpret_dict_type(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='labels', optional=True, type='dict')]
        body = {'labels': {'one': 'value'}}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_value_is_not_dict_as_per_specs(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='labels', optional=True, type='dict')]
        body = {'labels': 1}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_not_allow_both_type_and_allow_empty_in_a_spec(self):
        if False:
            print('Hello World!')
        specification = [dict(name='labels', optional=True, type='dict', allow_empty=True)]
        body = {'labels': 1}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpValidationSpecificationException):
            validator.validate(body)

    def test_validate_should_allow_type_and_optional_in_a_spec(self):
        if False:
            print('Hello World!')
        specification = [dict(name='labels', optional=True, type='dict')]
        body = {'labels': {}}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_union_field_is_not_found(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='an_union', type='union', optional=False, fields=[dict(name='variant_1', regexp='^.+$', optional=False, allow_empty=False)])]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_there_is_no_nested_field_for_union(self):
        if False:
            print('Hello World!')
        specification = [dict(name='an_union', type='union', optional=False, fields=[])]
        body = {}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpValidationSpecificationException):
            validator.validate(body)

    def test_validate_should_interpret_union_with_one_field(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='an_union', type='union', fields=[dict(name='variant_1', regexp='^.+$')])]
        body = {'variant_1': 'abc', 'variant_2': 'def'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_if_both_field_of_union_is_present(self):
        if False:
            while True:
                i = 10
        specification = [dict(name='an_union', type='union', fields=[dict(name='variant_1', regexp='^.+$'), dict(name='variant_2', regexp='^.+$')])]
        body = {'variant_1': 'abc', 'variant_2': 'def'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_validate_when_value_matches_regex(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='an_union', type='union', fields=[dict(name='variant_1', regexp='[^a-z]')])]
        body = {'variant_1': '12'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_fail_when_value_does_not_match_regex(self):
        if False:
            return 10
        specification = [dict(name='an_union', type='union', fields=[dict(name='variant_1', regexp='[^a-z]')])]
        body = {'variant_1': 'abc'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_raise_if_custom_validation_is_not_true(self):
        if False:
            for i in range(10):
                print('nop')

        def _int_equal_to_zero(value):
            if False:
                while True:
                    i = 10
            if int(value) != 0:
                raise GcpFieldValidationException('The available memory has to be equal to 0')
        specification = [dict(name='availableMemoryMb', custom_validation=_int_equal_to_zero)]
        body = {'availableMemoryMb': 1}
        validator = GcpBodyFieldValidator(specification, 'v1')
        with pytest.raises(GcpFieldValidationException):
            validator.validate(body)

    def test_validate_should_not_raise_if_custom_validation_is_true(self):
        if False:
            for i in range(10):
                print('nop')

        def _int_equal_to_zero(value):
            if False:
                return 10
            if int(value) != 0:
                raise GcpFieldValidationException('The available memory has to be equal to 0')
        specification = [dict(name='availableMemoryMb', custom_validation=_int_equal_to_zero)]
        body = {'availableMemoryMb': 0}
        validator = GcpBodyFieldValidator(specification, 'v1')
        assert validator.validate(body) is None

    def test_validate_should_validate_group_of_specs(self):
        if False:
            i = 10
            return i + 15
        specification = [dict(name='name', allow_empty=False), dict(name='description', allow_empty=False, optional=True), dict(name='labels', optional=True, type='dict'), dict(name='an_union', type='union', fields=[dict(name='variant_1', regexp='^.+$'), dict(name='variant_2', regexp='^.+$', api_version='v1beta2'), dict(name='variant_3', type='dict', fields=[dict(name='url', regexp='^.+$')]), dict(name='variant_4')])]
        body = {'variant_1': 'abc', 'name': 'bigquery'}
        validator = GcpBodyFieldValidator(specification, 'v1')
        validator.validate(body)