"""Tests for parameter domain objects."""
from __future__ import annotations
from core import feconf
from core import utils
from core.domain import object_registry
from core.domain import param_domain
from core.tests import test_utils

class ParameterDomainUnitTests(test_utils.GenericTestBase):
    """Tests for parameter domain objects."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.sample_customization_args: param_domain.CustomizationArgsDictWithValue = {'value': '5', 'parse_with_jinja': True}

    def test_param_spec_validation(self) -> None:
        if False:
            while True:
                i = 10
        'Test validation of param specs.'
        param_spec = param_domain.ParamSpec('Real')
        with self.assertRaisesRegex(utils.ValidationError, 'is not among the supported object types'):
            param_spec.validate()
        param_spec.obj_type = 'UnicodeString'
        param_spec.validate()

    def test_supported_object_types_exist_in_registry(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the supported object types of param specs.'
        for obj_type in feconf.SUPPORTED_OBJ_TYPES:
            object_registry.Registry.get_object_class_by_type(obj_type)

    def test_param_change_validation(self) -> None:
        if False:
            return 10
        'Test validation of parameter changes.'
        with self.assertRaisesRegex(utils.ValidationError, 'Only parameter names'):
            param_domain.ParamChange('¡hola', 'Copier', self.sample_customization_args).validate()
        with self.assertRaisesRegex(utils.ValidationError, 'Expected generator ID to be a string'):
            param_domain.ParamChange('abc', 123, self.sample_customization_args).validate()
        with self.assertRaisesRegex(utils.ValidationError, 'Invalid generator ID'):
            param_domain.ParamChange('abc', 'InvalidGenerator', self.sample_customization_args).validate()
        with self.assertRaisesRegex(utils.ValidationError, 'Expected a dict'):
            param_domain.ParamChange('abc', 'Copier', ['a', 'b']).validate()
        with self.assertRaisesRegex(utils.ValidationError, 'Expected param_change name to be a string, received'):
            param_domain.ParamChange(3, 'Copier', self.sample_customization_args).validate()
        with self.assertRaisesRegex(Exception, 'Invalid parameter change customization_arg name:'):
            customization_args_dict = {1: '1'}
            param_domain.ParamChange('abc', 'Copier', customization_args_dict).validate()

    def test_param_spec_to_dict(self) -> None:
        if False:
            while True:
                i = 10
        sample_dict = {'obj_type': 'UnicodeString'}
        param_spec = param_domain.ParamSpec(sample_dict['obj_type'])
        self.assertEqual(param_spec.to_dict(), sample_dict)

    def test_param_spec_from_dict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        sample_dict: param_domain.ParamSpecDict = {'obj_type': 'UnicodeString'}
        param_spec = param_domain.ParamSpec.from_dict(sample_dict)
        self.assertEqual(param_spec.to_dict(), sample_dict)

    def test_param_change_class(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the ParamChange class.'
        param_change = param_domain.ParamChange('abc', 'Copier', {'value': '3', 'parse_with_jinja': True})
        param_change.validate()
        self.assertEqual(param_change.name, 'abc')
        self.assertEqual(param_change.generator.id, 'Copier')
        self.assertEqual(param_change.to_dict(), {'name': 'abc', 'generator_id': 'Copier', 'customization_args': {'value': '3', 'parse_with_jinja': True}})
        self.assertEqual(param_change.get_value({}), '3')

    def test_param_change_from_dict(self) -> None:
        if False:
            print('Hello World!')
        sample_dict: param_domain.ParamChangeDict = {'name': 'abc', 'generator_id': 'Copier', 'customization_args': self.sample_customization_args}
        param_change = param_domain.ParamChange.from_dict(sample_dict)
        param_change.validate()
        self.assertEqual(param_change.to_dict(), sample_dict)