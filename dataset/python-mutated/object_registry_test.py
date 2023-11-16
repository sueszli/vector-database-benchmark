"""Tests for services relating to typed objects."""
from __future__ import annotations
from core.domain import interaction_registry
from core.domain import object_registry
from core.tests import test_utils
from extensions.objects.models import objects

class ObjectRegistryUnitTests(test_utils.GenericTestBase):
    """Test the Registry class in object_registry."""

    def test_get_object_class_by_type_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests the normal behavior of get_object_class_by_type().'
        self.assertEqual(object_registry.Registry.get_object_class_by_type('Int').__name__, 'Int')

    def test_fake_class_is_not_gettable(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that trying to retrieve a fake class raises an error.'
        with self.assertRaisesRegex(TypeError, 'not a valid object class'):
            object_registry.Registry.get_object_class_by_type('FakeClass')

    def test_base_object_is_not_gettable(self) -> None:
        if False:
            print('Hello World!')
        'Tests that BaseObject exists and cannot be set as an obj_type.'
        assert getattr(objects, 'BaseObject')
        with self.assertRaisesRegex(TypeError, 'not a valid object class'):
            object_registry.Registry.get_object_class_by_type('BaseObject')

class ObjectDefaultValuesUnitTests(test_utils.GenericTestBase):
    """Test that the default value of objects recorded in
    extensions/objects/object_defaults.json correspond to
    the defined default values in objects.py for all objects that
    are used in rules.
    """

    def test_all_rule_input_fields_have_default_values(self) -> None:
        if False:
            return 10
        'Checks that all rule input fields have a default value, and this\n        is provided in get_default_values().\n        '
        interactions = interaction_registry.Registry.get_all_interactions()
        object_default_vals = object_registry.get_default_object_values()
        for interaction in interactions:
            for rule_name in interaction.rules_dict:
                param_list = interaction.get_rule_param_list(rule_name)
                for (_, param_obj_type) in param_list:
                    param_obj_type_name = param_obj_type.__name__
                    default_value = param_obj_type.default_value
                    self.assertIsNotNone(default_value, msg='No default value specified for object class %s.' % param_obj_type_name)
                    self.assertIn(param_obj_type_name, object_default_vals)
                    self.assertEqual(default_value, object_default_vals[param_obj_type_name])

    def test_get_object_default_values_is_valid(self) -> None:
        if False:
            return 10
        'Checks that the default values provided by get_default_values()\n        correspond to the ones defined in objects.py.\n        '
        object_default_vals = object_registry.get_default_object_values()
        all_object_classes = object_registry.Registry.get_all_object_classes()
        for (obj_type, default_value) in object_default_vals.items():
            self.assertIn(obj_type, all_object_classes)
            self.assertEqual(default_value, all_object_classes[obj_type].default_value)