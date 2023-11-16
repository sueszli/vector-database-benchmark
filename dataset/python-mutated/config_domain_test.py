"""Tests for the config property registry."""
from __future__ import annotations
from core import feconf
from core import schema_utils_test
from core import utils
from core.domain import config_domain
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import config_models
(config_models,) = models.Registry.import_models([models.Names.CONFIG])

class ConfigPropertyChangeTests(test_utils.GenericTestBase):

    def test_config_property_change_object_with_missing_cmd(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(utils.ValidationError, 'Missing cmd key in change dict'):
            config_domain.ConfigPropertyChange({'invalid': 'data'})

    def test_config_property_change_object_with_invalid_cmd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(utils.ValidationError, 'Command invalid is not allowed'):
            config_domain.ConfigPropertyChange({'cmd': 'invalid'})

    def test_config_property_change_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(utils.ValidationError, 'The following required attributes are missing: new_value'):
            config_domain.ConfigPropertyChange({'cmd': 'change_property_value'})

    def test_config_property_change_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(utils.ValidationError, 'The following extra attributes are present: invalid'):
            config_domain.ConfigPropertyChange({'cmd': 'change_property_value', 'new_value': 'new_value', 'invalid': 'invalid'})

    def test_config_property_change_object_with_change_property_value(self) -> None:
        if False:
            while True:
                i = 10
        config_property_change_object = config_domain.ConfigPropertyChange({'cmd': 'change_property_value', 'new_value': 'new_value'})
        self.assertEqual(config_property_change_object.cmd, 'change_property_value')
        self.assertEqual(config_property_change_object.new_value, 'new_value')

    def test_to_dict(self) -> None:
        if False:
            return 10
        config_property_change_dict = {'cmd': 'change_property_value', 'new_value': 'new_value'}
        config_property_change_object = config_domain.ConfigPropertyChange(config_property_change_dict)
        self.assertEqual(config_property_change_object.to_dict(), config_property_change_dict)

class ConfigPropertyRegistryTests(test_utils.GenericTestBase):
    """Tests for the config property registry."""

    def test_config_property_schemas_are_valid(self) -> None:
        if False:
            while True:
                i = 10
        for property_name in config_domain.Registry.get_all_config_property_names():
            config_property = config_domain.Registry.get_config_property(property_name)
            assert config_property is not None
            schema_utils_test.validate_schema(config_property.schema)
        schemas = config_domain.Registry.get_config_property_schemas()
        for property_name in config_domain.Registry.get_all_config_property_names():
            schema_utils_test.validate_schema(schemas[property_name]['schema'])

    def test_raises_error_if_invalid_config_property_fetched_with_strict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(Exception, 'No config property exists for the given property name: Invalid_config_name'):
            config_domain.Registry.get_config_property('Invalid_config_name', strict=True)

    def test_get_exception_creating_new_config_property_with_existing_name(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'Property with name classroom_pages_data already exists'):
            config_domain.ConfigProperty('classroom_pages_data', config_domain.SET_OF_CLASSROOM_DICTS_SCHEMA, 'The details for each classroom page.', [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': '', 'topic_list_intro': ''}])

    def test_config_property_with_new_config_property_model(self) -> None:
        if False:
            while True:
                i = 10
        config_model = config_models.ConfigPropertyModel(id='config_model', value='new_value')
        config_model.commit(feconf.SYSTEM_COMMITTER_ID, [])
        retrieved_model = config_domain.ConfigProperty('config_model', config_domain.BOOL_SCHEMA, 'description', False)
        self.assertEqual(retrieved_model.value, 'new_value')
        self.assertEqual(retrieved_model.description, 'description')