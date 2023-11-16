"""Tests for methods in the interaction registry."""
from __future__ import annotations
import json
import os
from core import feconf
from core import schema_utils
from core import utils
from core.domain import exp_services
from core.domain import interaction_registry
from core.tests import test_utils
from extensions.interactions import base
from typing import Any, Dict, Final
EXPECTED_TERMINAL_INTERACTIONS_COUNT: Final = 1

class InteractionDependencyTests(test_utils.GenericTestBase):
    """Tests for the calculation of dependencies for interactions."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.login(self.EDITOR_EMAIL)

    def test_deduplication_of_dependency_ids(self) -> None:
        if False:
            print('Hello World!')
        self.assertItemsEqual(interaction_registry.Registry.get_deduplicated_dependency_ids(['CodeRepl']), ['skulpt', 'codemirror'])
        self.assertItemsEqual(interaction_registry.Registry.get_deduplicated_dependency_ids(['CodeRepl', 'CodeRepl', 'CodeRepl']), ['skulpt', 'codemirror'])
        self.assertItemsEqual(interaction_registry.Registry.get_deduplicated_dependency_ids(['CodeRepl', 'AlgebraicExpressionInput']), ['skulpt', 'codemirror', 'guppy', 'nerdamer'])

    def test_no_dependencies_in_non_exploration_pages(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        response = self.get_html_response(feconf.LIBRARY_INDEX_URL)
        response.mustcontain(no=['dependency_html.html'])

    def test_dependencies_loaded_in_exploration_editor(self) -> None:
        if False:
            print('Hello World!')
        exp_services.load_demo('0')
        response = self.get_html_response('/create/0')
        response.mustcontain('dependency_html.html')
        self.logout()

class InteractionRegistryUnitTests(test_utils.GenericTestBase):
    """Test for the interaction registry."""

    def test_interaction_registry(self) -> None:
        if False:
            return 10
        'Do some sanity checks on the interaction registry.'
        self.assertEqual({type(i).__name__ for i in interaction_registry.Registry.get_all_interactions()}, set(interaction_registry.Registry.get_all_interaction_ids()))
        with self.swap(interaction_registry.Registry, '_interactions', {}):
            self.assertEqual({type(i).__name__ for i in interaction_registry.Registry.get_all_interactions()}, set(interaction_registry.Registry.get_all_interaction_ids()))

    def test_get_all_specs(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the get_all_specs() method.'
        specs_dict = interaction_registry.Registry.get_all_specs()
        self.assertEqual(set(specs_dict.keys()), set(interaction_registry.Registry.get_all_interaction_ids()))
        terminal_interactions_count = 0
        for item in specs_dict.values():
            self.assertIn(item['display_mode'], base.ALLOWED_DISPLAY_MODES)
            self.assertTrue(isinstance(item['is_terminal'], bool))
            if item['is_terminal']:
                terminal_interactions_count += 1
        self.assertEqual(terminal_interactions_count, EXPECTED_TERMINAL_INTERACTIONS_COUNT)

    def test_interaction_specs_json_sync_all_specs(self) -> None:
        if False:
            while True:
                i = 10
        'Test to ensure that the interaction_specs.json file is upto date\n        with additions in the individual interaction files.\n        '
        all_specs = interaction_registry.Registry.get_all_specs()
        spec_file = os.path.join('extensions', 'interactions', 'interaction_specs.json')
        with utils.open_file(spec_file, 'r') as f:
            specs_from_json = json.loads(f.read())
        self.assertDictEqual(all_specs, specs_from_json)

    def test_interaction_specs_customization_arg_specs_names_are_valid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test to ensure that all customization argument names in\n        interaction specs only include alphabetic letters and are\n        lowerCamelCase. This is because these properties are involved in the\n        generation of content_ids for customization arguments.\n        '
        all_specs = interaction_registry.Registry.get_all_specs()
        ca_names_in_schema = []

        def traverse_schema_to_find_names(schema: Dict[str, Any]) -> None:
            if False:
                i = 10
                return i + 15
            "Recursively traverses the schema to find all name fields.\n            Recursion is required because names can be nested within\n            'type: dict' inside a schema.\n\n            Args:\n                schema: dict. The schema to traverse.\n            "
            if 'name' in schema:
                ca_names_in_schema.append(schema['name'])
            schema_type = schema['type']
            if schema_type == schema_utils.SCHEMA_TYPE_LIST:
                traverse_schema_to_find_names(schema['items'])
            elif schema_type == schema_utils.SCHEMA_TYPE_DICT:
                for schema_property in schema['properties']:
                    ca_names_in_schema.append(schema_property['name'])
                    traverse_schema_to_find_names(schema_property['schema'])
        for interaction_id in all_specs:
            for ca_spec in all_specs[interaction_id]['customization_arg_specs']:
                ca_names_in_schema.append(ca_spec['name'])
                traverse_schema_to_find_names(ca_spec['schema'])
        for name in ca_names_in_schema:
            self.assertTrue(name.isalpha())
            self.assertTrue(name[0].islower())

    def test_interaction_specs_customization_arg_default_values_are_valid(self) -> None:
        if False:
            print('Hello World!')
        'Test to ensure that all customization argument default values\n        that contain content_ids are properly set to None.\n        '
        all_specs = interaction_registry.Registry.get_all_specs()

        def traverse_schema_to_find_and_validate_subtitled_content(value: Any, schema: Dict[str, Any]) -> None:
            if False:
                while True:
                    i = 10
            'Recursively traverse the schema to find SubtitledHtml or\n            SubtitledUnicode contained or nested in value.\n\n            Args:\n                value: *. The value of the customization argument.\n                schema: dict. The customization argument schema.\n            '
            is_subtitled_html_spec = schema['type'] == schema_utils.SCHEMA_TYPE_CUSTOM and schema['obj_type'] == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML
            is_subtitled_unicode_spec = schema['type'] == schema_utils.SCHEMA_TYPE_CUSTOM and schema['obj_type'] == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE
            if is_subtitled_html_spec or is_subtitled_unicode_spec:
                self.assertIsNone(value['content_id'])
            elif schema['type'] == schema_utils.SCHEMA_TYPE_LIST:
                for x in value:
                    traverse_schema_to_find_and_validate_subtitled_content(x, schema['items'])
            elif schema['type'] == schema_utils.SCHEMA_TYPE_DICT:
                for schema_property in schema['properties']:
                    traverse_schema_to_find_and_validate_subtitled_content(x[schema_property.name], schema_property['schema'])
        for interaction_id in all_specs:
            for ca_spec in all_specs[interaction_id]['customization_arg_specs']:
                traverse_schema_to_find_and_validate_subtitled_content(ca_spec['default_value'], ca_spec['schema'])

    def test_get_all_specs_for_state_schema_version_for_unsaved_version(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(IOError, 'No specs JSON file found for state schema'):
            interaction_registry.Registry.get_all_specs_for_state_schema_version(10)

    def test_get_interaction_by_id_raises_error_for_none_interaction_id(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'No interaction exists for the None interaction_id.'):
            interaction_registry.Registry.get_interaction_by_id(None)