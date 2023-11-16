"""Unit tests for core.domain.rte_component_registry."""
from __future__ import annotations
import inspect
import os
import pkgutil
import re
import string
import struct
from core import feconf
from core import schema_utils
from core import schema_utils_test
from core import utils
from core.constants import constants
from core.domain import object_registry
from core.domain import rte_component_registry
from core.tests import test_utils
from typing import Final, List, Tuple, Type
IGNORED_FILE_SUFFIXES: Final = ['.pyc', '.DS_Store']
RTE_THUMBNAIL_HEIGHT_PX: Final = 16
RTE_THUMBNAIL_WIDTH_PX: Final = 16
_COMPONENT_CONFIG_SCHEMA: List[Tuple[str, Type[object]]] = [('backend_id', str), ('category', str), ('description', str), ('frontend_id', str), ('tooltip', str), ('icon_data_url', str), ('requires_fs', bool), ('is_block_element', bool), ('customization_arg_specs', list)]

class RteComponentUnitTests(test_utils.GenericTestBase):
    """Tests that all the default RTE components are valid."""

    def _is_camel_cased(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        'Check whether a name is in CamelCase.'
        return bool(name and name[0] in string.ascii_uppercase)

    def _is_alphanumeric_string(self, input_string: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Check whether a string is alphanumeric.'
        return bool(re.compile('^[a-zA-Z0-9_]+$').match(input_string))

    def _validate_customization_arg_specs(self, customization_arg_specs: List[rte_component_registry.CustomizationArgSpecDict]) -> None:
        if False:
            print('Hello World!')
        'Validates the given customization arg specs.'
        for ca_spec in customization_arg_specs:
            self.assertEqual(set(ca_spec.keys()), set(['name', 'description', 'schema', 'default_value']))
            self.assertTrue(isinstance(ca_spec['name'], str))
            self.assertTrue(self._is_alphanumeric_string(ca_spec['name']))
            self.assertTrue(isinstance(ca_spec['description'], str))
            self.assertGreater(len(ca_spec['description']), 0)
            schema_utils_test.validate_schema(ca_spec['schema'])
            self.assertEqual(ca_spec['default_value'], schema_utils.normalize_against_schema(ca_spec['default_value'], ca_spec['schema'], apply_custom_validators=False))
            if ca_spec['schema']['type'] == 'custom':
                if ca_spec['schema']['obj_type'] == 'SanitizedUrl':
                    self.assertEqual(ca_spec['default_value'], '')
                else:
                    obj_class = object_registry.Registry.get_object_class_by_type(ca_spec['schema']['obj_type'])
                    self.assertEqual(ca_spec['default_value'], obj_class.normalize(ca_spec['default_value']))

    def _listdir_omit_ignored(self, directory: str) -> List[str]:
        if False:
            print('Hello World!')
        "List all files and directories within 'directory', omitting the ones\n        whose name ends in one of the IGNORED_FILE_SUFFIXES.\n        "
        names = os.listdir(directory)
        for suffix in IGNORED_FILE_SUFFIXES:
            names = [name for name in names if not name.endswith(suffix)]
        return names

    def test_image_thumbnails_for_rte_components(self) -> None:
        if False:
            return 10
        'Test the thumbnails for the RTE component icons.'
        rte_components = rte_component_registry.Registry.get_all_rte_components()
        for (component_name, component_specs) in rte_components.items():
            generated_image_filepath = os.path.join(os.getcwd(), feconf.RTE_EXTENSIONS_DIR, component_name, '%s.png' % component_name)
            relative_icon_data_url = component_specs['icon_data_url'][1:]
            defined_image_filepath = os.path.join(os.getcwd(), feconf.EXTENSIONS_DIR_PREFIX, 'extensions', relative_icon_data_url)
            self.assertEqual(generated_image_filepath, defined_image_filepath)
            with utils.open_file(generated_image_filepath, 'rb', encoding=None) as f:
                img_data = f.read()
                (width, height) = struct.unpack('>LL', img_data[16:24])
                self.assertEqual(int(width), RTE_THUMBNAIL_WIDTH_PX)
                self.assertEqual(int(height), RTE_THUMBNAIL_HEIGHT_PX)

    def test_rte_components_are_valid(self) -> None:
        if False:
            return 10
        'Test that the default RTE components are valid.'
        rte_components = rte_component_registry.Registry.get_all_rte_components()
        for (component_id, component_specs) in rte_components.items():
            hyphenated_component_id = utils.camelcase_to_hyphenated(component_id)
            self.assertTrue(self._is_camel_cased(component_id))
            component_dir = os.path.join(feconf.RTE_EXTENSIONS_DIR, component_id)
            self.assertTrue(os.path.isdir(component_dir))
            dir_contents = self._listdir_omit_ignored(component_dir)
            self.assertLessEqual(len(dir_contents), 5)
            directives_dir = os.path.join(component_dir, 'directives')
            png_file = os.path.join(component_dir, '%s.png' % component_id)
            webdriverio_file = os.path.join(component_dir, 'webdriverio.js')
            self.assertTrue(os.path.isdir(directives_dir))
            self.assertTrue(os.path.isfile(png_file))
            self.assertTrue(os.path.isfile(webdriverio_file))
            main_ts_file = os.path.join(directives_dir, 'oppia-noninteractive-%s.component.ts' % hyphenated_component_id)
            main_html_file = os.path.join(directives_dir, '%s.component.html' % hyphenated_component_id)
            self.assertTrue(os.path.isfile(main_ts_file))
            self.assertTrue(os.path.isfile(main_html_file))
            ts_file_content = utils.get_file_contents(main_ts_file)
            self.assertIn('oppiaNoninteractive%s' % component_id, ts_file_content)
            self.assertNotIn('<script>', ts_file_content)
            self.assertNotIn('</script>', ts_file_content)
            for (item, item_type) in _COMPONENT_CONFIG_SCHEMA:
                self.assertTrue(isinstance(component_specs.get(item), item_type))
                if item_type == str:
                    self.assertTrue(component_specs.get(item))
            self._validate_customization_arg_specs(component_specs['customization_arg_specs'])

    def test_require_file_contains_all_imports(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the rich_text_components.html file contains script-imports\n        for all directives of all RTE components.\n        '
        rtc_ts_filenames: List[str] = []
        for component_id in feconf.ALLOWED_RTE_EXTENSIONS:
            component_dir = os.path.join(feconf.RTE_EXTENSIONS_DIR, component_id)
            directives_dir = os.path.join(component_dir, 'directives')
            directive_filenames = os.listdir(directives_dir)
            rtc_ts_filenames.extend((filename for filename in directive_filenames if filename.endswith('.ts') and (not filename.endswith('.spec.ts'))))
        rtc_ts_file = os.path.join(feconf.RTE_EXTENSIONS_DIR, 'richTextComponentsRequires.ts')
        with utils.open_file(rtc_ts_file, 'r') as f:
            rtc_require_file_contents = f.read()
        for rtc_ts_filename in rtc_ts_filenames:
            self.assertIn(rtc_ts_filename, rtc_require_file_contents)

class RteComponentRegistryUnitTests(test_utils.GenericTestBase):
    """Tests the methods in RteComponentRegistry."""

    def test_get_all_rte_components(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test get_all_rte_components method.'
        obtained_components = list(rte_component_registry.Registry.get_all_rte_components().keys())
        actual_components = [name for name in os.listdir('./extensions/rich_text_components') if os.path.isdir(os.path.join('./extensions/rich_text_components', name)) and name != '__pycache__']
        self.assertEqual(set(obtained_components), set(actual_components))

    def test_get_tag_list_with_attrs(self) -> None:
        if False:
            return 10
        'Test get_tag_list_with_attrs method.'
        obtained_tag_list_with_attrs = rte_component_registry.Registry.get_tag_list_with_attrs()
        actual_tag_list_with_attrs = {}
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        for component_spec in component_specs.values():
            tag_name = 'oppia-noninteractive-%s' % component_spec['frontend_id']
            attr_names = ['%s-with-value' % attr['name'] for attr in component_spec['customization_arg_specs']]
            actual_tag_list_with_attrs[tag_name] = attr_names
        self.assertEqual(set(obtained_tag_list_with_attrs.keys()), set(actual_tag_list_with_attrs.keys()))
        for (key, attrs) in obtained_tag_list_with_attrs.items():
            self.assertEqual(set(attrs), set(actual_tag_list_with_attrs[key]))

    def test_get_component_types_to_component_classes(self) -> None:
        if False:
            while True:
                i = 10
        'Test get_component_types_to_component_classes method.'
        component_types_to_component_classes = rte_component_registry.Registry.get_component_types_to_component_classes()
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        obtained_component_tags = list(component_types_to_component_classes.keys())
        actual_component_tags = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values()]
        self.assertEqual(set(obtained_component_tags), set(actual_component_tags))
        obtained_component_class_names = [component_class.__name__ for component_class in list(component_types_to_component_classes.values())]
        actual_component_class_names = []
        rte_path = [feconf.RTE_EXTENSIONS_DIR]
        for (loader, name, _) in pkgutil.iter_modules(path=rte_path):
            if name == 'components':
                fetched_module = loader.find_module(name)
                assert fetched_module is not None
                module = fetched_module.load_module(name)
                break
        for (name, obj) in inspect.getmembers(module):
            if inspect.isclass(obj) and name != 'BaseRteComponent':
                actual_component_class_names.append(name)
        self.assertEqual(set(obtained_component_class_names), set(actual_component_class_names))

    def test_get_component_tag_names(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test get_component_tag_names method.'
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        keys = ['is_block_element', 'is_complex']
        expected_values = [True, False]
        for key in keys:
            for expected_value in expected_values:
                actual_component_tag_names = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values() if component_spec.get(key) == expected_value]
                obtained_component_tag_names = rte_component_registry.Registry.get_component_tag_names(key, expected_value)
                self.assertEqual(set(actual_component_tag_names), set(obtained_component_tag_names))

    def test_get_inline_component_tag_names(self) -> None:
        if False:
            while True:
                i = 10
        'Test get_inline_component_tag_names method.'
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        obtained_inline_component_tag_names = rte_component_registry.Registry.get_inline_component_tag_names()
        actual_inline_component_tag_names = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values() if not component_spec['is_block_element']]
        self.assertEqual(set(actual_inline_component_tag_names), set(obtained_inline_component_tag_names))

    def test_inline_rte_components_list(self) -> None:
        if False:
            i = 10
            return i + 15
        inline_component_tag_names = rte_component_registry.Registry.get_inline_component_tag_names()
        inline_component_tag_names_from_constant = ['oppia-noninteractive-%s' % element_id for element_id in constants.INLINE_RTE_COMPONENTS]
        self.assertEqual(set(inline_component_tag_names), set(inline_component_tag_names_from_constant))

    def test_get_block_component_tag_names(self) -> None:
        if False:
            while True:
                i = 10
        'Test get_block_component_tag_names method.'
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        obtained_block_component_tag_names = rte_component_registry.Registry.get_block_component_tag_names()
        actual_block_component_tag_names = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values() if component_spec['is_block_element']]
        self.assertEqual(set(actual_block_component_tag_names), set(obtained_block_component_tag_names))

    def test_get_simple_component_tag_names(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test get_simple_component_tag_names method.'
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        obtained_simple_component_tag_names = rte_component_registry.Registry.get_simple_component_tag_names()
        actual_simple_component_tag_names = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values() if not component_spec['is_complex']]
        self.assertEqual(set(actual_simple_component_tag_names), set(obtained_simple_component_tag_names))

    def test_get_complex_component_tag_names(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test get_complex_component_tag_names method.'
        component_specs = rte_component_registry.Registry.get_all_rte_components()
        obtained_complex_component_tag_names = rte_component_registry.Registry.get_complex_component_tag_names()
        actual_complex_component_tag_names = ['oppia-noninteractive-%s' % component_spec['frontend_id'] for component_spec in component_specs.values() if component_spec['is_complex']]
        self.assertEqual(set(actual_complex_component_tag_names), set(obtained_complex_component_tag_names))