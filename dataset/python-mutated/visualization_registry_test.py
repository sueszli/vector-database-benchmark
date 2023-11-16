"""Tests for methods in the visualization registry."""
from __future__ import annotations
import importlib
import inspect
import re
from core.domain import visualization_registry
from core.tests import test_utils

class VisualizationRegistryUnitTests(test_utils.GenericTestBase):
    """Test for the visualization registry."""

    def test_visualization_registry(self) -> None:
        if False:
            i = 10
            return i + 15
        'Sanity checks on the visualization registry.'
        self.assertGreater(len(visualization_registry.Registry.get_all_visualization_ids()), 0)

    def test_get_visualization_class_with_invalid_id_raises_error(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'is not a valid visualization id.'):
            visualization_registry.Registry.get_visualization_class('invalid_visualization_id')

    def test_visualization_class_with_invalid_option_names(self) -> None:
        if False:
            return 10
        sorted_tiles = visualization_registry.Registry.get_visualization_class('SortedTiles')
        sorted_tiles_instance = sorted_tiles('AnswerFrequencies', {}, True)
        with self.assertRaisesRegex(Exception, re.escape("For visualization SortedTiles, expected option names ['header', 'use_percentages']; received names []")):
            sorted_tiles_instance.validate()

    def test_visualization_class_with_invalid_option_value(self) -> None:
        if False:
            return 10
        sorted_tiles = visualization_registry.Registry.get_visualization_class('SortedTiles')
        option_names = {'header': 'Pretty Tiles!', 'use_percentages': 'invalid_value'}
        sorted_tiles_instance = sorted_tiles('AnswerFrequencies', option_names, True)
        with self.assertRaisesRegex(Exception, 'Expected bool, received invalid_value'):
            sorted_tiles_instance.validate()

    def test_visualization_class_with_invalid_addressed_info_is_supported_value(self) -> None:
        if False:
            print('Hello World!')
        sorted_tiles = visualization_registry.Registry.get_visualization_class('SortedTiles')
        option_names = {'header': 'Pretty Tiles!', 'use_percentages': True}
        sorted_tiles_instance = sorted_tiles('AnswerFrequencies', option_names, 'invalid_value')
        with self.assertRaisesRegex(Exception, 'For visualization SortedTiles, expected a bool value for addressed_info_is_supported; received invalid_value'):
            sorted_tiles_instance.validate()

    def test_get_all_visualization_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        visualization_ids = visualization_registry.Registry.get_all_visualization_ids()
        expected_visualizations = ['FrequencyTable', 'ClickHexbins', 'EnumeratedFrequencyTable', 'SortedTiles']
        self.assertEqual(sorted(visualization_ids), sorted(expected_visualizations))

class VisualizationsNameTests(test_utils.GenericTestBase):

    def test_visualization_names(self) -> None:
        if False:
            return 10
        'This function checks for duplicate visualizations.'
        all_python_files = self.get_all_python_files()
        all_visualizations = []
        for file_name in all_python_files:
            python_module = importlib.import_module(file_name)
            for (name, clazz) in inspect.getmembers(python_module, predicate=inspect.isclass):
                all_base_classes = [base_class.__name__ for base_class in inspect.getmro(clazz)]
                if 'BaseVisualization' in all_base_classes:
                    all_visualizations.append(name)
        expected_visualizations = ['BaseVisualization', 'FrequencyTable', 'EnumeratedFrequencyTable', 'ClickHexbins', 'SortedTiles']
        self.assertEqual(sorted(all_visualizations), sorted(expected_visualizations))