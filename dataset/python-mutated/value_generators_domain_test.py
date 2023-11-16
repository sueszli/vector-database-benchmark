"""Unit tests for core.domain.value_generators_domain."""
from __future__ import annotations
import importlib
import inspect
import re
from core.domain import value_generators_domain
from core.tests import test_utils
from extensions.value_generators.models import generators

class ValueGeneratorsUnitTests(test_utils.GenericTestBase):
    """Test the value generator registry."""

    def test_registry_generator_not_found(self) -> None:
        if False:
            while True:
                i = 10
        "Tests that get_generator_class_by_id raises exception\n        when it isn't found.\n        "
        generator_id = 'aajfejaekj'
        with self.assertRaisesRegex(KeyError, generator_id):
            value_generators_domain.Registry.get_generator_class_by_id(generator_id)

    def test_value_generator_registry(self) -> None:
        if False:
            i = 10
            return i + 15
        copier_id = 'Copier'
        copier = value_generators_domain.Registry.get_generator_class_by_id(copier_id)
        self.assertEqual(copier().id, copier_id)
        all_generator_classes = value_generators_domain.Registry.get_all_generator_classes()
        self.assertEqual(len(all_generator_classes), 2)

    def test_generate_value_of_base_value_generator_raises_error(self) -> None:
        if False:
            while True:
                i = 10
        base_generator = value_generators_domain.BaseValueGenerator()
        with self.assertRaisesRegex(NotImplementedError, re.escape('generate_value() method has not yet been implemented')):
            base_generator.generate_value()

    def test_registry_template_random_selector_contents(self) -> None:
        if False:
            print('Hello World!')
        contents_registry = '<schema-based-editor [schema]="SCHEMA" [(ngModel)]="customizationArgs.list_of_values">\n</schema-based-editor>\n'
        class_object = value_generators_domain.Registry()
        self.assertEqual(contents_registry, class_object.get_generator_class_by_id('RandomSelector').get_html_template())

    def test_registry_template_copier_contents(self) -> None:
        if False:
            print('Hello World!')
        contents_registry = '<span class="d-inline-block align-middle">\n  <object-editor [objType]="objType" [initArgs]="initArgs" [(value)]="customizationArgs.value" [alwaysEditable]="true">\n  </object-editor>\n</span>\n'
        class_object = value_generators_domain.Registry()
        self.assertEqual(contents_registry, class_object.get_generator_class_by_id('Copier').get_html_template())

    def test_get_value_generator_classes_not_subclass(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test  that the value generator registry discovers all classes\n        correctly and excludes classes that are not subclasses of\n        BaseValueGenerator.\n        '

        class MockCopier:
            """This is a dummy class for self.swap to test  that the value
            generator registry discovers all classes correctly and excludes
            classes that are not subclasses of BaseValueGenerator.
            We need to have a class in the returned list of value generators
            that isn't a subclass of BaseValueGenerator to test.
            """
            pass
        module = importlib.import_module('extensions.value_generators.models.generators')
        expected_generators = {'RandomSelector': type(generators.RandomSelector())}
        with self.swap(module, 'Copier', MockCopier):
            value_generators = value_generators_domain.Registry.get_all_generator_classes()
        self.assertEqual(expected_generators, value_generators)

class ValueGeneratorNameTests(test_utils.GenericTestBase):

    def test_value_generator_names(self) -> None:
        if False:
            i = 10
            return i + 15
        'This function checks for duplicate value generators.'
        all_python_files = self.get_all_python_files()
        all_value_generators = []
        for file_name in all_python_files:
            python_module = importlib.import_module(file_name)
            for (name, clazz) in inspect.getmembers(python_module, predicate=inspect.isclass):
                all_base_classes = [base_class.__name__ for base_class in inspect.getmro(clazz)]
                if 'BaseValueGenerator' in all_base_classes:
                    all_value_generators.append(name)
        expected_value_generators = ['BaseValueGenerator', 'Copier', 'RandomSelector']
        self.assertEqual(sorted(all_value_generators), sorted(expected_value_generators))