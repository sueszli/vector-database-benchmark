"""Tests for the translatable object registry."""
from __future__ import annotations
from core.domain import translatable_object_registry
from core.tests import test_utils
from extensions.objects.models import objects

class TranslatableObjectRegistryUnitTests(test_utils.GenericTestBase):
    """Test the Registry class in translatable_object_registry."""

    def test_get_object_class_method(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests the normal behavior of get_object_class().'
        retrieved_class = translatable_object_registry.Registry.get_object_class('TranslatableHtml')
        self.assertEqual(retrieved_class.__name__, 'TranslatableHtml')

    def test_nontranslatable_class_is_not_gettable(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that trying to retrieve a non-translatable class raises an\n        error.\n        '
        with self.assertRaisesRegex(TypeError, 'not a valid translatable object class'):
            translatable_object_registry.Registry.get_object_class('Int')

    def test_fake_class_is_not_gettable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that trying to retrieve a fake class raises an error.'
        with self.assertRaisesRegex(TypeError, 'not a valid translatable object class'):
            translatable_object_registry.Registry.get_object_class('FakeClass')

    def test_base_objects_are_not_gettable(self) -> None:
        if False:
            return 10
        'Tests that the base objects exist but are not included in the\n        registry.\n        '
        assert getattr(objects, 'BaseObject')
        with self.assertRaisesRegex(TypeError, 'not a valid translatable object class'):
            translatable_object_registry.Registry.get_object_class('BaseObject')
        assert getattr(objects, 'BaseTranslatableObject')
        with self.assertRaisesRegex(TypeError, 'not a valid translatable object class'):
            translatable_object_registry.Registry.get_object_class('BaseTranslatableObject')

    def test_get_translatable_object_classes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests the normal behavior of get_translatable_object_classes().'
        class_names_to_classes = translatable_object_registry.Registry.get_all_class_names()
        self.assertEqual(class_names_to_classes, ['TranslatableHtml', 'TranslatableSetOfNormalizedString', 'TranslatableSetOfUnicodeString', 'TranslatableUnicodeString'])