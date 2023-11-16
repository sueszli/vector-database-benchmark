from unittest.mock import patch
from django.conf import settings
from django.core import checks
from django.db import models
from django.test import TestCase, override_settings
from wagtail.models import Locale
from wagtail.test.i18n.models import ClusterableTestModel, ClusterableTestModelChild, ClusterableTestModelTranslatableChild, InheritedTestModel, TestModel

def make_test_instance(model=None, **kwargs):
    if False:
        print('Hello World!')
    if model is None:
        model = TestModel
    return model.objects.create(**kwargs)

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestTranslatableMixin(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        language_codes = dict(settings.LANGUAGES).keys()
        for language_code in language_codes:
            Locale.objects.get_or_create(language_code=language_code)
        self.locale = Locale.objects.get(language_code='en')
        self.another_locale = Locale.objects.get(language_code='fr')
        self.main_instance = make_test_instance(locale=self.locale, title='Main Model')
        self.translated_model = make_test_instance(locale=self.another_locale, translation_key=self.main_instance.translation_key, title='Translated Model')
        make_test_instance()

    def test_get_translations_inclusive_false(self):
        if False:
            print('Hello World!')
        self.assertSequenceEqual(list(self.main_instance.get_translations()), [self.translated_model])

    def test_get_translations_inclusive_true(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(list(self.main_instance.get_translations(inclusive=True)), [self.main_instance, self.translated_model])

    def test_get_translation(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.main_instance.get_translation(self.locale), self.main_instance)

    def test_get_translation_using_locale_id(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.main_instance.get_translation(self.locale.id), self.main_instance)

    def test_get_translation_or_none_return_translation(self):
        if False:
            while True:
                i = 10
        with patch.object(self.main_instance, 'get_translation') as mock_get_translation:
            mock_get_translation.return_value = self.translated_model
            self.assertEqual(self.main_instance.get_translation_or_none(self.another_locale), self.translated_model)

    def test_get_translation_or_none_return_none(self):
        if False:
            i = 10
            return i + 15
        self.translated_model.delete()
        with patch.object(self.main_instance, 'get_translation') as mock_get_translation:
            mock_get_translation.side_effect = self.main_instance.DoesNotExist
            self.assertIsNone(self.main_instance.get_translation_or_none(self.another_locale))

    def test_has_translation_when_exists(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.main_instance.has_translation(self.locale))

    def test_has_translation_when_exists_using_locale_id(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.main_instance.has_translation(self.locale.id))

    def test_has_translation_when_none_exists(self):
        if False:
            return 10
        self.translated_model.delete()
        self.assertFalse(self.main_instance.has_translation(self.another_locale))

    def test_copy_for_translation(self):
        if False:
            print('Hello World!')
        self.translated_model.delete()
        copy = self.main_instance.copy_for_translation(locale=self.another_locale)
        self.assertNotEqual(copy, self.main_instance)
        self.assertEqual(copy.translation_key, self.main_instance.translation_key)
        self.assertEqual(copy.locale, self.another_locale)
        self.assertEqual('Main Model', copy.title)

    def test_get_translation_model(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.main_instance.get_translation_model(), TestModel)
        inherited_model = make_test_instance(model=InheritedTestModel)
        self.assertEqual(inherited_model.get_translation_model(), TestModel)

    def test_copy_inherited_model_for_translation(self):
        if False:
            print('Hello World!')
        instance = make_test_instance(model=InheritedTestModel)
        copy = instance.copy_for_translation(locale=self.another_locale)
        self.assertNotEqual(copy, instance)
        self.assertEqual(copy.translation_key, instance.translation_key)
        self.assertEqual(copy.locale, self.another_locale)

    def test_copy_clusterable_model_for_translation(self):
        if False:
            while True:
                i = 10
        instance = ClusterableTestModel.objects.create(title='A test clusterable model', children=[ClusterableTestModelChild(field='A non-translatable child object')], translatable_children=[ClusterableTestModelTranslatableChild(field='A translatable child object')])
        copy = instance.copy_for_translation(locale=self.another_locale)
        instance_child = instance.children.get()
        copy_child = copy.children.get()
        instance_translatable_child = instance.translatable_children.get()
        copy_translatable_child = copy.translatable_children.get()
        self.assertNotEqual(copy, instance)
        self.assertEqual(copy.translation_key, instance.translation_key)
        self.assertEqual(copy.locale, self.another_locale)
        self.assertNotEqual(copy_child, instance_child)
        self.assertEqual(copy_child.field, 'A non-translatable child object')
        self.assertNotEqual(copy_translatable_child, instance_translatable_child)
        self.assertEqual(copy_translatable_child.field, 'A translatable child object')
        self.assertEqual(copy_translatable_child.translation_key, instance_translatable_child.translation_key)
        self.assertEqual(copy_translatable_child.locale, self.another_locale)

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocalized(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.en_locale = Locale.objects.get()
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.en_instance = make_test_instance(locale=self.en_locale, title='Main Model')
        self.fr_instance = make_test_instance(locale=self.fr_locale, translation_key=self.en_instance.translation_key, title='Main Model')

    def test_localized_same_language(self):
        if False:
            while True:
                i = 10
        with self.assertNumQueries(1):
            instance = self.en_instance.localized
        self.assertEqual(instance, self.en_instance)

    def test_localized_different_language(self):
        if False:
            while True:
                i = 10
        with self.assertNumQueries(2):
            instance = self.fr_instance.localized
        self.assertEqual(instance, self.en_instance)

class TestSystemChecks(TestCase):

    def test_unique_together_raises_no_error(self):
        if False:
            return 10
        errors = TestModel.check()
        self.assertEqual(len(errors), 0)

    def test_unique_constraint_raises_no_error(self):
        if False:
            for i in range(10):
                print('nop')
        previous_unique_together = TestModel._meta.unique_together
        try:
            TestModel._meta.unique_together = []
            TestModel._meta.constraints = [models.UniqueConstraint(fields=['translation_key', 'locale'], name='unique_translation_key_locale_%(app_label)s_%(class)s')]
            errors = TestModel.check()
        finally:
            TestModel._meta.unique_together = previous_unique_together
            TestModel._meta.constraints = []
        self.assertEqual(len(errors), 0)

    def test_raises_error_if_both_unique_constraint_and_unique_together_are_missing(self):
        if False:
            while True:
                i = 10
        previous_unique_together = TestModel._meta.unique_together
        try:
            TestModel._meta.unique_together = []
            errors = TestModel.check()
        finally:
            TestModel._meta.unique_together = previous_unique_together
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], checks.Error)
        self.assertEqual(errors[0].id, 'wagtailcore.E003')
        self.assertEqual(errors[0].msg, "i18n.TestModel is missing a UniqueConstraint for the fields: ('translation_key', 'locale').")
        self.assertEqual(errors[0].hint, "Add models.UniqueConstraint(fields=('translation_key', 'locale'), name='unique_translation_key_locale_i18n_testmodel') to TestModel.Meta.constraints.")

    def test_error_with_both_unique_constraint_and_unique_together(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            TestModel._meta.constraints = [models.UniqueConstraint(fields=['translation_key', 'locale'], name='unique_translation_key_locale_%(app_label)s_%(class)s')]
            errors = TestModel.check()
        finally:
            TestModel._meta.constraints = []
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], checks.Error)
        self.assertEqual(errors[0].id, 'wagtailcore.E003')
        self.assertEqual(errors[0].msg, "i18n.TestModel should not have both UniqueConstraint and unique_together for: ('translation_key', 'locale').")
        self.assertEqual(errors[0].hint, 'Remove unique_together in favor of UniqueConstraint.')