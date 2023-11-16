"""Tests for translation fetchers."""
from __future__ import annotations
from core import feconf
from core.domain import translation_domain
from core.domain import translation_fetchers
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import translation_models
(translation_models,) = models.Registry.import_models([models.Names.TRANSLATION])

class MachineTranslationFetchersTests(test_utils.GenericTestBase):

    def test_get_translation_from_model(self) -> None:
        if False:
            return 10
        model_id = translation_models.MachineTranslationModel.create('en', 'es', 'hello world', 'hola mundo')
        assert model_id is not None
        model_instance = translation_models.MachineTranslationModel.get(model_id)
        assert model_instance is not None
        self.assertEqual(translation_fetchers.get_translation_from_model(model_instance).to_dict(), translation_domain.MachineTranslation('en', 'es', 'hello world', 'hola mundo').to_dict())

    def test_get_machine_translation_with_no_translation_returns_none(self) -> None:
        if False:
            while True:
                i = 10
        translation = translation_fetchers.get_machine_translation('en', 'es', 'untranslated_text')
        self.assertIsNone(translation)

    def test_get_machine_translation_for_cached_translation_returns_from_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        translation_models.MachineTranslationModel.create('en', 'es', 'hello world', 'hola mundo')
        translation = translation_fetchers.get_machine_translation('en', 'es', 'hello world')
        assert translation is not None
        self.assertEqual(translation.translated_text, 'hola mundo')

class EntityTranslationFetchersTests(test_utils.GenericTestBase):

    def test_get_all_entity_translation_objects_for_entity_returns_correclty(self) -> None:
        if False:
            print('Hello World!')
        exp_id = 'exp1'
        entity_translations = translation_fetchers.get_all_entity_translations_for_entity(feconf.TranslatableEntityType.EXPLORATION, exp_id, 5)
        self.assertEqual(len(entity_translations), 0)
        language_codes = ['hi', 'bn']
        for language_code in language_codes:
            translation_models.EntityTranslationsModel.create_new('exploration', exp_id, 5, language_code, {}).put()
        entity_translations = translation_fetchers.get_all_entity_translations_for_entity(feconf.TranslatableEntityType.EXPLORATION, exp_id, 5)
        self.assertEqual(len(entity_translations), 2)
        self.assertItemsEqual([entity_translation.language_code for entity_translation in entity_translations], language_codes)

    def test_get_entity_translation_returns_correctly(self) -> None:
        if False:
            return 10
        exp_id = 'exp1'
        translation_models.EntityTranslationsModel.create_new('exploration', exp_id, 5, 'hi', {}).put()
        entity_translation = translation_fetchers.get_entity_translation(feconf.TranslatableEntityType.EXPLORATION, exp_id, 5, 'hi')
        self.assertEqual(entity_translation.language_code, 'hi')

    def test_get_entity_translation_creates_empty_object(self) -> None:
        if False:
            return 10
        exp_id = 'exp1'
        entity_translation = translation_fetchers.get_entity_translation(feconf.TranslatableEntityType.EXPLORATION, exp_id, 5, 'hi')
        self.assertEqual(entity_translation.language_code, 'hi')
        self.assertEqual(entity_translation.translations, {})