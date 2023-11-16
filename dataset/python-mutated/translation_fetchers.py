"""Functions for converting translation models into domain objects."""
from __future__ import annotations
from core import feconf
from core.domain import translation_domain
from core.platform import models
from typing import List, Optional
MYPY = False
if MYPY:
    from mypy_imports import translation_models
(translation_models,) = models.Registry.import_models([models.Names.TRANSLATION])

def get_translation_from_model(translation_model: translation_models.MachineTranslationModel) -> translation_domain.MachineTranslation:
    if False:
        while True:
            i = 10
    'Returns a MachineTranslation object given a\n    MachineTranslationModel loaded from the datastore.\n\n    Args:\n        translation_model: MachineTranslationModel. The\n            MachineTranslationModel loaded from the datastore.\n\n    Returns:\n        MachineTranslation. A MachineTranslation object corresponding to\n        the given MachineTranslationModel.\n    '
    return translation_domain.MachineTranslation(translation_model.source_language_code, translation_model.target_language_code, translation_model.source_text, translation_model.translated_text)

def get_machine_translation(source_language_code: str, target_language_code: str, source_text: str) -> Optional[translation_domain.MachineTranslation]:
    if False:
        return 10
    'Gets MachineTranslation by language codes and source text.\n    Returns None if no translation exists for the given parameters.\n\n    Args:\n        source_language_code: str. The language code for the source text\n            language. Must be different from target_language_code.\n        target_language_code: str. The language code for the target translation\n            language. Must be different from source_language_code.\n        source_text: str. The untranslated source text.\n\n    Returns:\n        MachineTranslation|None. The MachineTranslation\n        if a translation exists or None if no translation is found.\n    '
    translation_model = translation_models.MachineTranslationModel.get_machine_translation(source_language_code, target_language_code, source_text)
    if translation_model is None:
        return None
    return get_translation_from_model(translation_model)

def _get_entity_translation_from_model(entity_translation_model: translation_models.EntityTranslationsModel) -> translation_domain.EntityTranslation:
    if False:
        while True:
            i = 10
    'Returns the EntityTranslation domain object from its model representation\n    (EntityTranslationsModel).\n\n    Args:\n        entity_translation_model: EntityTranslatioModel. An instance of\n            EntityTranslationsModel.\n\n    Returns:\n        EntityTranslation. An instance of EntityTranslation object, created from\n        its model.\n    '
    entity_translation = translation_domain.EntityTranslation.from_dict({'entity_id': entity_translation_model.entity_id, 'entity_type': entity_translation_model.entity_type, 'entity_version': entity_translation_model.entity_version, 'language_code': entity_translation_model.language_code, 'translations': entity_translation_model.translations})
    return entity_translation

def get_all_entity_translations_for_entity(entity_type: feconf.TranslatableEntityType, entity_id: str, entity_version: int) -> List[translation_domain.EntityTranslation]:
    if False:
        i = 10
        return i + 15
    'Returns a list of entity translation domain objects.\n\n    Args:\n        entity_type: TranslatableEntityType. The type of the entity whose\n            translations are to be fetched.\n        entity_id: str. The ID of the entity whose translations are to be\n            fetched.\n        entity_version: int. The version of the entity whose translations\n            are to be fetched.\n\n    Returns:\n        list(EnitityTranslation). A list of EntityTranslation domain objects.\n    '
    entity_translation_models = translation_models.EntityTranslationsModel.get_all_for_entity(entity_type, entity_id, entity_version)
    entity_translation_objects = []
    for model in entity_translation_models:
        domain_object = _get_entity_translation_from_model(model)
        entity_translation_objects.append(domain_object)
    return entity_translation_objects

def get_entity_translation(entity_type: feconf.TranslatableEntityType, entity_id: str, entity_version: int, language_code: str) -> translation_domain.EntityTranslation:
    if False:
        for i in range(10):
            print('nop')
    'Returns a unique entity translation domain object.\n\n    Args:\n        entity_type: TranslatableEntityType. The type of the entity.\n        entity_id: str. The ID of the entity.\n        entity_version: int. The version of the entity.\n        language_code: str. The language code for the entity.\n\n    Returns:\n        EntityTranslation. An instance of entity translations.\n    '
    entity_translation_model = translation_models.EntityTranslationsModel.get_model(entity_type, entity_id, entity_version, language_code)
    if entity_translation_model:
        domain_object = _get_entity_translation_from_model(entity_translation_model)
        return domain_object
    return translation_domain.EntityTranslation.create_empty(entity_type, entity_id, language_code, entity_version=entity_version)