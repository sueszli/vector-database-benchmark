"""Domain objects related to translations."""
from __future__ import annotations
import enum
from core import feconf
from core import utils
from core.constants import constants
from typing import Dict, List, Optional, Union
from typing_extensions import Final, TypedDict
from core.domain import translatable_object_registry

class ContentType(enum.Enum):
    """Represents all possible content types in the State."""
    CONTENT = 'content'
    INTERACTION = 'interaction'
    DEFAULT_OUTCOME = 'default_outcome'
    CUSTOMIZATION_ARG = 'ca'
    RULE = 'rule'
    FEEDBACK = 'feedback'
    HINT = 'hint'
    SOLUTION = 'solution'

class TranslatableContentFormat(enum.Enum):
    """Represents all possible data types for any translatable content."""
    HTML = 'html'
    UNICODE_STRING = 'unicode'
    SET_OF_NORMALIZED_STRING = 'set_of_normalized_string'
    SET_OF_UNICODE_STRING = 'set_of_unicode_string'

    @classmethod
    def is_data_format_list(cls, data_format: str) -> bool:
        if False:
            while True:
                i = 10
        'Checks whether the content of translation with given format is of\n        a list type.\n\n        Args:\n            data_format: str. The format of the translation.\n\n        Returns:\n            bool. Whether the content of translation is a list.\n        '
        return data_format in (cls.SET_OF_NORMALIZED_STRING.value, cls.SET_OF_UNICODE_STRING.value)

class TranslatableContentDict(TypedDict):
    """Dictionary representing TranslatableContent object."""
    content_id: str
    content_value: feconf.ContentValueType
    content_type: str
    content_format: str
    interaction_id: Optional[str]
    rule_type: Optional[str]

class TranslatableContent:
    """TranslatableContent represents a content of a translatable object which
    can be translated into multiple languages.
    """

    def __init__(self, content_id: str, content_type: ContentType, content_format: TranslatableContentFormat, content_value: feconf.ContentValueType, interaction_id: Optional[str]=None, rule_type: Optional[str]=None) -> None:
        if False:
            return 10
        'Constructs an TranslatableContent domain object.\n\n        Args:\n            content_id: str. The id of the corresponding translatable content\n                value.\n            content_type: TranslatableContentFormat. The type of the\n                corresponding content value.\n            content_format: TranslatableContentFormat. The format of the\n                content.\n            content_value: ContentValueType. The content value which can be\n                translated.\n            interaction_id: str|None. The ID of the interaction in which the\n                content is used.\n            rule_type: str|None. The rule type of the answer group in which the\n                content is used.\n        '
        self.content_id = content_id
        self.content_type = content_type
        self.content_format = content_format
        self.content_value = content_value
        self.interaction_id = interaction_id
        self.rule_type = rule_type

    def to_dict(self) -> TranslatableContentDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns the dict representation of TranslatableContent object.\n\n        Returns:\n            TranslatableContentDict. The dict representation of\n            TranslatableContent.\n        '
        return {'content_id': self.content_id, 'content_type': self.content_type.value, 'content_format': self.content_format.value, 'content_value': self.content_value, 'interaction_id': self.interaction_id, 'rule_type': self.rule_type}

    def is_data_format_list(self) -> bool:
        if False:
            return 10
        'Checks whether the content is of a list type.\n\n        Returns:\n            bool. Whether the content is a list.\n        '
        return TranslatableContentFormat.is_data_format_list(self.content_format.value)

class TranslatedContent:
    """Class representing a translation of translatable content. For example,
    if translatable content 'A' is translated into 'B' in a language other than
    English, then 'B' is a TranslatedContent instance that represents this
    class.
    A (TranslatableContent) -----(translation)-----> B (TranslatedContent).

    Args:
        content_value: ContentValueType. Represents translation of translatable
            content.
        content_format: TranslatableContentFormat. The format of the content.
        needs_update: bool. Whether the translation needs an update or not.
    """

    def __init__(self, content_value: feconf.ContentValueType, content_format: TranslatableContentFormat, needs_update: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructor for the TranslatedContent object.\n\n        Args:\n            content_value: ContentValueType. The content value which can be\n                translated.\n            content_format: TranslatableContentFormat. The format of the\n                content.\n            needs_update: bool. Whether the translated content needs update.\n        '
        self.content_value = content_value
        self.content_format = content_format
        self.needs_update = needs_update

    def to_dict(self) -> feconf.TranslatedContentDict:
        if False:
            for i in range(10):
                print('nop')
        "Returns the dict representation of TranslatedContent object.\n\n        Returns:\n            TranslatedContentDict. A dict, mapping content_value and\n            needs_update of a TranslatableContent instance to\n            corresponding keys 'content_value' and 'needs_update'.\n        "
        return {'content_value': self.content_value, 'content_format': self.content_format.value, 'needs_update': self.needs_update}

    @classmethod
    def from_dict(cls, translated_content_dict: feconf.TranslatedContentDict) -> TranslatedContent:
        if False:
            print('Hello World!')
        'Returns the TranslatedContent object.'
        return cls(translated_content_dict['content_value'], TranslatableContentFormat(translated_content_dict['content_format']), translated_content_dict['needs_update'])

class TranslatableContentsCollection:
    """A class to collect all TranslatableContents from a translatable object
    and map them with their corresponding content-ids.
    """
    content_id_to_translatable_content: Dict[str, TranslatableContent]

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        'Constructs a TranslatableContentsCollection object.'
        self.content_id_to_translatable_content = {}

    def add_translatable_field(self, content_id: str, content_type: ContentType, content_format: TranslatableContentFormat, content_value: feconf.ContentValueType, interaction_id: Optional[str]=None, rule_type: Optional[str]=None) -> None:
        if False:
            return 10
        "Adds translatable field parameter to\n        'content_id_to_translatable_content' dict.\n\n        Args:\n            content_id: str. The id of the corresponding translatable content\n                value.\n            content_type: TranslatableContentFormat. The type of the\n                corresponding content value.\n            content_format: TranslatableContentFormat. The format of the\n                content.\n            content_value: ContentValueType. The content value which can be\n                translated.\n            interaction_id: str|None. The ID of the interaction in which the\n                content is used.\n            rule_type: str|None. The rule type of the answer group in which the\n                content is used.\n\n        Raises:\n            Exception. The content_id_to_translatable_content dict already\n                contains the content_id.\n        "
        if content_id in self.content_id_to_translatable_content:
            raise Exception('Content_id %s already exists in the TranslatableContentsCollection.' % content_id)
        self.content_id_to_translatable_content[content_id] = TranslatableContent(content_id, content_type, content_format, content_value, interaction_id, rule_type)

    def add_fields_from_translatable_object(self, translatable_object: BaseTranslatableObject, **kwargs: Optional[str]) -> None:
        if False:
            return 10
        "Adds translatable fields from a translatable object parameter to\n        'content_id_to_translatable_content' dict.\n\n        NOTE: The functions take the entire translatable object as a param, as\n        the process to fetch translatable collections from different objects\n        are the same, and keeping this logic in one place will help avoid\n        duplicate patterns in the callsite. It will also help the callsite\n        look cleaner.\n\n        Args:\n            translatable_object: BaseTranslatableObject. An instance of\n                BaseTranslatableObject class.\n            **kwargs: *. The keyword args for registring translatable object.\n        "
        self.content_id_to_translatable_content.update(translatable_object.get_translatable_contents_collection(**kwargs).content_id_to_translatable_content)

class BaseTranslatableObject:
    """Base class for all translatable objects which contain translatable
    fields/objects. For example, a State is a translatable object in
    Exploration, a Hint is a translatable object in State, and a hint_content
    is a translatable field in Hint. So Exploration, State, and Hint all should
    be child classes of BaseTranslatableObject.
    """

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> TranslatableContentsCollection:
        if False:
            return 10
        'Get all translatable fields in a translatable object.\n\n        Raises:\n            NotImplementedError. The derived child class must implement the\n                necessary logic to get all translatable fields in a\n                translatable object.\n        '
        raise NotImplementedError('Must be implemented in subclasses.')

    def get_translatable_content_ids(self) -> List[str]:
        if False:
            while True:
                i = 10
        "Get all translatable content's Ids.\n\n        Returns:\n            list(str). A list of translatable content's Id.\n        "
        content_collection = self.get_translatable_contents_collection()
        return list(content_collection.content_id_to_translatable_content.keys())

    def get_all_contents_which_need_translations(self, entity_translation: Union[EntityTranslation, None]=None) -> Dict[str, TranslatableContent]:
        if False:
            i = 10
            return i + 15
        'Returns a list of TranslatableContent instances which need new or\n        updated translations.\n\n        Args:\n            entity_translation: EntityTranslation. An object storing the\n                existing translations of an entity.\n\n        Returns:\n            list(TranslatableContent). Returns a list of TranslatableContent.\n        '
        if entity_translation is None:
            entity_translation = EntityTranslation.create_empty(entity_type=feconf.TranslatableEntityType.EXPLORATION, entity_id='', language_code='')
        translatable_content_list = self.get_translatable_contents_collection().content_id_to_translatable_content.values()
        content_id_to_translatable_content = {}
        for translatable_content in translatable_content_list:
            content_value = translatable_content.content_value
            if content_value == '':
                continue
            if translatable_content.content_id not in entity_translation.translations:
                content_id_to_translatable_content[translatable_content.content_id] = translatable_content
            elif entity_translation.translations[translatable_content.content_id].needs_update:
                content_id_to_translatable_content[translatable_content.content_id] = translatable_content
        return content_id_to_translatable_content

    def get_translation_count(self, entity_translation: EntityTranslation) -> int:
        if False:
            return 10
        'Returs the number of updated translations avialable.\n\n        Args:\n            entity_translation: EntityTranslation. The translation object\n                containing translations.\n\n        Returns:\n            int. The number of translatable contnet for which translations are\n            available in the given translation object.\n        '
        count = 0
        for content_id in self.get_all_contents_which_need_translations():
            if not content_id in entity_translation.translations:
                continue
            if not entity_translation.translations[content_id].needs_update:
                count += 1
        return count

    def are_translations_displayable(self, entity_translation: EntityTranslation) -> bool:
        if False:
            while True:
                i = 10
        "Whether the given EntityTranslation in the given lanaguage is\n        displayable.\n\n        A language's translations are ready to be displayed if there are less\n        than five missing or update-needed translations. In addition, all\n        rule-related translations must be present.\n\n        Args:\n            entity_translation: EntityTranslation. An object storing the\n                existing translations of an entity.\n\n        Returns:\n            list(TranslatableContent). Returns a list of TranslatableContent.\n        "
        content_id_to_translatable_content = self.get_translatable_contents_collection().content_id_to_translatable_content
        for (content_id, translatable_content) in content_id_to_translatable_content.items():
            if translatable_content.content_type == ContentType.RULE and (not content_id in entity_translation.translations):
                return False
        translatable_content_count = self.get_content_count()
        translated_content_count = self.get_translation_count(entity_translation)
        translations_missing_count = translatable_content_count - translated_content_count
        return translations_missing_count < feconf.MIN_ALLOWED_MISSING_OR_UPDATE_NEEDED_WRITTEN_TRANSLATIONS

    def get_content_count(self) -> int:
        if False:
            print('Hello World!')
        'Returns the total number of distinct content fields available in the\n        exploration which are user facing and can be translated into\n        different languages.\n\n        (The content field includes state content, feedback, hints, solutions.)\n\n        Returns:\n            int. The total number of distinct content fields available inside\n            the exploration.\n        '
        return len(self.get_all_contents_which_need_translations())

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Gets all html content strings used in the object.\n\n        Returns:\n            list(str). The list of html content strings.\n        '
        html_list = []
        content_collection = self.get_translatable_contents_collection()
        translatable_contents = content_collection.content_id_to_translatable_content.values()
        for translatable_content in translatable_contents:
            if translatable_content.content_format == TranslatableContentFormat.HTML:
                assert isinstance(translatable_content.content_value, str)
                html_list.append(translatable_content.content_value)
        return html_list

    def validate_translatable_contents(self, next_content_id_index: int) -> None:
        if False:
            i = 10
            return i + 15
        'Validates the content Ids of the translatable contents.\n\n        Args:\n            next_content_id_index: int. The index for generating the Id\n                for a content.\n        '
        content_id_to_translatable_content = self.get_translatable_contents_collection().content_id_to_translatable_content
        for content_id in content_id_to_translatable_content.keys():
            content_id_suffix = content_id.split('_')[-1]
            if content_id_suffix.isdigit() and int(content_id_suffix) > next_content_id_index:
                raise utils.ValidationError('Expected all content id indexes to be less than the "next content id index(%s)", but received content id %s' % (next_content_id_index, content_id))

class EntityTranslationDict(TypedDict):
    """Dictionary representing the EntityTranslation object."""
    entity_id: str
    entity_type: str
    entity_version: int
    language_code: str
    translations: Dict[str, feconf.TranslatedContentDict]

class EntityTranslation:
    """A domain object to store all translations for a given versioned-entity
    in a given language.

    NOTE: This domain object corresponds to EntityTranslationsModel in the
    storage layer.

    Args:
        entity_id: str. The id of the corresponding entity.
        entity_type: TranslatableEntityType. The type
            of the corresponding entity.
        entity_version: str. The version of the corresponding entity.
        language_code: str. The language code for the corresponding entity.
        translations: dict(str, TranslatedContent). A dict representing
            content-id as keys and TranslatedContent instance as values.
    """

    def __init__(self, entity_id: str, entity_type: feconf.TranslatableEntityType, entity_version: int, language_code: str, translations: Dict[str, TranslatedContent]):
        if False:
            while True:
                i = 10
        'Constructs an TranslatableContent domain object.\n\n        Args:\n            entity_id: str. The ID of the entity.\n            entity_type: TranslatableEntityType. The type of the entity.\n            entity_version: int. The version of the entity.\n            language_code: str. The langauge code for the translated contents\n                language.\n            translations: dict(str, TranslatedContent). The translations dict\n                containing content_id as key and TranslatedContent as value.\n        '
        self.entity_id = entity_id
        self.entity_type = entity_type.value
        self.entity_version = entity_version
        self.language_code = language_code
        self.translations = translations

    def to_dict(self) -> EntityTranslationDict:
        if False:
            print('Hello World!')
        'Returns the dict representation of the EntityTranslation object.\n\n        Returns:\n            EntityTranslationDict. The dict representation of the\n            EntityTranslation object.\n        '
        translations_dict = {content_id: translated_content.to_dict() for (content_id, translated_content) in self.translations.items()}
        return {'entity_id': self.entity_id, 'entity_type': self.entity_type, 'entity_version': self.entity_version, 'language_code': self.language_code, 'translations': translations_dict}

    @classmethod
    def from_dict(cls, entity_translation_dict: EntityTranslationDict) -> EntityTranslation:
        if False:
            while True:
                i = 10
        'Creates the EntityTranslation from the given dict.\n\n        Args:\n            entity_translation_dict: EntityTranslationDict. The dict\n                representation of the EntityTranslation object.\n\n        Returns:\n            EntityTranslation. The EntityTranslation object created using the\n            given dict.\n        '
        translations_dict = entity_translation_dict['translations']
        content_id_to_translated_content = {}
        for (content_id, translated_content) in translations_dict.items():
            content_id_to_translated_content[content_id] = TranslatedContent.from_dict(translated_content)
        return cls(entity_translation_dict['entity_id'], feconf.TranslatableEntityType(entity_translation_dict['entity_type']), entity_translation_dict['entity_version'], entity_translation_dict['language_code'], content_id_to_translated_content)

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates the EntityTranslation object.'
        if not isinstance(self.entity_type, str):
            raise utils.ValidationError('entity_type must be a string, recieved %r' % self.entity_type)
        if not isinstance(self.entity_id, str):
            raise utils.ValidationError('entity_id must be a string, recieved %r' % self.entity_id)
        if not isinstance(self.entity_version, int):
            raise utils.ValidationError('entity_version must be an int, recieved %r' % self.entity_version)
        if not isinstance(self.language_code, str):
            raise utils.ValidationError('language_code must be a string, recieved %r' % self.language_code)
        for (content_id, translated_content) in self.translations.items():
            if not isinstance(content_id, str):
                raise utils.ValidationError('content_id must be a string, recieved %r' % content_id)
            if not isinstance(translated_content.needs_update, bool):
                raise utils.ValidationError('needs_update must be a bool, recieved %r' % translated_content.needs_update)

    def add_translation(self, content_id: str, content_value: feconf.ContentValueType, content_format: TranslatableContentFormat, needs_update: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds new TranslatedContent in the object.\n\n        Args:\n            content_id: str. The ID of the content.\n            content_value: ContentValueType. The translation content.\n            content_format: TranslatableContentFormat. The format of the\n                content.\n            needs_update: bool. Whether the translation needs update.\n        '
        self.translations[content_id] = TranslatedContent(content_value, content_format, needs_update)

    def remove_translations(self, content_ids: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Remove translations for the given list of content Ids.\n\n        Args:\n            content_ids: list(str). The list of content Ids for removing\n                translations.\n        '
        for content_id in content_ids:
            if content_id in self.translations:
                del self.translations[content_id]

    def mark_translations_needs_update(self, content_ids: List[str]) -> None:
        if False:
            return 10
        'Marks translation needs update for the given list of content Ids.\n\n        Args:\n            content_ids: list(str). The list of content Ids for to mark their\n                translation needs update.\n        '
        for content_id in content_ids:
            if content_id in self.translations:
                self.translations[content_id].needs_update = True

    @classmethod
    def create_empty(cls, entity_type: feconf.TranslatableEntityType, entity_id: str, language_code: str, entity_version: int=0) -> EntityTranslation:
        if False:
            while True:
                i = 10
        'Creates a new and empty EntityTranslation object.'
        return cls(entity_id=entity_id, entity_type=entity_type, entity_version=entity_version, language_code=language_code, translations={})

class MachineTranslation:
    """Domain object for machine translation of exploration content."""

    def __init__(self, source_language_code: str, target_language_code: str, source_text: str, translated_text: str) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a MachineTranslation domain object.\n\n        Args:\n            source_language_code: str. The language code for the source text\n                language. Must be different from target_language_code.\n            target_language_code: str. The language code for the target\n                translation language. Must be different from\n                source_language_code.\n            source_text: str. The untranslated source text.\n            translated_text: str. The machine generated translation of the\n                source text into the target language.\n        '
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code
        self.source_text = source_text
        self.translated_text = translated_text

    def validate(self) -> None:
        if False:
            return 10
        'Validates properties of the MachineTranslation.\n\n        Raises:\n            ValidationError. One or more attributes of the MachineTranslation\n                are invalid.\n        '
        if not utils.is_supported_audio_language_code(self.source_language_code) and (not utils.is_valid_language_code(self.source_language_code)):
            raise utils.ValidationError('Invalid source language code: %s' % self.source_language_code)
        if not utils.is_supported_audio_language_code(self.target_language_code) and (not utils.is_valid_language_code(self.target_language_code)):
            raise utils.ValidationError('Invalid target language code: %s' % self.target_language_code)
        if self.source_language_code == self.target_language_code:
            raise utils.ValidationError('Expected source_language_code to be different from target_language_code: "%s" = "%s"' % (self.source_language_code, self.target_language_code))

    def to_dict(self) -> Dict[str, str]:
        if False:
            return 10
        'Converts the MachineTranslation domain instance into a dictionary\n        form with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the MachineTranslation class\n            information in a dictionary form.\n        '
        return {'source_language_code': self.source_language_code, 'target_language_code': self.target_language_code, 'source_text': self.source_text, 'translated_text': self.translated_text}

class WrittenTranslationDict(TypedDict):
    """Dictionary representing the WrittenTranslation object."""
    data_format: str
    translation: Union[str, List[str]]
    needs_update: bool

class WrittenTranslation:
    """Value object representing a written translation for a content.

    Here, "content" could mean a string or a list of strings. The latter arises,
    for example, in the case where we are checking for equality of a learner's
    answer against a given set of strings. In such cases, the number of strings
    in the translation of the original object may not be the same as the number
    of strings in the original object.
    """
    DATA_FORMAT_HTML: Final = 'html'
    DATA_FORMAT_UNICODE_STRING: Final = 'unicode'
    DATA_FORMAT_SET_OF_NORMALIZED_STRING: Final = 'set_of_normalized_string'
    DATA_FORMAT_SET_OF_UNICODE_STRING: Final = 'set_of_unicode_string'
    DATA_FORMAT_TO_TRANSLATABLE_OBJ_TYPE: Dict[str, translatable_object_registry.TranslatableObjectNames] = {DATA_FORMAT_HTML: 'TranslatableHtml', DATA_FORMAT_UNICODE_STRING: 'TranslatableUnicodeString', DATA_FORMAT_SET_OF_NORMALIZED_STRING: 'TranslatableSetOfNormalizedString', DATA_FORMAT_SET_OF_UNICODE_STRING: 'TranslatableSetOfUnicodeString'}

    def __init__(self, data_format: str, translation: Union[str, List[str]], needs_update: bool) -> None:
        if False:
            return 10
        'Initializes a WrittenTranslation domain object.\n\n        Args:\n            data_format: str. One of the keys in\n                DATA_FORMAT_TO_TRANSLATABLE_OBJ_TYPE. Indicates the\n                type of the field (html, unicode, etc.).\n            translation: str|list(str). A user-submitted string or list of\n                strings that matches the given data format.\n            needs_update: bool. Whether the translation is marked as needing\n                review.\n        '
        self.data_format = data_format
        self.translation = translation
        self.needs_update = needs_update

    def to_dict(self) -> WrittenTranslationDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this WrittenTranslation domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of WrittenTranslation instance.\n        '
        return {'data_format': self.data_format, 'translation': self.translation, 'needs_update': self.needs_update}

    @classmethod
    def from_dict(cls, written_translation_dict: WrittenTranslationDict) -> WrittenTranslation:
        if False:
            print('Hello World!')
        'Return a WrittenTranslation domain object from a dict.\n\n        Args:\n            written_translation_dict: dict. The dict representation of\n                WrittenTranslation object.\n\n        Returns:\n            WrittenTranslation. The corresponding WrittenTranslation domain\n            object.\n        '
        return cls(written_translation_dict['data_format'], written_translation_dict['translation'], written_translation_dict['needs_update'])

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates properties of the WrittenTranslation, normalizing the\n        translation if needed.\n\n        Raises:\n            ValidationError. One or more attributes of the WrittenTranslation\n                are invalid.\n        '
        if self.data_format not in self.DATA_FORMAT_TO_TRANSLATABLE_OBJ_TYPE:
            raise utils.ValidationError('Invalid data_format: %s' % self.data_format)
        translatable_class_name = self.DATA_FORMAT_TO_TRANSLATABLE_OBJ_TYPE[self.data_format]
        translatable_obj_class = translatable_object_registry.Registry.get_object_class(translatable_class_name)
        self.translation = translatable_obj_class.normalize_value(self.translation)
        if not isinstance(self.needs_update, bool):
            raise utils.ValidationError('Expected needs_update to be a bool, received %s' % self.needs_update)

class WrittenTranslationsDict(TypedDict):
    """Dictionary representing the WrittenTranslations object."""
    translations_mapping: Dict[str, Dict[str, WrittenTranslationDict]]

class WrittenTranslations:
    """Value object representing a content translations which stores
    translated contents of all state contents (like hints, feedback etc.) in
    different languages linked through their content_id.
    """

    def __init__(self, translations_mapping: Dict[str, Dict[str, WrittenTranslation]]) -> None:
        if False:
            return 10
        'Initializes a WrittenTranslations domain object.\n\n        Args:\n            translations_mapping: dict. A dict mapping the content Ids\n                to the dicts which is the map of abbreviated code of the\n                languages to WrittenTranslation objects.\n        '
        self.translations_mapping = translations_mapping

    def to_dict(self) -> WrittenTranslationsDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this WrittenTranslations domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of WrittenTranslations instance.\n        '
        translations_mapping: Dict[str, Dict[str, WrittenTranslationDict]] = {}
        for (content_id, language_code_to_written_translation) in self.translations_mapping.items():
            translations_mapping[content_id] = {}
            for (language_code, written_translation) in language_code_to_written_translation.items():
                translations_mapping[content_id][language_code] = written_translation.to_dict()
        written_translations_dict: WrittenTranslationsDict = {'translations_mapping': translations_mapping}
        return written_translations_dict

    @classmethod
    def from_dict(cls, written_translations_dict: WrittenTranslationsDict) -> WrittenTranslations:
        if False:
            print('Hello World!')
        'Returns a WrittenTranslations domain object from a dict.\n\n        Args:\n            written_translations_dict: dict. The dict representation of\n                WrittenTranslations object.\n\n        Returns:\n            WrittenTranslations. The corresponding WrittenTranslations domain\n            object.\n        '
        translations_mapping: Dict[str, Dict[str, WrittenTranslation]] = {}
        for (content_id, language_code_to_written_translation) in written_translations_dict['translations_mapping'].items():
            translations_mapping[content_id] = {}
            for (language_code, written_translation) in language_code_to_written_translation.items():
                translations_mapping[content_id][language_code] = WrittenTranslation.from_dict(written_translation)
        return cls(translations_mapping)

    def validate(self, expected_content_id_list: Optional[List[str]]) -> None:
        if False:
            i = 10
            return i + 15
        'Validates properties of the WrittenTranslations.\n\n        Args:\n            expected_content_id_list: list(str)|None. A list of content id which\n                are expected to be inside they WrittenTranslations.\n\n        Raises:\n            ValidationError. One or more attributes of the WrittenTranslations\n                are invalid.\n        '
        if expected_content_id_list is not None:
            if not set(self.translations_mapping.keys()) == set(expected_content_id_list):
                raise utils.ValidationError('Expected state written_translations to match the listed content ids %s, found %s' % (expected_content_id_list, list(self.translations_mapping.keys())))
        for (content_id, language_code_to_written_translation) in self.translations_mapping.items():
            if not isinstance(content_id, str):
                raise utils.ValidationError('Expected content_id to be a string, received %s' % content_id)
            if not isinstance(language_code_to_written_translation, dict):
                raise utils.ValidationError('Expected content_id value to be a dict, received %s' % language_code_to_written_translation)
            for (language_code, written_translation) in language_code_to_written_translation.items():
                if not isinstance(language_code, str):
                    raise utils.ValidationError('Expected language_code to be a string, received %s' % language_code)
                allowed_language_codes = [language['id'] for language in constants.SUPPORTED_AUDIO_LANGUAGES]
                if language_code not in allowed_language_codes:
                    raise utils.ValidationError('Invalid language_code: %s' % language_code)
                written_translation.validate()

    def add_content_id_for_translation(self, content_id: str) -> None:
        if False:
            print('Hello World!')
        "Adds a content id as a key for the translation into the\n        content_translation dict.\n\n        Args:\n            content_id: str. The id representing a subtitled html.\n\n        Raises:\n            Exception. The content id isn't a string.\n        "
        if not isinstance(content_id, str):
            raise Exception('Expected content_id to be a string, received %s' % content_id)
        if content_id in self.translations_mapping:
            raise Exception('The content_id %s already exist.' % content_id)
        self.translations_mapping[content_id] = {}

    def delete_content_id_for_translation(self, content_id: str) -> None:
        if False:
            i = 10
            return i + 15
        "Deletes a content id from the content_translation dict.\n\n        Args:\n            content_id: str. The id representing a subtitled html.\n\n        Raises:\n            Exception. The content id isn't a string.\n        "
        if not isinstance(content_id, str):
            raise Exception('Expected content_id to be a string, received %s' % content_id)
        if content_id not in self.translations_mapping:
            raise Exception('The content_id %s does not exist.' % content_id)
        self.translations_mapping.pop(content_id, None)

class ContentIdGenerator:
    """Class to generate the content-id for a translatable content based on the
    next_content_id_index variable.
    """

    def __init__(self, start_index: int=0) -> None:
        if False:
            return 10
        'Constructs an ContentIdGenerator object.'
        self.next_content_id_index = start_index

    def generate(self, content_type: ContentType, extra_prefix: Optional[str]=None) -> str:
        if False:
            return 10
        'Generates the new content-id from the next content id.'
        content_id = content_type.value + '_'
        if extra_prefix:
            content_id += extra_prefix + '_'
        content_id += str(self.next_content_id_index)
        self.next_content_id_index += 1
        return content_id