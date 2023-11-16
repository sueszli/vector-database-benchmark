"""Domain objects for a collection and its constituents.

Domain objects capture domain-specific logic and are agnostic of how the
objects they represent are stored. All methods and properties in this file
should therefore be independent of the specific storage models used.
"""
from __future__ import annotations
import datetime
import json
import re
import string
from core import feconf
from core import utils
from core.constants import constants
from core.domain import change_domain
from typing import Dict, Final, List, Literal, Optional, TypedDict, cast
COLLECTION_PROPERTY_TITLE: Final = 'title'
COLLECTION_PROPERTY_CATEGORY: Final = 'category'
COLLECTION_PROPERTY_OBJECTIVE: Final = 'objective'
COLLECTION_PROPERTY_LANGUAGE_CODE: Final = 'language_code'
COLLECTION_PROPERTY_TAGS: Final = 'tags'
COLLECTION_NODE_PROPERTY_PREREQUISITE_SKILL_IDS: Final = 'prerequisite_skill_ids'
COLLECTION_NODE_PROPERTY_ACQUIRED_SKILL_IDS: Final = 'acquired_skill_ids'
COLLECTION_NODE_PROPERTY_PREREQUISITE_SKILLS: Final = 'prerequisite_skills'
COLLECTION_NODE_PROPERTY_ACQUIRED_SKILLS: Final = 'acquired_skills'
CMD_CREATE_NEW: Final = 'create_new'
CMD_ADD_COLLECTION_NODE: Final = 'add_collection_node'
CMD_DELETE_COLLECTION_NODE: Final = 'delete_collection_node'
CMD_SWAP_COLLECTION_NODES: Final = 'swap_nodes'
CMD_EDIT_COLLECTION_PROPERTY: Final = 'edit_collection_property'
CMD_EDIT_COLLECTION_NODE_PROPERTY: Final = 'edit_collection_node_property'
CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION: Final = 'migrate_schema_to_latest_version'
CMD_ADD_COLLECTION_SKILL: Final = 'add_collection_skill'
CMD_DELETE_COLLECTION_SKILL: Final = 'delete_collection_skill'
CMD_ADD_QUESTION_ID_TO_SKILL: Final = 'add_question_id_to_skill'
CMD_REMOVE_QUESTION_ID_FROM_SKILL: Final = 'remove_question_id_from_skill'
_SKILL_ID_PREFIX: Final = 'skill'

class CollectionChange(change_domain.BaseChange):
    """Domain object class for a change to a collection.

    IMPORTANT: Ensure that all changes to this class (and how these cmds are
    interpreted in general) preserve backward-compatibility with the
    collection snapshots in the datastore. Do not modify the definitions of
    cmd keys that already exist.

    The allowed commands, together with the attributes:
        - 'add_collection_node' (with exploration_id)
        - 'delete_collection_node' (with exploration_id)
        - 'edit_collection_node_property' (with exploration_id,
            property_name, new_value and, optionally, old_value)
        - 'edit_collection_property' (with property_name, new_value
            and, optionally, old_value)
        - 'migrate_schema' (with from_version and to_version)
    For a collection, property_name must be one of
    COLLECTION_PROPERTIES.
    """
    COLLECTION_PROPERTIES: List[str] = [COLLECTION_PROPERTY_TITLE, COLLECTION_PROPERTY_CATEGORY, COLLECTION_PROPERTY_OBJECTIVE, COLLECTION_PROPERTY_LANGUAGE_CODE, COLLECTION_PROPERTY_TAGS]
    ALLOWED_COMMANDS: List[feconf.ValidCmdDict] = [{'name': CMD_CREATE_NEW, 'required_attribute_names': ['category', 'title'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_ADD_COLLECTION_NODE, 'required_attribute_names': ['exploration_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_DELETE_COLLECTION_NODE, 'required_attribute_names': ['exploration_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_SWAP_COLLECTION_NODES, 'required_attribute_names': ['first_index', 'second_index'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_EDIT_COLLECTION_PROPERTY, 'required_attribute_names': ['property_name', 'new_value'], 'optional_attribute_names': ['old_value'], 'user_id_attribute_names': [], 'allowed_values': {'property_name': COLLECTION_PROPERTIES}, 'deprecated_values': {}}, {'name': CMD_EDIT_COLLECTION_NODE_PROPERTY, 'required_attribute_names': ['exploration_id', 'property_name', 'new_value'], 'optional_attribute_names': ['old_value'], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION, 'required_attribute_names': ['from_version', 'to_version'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_ADD_COLLECTION_SKILL, 'required_attribute_names': ['name'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_DELETE_COLLECTION_SKILL, 'required_attribute_names': ['skill_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_ADD_QUESTION_ID_TO_SKILL, 'required_attribute_names': ['question_id', 'skill_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_REMOVE_QUESTION_ID_FROM_SKILL, 'required_attribute_names': ['question_id', 'skill_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}]

class CreateNewCollectionCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_CREATE_NEW command.
    """
    category: str
    title: str

class AddCollectionNodeCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_ADD_COLLECTION_NODE command.
    """
    exploration_id: str

class DeleteCollectionNodeCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_DELETE_COLLECTION_NODE command.
    """
    exploration_id: str

class SwapCollectionNodesCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_SWAP_COLLECTION_NODES command.
    """
    first_index: int
    second_index: int

class EditCollectionPropertyTitleCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_PROPERTY command with
    COLLECTION_PROPERTY_TITLE as allowed value.
    """
    property_name: Literal['title']
    new_value: str
    old_value: str

class EditCollectionPropertyCategoryCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_PROPERTY command with
    COLLECTION_PROPERTY_CATEGORY as allowed value.
    """
    property_name: Literal['category']
    new_value: str
    old_value: str

class EditCollectionPropertyObjectiveCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_PROPERTY command with
    COLLECTION_PROPERTY_OBJECTIVE as allowed value.
    """
    property_name: Literal['objective']
    new_value: str
    old_value: str

class EditCollectionPropertyLanguageCodeCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_PROPERTY command with
    COLLECTION_PROPERTY_LANGUAGE_CODE as allowed value.
    """
    property_name: Literal['language_code']
    new_value: str
    old_value: str

class EditCollectionPropertyTagsCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_PROPERTY command with
    COLLECTION_PROPERTY_TAGS as allowed value.
    """
    property_name: Literal['tags']
    new_value: List[str]
    old_value: List[str]

class EditCollectionNodePropertyCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_EDIT_COLLECTION_NODE_PROPERTY command.
    """
    exploration_id: str
    property_name: str
    new_value: str
    old_value: str

class MigrateSchemaToLatestVersionCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION command.
    """
    from_version: int
    to_version: int

class AddCollectionSkillCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_ADD_COLLECTION_SKILL command.
    """
    name: str

class DeleteCollectionSkillCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_DELETE_COLLECTION_SKILL command.
    """
    skill_id: str

class AddQuestionIdToSkillCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_ADD_QUESTION_ID_TO_SKILL command.
    """
    question_id: str
    skill_id: str

class RemoveQuestionIdFromSkillCmd(CollectionChange):
    """Class representing the CollectionChange's
    CMD_ADD_QUESTION_ID_TO_SKILL command.
    """
    question_id: str
    skill_id: str

class CollectionNodeDict(TypedDict):
    """Dictionary representing the CollectionNode object."""
    exploration_id: str

class CollectionNode:
    """Domain object describing a node in the exploration graph of a
    collection. The node contains the reference to
    its exploration (its ID).
    """

    def __init__(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Initializes a CollectionNode domain object.\n\n        Args:\n            exploration_id: str. A valid ID of an exploration referenced by\n                this node.\n        '
        self.exploration_id = exploration_id

    def to_dict(self) -> CollectionNodeDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this CollectionNode domain object.\n\n        Returns:\n            dict. A dict, mapping all fields (exploration_id,\n            prerequisite_skill_ids, acquired_skill_ids) of CollectionNode\n            instance.\n        '
        return {'exploration_id': self.exploration_id}

    @classmethod
    def from_dict(cls, node_dict: CollectionNodeDict) -> CollectionNode:
        if False:
            for i in range(10):
                print('nop')
        'Return a CollectionNode domain object from a dict.\n\n        Args:\n            node_dict: dict. The dict representation of CollectionNode object.\n\n        Returns:\n            CollectionNode. The corresponding CollectionNode domain object.\n        '
        return cls(node_dict['exploration_id'])

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates various properties of the collection node.\n\n        Raises:\n            ValidationError. One or more attributes of the collection node are\n                invalid.\n        '
        if not isinstance(self.exploration_id, str):
            raise utils.ValidationError('Expected exploration ID to be a string, received %s' % self.exploration_id)

    @classmethod
    def create_default_node(cls, exploration_id: str) -> CollectionNode:
        if False:
            while True:
                i = 10
        'Returns a CollectionNode domain object with default values.\n\n        Args:\n            exploration_id: str. The id of the exploration.\n\n        Returns:\n            CollectionNode. The CollectionNode domain object with default\n            value.\n        '
        return cls(exploration_id)

class CollectionDict(TypedDict):
    """Dictionary representing the Collection object."""
    id: str
    title: str
    category: str
    objective: str
    language_code: str
    tags: List[str]
    schema_version: int
    nodes: List[CollectionNodeDict]

class SerializableCollectionDict(CollectionDict):
    """Dictionary representing the serializable Collection object."""
    version: int
    created_on: str
    last_updated: str

class VersionedCollectionDict(TypedDict):
    """Dictionary representing versioned Collection object."""
    schema_version: int
    collection_contents: CollectionDict

class Collection:
    """Domain object for an Oppia collection."""

    def __init__(self, collection_id: str, title: str, category: str, objective: str, language_code: str, tags: List[str], schema_version: int, nodes: List[CollectionNode], version: int, created_on: Optional[datetime.datetime]=None, last_updated: Optional[datetime.datetime]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Constructs a new collection given all the information necessary to\n        represent a collection.\n\n        Note: The schema_version represents the version of any underlying\n        dictionary or list structures stored within the collection. In\n        particular, the schema for CollectionNodes is represented by this\n        version. If the schema for CollectionNode changes, then a migration\n        function will need to be added to this class to convert from the\n        current schema version to the new one. This function should be called\n        in both from_yaml in this class and\n        collection_services._migrate_collection_contents_to_latest_schema.\n        feconf.CURRENT_COLLECTION_SCHEMA_VERSION should be incremented and the\n        new value should be saved in the collection after the migration\n        process, ensuring it represents the latest schema version.\n\n        Args:\n            collection_id: str. The unique id of the collection.\n            title: str. The title of the collection.\n            category: str. The category of the collection.\n            objective: str. The objective of the collection.\n            language_code: str. The language code of the collection (like 'en'\n                for English).\n            tags: list(str). The list of tags given to the collection.\n            schema_version: int. The schema version for the collection.\n            nodes: list(CollectionNode). The list of nodes present in the\n                collection.\n            version: int. The version of the collection.\n            created_on: datetime.datetime. Date and time when the collection is\n                created.\n            last_updated: datetime.datetime. Date and time when the\n                collection was last updated.\n        "
        self.id = collection_id
        self.title = title
        self.category = category
        self.objective = objective
        self.language_code = language_code
        self.tags = tags
        self.schema_version = schema_version
        self.nodes = nodes
        self.version = version
        self.created_on = created_on
        self.last_updated = last_updated

    def to_dict(self) -> CollectionDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this Collection domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Collection instance.\n        '
        return {'id': self.id, 'title': self.title, 'category': self.category, 'objective': self.objective, 'language_code': self.language_code, 'tags': self.tags, 'schema_version': self.schema_version, 'nodes': [node.to_dict() for node in self.nodes]}

    @classmethod
    def create_default_collection(cls, collection_id: str, title: str=feconf.DEFAULT_COLLECTION_TITLE, category: str=feconf.DEFAULT_COLLECTION_CATEGORY, objective: str=feconf.DEFAULT_COLLECTION_OBJECTIVE, language_code: str=constants.DEFAULT_LANGUAGE_CODE) -> Collection:
        if False:
            print('Hello World!')
        "Returns a Collection domain object with default values.\n\n        Args:\n            collection_id: str. The unique id of the collection.\n            title: str. The title of the collection.\n            category: str. The category of the collection.\n            objective: str. The objective of the collection.\n            language_code: str. The language code of the collection (like 'en'\n                for English).\n\n        Returns:\n            Collection. The Collection domain object with the default\n            values.\n        "
        return cls(collection_id, title, category, objective, language_code, [], feconf.CURRENT_COLLECTION_SCHEMA_VERSION, [], 0)

    @classmethod
    def from_dict(cls, collection_dict: CollectionDict, collection_version: int=0, collection_created_on: Optional[datetime.datetime]=None, collection_last_updated: Optional[datetime.datetime]=None) -> Collection:
        if False:
            for i in range(10):
                print('nop')
        'Return a Collection domain object from a dict.\n\n        Args:\n            collection_dict: dict. The dictionary representation of the\n                collection.\n            collection_version: int. The version of the collection.\n            collection_created_on: datetime.datetime. Date and time when the\n                collection is created.\n            collection_last_updated: datetime.datetime. Date and time when\n                the collection is updated last time.\n\n        Returns:\n            Collection. The corresponding Collection domain object.\n        '
        collection = cls(collection_dict['id'], collection_dict['title'], collection_dict['category'], collection_dict['objective'], collection_dict['language_code'], collection_dict['tags'], collection_dict['schema_version'], [CollectionNode.from_dict(node_dict) for node_dict in collection_dict['nodes']], collection_version, collection_created_on, collection_last_updated)
        return collection

    @classmethod
    def deserialize(cls, json_string: str) -> Collection:
        if False:
            return 10
        'Returns a Collection domain object decoded from a JSON string.\n\n        Args:\n            json_string: str. A JSON-encoded string that can be\n                decoded into a dictionary representing a Collection.\n                Only call on strings that were created using serialize().\n\n        Returns:\n            Collection. The corresponding Collection domain object.\n        '
        collection_dict = json.loads(json_string)
        created_on = utils.convert_string_to_naive_datetime_object(collection_dict['created_on']) if 'created_on' in collection_dict else None
        last_updated = utils.convert_string_to_naive_datetime_object(collection_dict['last_updated']) if 'last_updated' in collection_dict else None
        collection = cls.from_dict(collection_dict, collection_version=collection_dict['version'], collection_created_on=created_on, collection_last_updated=last_updated)
        return collection

    def serialize(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the object serialized as a JSON string.\n\n        Returns:\n            str. JSON-encoded str encoding all of the information composing\n            the object.\n        '
        collection_dict: SerializableCollectionDict = self.to_dict()
        collection_dict['version'] = self.version
        if self.created_on:
            collection_dict['created_on'] = utils.convert_naive_datetime_to_string(self.created_on)
        if self.last_updated:
            collection_dict['last_updated'] = utils.convert_naive_datetime_to_string(self.last_updated)
        return json.dumps(collection_dict)

    def to_yaml(self) -> str:
        if False:
            i = 10
            return i + 15
        'Convert the Collection domain object into YAML.\n\n        Returns:\n            str. The YAML representation of this Collection.\n        '
        collection_dict = self.to_dict()
        del collection_dict['id']
        return utils.yaml_from_dict(collection_dict)

    @classmethod
    def _convert_v1_dict_to_v2_dict(cls, collection_dict: CollectionDict) -> CollectionDict:
        if False:
            print('Hello World!')
        'Converts a v1 collection dict into a v2 collection dict.\n\n        Adds a language code, and tags.\n\n        Args:\n            collection_dict: dict. The dict representation of a collection with\n                schema version v1.\n\n        Returns:\n            dict. The dict representation of the Collection domain object,\n            following schema version v2.\n        '
        collection_dict['schema_version'] = 2
        collection_dict['language_code'] = constants.DEFAULT_LANGUAGE_CODE
        collection_dict['tags'] = []
        return collection_dict

    @classmethod
    def _convert_v2_dict_to_v3_dict(cls, collection_dict: CollectionDict) -> CollectionDict:
        if False:
            print('Hello World!')
        'Converts a v2 collection dict into a v3 collection dict.\n\n        This function does nothing as the collection structure is changed in\n        collection_services.get_collection_from_model.\n\n        Args:\n            collection_dict: dict. The dict representation of a collection with\n                schema version v2.\n\n        Returns:\n            dict. The dict representation of the Collection domain object,\n            following schema version v3.\n        '
        collection_dict['schema_version'] = 3
        return collection_dict

    @classmethod
    def _convert_v3_dict_to_v4_dict(cls, collection_dict: CollectionDict) -> CollectionDict:
        if False:
            print('Hello World!')
        'Converts a v3 collection dict into a v4 collection dict.\n\n        This migrates the structure of skills, see the docstring in\n        _convert_collection_contents_v3_dict_to_v4_dict.\n        '
        new_collection_dict = cls._convert_collection_contents_v3_dict_to_v4_dict(collection_dict)
        collection_dict['skills'] = new_collection_dict['skills']
        collection_dict['next_skill_id'] = new_collection_dict['next_skill_id']
        collection_dict['schema_version'] = 4
        return collection_dict

    @classmethod
    def _convert_v4_dict_to_v5_dict(cls, collection_dict: CollectionDict) -> CollectionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts a v4 collection dict into a v5 collection dict.\n\n        This changes the field name of next_skill_id to next_skill_index.\n        '
        cls._convert_collection_contents_v4_dict_to_v5_dict(collection_dict)
        collection_dict['schema_version'] = 5
        return collection_dict

    @classmethod
    def _convert_v5_dict_to_v6_dict(cls, collection_dict: CollectionDict) -> CollectionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts a v5 collection dict into a v6 collection dict.\n\n        This changes the structure of each node to not include skills as well\n        as remove skills from the Collection model itself.\n        '
        del collection_dict['skills']
        del collection_dict['next_skill_index']
        collection_dict['schema_version'] = 6
        return collection_dict

    @classmethod
    def _migrate_to_latest_yaml_version(cls, yaml_content: str) -> CollectionDict:
        if False:
            return 10
        "Return the YAML content of the collection in the latest schema\n        format.\n\n        Args:\n            yaml_content: str. The YAML representation of the collection.\n\n        Returns:\n            Dict. The dictionary representation of the collection in which\n            the latest YAML representation of the collection and latest\n            schema format is used.\n\n        Raises:\n            InvalidInputException. The 'yaml_content' or the schema version\n                is not specified.\n            Exception. The collection schema version is not valid.\n        "
        try:
            collection_dict = cast(CollectionDict, utils.dict_from_yaml(yaml_content))
        except utils.InvalidInputException as e:
            raise utils.InvalidInputException('Please ensure that you are uploading a YAML text file, not a zip file. The YAML parser returned the following error: %s' % e)
        collection_schema_version = collection_dict.get('schema_version')
        if collection_schema_version is None:
            raise utils.InvalidInputException('Invalid YAML file: no schema version specified.')
        if not 1 <= collection_schema_version <= feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
            raise Exception('Sorry, we can only process v1 to v%s collection YAML files at present.' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION)
        while collection_schema_version < feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
            conversion_fn = getattr(cls, '_convert_v%s_dict_to_v%s_dict' % (collection_schema_version, collection_schema_version + 1))
            collection_dict = conversion_fn(collection_dict)
            collection_schema_version += 1
        return collection_dict

    @classmethod
    def from_yaml(cls, collection_id: str, yaml_content: str) -> Collection:
        if False:
            i = 10
            return i + 15
        'Converts a YAML string to a Collection domain object.\n\n        Args:\n            collection_id: str. The id of the collection.\n            yaml_content: str. The YAML representation of the collection.\n\n        Returns:\n            Collection. The corresponding collection domain object.\n        '
        collection_dict = cls._migrate_to_latest_yaml_version(yaml_content)
        collection_dict['id'] = collection_id
        return Collection.from_dict(collection_dict)

    @classmethod
    def _convert_collection_contents_v1_dict_to_v2_dict(cls, collection_contents: CollectionDict) -> CollectionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts from version 1 to 2. Does nothing since this migration only\n        changes the language code.\n\n        Args:\n            collection_contents: dict. A dict representing the collection\n                contents object to convert.\n\n        Returns:\n            dict. The updated collection_contents dict.\n        '
        return collection_contents

    @classmethod
    def _convert_collection_contents_v2_dict_to_v3_dict(cls, collection_contents: CollectionDict) -> CollectionDict:
        if False:
            while True:
                i = 10
        'Converts from version 2 to 3. Does nothing since the changes are\n        handled while loading the collection.\n\n        Args:\n            collection_contents: dict. A dict representing the collection\n                contents object to convert.\n\n        Returns:\n            dict. The updated collection_contents dict.\n        '
        return collection_contents

    @classmethod
    def _convert_collection_contents_v3_dict_to_v4_dict(cls, collection_contents: CollectionDict) -> CollectionDict:
        if False:
            print('Hello World!')
        'Converts from version 3 to 4.\n\n        Adds a skills dict and skill id counter. Migrates prerequisite_skills\n        and acquired_skills to prerequistite_skill_ids and acquired_skill_ids.\n        Then, gets skills in prerequisite_skill_ids and acquired_skill_ids in\n        nodes, and assigns them IDs.\n\n        Args:\n            collection_contents: dict. A dict representing the collection\n                contents object to convert.\n\n        Returns:\n            dict. The updated collection_contents dict.\n        '
        skill_names = set()
        for node in collection_contents['nodes']:
            skill_names.update(node['acquired_skills'])
            skill_names.update(node['prerequisite_skills'])
        skill_names_to_ids = {name: _SKILL_ID_PREFIX + str(index) for (index, name) in enumerate(sorted(skill_names))}
        collection_contents['nodes'] = [{'exploration_id': node['exploration_id'], 'prerequisite_skill_ids': [skill_names_to_ids[prerequisite_skill_name] for prerequisite_skill_name in node['prerequisite_skills']], 'acquired_skill_ids': [skill_names_to_ids[acquired_skill_name] for acquired_skill_name in node['acquired_skills']]} for node in collection_contents['nodes']]
        collection_contents['skills'] = {skill_id: {'name': skill_name, 'question_ids': []} for (skill_name, skill_id) in skill_names_to_ids.items()}
        collection_contents['next_skill_id'] = len(skill_names)
        return collection_contents

    @classmethod
    def _convert_collection_contents_v4_dict_to_v5_dict(cls, collection_contents: CollectionDict) -> CollectionDict:
        if False:
            i = 10
            return i + 15
        "Converts from version 4 to 5.\n\n        Converts next_skill_id to next_skill_index, since next_skill_id isn't\n        actually a skill ID.\n\n        Args:\n            collection_contents: dict. A dict representing the collection\n                contents object to convert.\n\n        Returns:\n            dict. The updated collection_contents dict.\n        "
        collection_contents['next_skill_index'] = collection_contents['next_skill_id']
        del collection_contents['next_skill_id']
        return collection_contents

    @classmethod
    def _convert_collection_contents_v5_dict_to_v6_dict(cls, collection_contents: CollectionDict) -> CollectionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts from version 5 to 6.\n\n        Removes skills from collection node.\n\n        Args:\n            collection_contents: dict. A dict representing the collection\n                contents object to convert.\n\n        Returns:\n            dict. The updated collection_contents dict.\n        '
        for node in collection_contents['nodes']:
            del node['prerequisite_skill_ids']
            del node['acquired_skill_ids']
        return collection_contents

    @classmethod
    def update_collection_contents_from_model(cls, versioned_collection_contents: VersionedCollectionDict, current_version: int) -> None:
        if False:
            i = 10
            return i + 15
        "Converts the states blob contained in the given\n        versioned_collection_contents dict from current_version to\n        current_version + 1. Note that the versioned_collection_contents being\n        passed in is modified in-place.\n\n        Args:\n            versioned_collection_contents: dict. A dict with two keys:\n                - schema_version: int. The schema version for the collection.\n                - collection_contents: dict. The dict comprising the collection\n                    contents.\n            current_version: int. The current collection schema version.\n\n        Raises:\n            Exception. The value of the key 'schema_version' in\n                versioned_collection_contents is not valid.\n        "
        if versioned_collection_contents['schema_version'] + 1 > feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
            raise Exception('Collection is version %d but current collection schema version is %d' % (versioned_collection_contents['schema_version'], feconf.CURRENT_COLLECTION_SCHEMA_VERSION))
        versioned_collection_contents['schema_version'] = current_version + 1
        conversion_fn = getattr(cls, '_convert_collection_contents_v%s_dict_to_v%s_dict' % (current_version, current_version + 1))
        versioned_collection_contents['collection_contents'] = conversion_fn(versioned_collection_contents['collection_contents'])

    @property
    def exploration_ids(self) -> List[str]:
        if False:
            return 10
        'Returns a list of all the exploration IDs that are part of this\n        collection.\n\n        Returns:\n            list(str). List of exploration IDs.\n        '
        return [node.exploration_id for node in self.nodes]

    @property
    def first_exploration_id(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the first element in the node list of the collection, which\n           corresponds to the first node that the user would encounter, or if\n           the collection is empty, returns None.\n\n        Returns:\n            str|None. The exploration ID of the first node, or None if the\n            collection is empty.\n        '
        if len(self.nodes) > 0:
            return self.nodes[0].exploration_id
        else:
            return None

    def get_next_exploration_id(self, completed_exp_ids: List[str]) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the first exploration id in the collection that has not yet\n           been completed by the learner, or if the collection is completed,\n           returns None.\n\n        Args:\n            completed_exp_ids: list(str). List of completed exploration\n                ids.\n\n        Returns:\n            str|None. The exploration ID of the next node,\n            or None if the collection is completed.\n        '
        for exp_id in self.exploration_ids:
            if exp_id not in completed_exp_ids:
                return exp_id
        return None

    def get_next_exploration_id_in_sequence(self, current_exploration_id: str) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Returns the exploration ID of the node just after the node\n           corresponding to the current exploration id. If the user is on the\n           last node, None is returned.\n\n        Args:\n            current_exploration_id: str. The id of exploration currently\n                completed.\n\n        Returns:\n            str|None. The exploration ID of the next node,\n            or None if the passed id is the last one in the collection.\n        '
        exploration_just_unlocked = None
        for index in range(len(self.nodes) - 1):
            if self.nodes[index].exploration_id == current_exploration_id:
                exploration_just_unlocked = self.nodes[index + 1].exploration_id
                break
        return exploration_just_unlocked

    @classmethod
    def is_demo_collection_id(cls, collection_id: str) -> bool:
        if False:
            while True:
                i = 10
        'Whether the collection id is that of a demo collection.\n\n        Args:\n            collection_id: str. The id of the collection.\n\n        Returns:\n            bool. True if the collection is a demo else False.\n        '
        return collection_id in feconf.DEMO_COLLECTIONS

    @property
    def is_demo(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether the collection is one of the demo collections.\n\n        Returs:\n            bool. True if the collection is a demo else False.\n        '
        return self.is_demo_collection_id(self.id)

    def update_title(self, title: str) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the title of the collection.\n\n        Args:\n            title: str. The new title of the collection.\n        '
        self.title = title

    def update_category(self, category: str) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the category of the collection.\n\n        Args:\n            category: str. The new category of the collection.\n        '
        self.category = category

    def update_objective(self, objective: str) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the objective of the collection.\n\n        Args:\n            objective: str. The new objective of the collection.\n        '
        self.objective = objective

    def update_language_code(self, language_code: str) -> None:
        if False:
            print('Hello World!')
        'Updates the language code of the collection.\n\n        Args:\n            language_code: str. The new language code of the collection.\n        '
        self.language_code = language_code

    def update_tags(self, tags: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Updates the tags of the collection.\n\n        Args:\n            tags: list(str). The new tags of the collection.\n        '
        self.tags = tags

    def _find_node(self, exploration_id: str) -> Optional[int]:
        if False:
            return 10
        'Returns the index of the collection node with the given exploration\n        id, or None if the exploration id is not in the nodes list.\n\n        Args:\n            exploration_id: str. The id of the exploration.\n\n        Returns:\n            int or None. The index of the corresponding node, or None if there\n            is no such node.\n        '
        for (ind, node) in enumerate(self.nodes):
            if node.exploration_id == exploration_id:
                return ind
        return None

    def get_node(self, exploration_id: str) -> Optional[CollectionNode]:
        if False:
            i = 10
            return i + 15
        'Retrieves a collection node from the collection based on an\n        exploration ID.\n\n        Args:\n            exploration_id: str. The id of the exploration.\n\n        Returns:\n            CollectionNode or None. If the list of nodes contains the given\n            exploration then it will return the corresponding node, else None.\n        '
        for node in self.nodes:
            if node.exploration_id == exploration_id:
                return node
        return None

    def add_node(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Adds a new node to the collection; the new node represents the given\n        exploration_id.\n\n        Args:\n            exploration_id: str. The id of the exploration.\n\n        Raises:\n            ValueError. The exploration is already part of the colletion.\n        '
        if self.get_node(exploration_id) is not None:
            raise ValueError('Exploration is already part of this collection: %s' % exploration_id)
        self.nodes.append(CollectionNode.create_default_node(exploration_id))

    def swap_nodes(self, first_index: int, second_index: int) -> None:
        if False:
            return 10
        'Swaps the values of 2 nodes in the collection.\n\n        Args:\n            first_index: int. Index of one of the nodes to be swapped.\n            second_index: int. Index of the other node to be swapped.\n\n        Raises:\n            ValueError. Both indices are the same number.\n        '
        if first_index == second_index:
            raise ValueError('Both indices point to the same collection node.')
        temp = self.nodes[first_index]
        self.nodes[first_index] = self.nodes[second_index]
        self.nodes[second_index] = temp

    def delete_node(self, exploration_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes the node corresponding to the given exploration from the\n        collection.\n\n        Args:\n            exploration_id: str. The id of the exploration.\n\n        Raises:\n            ValueError. The exploration is not part of the collection.\n        '
        node_index = self._find_node(exploration_id)
        if node_index is None:
            raise ValueError('Exploration is not part of this collection: %s' % exploration_id)
        del self.nodes[node_index]

    def validate(self, strict: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates all properties of this collection and its constituents.\n\n        Raises:\n            ValidationError. One or more attributes of the Collection are not\n                valid.\n        '
        if not isinstance(self.title, str):
            raise utils.ValidationError('Expected title to be a string, received %s' % self.title)
        utils.require_valid_name(self.title, 'the collection title', allow_empty=True)
        if not isinstance(self.category, str):
            raise utils.ValidationError('Expected category to be a string, received %s' % self.category)
        utils.require_valid_name(self.category, 'the collection category', allow_empty=True)
        if not isinstance(self.objective, str):
            raise utils.ValidationError('Expected objective to be a string, received %s' % self.objective)
        if not isinstance(self.language_code, str):
            raise utils.ValidationError('Expected language code to be a string, received %s' % self.language_code)
        if not self.language_code:
            raise utils.ValidationError("A language must be specified (in the 'Settings' tab).")
        if not utils.is_valid_language_code(self.language_code):
            raise utils.ValidationError('Invalid language code: %s' % self.language_code)
        if not isinstance(self.tags, list):
            raise utils.ValidationError('Expected tags to be a list, received %s' % self.tags)
        if len(set(self.tags)) < len(self.tags):
            raise utils.ValidationError('Expected tags to be unique, but found duplicates')
        for tag in self.tags:
            if not isinstance(tag, str):
                raise utils.ValidationError("Expected each tag to be a string, received '%s'" % tag)
            if not tag:
                raise utils.ValidationError('Tags should be non-empty.')
            if not re.match(constants.TAG_REGEX, tag):
                raise utils.ValidationError("Tags should only contain lowercase letters and spaces, received '%s'" % tag)
            if tag[0] not in string.ascii_lowercase or tag[-1] not in string.ascii_lowercase:
                raise utils.ValidationError("Tags should not start or end with whitespace, received  '%s'" % tag)
            if re.search('\\s\\s+', tag):
                raise utils.ValidationError("Adjacent whitespace in tags should be collapsed, received '%s'" % tag)
        if not isinstance(self.schema_version, int):
            raise utils.ValidationError('Expected schema version to be an integer, received %s' % self.schema_version)
        if self.schema_version != feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
            raise utils.ValidationError('Expected schema version to be %s, received %s' % (feconf.CURRENT_COLLECTION_SCHEMA_VERSION, self.schema_version))
        if not isinstance(self.nodes, list):
            raise utils.ValidationError('Expected nodes to be a list, received %s' % self.nodes)
        all_exp_ids = self.exploration_ids
        if len(set(all_exp_ids)) != len(all_exp_ids):
            raise utils.ValidationError('There are explorations referenced in the collection more than once.')
        for node in self.nodes:
            node.validate()
        if strict:
            if not self.title:
                raise utils.ValidationError('A title must be specified for the collection.')
            if not self.objective:
                raise utils.ValidationError('An objective must be specified for the collection.')
            if not self.category:
                raise utils.ValidationError('A category must be specified for the collection.')
            if not self.nodes:
                raise utils.ValidationError('Expected to have at least 1 exploration in the collection.')

class CollectionSummaryDict(TypedDict):
    """Dictionary representing the CollectionSummary object."""
    id: str
    title: str
    category: str
    objective: str
    language_code: str
    tags: List[str]
    status: str
    community_owned: bool
    owner_ids: List[str]
    editor_ids: List[str]
    viewer_ids: List[str]
    contributor_ids: List[str]
    contributors_summary: Dict[str, int]
    version: int
    collection_model_created_on: datetime.datetime
    collection_model_last_updated: datetime.datetime

class CollectionSummary:
    """Domain object for an Oppia collection summary."""

    def __init__(self, collection_id: str, title: str, category: str, objective: str, language_code: str, tags: List[str], status: str, community_owned: bool, owner_ids: List[str], editor_ids: List[str], viewer_ids: List[str], contributor_ids: List[str], contributors_summary: Dict[str, int], version: int, node_count: int, collection_model_created_on: datetime.datetime, collection_model_last_updated: datetime.datetime) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a CollectionSummary domain object.\n\n        Args:\n            collection_id: str. The unique id of the collection.\n            title: str. The title of the collection.\n            category: str. The category of the collection.\n            objective: str. The objective of the collection.\n            language_code: str. The language code of the collection.\n            tags: list(str). The tags given to the collection.\n            status: str. The status of the collection.\n            community_owned: bool. Whether the collection is community-owned.\n            owner_ids: list(str). List of the user ids who are the owner of\n                this collection.\n            editor_ids: list(str). List of the user ids of the users who have\n                access to edit this collection.\n            viewer_ids: list(str). List of the user ids of the users who have\n                view this collection.\n            contributor_ids: list(str). List of the user ids of the user who\n                have contributed to  this collection.\n            contributors_summary: dict. The summary given by the contributors\n                to the collection, user id as the key and summary as value.\n            version: int. The version of the collection.\n            node_count: int. The number of nodes present in the collection.\n            collection_model_created_on: datetime.datetime. Date and time when\n                the collection model is created.\n            collection_model_last_updated: datetime.datetime. Date and time\n                when the collection model was last updated.\n        '
        self.id = collection_id
        self.title = title
        self.category = category
        self.objective = objective
        self.language_code = language_code
        self.tags = tags
        self.status = status
        self.community_owned = community_owned
        self.owner_ids = owner_ids
        self.editor_ids = editor_ids
        self.viewer_ids = viewer_ids
        self.contributor_ids = contributor_ids
        self.contributors_summary = contributors_summary
        self.version = version
        self.node_count = node_count
        self.collection_model_created_on = collection_model_created_on
        self.collection_model_last_updated = collection_model_last_updated

    def to_dict(self) -> CollectionSummaryDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this CollectionSummary domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of CollectionSummary instance.\n        '
        return {'id': self.id, 'title': self.title, 'category': self.category, 'objective': self.objective, 'language_code': self.language_code, 'tags': self.tags, 'status': self.status, 'community_owned': self.community_owned, 'owner_ids': self.owner_ids, 'editor_ids': self.editor_ids, 'viewer_ids': self.viewer_ids, 'contributor_ids': self.contributor_ids, 'contributors_summary': self.contributors_summary, 'version': self.version, 'collection_model_created_on': self.collection_model_created_on, 'collection_model_last_updated': self.collection_model_last_updated}

    def validate(self) -> None:
        if False:
            return 10
        'Validates various properties of the CollectionSummary.\n\n        Raises:\n            ValidationError. One or more attributes of the CollectionSummary\n                are invalid.\n        '
        utils.require_valid_name(self.title, 'the collection title', allow_empty=True)
        utils.require_valid_name(self.category, 'the collection category', allow_empty=True)
        if not utils.is_valid_language_code(self.language_code):
            raise utils.ValidationError('Invalid language code: %s' % self.language_code)
        for tag in self.tags:
            if not tag:
                raise utils.ValidationError('Tags should be non-empty.')
            if not re.match(constants.TAG_REGEX, tag):
                raise utils.ValidationError("Tags should only contain lowercase letters and spaces, received '%s'" % tag)
            if tag[0] not in string.ascii_lowercase or tag[-1] not in string.ascii_lowercase:
                raise utils.ValidationError("Tags should not start or end with whitespace, received '%s'" % tag)
            if re.search('\\s\\s+', tag):
                raise utils.ValidationError("Adjacent whitespace in tags should be collapsed, received '%s'" % tag)
        if len(set(self.tags)) < len(self.tags):
            raise utils.ValidationError('Expected tags to be unique, but found duplicates')

    def is_editable_by(self, user_id: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if a given user may edit the collection.\n\n        Args:\n            user_id: str. User id of the user.\n\n        Returns:\n            bool. Whether the given user may edit the collection.\n        '
        return user_id in self.editor_ids or user_id in self.owner_ids or self.community_owned

    def is_private(self) -> bool:
        if False:
            return 10
        'Checks whether the collection is private.\n\n        Returns:\n            bool. Whether the collection is private.\n        '
        return bool(self.status == constants.ACTIVITY_STATUS_PRIVATE)

    def is_solely_owned_by_user(self, user_id: str) -> bool:
        if False:
            return 10
        'Checks whether the collection is solely owned by the user.\n\n        Args:\n            user_id: str. The id of the user.\n\n        Returns:\n            bool. Whether the collection is solely owned by the user.\n        '
        return user_id in self.owner_ids and len(self.owner_ids) == 1

    def does_user_have_any_role(self, user_id: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if a given user has any role within the collection.\n\n        Args:\n            user_id: str. User id of the user.\n\n        Returns:\n            bool. Whether the given user has any role in the collection.\n        '
        return user_id in self.owner_ids or user_id in self.editor_ids or user_id in self.viewer_ids

    def add_contribution_by_user(self, contributor_id: str) -> None:
        if False:
            print('Hello World!')
        'Add a new contributor to the contributors summary.\n\n        Args:\n            contributor_id: str. ID of the contributor to be added.\n        '
        if contributor_id not in constants.SYSTEM_USER_IDS:
            self.contributors_summary[contributor_id] = self.contributors_summary.get(contributor_id, 0) + 1
        self.contributor_ids = list(self.contributors_summary.keys())