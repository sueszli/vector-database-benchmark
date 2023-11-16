"""Domain objects for configuration properties."""
from __future__ import annotations
from core import schema_utils
from core.constants import constants
from core.domain import change_domain
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union, overload
from core.domain import caching_services
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import config_models
(config_models,) = models.Registry.import_models([models.Names.CONFIG])
AllowedDefaultValueTypes = Union[str, bool, float, Dict[str, str], List[str], List[Dict[str, Sequence[str]]], List[Dict[str, str]]]

class ConfigPropertySchemaDict(TypedDict):
    """Type representing the config property's schema dictionary."""
    schema: Dict[str, Any]
    description: str
    value: AllowedDefaultValueTypes
CMD_CHANGE_PROPERTY_VALUE = 'change_property_value'
LIST_OF_FEATURED_TRANSLATION_LANGUAGES_DICTS_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_DICT, 'properties': [{'name': 'language_code', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE, 'validators': [{'id': 'is_supported_audio_language_code'}]}}, {'name': 'explanation', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}]}}
SET_OF_STRINGS_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}, 'validators': [{'id': 'is_uniquified'}]}
SET_OF_CLASSROOM_DICTS_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_DICT, 'properties': [{'name': 'name', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE}}, {'name': 'url_fragment', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE, 'validators': [{'id': 'is_url_fragment'}, {'id': 'has_length_at_most', 'max_value': constants.MAX_CHARS_IN_CLASSROOM_URL_FRAGMENT}]}}, {'name': 'course_details', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE, 'ui_config': {'rows': 8}}}, {'name': 'topic_list_intro', 'schema': {'type': schema_utils.SCHEMA_TYPE_UNICODE, 'ui_config': {'rows': 5}}}, {'name': 'topic_ids', 'schema': {'type': schema_utils.SCHEMA_TYPE_LIST, 'items': {'type': schema_utils.SCHEMA_TYPE_UNICODE}, 'validators': [{'id': 'is_uniquified'}]}}]}}
BOOL_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_BOOL}
FLOAT_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_FLOAT}
INT_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_INT}
POSITIVE_INT_SCHEMA = {'type': schema_utils.SCHEMA_TYPE_CUSTOM, 'obj_type': 'PositiveInt'}

class ConfigPropertyChange(change_domain.BaseChange):
    """Domain object for changes made to a config property object.

    The allowed commands, together with the attributes:
        - 'change_property_value' (with new_value)
    """
    ALLOWED_COMMANDS = [{'name': CMD_CHANGE_PROPERTY_VALUE, 'required_attribute_names': ['new_value'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}]

class ChangePropertyValueCmd(ConfigPropertyChange):
    """Class representing the ConfigPropertyChange's
    CMD_CHANGE_PROPERTY_VALUE command.
    """
    new_value: str

class ConfigProperty:
    """A property with a name and a default value.

    NOTE TO DEVELOPERS: These config properties are deprecated. Do not reuse
    these names:
    - about_page_youtube_video_id.
    - admin_email_address.
    - admin_ids.
    - admin_usernames.
    - allow_yaml_file_upload.
    - always_ask_learners_for_answer_details.
    - banned_usernames.
    - banner_alt_text.
    - before_end_body_tag_hook.
    - before_end_head_tag_hook.
    - batch_index_for_mailchimp
    - carousel_slides_config.
    - checkpoints_feature_is_enabled.
    - classroom_page_is_accessible.
    - classroom_promos_are_enabled.
    - collection_editor_whitelist.
    - contact_email_address.
    - contribute_gallery_page_announcement.
    - contributor_dashboard_is_enabled.
    - contributor_dashboard_reviewer_emails_is_enabled.
    - default_twitter_share_message_editor.
    - disabled_explorations.
    - editor_page_announcement.
    - editor_prerequisites_agreement.
    - email_footer.
    - email_sender_name.
    - embedded_google_group_url.
    - enable_admin_notifications_for_reviewer_shortage.
    - featured_translation_languages.
    - full_site_url.
    - high_bounce_rate_task_minimum_exploration_starts.
    - high_bounce_rate_task_state_bounce_rate_creation_threshold.
    - high_bounce_rate_task_state_bounce_rate_obsoletion_threshold.
    - is_improvements_tab_enabled.
    - learner_groups_are_enabled.
    - list_of_default_tags_for_blog_post.
    - max_number_of_explorations_in_math_svgs_batch.
    - max_number_of_suggestions_per_reviewer.
    - max_number_of_svgs_in_math_svgs_batch.
    - max_number_of_tags_assigned_to_blog_post.
    - moderator_ids.
    - moderator_request_forum_url.
    - moderator_usernames.
    - notify_admins_suggestions_waiting_too_long_is_enabled.
    - promo_bar_enabled.
    - promo_bar_message.
    - publicize_exploration_email_html_body.
    - record_playthrough_probability.
    - sharing_options.
    - sharing_options_twitter_text.
    - show_translation_size.
    - sidebar_menu_additional_links.
    - signup_email_body_content.
    - signup_email_subject_content.
    - site_forum_url.
    - social_media_buttons.
    - splash_page_exploration_id.
    - splash_page_exploration_version.
    - splash_page_youtube_video_id.
    - ssl_challenge_responses.
    - unpublish_exploration_email_html_body.
    - vmid_shared_secret_key_mapping.
    - whitelisted_email_senders.
    - whitelisted_exploration_ids_for_playthroughs.
    """

    def __init__(self, name: str, schema: Dict[str, Any], description: str, default_value: AllowedDefaultValueTypes) -> None:
        if False:
            for i in range(10):
                print('nop')
        if Registry.get_config_property(name):
            raise Exception('Property with name %s already exists' % name)
        self._name = name
        self._schema = schema
        self._description = description
        self._default_value = self.normalize(default_value)
        Registry.init_config_property(self.name, self)

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the name of the configuration property.'
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns the schema of the configuration property.'
        return self._schema

    @property
    def description(self) -> str:
        if False:
            return 10
        'Returns the description of the configuration property.'
        return self._description

    @property
    def default_value(self) -> AllowedDefaultValueTypes:
        if False:
            i = 10
            return i + 15
        'Returns the default value of the configuration property.'
        return self._default_value

    @property
    def value(self) -> Any:
        if False:
            print('Hello World!')
        'Get the latest value from memcache, datastore, or use default.'
        memcached_items = caching_services.get_multi(caching_services.CACHE_NAMESPACE_CONFIG, None, [self.name])
        if self.name in memcached_items:
            return memcached_items[self.name]
        datastore_item = config_models.ConfigPropertyModel.get(self.name, strict=False)
        if datastore_item is not None:
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_CONFIG, None, {datastore_item.id: datastore_item.value})
            return datastore_item.value
        return self.default_value

    def set_value(self, committer_id: str, raw_value: Union[str, List[str]]) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the value of the property. In general, this should not be\n        called directly -- use config_services.set_property() instead.\n        '
        value = self.normalize(raw_value)
        model_instance = config_models.ConfigPropertyModel.get(self.name, strict=False)
        if model_instance is None:
            model_instance = config_models.ConfigPropertyModel(id=self.name)
        model_instance.value = value
        model_instance.commit(committer_id, [{'cmd': CMD_CHANGE_PROPERTY_VALUE, 'new_value': value}])
        caching_services.set_multi(caching_services.CACHE_NAMESPACE_CONFIG, None, {model_instance.id: model_instance.value})

    def normalize(self, value: AllowedDefaultValueTypes) -> AllowedDefaultValueTypes:
        if False:
            i = 10
            return i + 15
        'Validates the given object using the schema and normalizes if\n        necessary.\n\n        Args:\n            value: str. The value of the configuration property.\n\n        Returns:\n            instance. The normalized object.\n        '
        email_validators = [{'id': 'does_not_contain_email'}]
        normalized_value: AllowedDefaultValueTypes = schema_utils.normalize_against_schema(value, self._schema, global_validators=email_validators)
        return normalized_value

class Registry:
    """Registry of all configuration properties."""
    _config_registry: Dict[str, ConfigProperty] = {}

    @classmethod
    def init_config_property(cls, name: str, instance: ConfigProperty) -> None:
        if False:
            print('Hello World!')
        'Initializes _config_registry with keys as the property names and\n        values as instances of the specified property.\n\n        Args:\n            name: str. The name of the configuration property.\n            instance: *. The instance of the configuration property.\n        '
        cls._config_registry[name] = instance

    @overload
    @classmethod
    def get_config_property(cls, name: str) -> Optional[ConfigProperty]:
        if False:
            print('Hello World!')
        ...

    @overload
    @classmethod
    def get_config_property(cls, name: str, *, strict: Literal[True]) -> ConfigProperty:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    @classmethod
    def get_config_property(cls, name: str, *, strict: Literal[False]) -> Optional[ConfigProperty]:
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    def get_config_property(cls, name: str, strict: bool=False) -> Optional[ConfigProperty]:
        if False:
            i = 10
            return i + 15
        'Returns the instance of the specified name of the configuration\n        property.\n\n        Args:\n            name: str. The name of the configuration property.\n            strict: bool. Whether to fail noisily if no config property exist.\n\n        Returns:\n            instance. The instance of the specified configuration property.\n\n        Raises:\n            Exception. No config property exist for the given property name.\n        '
        config_property = cls._config_registry.get(name)
        if strict and config_property is None:
            raise Exception('No config property exists for the given property name: %s' % name)
        return config_property

    @classmethod
    def get_config_property_schemas(cls) -> Dict[str, ConfigPropertySchemaDict]:
        if False:
            i = 10
            return i + 15
        'Return a dict of editable config property schemas.\n\n        The keys of the dict are config property names. The values are dicts\n        with the following keys: schema, description, value.\n        '
        schemas_dict: Dict[str, ConfigPropertySchemaDict] = {}
        for (property_name, instance) in cls._config_registry.items():
            schemas_dict[property_name] = {'schema': instance.schema, 'description': instance.description, 'value': instance.value}
        return schemas_dict

    @classmethod
    def get_all_config_property_names(cls) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Return a list of all the config property names.\n\n        Returns:\n            list. The list of all config property names.\n        '
        return list(cls._config_registry)
CLASSROOM_PAGES_DATA = ConfigProperty('classroom_pages_data', SET_OF_CLASSROOM_DICTS_SCHEMA, 'The details for each classroom page.', [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [], 'course_details': '', 'topic_list_intro': ''}])