from __future__ import absolute_import
import copy
import re
import six
from oslo_config import cfg
from st2common import log as logging
from st2common.models.db.pack import ConfigDB
from st2common.persistence.pack import ConfigSchema
from st2common.persistence.pack import Config
from st2common.content import utils as content_utils
from st2common.util import jinja as jinja_utils
from st2common.util.templating import render_template_with_system_and_user_context
from st2common.util.config_parser import ContentPackConfigParser
from st2common.exceptions.db import StackStormDBObjectNotFoundError
__all__ = ['ContentPackConfigLoader']
LOG = logging.getLogger(__name__)

class ContentPackConfigLoader(object):
    """
    Class which loads and resolves all the config values and returns a dictionary of resolved values
    which can be passed to the resource.

    It loads and resolves values in the following order:

    1. Static values from <pack path>/config.yaml file
    2. Dynamic and or static values from /opt/stackstorm/configs/<pack name>.yaml file.

    Values are merged from left to right which means values from "<pack name>.yaml" file have
    precedence and override values from pack local config file.
    """

    def __init__(self, pack_name, user=None):
        if False:
            print('Hello World!')
        self.pack_name = pack_name
        self.user = user or cfg.CONF.system_user.user
        self.pack_path = content_utils.get_pack_base_path(pack_name=pack_name)
        self._config_parser = ContentPackConfigParser(pack_name=pack_name)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        try:
            config_db = Config.get_by_pack(value=self.pack_name)
        except StackStormDBObjectNotFoundError:
            config_db = ConfigDB(pack=self.pack_name, values={})
        try:
            config_schema_db = ConfigSchema.get_by_pack(value=self.pack_name)
        except StackStormDBObjectNotFoundError:
            config_schema_db = None
        config = self._get_values_for_config(config_schema_db=config_schema_db, config_db=config_db)
        result.update(config)
        return result

    def _get_values_for_config(self, config_schema_db, config_db):
        if False:
            for i in range(10):
                print('nop')
        schema_values = getattr(config_schema_db, 'attributes', {})
        config_values = getattr(config_db, 'values', {})
        config = copy.deepcopy(config_values or {})
        config = self._assign_dynamic_config_values(schema=schema_values, config=config)
        config = self._assign_default_values(schema=schema_values, config=config)
        return config

    @staticmethod
    def _get_object_properties_schema(object_schema, object_keys=None):
        if False:
            print('Hello World!')
        "\n        Create a schema for an object property using all of: properties,\n        patternProperties, and additionalProperties.\n\n        This 'flattens' properties, patternProperties, and additionalProperties\n        so that we can handle patternProperties and additionalProperties\n        as if they were defined in properties.\n        So, every key in object_keys will be assigned a schema\n        from properties, patternProperties, or additionalProperties.\n\n        NOTE: order of precedence: properties, patternProperties, additionalProperties\n        So, the additionalProperties schema is only used for keys that are not in\n        properties and that do not match any of the patterns in patternProperties.\n        And, patternProperties schemas only apply to keys missing from properties.\n\n        :rtype: ``dict``\n        "
        flattened_properties_schema = {key: {} for key in object_keys}
        properties_schema = object_schema.get('properties', {})
        flattened_properties_schema.update(properties_schema)
        extra_keys = set(object_keys) - set(properties_schema.keys())
        if not extra_keys:
            return flattened_properties_schema
        pattern_properties = object_schema.get('patternProperties', {})
        if pattern_properties and isinstance(pattern_properties, dict):
            pattern_properties = {re.compile(raw_pattern): pattern_schema for (raw_pattern, pattern_schema) in pattern_properties.items()}
            for key in list(extra_keys):
                key_schemas = []
                for (pattern, pattern_schema) in pattern_properties.items():
                    if pattern.search(key):
                        key_schemas.append(pattern_schema)
                if key_schemas:
                    composed_schema = {}
                    for schema in key_schemas:
                        composed_schema.update(schema)
                    flattened_properties_schema[key] = composed_schema
                    extra_keys.remove(key)
            if not extra_keys:
                return flattened_properties_schema
        additional_properties = object_schema.get('additionalProperties', {})
        if additional_properties and isinstance(additional_properties, dict):
            for key in extra_keys:
                flattened_properties_schema[key] = additional_properties
        return flattened_properties_schema

    @staticmethod
    def _get_array_items_schema(array_schema, items_count=0):
        if False:
            i = 10
            return i + 15
        "\n        Create a schema for array items using both additionalItems and items.\n\n        This 'flattens' items and additionalItems so that we can handle additionalItems\n        as if each additional item was defined in items.\n\n        The additionalItems schema will only be used if the items schema is shorter\n        than items_count. So, when additionalItems is defined, the items schema will be\n        extended to be at least as long as items_count.\n\n        :rtype: ``list``\n        "
        flattened_items_schema = []
        items_schema = array_schema.get('items', [])
        if isinstance(items_schema, dict):
            flattened_items_schema.extend([items_schema] * items_count)
        else:
            flattened_items_schema.extend(items_schema)
        flattened_items_schema_count = len(flattened_items_schema)
        if flattened_items_schema_count >= items_count:
            return flattened_items_schema
        additional_items = array_schema.get('additionalItems', {})
        if additional_items and isinstance(additional_items, dict):
            flattened_items_schema.extend([additional_items] * (items_count - flattened_items_schema_count))
        return flattened_items_schema

    def _assign_dynamic_config_values(self, schema, config, parent_keys=None):
        if False:
            i = 10
            return i + 15
        '\n        Assign dynamic config value for a particular config item if the ite utilizes a Jinja\n        expression for dynamic config values.\n\n        Note: This method mutates config argument in place.\n\n        :rtype: ``dict``\n        '
        parent_keys = parent_keys or []
        config_is_dict = isinstance(config, dict)
        config_is_list = isinstance(config, list)
        iterator = six.iteritems(config) if config_is_dict else enumerate(config)
        for (config_item_key, config_item_value) in iterator:
            if config_is_dict:
                schema_item = schema.get(config_item_key, {})
            if config_is_list and isinstance(schema, list):
                try:
                    schema_item = schema[config_item_key]
                except IndexError:
                    schema_item = {}
            elif config_is_list:
                schema_item = schema
            is_dictionary = isinstance(config_item_value, dict)
            is_list = isinstance(config_item_value, list)
            current_keys = parent_keys + [str(config_item_key)]
            if is_dictionary:
                properties_schema = self._get_object_properties_schema(schema_item, object_keys=config_item_value.keys())
                self._assign_dynamic_config_values(schema=properties_schema, config=config[config_item_key], parent_keys=current_keys)
            elif is_list:
                items_schema = self._get_array_items_schema(schema_item, items_count=len(config[config_item_key]))
                self._assign_dynamic_config_values(schema=items_schema, config=config[config_item_key], parent_keys=current_keys)
            else:
                is_jinja_expression = jinja_utils.is_jinja_expression(value=config_item_value)
                if is_jinja_expression:
                    full_config_item_key = '.'.join(current_keys)
                    value = self._get_datastore_value_for_expression(key=full_config_item_key, value=config_item_value, config_schema_item=schema_item)
                    config[config_item_key] = value
                else:
                    config[config_item_key] = config_item_value
        return config

    def _assign_default_values(self, schema, config):
        if False:
            i = 10
            return i + 15
        '\n        Assign default values for particular config if default values are provided in the config\n        schema and a value is not specified in the config.\n\n        Note: This method mutates config argument in place.\n\n        :rtype: ``dict|list``\n        '
        schema_is_dict = isinstance(schema, dict)
        iterator = schema.items() if schema_is_dict else enumerate(schema)
        for (schema_item_key, schema_item) in iterator:
            has_default_value = 'default' in schema_item
            if isinstance(config, dict):
                has_config_value = schema_item_key in config
            else:
                has_config_value = schema_item_key < len(config)
            default_value = schema_item.get('default', None)
            if has_default_value and (not has_config_value):
                config[schema_item_key] = default_value
            try:
                config_value = config[schema_item_key]
            except (KeyError, IndexError):
                config_value = None
            schema_item_type = schema_item.get('type', None)
            if schema_item_type == 'object':
                has_properties = schema_item.get('properties', None)
                has_pattern_properties = schema_item.get('patternProperties', None)
                has_additional_properties = schema_item.get('additionalProperties', None)
                if has_properties or has_pattern_properties or has_additional_properties:
                    if not config_value:
                        config_value = config[schema_item_key] = {}
                    properties_schema = self._get_object_properties_schema(schema_item, object_keys=config_value.keys())
                    self._assign_default_values(schema=properties_schema, config=config_value)
            elif schema_item_type == 'array':
                has_items = schema_item.get('items', None)
                has_additional_items = schema_item.get('additionalItems', None)
                if has_items or has_additional_items:
                    if not config_value:
                        config_value = config[schema_item_key] = []
                    items_schema = self._get_array_items_schema(schema_item, items_count=len(config_value))
                    self._assign_default_values(schema=items_schema, config=config_value)
        return config

    def _get_datastore_value_for_expression(self, key, value, config_schema_item=None):
        if False:
            while True:
                i = 10
        '\n        Retrieve datastore value by first resolving the datastore expression and then retrieving\n        the value from the datastore.\n\n        :param key: Full path to the config item key (e.g. "token" / "auth.settings.token", etc.)\n        '
        from st2common.services.config import deserialize_key_value
        config_schema_item = config_schema_item or {}
        secret = config_schema_item.get('secret', False)
        if secret or 'decrypt_kv' in value:
            LOG.audit('User %s is decrypting the value for key %s from the config within pack %s', self.user, key, self.pack_name, extra={'user': self.user, 'key_name': key, 'pack_name': self.pack_name, 'operation': 'pack_config_value_decrypt'})
        try:
            value = render_template_with_system_and_user_context(value=value, user=self.user)
        except Exception as e:
            exc_class = type(e)
            original_msg = six.text_type(e)
            msg = 'Failed to render dynamic configuration value for key "%s" with value "%s" for pack "%s" config: %s %s ' % (key, value, self.pack_name, exc_class, original_msg)
            raise RuntimeError(msg)
        if value:
            value = deserialize_key_value(value=value, secret=secret)
        else:
            value = None
        return value

def get_config(pack, user):
    if False:
        return 10
    'Returns config for given pack and user.'
    LOG.debug('Attempting to get config for pack "%s" and user "%s"' % (pack, user))
    if pack and user:
        LOG.debug('Pack and user found. Loading config.')
        config_loader = ContentPackConfigLoader(pack_name=pack, user=user)
        config = config_loader.get_config()
    else:
        config = {}
    LOG.debug('Config: %s', config)
    return config