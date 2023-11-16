import os
import jsonschema
import logging
from typing import List
import json
from ray._private.runtime_env.constants import RAY_RUNTIME_ENV_PLUGIN_SCHEMAS_ENV_VAR, RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX
logger = logging.getLogger(__name__)

class RuntimeEnvPluginSchemaManager:
    """This manager is used to load plugin json schemas."""
    default_schema_path = os.path.join(os.path.dirname(__file__), '../../runtime_env/schemas')
    schemas = {}
    loaded = False

    @classmethod
    def _load_schemas(cls, schema_paths: List[str]):
        if False:
            print('Hello World!')
        for schema_path in schema_paths:
            try:
                with open(schema_path) as f:
                    schema = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.error('Invalid runtime env schema %s, skip it.', schema_path)
                continue
            except OSError:
                logger.error('Cannot open runtime env schema %s, skip it.', schema_path)
                continue
            if 'title' not in schema:
                logger.error('No valid title in runtime env schema %s, skip it.', schema_path)
                continue
            if schema['title'] in cls.schemas:
                logger.error("The 'title' of runtime env schema %s conflicts with %s, skip it.", schema_path, cls.schemas[schema['title']])
                continue
            cls.schemas[schema['title']] = schema

    @classmethod
    def _load_default_schemas(cls):
        if False:
            print('Hello World!')
        schema_json_files = list()
        for (root, _, files) in os.walk(cls.default_schema_path):
            for f in files:
                if f.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                    schema_json_files.append(os.path.join(root, f))
            logger.debug(f'Loading the default runtime env schemas: {schema_json_files}.')
            cls._load_schemas(schema_json_files)

    @classmethod
    def _load_schemas_from_env_var(cls):
        if False:
            for i in range(10):
                print('nop')
        schema_paths = os.environ.get(RAY_RUNTIME_ENV_PLUGIN_SCHEMAS_ENV_VAR)
        if schema_paths:
            schema_json_files = list()
            for path in schema_paths.split(','):
                if path.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                    schema_json_files.append(path)
                elif os.path.isdir(path):
                    for (root, _, files) in os.walk(path):
                        for f in files:
                            if f.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                                schema_json_files.append(os.path.join(root, f))
            logger.info(f'Loading the runtime env schemas from env var: {schema_json_files}.')
            cls._load_schemas(schema_json_files)

    @classmethod
    def validate(cls, name, instance):
        if False:
            print('Hello World!')
        if not cls.loaded:
            cls._load_default_schemas()
            cls._load_schemas_from_env_var()
            cls.loaded = True
        if name in cls.schemas:
            jsonschema.validate(instance=instance, schema=cls.schemas[name])

    @classmethod
    def clear(cls):
        if False:
            i = 10
            return i + 15
        cls.schemas.clear()
        cls.loaded = False