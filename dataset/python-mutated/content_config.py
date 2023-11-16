"""Content configuration."""
from __future__ import annotations
import os
import pickle
import typing as t
from .constants import CONTROLLER_PYTHON_VERSIONS, SUPPORTED_PYTHON_VERSIONS
from .compat.packaging import PACKAGING_IMPORT_ERROR, SpecifierSet, Version
from .compat.yaml import YAML_IMPORT_ERROR, yaml_load
from .io import open_binary_file, read_text_file
from .util import ApplicationError, display
from .data import data_context
from .config import EnvironmentConfig, ContentConfig, ModulesConfig
MISSING = object()

def parse_modules_config(data: t.Any) -> ModulesConfig:
    if False:
        return 10
    'Parse the given dictionary as module config and return it.'
    if not isinstance(data, dict):
        raise Exception('config must be type `dict` not `%s`' % type(data))
    python_requires = data.get('python_requires', MISSING)
    if python_requires == MISSING:
        raise KeyError('python_requires is required')
    return ModulesConfig(python_requires=python_requires, python_versions=parse_python_requires(python_requires), controller_only=python_requires == 'controller')

def parse_content_config(data: t.Any) -> ContentConfig:
    if False:
        i = 10
        return i + 15
    'Parse the given dictionary as content config and return it.'
    if not isinstance(data, dict):
        raise Exception('config must be type `dict` not `%s`' % type(data))
    modules = parse_modules_config(data.get('modules', {}))
    python_versions = tuple((version for version in SUPPORTED_PYTHON_VERSIONS if version in CONTROLLER_PYTHON_VERSIONS or version in modules.python_versions))
    return ContentConfig(modules=modules, python_versions=python_versions)

def load_config(path: str) -> t.Optional[ContentConfig]:
    if False:
        return 10
    'Load and parse the specified config file and return the result or None if loading/parsing failed.'
    if YAML_IMPORT_ERROR:
        raise ApplicationError('The "PyYAML" module is required to parse config: %s' % YAML_IMPORT_ERROR)
    if PACKAGING_IMPORT_ERROR:
        raise ApplicationError('The "packaging" module is required to parse config: %s' % PACKAGING_IMPORT_ERROR)
    value = read_text_file(path)
    try:
        yaml_value = yaml_load(value)
    except Exception as ex:
        display.warning('Ignoring config "%s" due to a YAML parsing error: %s' % (path, ex))
        return None
    try:
        config = parse_content_config(yaml_value)
    except Exception as ex:
        display.warning('Ignoring config "%s" due a config parsing error: %s' % (path, ex))
        return None
    display.info('Loaded configuration: %s' % path, verbosity=1)
    return config

def get_content_config(args: EnvironmentConfig) -> ContentConfig:
    if False:
        print('Hello World!')
    '\n    Parse and return the content configuration (if any) for the current collection.\n    For ansible-core, a default configuration is used.\n    Results are cached.\n    '
    if args.host_path:
        args.content_config = deserialize_content_config(os.path.join(args.host_path, 'config.dat'))
    if args.content_config:
        return args.content_config
    collection_config_path = 'tests/config.yml'
    config = None
    if data_context().content.collection and os.path.exists(collection_config_path):
        config = load_config(collection_config_path)
    if not config:
        config = parse_content_config(dict(modules=dict(python_requires='default')))
    if not config.modules.python_versions:
        raise ApplicationError('This collection does not declare support for modules/module_utils on any known Python version.\nAnsible supports modules/module_utils on Python versions: %s\nThis collection provides the Python requirement: %s' % (', '.join(SUPPORTED_PYTHON_VERSIONS), config.modules.python_requires))
    args.content_config = config
    return config

def parse_python_requires(value: t.Any) -> tuple[str, ...]:
    if False:
        while True:
            i = 10
    "Parse the given 'python_requires' version specifier and return the matching Python versions."
    if not isinstance(value, str):
        raise ValueError('python_requires must must be of type `str` not type `%s`' % type(value))
    versions: tuple[str, ...]
    if value == 'default':
        versions = SUPPORTED_PYTHON_VERSIONS
    elif value == 'controller':
        versions = CONTROLLER_PYTHON_VERSIONS
    else:
        specifier_set = SpecifierSet(value)
        versions = tuple((version for version in SUPPORTED_PYTHON_VERSIONS if specifier_set.contains(Version(version))))
    return versions

def serialize_content_config(args: EnvironmentConfig, path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Serialize the content config to the given path. If the config has not been loaded, an empty config will be serialized.'
    with open_binary_file(path, 'wb') as config_file:
        pickle.dump(args.content_config, config_file)

def deserialize_content_config(path: str) -> ContentConfig:
    if False:
        for i in range(10):
            print('nop')
    'Deserialize content config from the path.'
    with open_binary_file(path) as config_file:
        return pickle.load(config_file)