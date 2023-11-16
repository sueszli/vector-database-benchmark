from __future__ import annotations
import warnings
from pathlib import Path
from dynaconf import default_settings
from dynaconf.constants import TOML_EXTENSIONS
from dynaconf.loaders.base import BaseLoader
from dynaconf.utils import object_merge
from dynaconf.vendor import toml
from dynaconf.vendor import tomllib

def load(obj, env=None, silent=True, key=None, filename=None, validate=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reads and loads in to "obj" a single key or all keys from source file.\n\n    :param obj: the settings instance\n    :param env: settings current env default=\'development\'\n    :param silent: if errors should raise\n    :param key: if defined load a single key, else load all in env\n    :param filename: Optional custom filename to load\n    :return: None\n    '
    try:
        loader = BaseLoader(obj=obj, env=env, identifier='toml', extensions=TOML_EXTENSIONS, file_reader=tomllib.load, string_reader=tomllib.loads, opener_params={'mode': 'rb'}, validate=validate)
        loader.load(filename=filename, key=key, silent=silent)
    except UnicodeDecodeError:
        '\n        NOTE: Compat functions exists to keep backwards compatibility with\n        the new tomllib library. The old library was called `toml` and\n        the new one is called `tomllib`.\n\n        The old lib uiri/toml allowed unicode characters and re-added files\n        as string.\n\n        The new tomllib (stdlib) does not allow unicode characters, only\n        utf-8 encoded, and read files as binary.\n\n        NOTE: In dynaconf 4.0.0 we will drop support for the old library\n        removing the compat functions and calling directly the new lib.\n        '
        loader = BaseLoader(obj=obj, env=env, identifier='toml', extensions=TOML_EXTENSIONS, file_reader=toml.load, string_reader=toml.loads, validate=validate)
        loader.load(filename=filename, key=key, silent=silent)
        warnings.warn('TOML files should have only UTF-8 encoded characters. starting on 4.0.0 dynaconf will stop allowing invalid chars.')

def write(settings_path, settings_data, merge=True):
    if False:
        i = 10
        return i + 15
    'Write data to a settings file.\n\n    :param settings_path: the filepath\n    :param settings_data: a dictionary with data\n    :param merge: boolean if existing file should be merged with new data\n    '
    settings_path = Path(settings_path)
    if settings_path.exists() and merge:
        try:
            with open(str(settings_path), 'rb') as open_file:
                object_merge(tomllib.load(open_file), settings_data)
        except UnicodeDecodeError:
            with open(str(settings_path), encoding=default_settings.ENCODING_FOR_DYNACONF) as open_file:
                object_merge(toml.load(open_file), settings_data)
    try:
        with open(str(settings_path), 'wb') as open_file:
            tomllib.dump(encode_nulls(settings_data), open_file)
    except UnicodeEncodeError:
        with open(str(settings_path), 'w', encoding=default_settings.ENCODING_FOR_DYNACONF) as open_file:
            toml.dump(encode_nulls(settings_data), open_file)
        warnings.warn('TOML files should have only UTF-8 encoded characters. starting on 4.0.0 dynaconf will stop allowing invalid chars.')

def encode_nulls(data):
    if False:
        return 10
    "TOML does not support `None` so this function transforms to '@none '."
    if data is None:
        return '@none '
    if isinstance(data, dict):
        return {key: encode_nulls(value) for (key, value) in data.items()}
    elif isinstance(data, (list, tuple)):
        return [encode_nulls(item) for item in data]
    return data