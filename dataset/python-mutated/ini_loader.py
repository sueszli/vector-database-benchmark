from __future__ import annotations
import io
from pathlib import Path
from dynaconf import default_settings
from dynaconf.constants import INI_EXTENSIONS
from dynaconf.loaders.base import BaseLoader
from dynaconf.utils import object_merge
try:
    from configobj import ConfigObj
except ImportError:
    ConfigObj = None

def load(obj, env=None, silent=True, key=None, filename=None, validate=False):
    if False:
        print('Hello World!')
    '\n    Reads and loads in to "obj" a single key or all keys from source file.\n\n    :param obj: the settings instance\n    :param env: settings current env default=\'development\'\n    :param silent: if errors should raise\n    :param key: if defined load a single key, else load all in env\n    :param filename: Optional custom filename to load\n    :return: None\n    '
    if ConfigObj is None:
        BaseLoader.warn_not_installed(obj, 'ini')
        return
    loader = BaseLoader(obj=obj, env=env, identifier='ini', extensions=INI_EXTENSIONS, file_reader=lambda fileobj: ConfigObj(fileobj).dict(), string_reader=lambda strobj: ConfigObj(strobj.split('\n')).dict(), validate=validate)
    loader.load(filename=filename, key=key, silent=silent)

def write(settings_path, settings_data, merge=True):
    if False:
        i = 10
        return i + 15
    'Write data to a settings file.\n\n    :param settings_path: the filepath\n    :param settings_data: a dictionary with data\n    :param merge: boolean if existing file should be merged with new data\n    '
    settings_path = Path(settings_path)
    if settings_path.exists() and merge:
        with open(str(settings_path), encoding=default_settings.ENCODING_FOR_DYNACONF) as open_file:
            object_merge(ConfigObj(open_file).dict(), settings_data)
    new = ConfigObj()
    new.update(settings_data)
    new.write(open(str(settings_path), 'bw'))