from __future__ import annotations
import io
import json
from pathlib import Path
from dynaconf import default_settings
from dynaconf.constants import JSON_EXTENSIONS
from dynaconf.loaders.base import BaseLoader
from dynaconf.utils import object_merge
from dynaconf.utils.parse_conf import try_to_encode
try:
    import commentjson
except ImportError:
    commentjson = None

def load(obj, env=None, silent=True, key=None, filename=None, validate=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reads and loads in to "obj" a single key or all keys from source file.\n\n    :param obj: the settings instance\n    :param env: settings current env default=\'development\'\n    :param silent: if errors should raise\n    :param key: if defined load a single key, else load all in env\n    :param filename: Optional custom filename to load\n    :return: None\n    '
    if obj.get('COMMENTJSON_ENABLED_FOR_DYNACONF') and commentjson:
        file_reader = commentjson.load
        string_reader = commentjson.loads
    else:
        file_reader = json.load
        string_reader = json.loads
    loader = BaseLoader(obj=obj, env=env, identifier='json', extensions=JSON_EXTENSIONS, file_reader=file_reader, string_reader=string_reader, validate=validate)
    loader.load(filename=filename, key=key, silent=silent)

def write(settings_path, settings_data, merge=True):
    if False:
        return 10
    'Write data to a settings file.\n\n    :param settings_path: the filepath\n    :param settings_data: a dictionary with data\n    :param merge: boolean if existing file should be merged with new data\n    '
    settings_path = Path(settings_path)
    if settings_path.exists() and merge:
        with open(str(settings_path), encoding=default_settings.ENCODING_FOR_DYNACONF) as open_file:
            object_merge(json.load(open_file), settings_data)
    with open(str(settings_path), 'w', encoding=default_settings.ENCODING_FOR_DYNACONF) as open_file:
        json.dump(settings_data, open_file, cls=DynaconfEncoder)

class DynaconfEncoder(json.JSONEncoder):
    """Transform Dynaconf custom types instances to json representation"""

    def default(self, o):
        if False:
            while True:
                i = 10
        return try_to_encode(o, callback=super().default)