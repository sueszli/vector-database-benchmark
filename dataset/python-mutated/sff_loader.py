from __future__ import annotations
from dynaconf.base import SourceMetadata

def load(obj, env=None, silent=True, key=None, filename=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reads and loads in to "obj" a single key or all keys from source\n    :param obj: the settings instance\n    :param env: settings current env (upper case) default=\'DEVELOPMENT\'\n    :param silent: if errors should raise\n    :param key: if defined load a single key, else load all from `env`\n    :param filename: Custom filename to load (useful for tests)\n    :return: None\n    '
    keys = []
    values = []
    found_file = obj.find_file('settings.sff')
    if not found_file:
        return
    with open(found_file) as settings_file:
        for line in settings_file.readlines():
            if line.startswith('#'):
                continue
            if line.startswith('KEYS:'):
                keys = line.strip('KEYS:').strip('\n').split(';')
            if line.startswith('VALUES:'):
                values = line.strip('VALUES:').strip('\n').split(';')
    data = dict(zip(keys, values))
    source_metadata = SourceMetadata('sff', found_file, 'default')
    if key:
        value = data.get(key.lower())
        obj.set(key, value, loader_identifier=source_metadata)
    else:
        obj.update(data, loader_identifier=source_metadata)
    obj._loaded_files.append(found_file)