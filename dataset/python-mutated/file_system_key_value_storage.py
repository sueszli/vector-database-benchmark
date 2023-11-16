import glob
from tempfile import gettempdir
import os
import errno
from cloudinary.cache.storage.key_value_storage import KeyValueStorage

class FileSystemKeyValueStorage(KeyValueStorage):
    """File-based key-value storage"""
    _item_ext = '.cldci'

    def __init__(self, root_path):
        if False:
            return 10
        '\n        Create a new Storage object.\n\n        All files will be stored under the root_path location\n\n        :param root_path: The base folder for all storage files\n        '
        if root_path is None:
            root_path = gettempdir()
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        self._root_path = root_path

    def get(self, key):
        if False:
            while True:
                i = 10
        if not self._exists(key):
            return None
        with open(self._get_key_full_path(key), 'r') as f:
            value = f.read()
        return value

    def set(self, key, value):
        if False:
            while True:
                i = 10
        with open(self._get_key_full_path(key), 'w') as f:
            f.write(value)
        return True

    def delete(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            os.remove(self._get_key_full_path(key))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        return True

    def clear(self):
        if False:
            return 10
        for cache_item_path in glob.iglob(os.path.join(self._root_path, '*' + self._item_ext)):
            os.remove(cache_item_path)
        return True

    def _get_key_full_path(self, key):
        if False:
            return 10
        '\n        Generate the file path for the key\n\n        :param key: The key\n\n        :return: The absolute path of the value file associated with the key\n        '
        return os.path.join(self._root_path, key + self._item_ext)

    def _exists(self, key):
        if False:
            while True:
                i = 10
        '\n        Indicate whether key exists\n\n        :param key: The key\n\n        :return: bool True if the file for the given key exists\n        '
        return os.path.isfile(self._get_key_full_path(key))