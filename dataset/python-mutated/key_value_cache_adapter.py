import json
from hashlib import sha1
from cloudinary.cache.adapter.cache_adapter import CacheAdapter
from cloudinary.cache.storage.key_value_storage import KeyValueStorage
from cloudinary.utils import check_property_enabled

class KeyValueCacheAdapter(CacheAdapter):
    """
    A cache adapter for a key-value storage type
    """

    def __init__(self, storage):
        if False:
            for i in range(10):
                print('nop')
        'Create a new adapter for the provided storage interface'
        if not isinstance(storage, KeyValueStorage):
            raise ValueError('An instance of valid KeyValueStorage must be provided')
        self._key_value_storage = storage

    @property
    def enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self._key_value_storage is not None

    @check_property_enabled
    def get(self, public_id, type, resource_type, transformation, format):
        if False:
            i = 10
            return i + 15
        key = self.generate_cache_key(public_id, type, resource_type, transformation, format)
        value_str = self._key_value_storage.get(key)
        return json.loads(value_str) if value_str else value_str

    @check_property_enabled
    def set(self, public_id, type, resource_type, transformation, format, value):
        if False:
            return 10
        key = self.generate_cache_key(public_id, type, resource_type, transformation, format)
        return self._key_value_storage.set(key, json.dumps(value))

    @check_property_enabled
    def delete(self, public_id, type, resource_type, transformation, format):
        if False:
            i = 10
            return i + 15
        return self._key_value_storage.delete(self.generate_cache_key(public_id, type, resource_type, transformation, format))

    @check_property_enabled
    def flush_all(self):
        if False:
            print('Hello World!')
        return self._key_value_storage.clear()

    @staticmethod
    def generate_cache_key(public_id, type, resource_type, transformation, format):
        if False:
            while True:
                i = 10
        '\n        Generates key-value storage key from parameters\n\n        :param public_id:       The public ID of the resource\n        :param type:            The storage type\n        :param resource_type:   The type of the resource\n        :param transformation:  The transformation string\n        :param format:          The format of the resource\n\n        :return: Resulting cache key\n        '
        valid_params = [p for p in [public_id, type, resource_type, transformation, format] if p]
        return sha1('/'.join(valid_params).encode('utf-8')).hexdigest()