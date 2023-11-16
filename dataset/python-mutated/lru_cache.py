import sys
from collections import OrderedDict
from deeplake.core.partial_reader import PartialReader
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from deeplake.core.chunk.base_chunk import BaseChunk
from typing import Any, Dict, Optional, Union
from deeplake.core.storage.provider import StorageProvider

def _get_nbytes(obj: Union[bytes, memoryview, DeepLakeMemoryObject]):
    if False:
        return 10
    if isinstance(obj, DeepLakeMemoryObject):
        return obj.nbytes
    return len(obj)

def obj_to_bytes(obj):
    if False:
        i = 10
        return i + 15
    if isinstance(obj, DeepLakeMemoryObject):
        obj = obj.tobytes()
    if isinstance(obj, memoryview):
        obj = bytes(obj)
    return obj

class LRUCache(StorageProvider):
    """LRU Cache that uses StorageProvider for caching"""

    def __init__(self, cache_storage: StorageProvider, next_storage: Optional[StorageProvider], cache_size: int):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the LRUCache. It can be chained with other LRUCache objects to create multilayer caches.\n\n        Args:\n            cache_storage (StorageProvider): The storage being used as the caching layer of the cache.\n                This should be a base provider such as MemoryProvider, LocalProvider or S3Provider but not another LRUCache.\n            next_storage (StorageProvider): The next storage layer of the cache.\n                This can either be a base provider (i.e. it is the final storage) or another LRUCache (i.e. in case of chained cache).\n                While reading data, all misses from cache would be retrieved from here.\n                While writing data, the data will be written to the next_storage when cache_storage is full or flush is called.\n            cache_size (int): The total space that can be used from the cache_storage in bytes.\n                This number may be less than the actual space available on the cache_storage.\n                Setting it to a higher value than actually available space may lead to unexpected behaviors.\n        '
        self.next_storage = next_storage
        self.cache_storage = cache_storage
        self.cache_size = cache_size
        self.lru_sizes: OrderedDict[str, int] = OrderedDict()
        self.dirty_keys: Dict[str, None] = OrderedDict() if sys.version_info < (3, 7) else {}
        self.cache_used = 0
        self.deeplake_objects: Dict[str, DeepLakeMemoryObject] = {}
        self.use_async = False

    def register_deeplake_object(self, path: str, obj: DeepLakeMemoryObject):
        if False:
            return 10
        'Registers a new object in the cache.'
        self.deeplake_objects[path] = obj

    def clear_deeplake_objects(self):
        if False:
            for i in range(10):
                print('nop')
        'Removes all DeepLakeMemoryObjects from the cache.'
        self.deeplake_objects.clear()

    def remove_deeplake_object(self, path: str):
        if False:
            for i in range(10):
                print('nop')
        'Removes a DeepLakeMemoryObject from the cache.'
        self.deeplake_objects.pop(path, None)

    def update_used_cache_for_path(self, path: str, new_size: int):
        if False:
            print('Hello World!')
        if new_size < 0:
            raise ValueError(f'`new_size` must be >= 0. Got: {new_size}')
        if path in self.lru_sizes:
            old_size = self.lru_sizes[path]
            self.cache_used -= old_size
        self.cache_used += new_size
        self.lru_sizes[path] = new_size

    def flush(self):
        if False:
            while True:
                i = 10
        'Writes data from cache_storage to next_storage. Only the dirty keys are written.\n        This is a cascading function and leads to data being written to the final storage in case of a chained cache.\n        '
        self.check_readonly()
        initial_autoflush = self.autoflush
        self.autoflush = False
        for (path, obj) in self.deeplake_objects.items():
            if obj.is_dirty:
                self[path] = obj
                obj.is_dirty = False
        if self.dirty_keys:
            if hasattr(self.next_storage, 'set_items') and self.use_async:
                d = {key: obj_to_bytes(self.cache_storage[key]) for key in self.dirty_keys}
                self.next_storage.set_items(d)
                self.dirty_keys.clear()
            else:
                for key in self.dirty_keys.copy():
                    self._forward(key)
                if self.next_storage is not None:
                    self.next_storage.flush()
        self.autoflush = initial_autoflush

    def get_deeplake_object(self, path: str, expected_class, meta: Optional[Dict]=None, url=False, partial_bytes: int=0):
        if False:
            for i in range(10):
                print('nop')
        "If the data at `path` was stored using the output of a DeepLakeMemoryObject's `tobytes` function,\n        this function will read it back into object form & keep the object in cache.\n\n        Args:\n            path (str): Path to the stored object.\n            expected_class (callable): The expected subclass of `DeepLakeMemoryObject`.\n            meta (dict, optional): Metadata associated with the stored object\n            url (bool): Get presigned url instead of downloading chunk (only for videos)\n            partial_bytes (int): Number of bytes to read from the beginning of the file. If 0, reads the whole file. Defaults to 0.\n\n        Raises:\n            ValueError: If the incorrect `expected_class` was provided.\n            ValueError: If the type of the data at `path` is invalid.\n            ValueError: If url is True but `expected_class` is not a subclass of BaseChunk.\n\n        Returns:\n            An instance of `expected_class` populated with the data.\n        "
        if partial_bytes != 0:
            assert issubclass(expected_class, BaseChunk)
            if path in self.lru_sizes:
                return self[path]
            buff = self.get_bytes(path, 0, partial_bytes)
            obj = expected_class.frombuffer(buff, meta, partial=True)
            obj.data_bytes = PartialReader(self, path, header_offset=obj.header_bytes)
            if obj.nbytes <= self.cache_size:
                self._insert_in_cache(path, obj)
            return obj
        if url:
            from deeplake.util.remove_cache import get_base_storage
            item = get_base_storage(self).get_presigned_url(path).encode('utf-8')
            if issubclass(expected_class, BaseChunk):
                obj = expected_class.frombuffer(item, meta, url=True)
                return obj
            else:
                raise ValueError('Expected class should be subclass of BaseChunk when url is True.')
        else:
            item = self[path]
        if isinstance(item, DeepLakeMemoryObject):
            if type(item) != expected_class:
                raise ValueError(f"'{path}' was expected to have the class '{expected_class.__name__}'. Instead, got: '{type(item)}'.")
            return item
        if isinstance(item, (bytes, memoryview)):
            obj = expected_class.frombuffer(item) if meta is None else expected_class.frombuffer(item, meta)
            if obj.nbytes <= self.cache_size:
                self._insert_in_cache(path, obj)
            return obj
        raise ValueError(f"Item at '{path}' got an invalid type: '{type(item)}'.")

    def __getitem__(self, path: str):
        if False:
            for i in range(10):
                print('nop')
        "If item is in cache_storage, retrieves from there and returns.\n        If item isn't in cache_storage, retrieves from next storage, stores in cache_storage (if possible) and returns.\n\n        Args:\n            path (str): The path relative to the root of the underlying storage.\n\n        Raises:\n            KeyError: if an object is not found at the path.\n\n        Returns:\n            bytes: The bytes of the object present at the path.\n        "
        if path in self.deeplake_objects:
            if path in self.lru_sizes:
                self.lru_sizes.move_to_end(path)
            return self.deeplake_objects[path]
        elif path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)
            return self.cache_storage[path]
        else:
            if self.next_storage is not None:
                result = self.next_storage[path]
                if _get_nbytes(result) <= self.cache_size:
                    self._insert_in_cache(path, result)
                return result
            raise KeyError(path)

    def get_bytes(self, path: str, start_byte: Optional[int]=None, end_byte: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        'Gets the object present at the path within the given byte range.\n\n        Args:\n            path (str): The path relative to the root of the provider.\n            start_byte (int, optional): If only specific bytes starting from start_byte are required.\n            end_byte (int, optional): If only specific bytes up to end_byte are required.\n\n        Returns:\n            bytes: The bytes of the object present at the path within the given byte range.\n\n        Raises:\n            InvalidBytesRequestedError: If `start_byte` > `end_byte` or `start_byte` < 0 or `end_byte` < 0.\n            KeyError: If an object is not found at the path.\n        '
        if path in self.deeplake_objects:
            if path in self.lru_sizes:
                self.lru_sizes.move_to_end(path)
            return self.deeplake_objects[path].tobytes()[start_byte:end_byte]
        elif path in self.lru_sizes and (not (isinstance(self.cache_storage[path], BaseChunk) and self.cache_storage[path].is_partially_read_chunk)):
            self.lru_sizes.move_to_end(path)
            return self.cache_storage[path][start_byte:end_byte]
        else:
            if self.next_storage is not None:
                return self.next_storage.get_bytes(path, start_byte, end_byte)
            raise KeyError(path)

    def __setitem__(self, path: str, value: Union[bytes, DeepLakeMemoryObject]):
        if False:
            for i in range(10):
                print('nop')
        'Puts the item in the cache_storage (if possible), else writes to next_storage.\n\n        Args:\n            path (str): the path relative to the root of the underlying storage.\n            value (bytes): the value to be assigned at the path.\n\n        Raises:\n            ReadOnlyError: If the provider is in read-only mode.\n        '
        self.check_readonly()
        if path in self.deeplake_objects:
            self.deeplake_objects[path].is_dirty = False
        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size
        if _get_nbytes(value) <= self.cache_size:
            self._insert_in_cache(path, value)
            self.dirty_keys[path] = None
        else:
            self._forward_value(path, value)
        self.maybe_flush()

    def __delitem__(self, path: str):
        if False:
            print('Hello World!')
        'Deletes the object present at the path from the cache and the underlying storage.\n\n        Args:\n            path (str): the path to the object relative to the root of the provider.\n\n        Raises:\n            KeyError: If an object is not found at the path.\n            ReadOnlyError: If the provider is in read-only mode.\n        '
        self.check_readonly()
        deleted_from_cache = False
        if path in self.deeplake_objects:
            self.remove_deeplake_object(path)
            deleted_from_cache = True
        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size
            del self.cache_storage[path]
            self.dirty_keys.pop(path, None)
            deleted_from_cache = True
        try:
            if self.next_storage is not None:
                del self.next_storage[path]
            else:
                raise KeyError(path)
        except KeyError:
            if not deleted_from_cache:
                raise

    def clear_cache(self):
        if False:
            while True:
                i = 10
        "Flushes the content of all the cache layers if not in read mode and and then deletes contents of all the layers of it.\n        This doesn't delete data from the actual storage.\n        "
        self._flush_if_not_read_only()
        self.clear_cache_without_flush()

    def clear_cache_without_flush(self):
        if False:
            while True:
                i = 10
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()
        self.deeplake_objects.clear()
        if self.next_storage is not None and hasattr(self.next_storage, 'clear_cache'):
            self.next_storage.clear_cache()

    def clear(self, prefix=''):
        if False:
            for i in range(10):
                print('nop')
        'Deletes ALL the data from all the layers of the cache and the actual storage.\n        This is an IRREVERSIBLE operation. Data once deleted can not be recovered.\n        '
        self.check_readonly()
        if prefix:
            rm = [path for path in self.deeplake_objects if path.startswith(prefix)]
            for path in rm:
                self.remove_deeplake_object(path)
            rm = [path for path in self.lru_sizes if path.startswith(prefix)]
            for path in rm:
                size = self.lru_sizes.pop(path)
                self.cache_used -= size
                self.dirty_keys.pop(path, None)
        else:
            self.cache_used = 0
            self.lru_sizes.clear()
            self.dirty_keys.clear()
            self.deeplake_objects.clear()
        self.cache_storage.clear(prefix=prefix)
        if self.next_storage is not None:
            self.next_storage.clear(prefix=prefix)

    def __len__(self):
        if False:
            return 10
        'Returns the number of files present in the cache and the underlying storage.\n\n        Returns:\n            int: the number of files present inside the root.\n        '
        return len(self._all_keys())

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Generator function that iterates over the keys of the cache and the underlying storage.\n\n        Yields:\n            str: the path of the object that it is iterating over, relative to the root of the provider.\n        '
        yield from self._all_keys()

    def _forward(self, path):
        if False:
            print('Hello World!')
        'Forward the value at a given path to the next storage, and un-marks its key.'
        if self.next_storage is not None:
            self._forward_value(path, self.cache_storage[path])

    def _forward_value(self, path, value):
        if False:
            for i in range(10):
                print('nop')
        'Forwards a path-value pair to the next storage, and un-marks its key.\n\n        Args:\n            path (str): the path to the object relative to the root of the provider.\n            value (bytes, DeepLakeMemoryObject): the value to send to the next storage.\n        '
        if self.next_storage is not None:
            self.dirty_keys.pop(path, None)
            if isinstance(value, DeepLakeMemoryObject):
                self.next_storage[path] = value.tobytes()
            else:
                self.next_storage[path] = value

    def _free_up_space(self, extra_size: int):
        if False:
            for i in range(10):
                print('nop')
        'Helper function that frees up space the requred space in cache.\n            No action is taken if there is sufficient space in the cache.\n\n        Args:\n            extra_size (int): the space that needs is required in bytes.\n        '
        while self.cache_used > 0 and extra_size + self.cache_used > self.cache_size:
            self._pop_from_cache()

    def _pop_from_cache(self):
        if False:
            for i in range(10):
                print('nop')
        'Helper function that pops the least recently used key, value pair from the cache'
        (key, itemsize) = self.lru_sizes.popitem(last=False)
        if key in self.dirty_keys:
            self._forward(key)
        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _insert_in_cache(self, path: str, value: Union[bytes, DeepLakeMemoryObject]):
        if False:
            for i in range(10):
                print('nop')
        'Helper function that adds a key value pair to the cache.\n\n        Args:\n            path (str): the path relative to the root of the underlying storage.\n            value (bytes): the value to be assigned at the path.\n\n        Raises:\n            ReadOnlyError: If the provider is in read-only mode.\n        '
        self._free_up_space(_get_nbytes(value))
        self.cache_storage[path] = value
        self.update_used_cache_for_path(path, _get_nbytes(value))

    def _all_keys(self):
        if False:
            for i in range(10):
                print('nop')
        'Helper function that lists all the objects present in the cache and the underlying storage.\n\n        Returns:\n            set: set of all the objects found in the cache and the underlying storage.\n        '
        key_set = set()
        if self.next_storage is not None:
            key_set = self.next_storage._all_keys()
        key_set = set().union(key_set, self.cache_storage._all_keys())
        for (path, obj) in self.deeplake_objects.items():
            if obj.is_dirty:
                key_set.add(path)
        return key_set

    def _flush_if_not_read_only(self):
        if False:
            while True:
                i = 10
        'Flushes the cache if not in read-only mode.'
        if not self.read_only:
            self.flush()

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the state of the cache, for pickling'
        self._flush_if_not_read_only()
        return {'next_storage': self.next_storage, 'cache_storage': self.cache_storage, 'cache_size': self.cache_size, 'use_async': self.use_async}

    def __setstate__(self, state: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        'Recreates a cache with the same configuration as the state.\n\n        Args:\n            state (dict): The state to be used to recreate the cache.\n\n        Note:\n            While restoring the cache, we reset its contents.\n            In case the cache storage was local/s3 and is still accessible when unpickled (if same machine/s3 creds present respectively), the earlier cache contents are no longer accessible.\n        '
        self.next_storage = state['next_storage']
        self.cache_storage = state['cache_storage']
        self.cache_size = state['cache_size']
        self.use_async = state['use_async']
        self.lru_sizes = OrderedDict()
        self.dirty_keys = OrderedDict()
        self.cache_used = 0
        self.deeplake_objects = {}

    def get_object_size(self, key: str) -> int:
        if False:
            i = 10
            return i + 15
        if key in self.deeplake_objects:
            return self.deeplake_objects[key].nbytes
        try:
            return self.cache_storage.get_object_size(key)
        except KeyError:
            if self.next_storage is not None:
                return self.next_storage.get_object_size(key)
            raise