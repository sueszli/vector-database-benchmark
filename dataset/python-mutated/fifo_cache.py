"""A simple first-in-first-out (FIFO) cache."""
from __future__ import absolute_import
from collections import deque

class FIFOCache(dict):
    """A class which manages a cache of entries, removing old ones."""

    def __init__(self, max_cache=100, after_cleanup_count=None):
        if False:
            i = 10
            return i + 15
        dict.__init__(self)
        self._max_cache = max_cache
        if after_cleanup_count is None:
            self._after_cleanup_count = self._max_cache * 8 / 10
        else:
            self._after_cleanup_count = min(after_cleanup_count, self._max_cache)
        self._cleanup = {}
        self._queue = deque()

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        'Add a value to the cache, there will be no cleanup function.'
        self.add(key, value, cleanup=None)

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        remove = getattr(self._queue, 'remove', None)
        if remove is not None:
            remove(key)
        else:
            self._queue = deque([k for k in self._queue if k != key])
        self._remove(key)

    def add(self, key, value, cleanup=None):
        if False:
            print('Hello World!')
        "Add a new value to the cache.\n\n        Also, if the entry is ever removed from the queue, call cleanup.\n        Passing it the key and value being removed.\n\n        :param key: The key to store it under\n        :param value: The object to store\n        :param cleanup: None or a function taking (key, value) to indicate\n                        'value' should be cleaned up\n        "
        if key in self:
            del self[key]
        self._queue.append(key)
        dict.__setitem__(self, key, value)
        if cleanup is not None:
            self._cleanup[key] = cleanup
        if len(self) > self._max_cache:
            self.cleanup()

    def cache_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the number of entries we will cache.'
        return self._max_cache

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        'Clear the cache until it shrinks to the requested size.\n\n        This does not completely wipe the cache, just makes sure it is under\n        the after_cleanup_count.\n        '
        while len(self) > self._after_cleanup_count:
            self._remove_oldest()
        if len(self._queue) != len(self):
            raise AssertionError('The length of the queue should always equal the length of the dict. %s != %s' % (len(self._queue), len(self)))

    def clear(self):
        if False:
            print('Hello World!')
        'Clear out all of the cache.'
        while self:
            self._remove_oldest()

    def _remove(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Remove an entry, making sure to call any cleanup function.'
        cleanup = self._cleanup.pop(key, None)
        val = dict.pop(self, key)
        if cleanup is not None:
            cleanup(key, val)
        return val

    def _remove_oldest(self):
        if False:
            return 10
        'Remove the oldest entry.'
        key = self._queue.popleft()
        self._remove(key)

    def resize(self, max_cache, after_cleanup_count=None):
        if False:
            for i in range(10):
                print('nop')
        'Increase/decrease the number of cached entries.\n\n        :param max_cache: The maximum number of entries to cache.\n        :param after_cleanup_count: After cleanup, we should have at most this\n            many entries. This defaults to 80% of max_cache.\n        '
        self._max_cache = max_cache
        if after_cleanup_count is None:
            self._after_cleanup_count = max_cache * 8 / 10
        else:
            self._after_cleanup_count = min(max_cache, after_cleanup_count)
        if len(self) > self._max_cache:
            self.cleanup()

    def copy(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self.copy)

    def pop(self, key, default=None):
        if False:
            print('Hello World!')
        raise NotImplementedError(self.pop)

    def popitem(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(self.popitem)

    def setdefault(self, key, defaultval=None):
        if False:
            print('Hello World!')
        'similar to dict.setdefault'
        if key in self:
            return self[key]
        self[key] = defaultval
        return defaultval

    def update(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Similar to dict.update()'
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                for (key, val) in arg.iteritems():
                    self.add(key, val)
            else:
                for (key, val) in args[0]:
                    self.add(key, val)
        elif len(args) > 1:
            raise TypeError('update expected at most 1 argument, got %d' % len(args))
        if kwargs:
            for (key, val) in kwargs.iteritems():
                self.add(key, val)

class FIFOSizeCache(FIFOCache):
    """An FIFOCache that removes things based on the size of the values.

    This differs in that it doesn't care how many actual items there are,
    it restricts the cache to be cleaned based on the size of the data.
    """

    def __init__(self, max_size=1024 * 1024, after_cleanup_size=None, compute_size=None):
        if False:
            while True:
                i = 10
        "Create a new FIFOSizeCache.\n\n        :param max_size: The max number of bytes to store before we start\n            clearing out entries.\n        :param after_cleanup_size: After cleaning up, shrink everything to this\n            size (defaults to 80% of max_size).\n        :param compute_size: A function to compute the size of a value. If\n            not supplied we default to 'len'.\n        "
        FIFOCache.__init__(self, max_cache=max_size)
        self._max_size = max_size
        if after_cleanup_size is None:
            self._after_cleanup_size = self._max_size * 8 / 10
        else:
            self._after_cleanup_size = min(after_cleanup_size, self._max_size)
        self._value_size = 0
        self._compute_size = compute_size
        if compute_size is None:
            self._compute_size = len

    def add(self, key, value, cleanup=None):
        if False:
            for i in range(10):
                print('nop')
        "Add a new value to the cache.\n\n        Also, if the entry is ever removed from the queue, call cleanup.\n        Passing it the key and value being removed.\n\n        :param key: The key to store it under\n        :param value: The object to store, this value by itself is >=\n            after_cleanup_size, then we will not store it at all.\n        :param cleanup: None or a function taking (key, value) to indicate\n                        'value' sohuld be cleaned up.\n        "
        if key in self:
            del self[key]
        value_len = self._compute_size(value)
        if value_len >= self._after_cleanup_size:
            return
        self._queue.append(key)
        dict.__setitem__(self, key, value)
        if cleanup is not None:
            self._cleanup[key] = cleanup
        self._value_size += value_len
        if self._value_size > self._max_size:
            self.cleanup()

    def cache_size(self):
        if False:
            print('Hello World!')
        'Get the number of bytes we will cache.'
        return self._max_size

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        'Clear the cache until it shrinks to the requested size.\n\n        This does not completely wipe the cache, just makes sure it is under\n        the after_cleanup_size.\n        '
        while self._value_size > self._after_cleanup_size:
            self._remove_oldest()

    def _remove(self, key):
        if False:
            i = 10
            return i + 15
        'Remove an entry, making sure to maintain the invariants.'
        val = FIFOCache._remove(self, key)
        self._value_size -= self._compute_size(val)
        return val

    def resize(self, max_size, after_cleanup_size=None):
        if False:
            return 10
        'Increase/decrease the amount of cached data.\n\n        :param max_size: The maximum number of bytes to cache.\n        :param after_cleanup_size: After cleanup, we should have at most this\n            many bytes cached. This defaults to 80% of max_size.\n        '
        FIFOCache.resize(self, max_size)
        self._max_size = max_size
        if after_cleanup_size is None:
            self._after_cleanup_size = max_size * 8 / 10
        else:
            self._after_cleanup_size = min(max_size, after_cleanup_size)
        if self._value_size > self._max_size:
            self.cleanup()