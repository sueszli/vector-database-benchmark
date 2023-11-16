from __future__ import annotations
from ansible.plugins.cache import BaseCacheModule
DOCUMENTATION = '\n    cache: none\n    short_description: write-only cache (no cache)\n    description:\n        - No caching at all\n    version_added: historical\n    author: core team (@ansible-core)\n'

class CacheModule(BaseCacheModule):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.empty = {}

    def get(self, key):
        if False:
            i = 10
            return i + 15
        return self.empty.get(key)

    def set(self, key, value):
        if False:
            i = 10
            return i + 15
        return value

    def keys(self):
        if False:
            return 10
        return self.empty.keys()

    def contains(self, key):
        if False:
            return 10
        return key in self.empty

    def delete(self, key):
        if False:
            return 10
        del self.emtpy[key]

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self.empty = {}

    def copy(self):
        if False:
            return 10
        return self.empty.copy()

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.copy()

    def __setstate__(self, data):
        if False:
            while True:
                i = 10
        self.empty = data