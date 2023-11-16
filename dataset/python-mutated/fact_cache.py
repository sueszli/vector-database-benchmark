from __future__ import annotations
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.plugins.loader import cache_loader
from ansible.utils.display import Display
display = Display()

class FactCache(MutableMapping):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self._plugin = cache_loader.get(C.CACHE_PLUGIN)
        if not self._plugin:
            raise AnsibleError('Unable to load the facts cache plugin (%s).' % C.CACHE_PLUGIN)
        super(FactCache, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if False:
            return 10
        if not self._plugin.contains(key):
            raise KeyError
        return self._plugin.get(key)

    def __setitem__(self, key, value):
        if False:
            return 10
        self._plugin.set(key, value)

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        self._plugin.delete(key)

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._plugin.contains(key)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._plugin.keys())

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._plugin.keys())

    def copy(self):
        if False:
            print('Hello World!')
        ' Return a primitive copy of the keys and values from the cache. '
        return dict(self)

    def keys(self):
        if False:
            while True:
                i = 10
        return self._plugin.keys()

    def flush(self):
        if False:
            print('Hello World!')
        ' Flush the fact cache of all keys. '
        self._plugin.flush()

    def first_order_merge(self, key, value):
        if False:
            print('Hello World!')
        host_facts = {key: value}
        try:
            host_cache = self._plugin.get(key)
            if host_cache:
                host_cache.update(value)
                host_facts[key] = host_cache
        except KeyError:
            pass
        super(FactCache, self).update(host_facts)