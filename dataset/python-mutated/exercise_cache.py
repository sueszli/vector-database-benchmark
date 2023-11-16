from __future__ import annotations
DOCUMENTATION = "\n    inventory: exercise_cache\n    short_description: run tests against the specified cache plugin\n    description:\n      - This plugin doesn't modify inventory.\n      - Load a cache plugin and test the inventory cache interface is dict-like.\n      - Most inventory cache write methods only apply to the in-memory cache.\n      - The 'flush' and 'set_cache' methods should be used to apply changes to the backing cache plugin.\n      - The inventory cache read methods prefer the in-memory cache, and fall back to reading from the cache plugin.\n    extends_documentation_fragment:\n      - inventory_cache\n    options:\n      plugin:\n        required: true\n        description: name of the plugin (exercise_cache)\n    cache_timeout:\n        ini: []\n        env: []\n        cli: []\n        default: 0  # never expire\n"
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable
from ansible.utils.display import Display
from time import sleep
display = Display()

class InventoryModule(BaseInventoryPlugin, Cacheable):
    NAME = 'exercise_cache'
    test_cache_methods = ['test_plugin_name', 'test_update_cache_if_changed', 'test_set_cache', 'test_load_whole_cache', 'test_iter', 'test_len', 'test_get_missing_key', 'test_get_expired_key', 'test_initial_get', 'test_get', 'test_items', 'test_keys', 'test_values', 'test_pop', 'test_del', 'test_set', 'test_update', 'test_flush']

    def verify_file(self, path):
        if False:
            i = 10
            return i + 15
        if not path.endswith(('exercise_cache.yml', 'exercise_cache.yaml')):
            return False
        return super(InventoryModule, self).verify_file(path)

    def parse(self, inventory, loader, path, cache=None):
        if False:
            while True:
                i = 10
        super(InventoryModule, self).parse(inventory, loader, path)
        self._read_config_data(path)
        try:
            self.exercise_test_cache()
        except AnsibleError:
            raise
        except Exception as e:
            raise AnsibleError('Failed to run cache tests: {0}'.format(e)) from e

    def exercise_test_cache(self):
        if False:
            i = 10
            return i + 15
        failed = []
        for test_name in self.test_cache_methods:
            try:
                getattr(self, test_name)()
            except AssertionError:
                failed.append(test_name)
            finally:
                self.cache.flush()
                self.cache.update_cache_if_changed()
        if failed:
            raise AnsibleError(f"Cache tests failed: {', '.join(failed)}")

    def test_equal(self, a, b):
        if False:
            i = 10
            return i + 15
        try:
            assert a == b
        except AssertionError:
            display.warning(f'Assertion {a} == {b} failed')
            raise

    def test_plugin_name(self):
        if False:
            while True:
                i = 10
        self.test_equal(self.cache._plugin_name, self.get_option('cache_plugin'))

    def test_update_cache_if_changed(self):
        if False:
            print('Hello World!')
        self.cache._retrieved = {}
        self.cache._cache = {'foo': 'bar'}
        self.cache.update_cache_if_changed()
        self.test_equal(self.cache._retrieved, {'foo': 'bar'})
        self.test_equal(self.cache._cache, {'foo': 'bar'})

    def test_set_cache(self):
        if False:
            i = 10
            return i + 15
        cache_key1 = 'key1'
        cache1 = {'hosts': {'h1': {'foo': 'bar'}}}
        cache_key2 = 'key2'
        cache2 = {'hosts': {'h2': {}}}
        self.cache._cache = {cache_key1: cache1, cache_key2: cache2}
        self.cache.set_cache()
        self.test_equal(self.cache._plugin.contains(cache_key1), True)
        self.test_equal(self.cache._plugin.get(cache_key1), cache1)
        self.test_equal(self.cache._plugin.contains(cache_key2), True)
        self.test_equal(self.cache._plugin.get(cache_key2), cache2)

    def test_load_whole_cache(self):
        if False:
            while True:
                i = 10
        cache_data = {'key1': {'hosts': {'h1': {'foo': 'bar'}}}, 'key2': {'hosts': {'h2': {}}}}
        self.cache._cache = cache_data
        self.cache.set_cache()
        self.cache._cache = {}
        self.cache.load_whole_cache()
        self.test_equal(self.cache._cache, cache_data)

    def test_iter(self):
        if False:
            return 10
        cache_data = {'key1': {'hosts': {'h1': {'foo': 'bar'}}}, 'key2': {'hosts': {'h2': {}}}}
        self.cache._cache = cache_data
        self.test_equal(sorted(list(self.cache)), ['key1', 'key2'])

    def test_len(self):
        if False:
            return 10
        cache_data = {'key1': {'hosts': {'h1': {'foo': 'bar'}}}, 'key2': {'hosts': {'h2': {}}}}
        self.cache._cache = cache_data
        self.test_equal(len(self.cache), 2)

    def test_get_missing_key(self):
        if False:
            while True:
                i = 10
        try:
            self.cache['keyerror']
        except KeyError:
            pass
        else:
            assert False
        self.test_equal(self.cache.get('missing'), None)
        self.test_equal(self.cache.get('missing', 'default'), 'default')

    def _setup_expired(self):
        if False:
            print('Hello World!')
        self.cache._cache = {'expired': True}
        self.cache.set_cache()
        self.cache._cache = {}
        self.cache._retrieved = {}
        self.cache._plugin._cache = {}
        self.cache._plugin.set_option('timeout', 1)
        self.cache._plugin._timeout = 1
        sleep(2)

    def _cleanup_expired(self):
        if False:
            i = 10
            return i + 15
        self.cache._plugin.set_option('timeout', 0)
        self.cache._plugin._timeout = 0

    def test_get_expired_key(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self.cache._plugin, '_timeout'):
            return
        self._setup_expired()
        try:
            self.cache['expired']
        except KeyError:
            pass
        else:
            assert False
        finally:
            self._cleanup_expired()
        self._setup_expired()
        try:
            self.test_equal(self.cache.get('expired'), None)
            self.test_equal(self.cache.get('expired', 'default'), 'default')
        finally:
            self._cleanup_expired()

    def test_initial_get(self):
        if False:
            return 10
        k1 = {'hosts': {'h1': {'foo': 'bar'}}}
        k2 = {'hosts': {'h2': {}}}
        self.cache._cache = {'key1': k1, 'key2': k2}
        self.cache.set_cache()
        self.cache._cache = {}
        self.cache._retrieved = {}
        self.cache._plugin._cache = {}
        self.test_equal(self.cache['key1'], k1)
        self.cache._cache = {}
        self.cache._retrieved = {}
        self.cache._plugin._cache = {}
        self.test_equal(self.cache.get('key1'), k1)

    def test_get(self):
        if False:
            print('Hello World!')
        k1 = {'hosts': {'h1': {'foo': 'bar'}}}
        k2 = {'hosts': {'h2': {}}}
        self.cache._cache = {'key1': k1, 'key2': k2}
        self.cache.set_cache()
        self.test_equal(self.cache['key1'], k1)
        self.test_equal(self.cache.get('key1'), k1)

    def test_items(self):
        if False:
            print('Hello World!')
        self.test_equal(self.cache.items(), {}.items())
        test_items = {'hosts': {'host1': {'foo': 'bar'}}}
        self.cache._cache = test_items
        self.test_equal(self.cache.items(), test_items.items())

    def test_keys(self):
        if False:
            print('Hello World!')
        self.test_equal(self.cache.keys(), {}.keys())
        test_items = {'hosts': {'host1': {'foo': 'bar'}}}
        self.cache._cache = test_items
        self.test_equal(self.cache.keys(), test_items.keys())

    def test_values(self):
        if False:
            i = 10
            return i + 15
        self.test_equal(list(self.cache.values()), list({}.values()))
        test_items = {'hosts': {'host1': {'foo': 'bar'}}}
        self.cache._cache = test_items
        self.test_equal(list(self.cache.values()), list(test_items.values()))

    def test_pop(self):
        if False:
            i = 10
            return i + 15
        try:
            self.cache.pop('missing')
        except KeyError:
            pass
        else:
            assert False
        self.test_equal(self.cache.pop('missing', 'default'), 'default')
        self.cache._cache = {'cache_key': 'cache'}
        self.test_equal(self.cache.pop('cache_key'), 'cache')
        cache_key1 = 'key1'
        cache1 = {'hosts': {'h1': {'foo': 'bar'}}}
        cache_key2 = 'key2'
        cache2 = {'hosts': {'h2': {}}}
        self.cache._cache = {cache_key1: cache1, cache_key2: cache2}
        self.cache.set_cache()
        self.test_equal(self.cache.pop('key1'), cache1)
        self.test_equal(self.cache._cache, {cache_key2: cache2})
        self.test_equal(self.cache._plugin._cache, {cache_key1: cache1, cache_key2: cache2})

    def test_del(self):
        if False:
            i = 10
            return i + 15
        try:
            del self.cache['missing']
        except KeyError:
            pass
        else:
            assert False
        cache_key1 = 'key1'
        cache1 = {'hosts': {'h1': {'foo': 'bar'}}}
        cache_key2 = 'key2'
        cache2 = {'hosts': {'h2': {}}}
        self.cache._cache = {cache_key1: cache1, cache_key2: cache2}
        self.cache.set_cache()
        del self.cache['key1']
        self.test_equal(self.cache._cache, {cache_key2: cache2})
        self.test_equal(self.cache._plugin._cache, {cache_key1: cache1, cache_key2: cache2})

    def test_set(self):
        if False:
            return 10
        cache_key = 'key1'
        hosts = {'hosts': {'h1': {'foo': 'bar'}}}
        self.cache[cache_key] = hosts
        self.test_equal(self.cache._cache, {cache_key: hosts})
        self.test_equal(self.cache._plugin._cache, {})

    def test_update(self):
        if False:
            print('Hello World!')
        cache_key1 = 'key1'
        cache1 = {'hosts': {'h1': {'foo': 'bar'}}}
        cache_key2 = 'key2'
        cache2 = {'hosts': {'h2': {}}}
        self.cache._cache = {cache_key1: cache1}
        self.cache.update({cache_key2: cache2})
        self.test_equal(self.cache._cache, {cache_key1: cache1, cache_key2: cache2})

    def test_flush(self):
        if False:
            return 10
        cache_key1 = 'key1'
        cache1 = {'hosts': {'h1': {'foo': 'bar'}}}
        cache_key2 = 'key2'
        cache2 = {'hosts': {'h2': {}}}
        self.cache._cache = {cache_key1: cache1, cache_key2: cache2}
        self.cache.set_cache()
        self.cache.flush()
        self.test_equal(self.cache._cache, {})
        self.test_equal(self.cache._plugin._cache, {})