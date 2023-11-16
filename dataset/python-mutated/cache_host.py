from __future__ import annotations
DOCUMENTATION = '\n    inventory: cache_host\n    short_description: add a host to inventory and cache it\n    description: add a host to inventory and cache it\n    extends_documentation_fragment:\n      - inventory_cache\n    options:\n      plugin:\n        required: true\n        description: name of the plugin (cache_host)\n'
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable
import random

class InventoryModule(BaseInventoryPlugin, Cacheable):
    NAME = 'cache_host'

    def verify_file(self, path):
        if False:
            i = 10
            return i + 15
        if not path.endswith(('cache_host.yml', 'cache_host.yaml')):
            return False
        return super(InventoryModule, self).verify_file(path)

    def parse(self, inventory, loader, path, cache=None):
        if False:
            i = 10
            return i + 15
        super(InventoryModule, self).parse(inventory, loader, path)
        self._read_config_data(path)
        cache_key = self.get_cache_key(path)
        read_cache = self.get_option('cache') and cache
        update_cache = self.get_option('cache') and (not cache)
        host = None
        if read_cache:
            try:
                host = self._cache[cache_key]
            except KeyError:
                update_cache = True
        if host is None:
            host = 'testhost{0}'.format(random.randint(0, 50))
        self.inventory.add_host(host, 'all')
        if update_cache:
            self._cache[cache_key] = host