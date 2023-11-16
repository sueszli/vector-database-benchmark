import time
from ..base.addon import BaseAddon

class Interface:

    def __init__(self, address):
        if False:
            print('Hello World!')
        self.address = address
        self.history = {}

    def last_plugin_access(self, plugin_name, account):
        if False:
            return 10
        if (plugin_name, account) in self.history:
            return self.history[plugin_name, account]
        else:
            return 0

    def use_for(self, plugin_name, account):
        if False:
            while True:
                i = 10
        self.history[plugin_name, account] = time.time()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Interface - {}>'.format(self.address)

class MultiHome(BaseAddon):
    __name__ = 'MultiHome'
    __type__ = 'addon'
    __version__ = '0.21'
    __status__ = 'testing'
    __config__ = [('enabled', 'bool', 'Activated', False), ('interfaces', 'str', 'Interfaces', 'None')]
    __description__ = 'IP address changer'
    __license__ = 'GPLv3'
    __authors__ = [('mkaay', 'mkaay@mkaay.de'), ('GammaC0de', 'nitzo2001{AT]yahoo[DOT]com')]

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.interfaces = []
        self.old_get_request = None
        self.parse_interfaces(self.config.get('interfaces').split(';'))
        if not self.interfaces:
            self.parse_interfaces([self.pyload.config.get('download', 'interface')])
            self.pyload.config.set_plugin(self.__name__, 'interfaces', self.to_config())

    def to_config(self):
        if False:
            print('Hello World!')
        return ';'.join((i.address for i in self.interfaces))

    def parse_interfaces(self, interfaces):
        if False:
            return 10
        for interface in interfaces:
            if not interface or str(interface).lower() == 'none':
                continue
            self.interfaces.append(Interface(interface))

    def activate(self):
        if False:
            print('Hello World!')
        self.old_get_request = self.pyload.request_factory.get_request
        new_get_request = self.build_get_request()
        self.pyload.request_factory.get_request = lambda *args: new_get_request(*args)

    def best_interface(self, plugin_name, account):
        if False:
            for i in range(10):
                print('nop')
        best = None
        for interface in self.interfaces:
            if not best or interface.last_plugin_access(plugin_name, account) < best.last_plugin_access(plugin_name, account):
                best = interface
        return best

    def get_request(self, plugin_name, account=None):
        if False:
            for i in range(10):
                print('nop')
        iface = self.best_interface(plugin_name, account)
        if iface is None:
            self.log_warning(self._('Best interface not found'))
            return self.old_get_request(plugin_name, account)
        iface.use_for(plugin_name, account)
        self.pyload.request_factory.iface = lambda : iface.address
        self.log_debug('Using address', iface.address)
        return self.old_get_request(plugin_name, account)

    def build_get_request(self):
        if False:
            return 10

        def resfunc(*args):
            if False:
                i = 10
                return i + 15
            return self.get_request(*args)
        return resfunc