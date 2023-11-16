import logging
import sys
import os
sys.path.insert(0, '../ajenti-core')
import aj
import aj.config
import aj.entry
import aj.log
import aj.plugins

class TestConfig(aj.config.BaseConfig):

    def __init__(self):
        if False:
            return 10
        aj.config.BaseConfig.__init__(self)
        self.data = {'bind': {'mode': 'tcp', 'host': '0.0.0.0', 'port': 8000}, 'color': 'blue', 'name': 'test', 'ssl': {'enable': False}, 'email': {'enable': False}}

    def load(self):
        if False:
            while True:
                i = 10
        pass

    def save(self):
        if False:
            i = 10
            return i + 15
        pass
aj.log.init_console(logging.WARN)
os.makedirs('/etc/ajenti', exist_ok=True)
aj.entry.start(config=TestConfig(), dev_mode=False, debug_mode=True, autologin=True, product_name='ajenti', daemonize=False, plugin_providers=[aj.plugins.DirectoryPluginProvider('../plugins')])