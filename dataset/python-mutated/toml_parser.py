import os.path
from configparser import ConfigParser
try:
    import toml
except ImportError:
    toml = False
from .base_parser import BaseParser
from ..freezing import recursively_freeze

class LuigiTomlParser(BaseParser, ConfigParser):
    NO_DEFAULT = object()
    enabled = bool(toml)
    data = dict()
    _instance = None
    _config_paths = ['/etc/luigi/luigi.toml', 'luigi.toml']

    @staticmethod
    def _update_data(data, new_data):
        if False:
            i = 10
            return i + 15
        if not new_data:
            return data
        if not data:
            return new_data
        for (section, content) in new_data.items():
            if section not in data:
                data[section] = dict()
            data[section].update(content)
        return data

    def read(self, config_paths):
        if False:
            i = 10
            return i + 15
        self.data = dict()
        for path in config_paths:
            if os.path.isfile(path):
                self.data = self._update_data(self.data, toml.load(path))
        for (section, content) in self.data.items():
            for (key, value) in content.items():
                if isinstance(value, dict):
                    self.data[section][key] = recursively_freeze(value)
        return self.data

    def get(self, section, option, default=NO_DEFAULT, **kwargs):
        if False:
            return 10
        try:
            return self.data[section][option]
        except KeyError:
            if default is self.NO_DEFAULT:
                raise
            return default

    def getboolean(self, section, option, default=NO_DEFAULT):
        if False:
            print('Hello World!')
        return self.get(section, option, default)

    def getint(self, section, option, default=NO_DEFAULT):
        if False:
            print('Hello World!')
        return self.get(section, option, default)

    def getfloat(self, section, option, default=NO_DEFAULT):
        if False:
            print('Hello World!')
        return self.get(section, option, default)

    def getintdict(self, section):
        if False:
            print('Hello World!')
        return self.data.get(section, {})

    def set(self, section, option, value=None):
        if False:
            while True:
                i = 10
        if section not in self.data:
            self.data[section] = {}
        self.data[section][option] = value

    def has_option(self, section, option):
        if False:
            for i in range(10):
                print('nop')
        return section in self.data and option in self.data[section]

    def __getitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.data[name]