import hashlib
import json
import pipenv.vendor.tomlkit as tomlkit
from .models import DataView, Hash, Requires, PipfileSection, Pipenv, PackageCollection, ScriptCollection, SourceCollection
PIPFILE_SECTIONS = {'source': SourceCollection, 'packages': PackageCollection, 'dev-packages': PackageCollection, 'requires': Requires, 'scripts': ScriptCollection, 'pipfile': PipfileSection, 'pipenv': Pipenv}
DEFAULT_SOURCE_TOML = '[[source]]\nname = "pypi"\nurl = "https://pypi.org/simple"\nverify_ssl = true\n'

class Pipfile(DataView):
    """Representation of a Pipfile.
    """
    __SCHEMA__ = {}

    @classmethod
    def validate(cls, data):
        if False:
            for i in range(10):
                print('nop')
        for (key, klass) in PIPFILE_SECTIONS.items():
            if key not in data:
                continue
            klass.validate(data[key])
        package_categories = set(data.keys()) - set(PIPFILE_SECTIONS.keys())
        for category in package_categories:
            PackageCollection.validate(data[category])

    @classmethod
    def load(cls, f, encoding=None):
        if False:
            return 10
        content = f.read()
        if encoding is not None:
            content = content.decode(encoding)
        data = tomlkit.loads(content)
        if 'source' not in data:
            sep = '' if content.startswith('\n') else '\n'
            content = DEFAULT_SOURCE_TOML + sep + content
        data = tomlkit.loads(content)
        return cls(data)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        value = self._data[key]
        try:
            return PIPFILE_SECTIONS[key](value)
        except KeyError:
            return value

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        if isinstance(value, DataView):
            self._data[key] = value._data
        else:
            self._data[key] = value

    def get_hash(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'_meta': {'sources': self._data['source'], 'requires': self._data.get('requires', {})}, 'default': self._data.get('packages', {}), 'develop': self._data.get('dev-packages', {})}
        for (category, values) in self._data.items():
            if category in PIPFILE_SECTIONS or category in ('default', 'develop', 'pipenv'):
                continue
            data[category] = values
        content = json.dumps(data, sort_keys=True, separators=(',', ':'))
        if isinstance(content, str):
            content = content.encode('utf-8')
        return Hash.from_hash(hashlib.sha256(content))

    def dump(self, f, encoding=None):
        if False:
            i = 10
            return i + 15
        content = tomlkit.dumps(self._data)
        if encoding is not None:
            content = content.encode(encoding)
        f.write(content)

    @property
    def sources(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self['source']
        except KeyError:
            raise AttributeError('sources')

    @sources.setter
    def sources(self, value):
        if False:
            while True:
                i = 10
        self['source'] = value

    @property
    def source(self):
        if False:
            return 10
        try:
            return self['source']
        except KeyError:
            raise AttributeError('source')

    @source.setter
    def source(self, value):
        if False:
            i = 10
            return i + 15
        self['source'] = value

    @property
    def packages(self):
        if False:
            return 10
        try:
            return self['packages']
        except KeyError:
            raise AttributeError('packages')

    @packages.setter
    def packages(self, value):
        if False:
            while True:
                i = 10
        self['packages'] = value

    @property
    def dev_packages(self):
        if False:
            while True:
                i = 10
        try:
            return self['dev-packages']
        except KeyError:
            raise AttributeError('dev-packages')

    @dev_packages.setter
    def dev_packages(self, value):
        if False:
            i = 10
            return i + 15
        self['dev-packages'] = value

    @property
    def requires(self):
        if False:
            print('Hello World!')
        try:
            return self['requires']
        except KeyError:
            raise AttributeError('requires')

    @requires.setter
    def requires(self, value):
        if False:
            print('Hello World!')
        self['requires'] = value

    @property
    def scripts(self):
        if False:
            return 10
        try:
            return self['scripts']
        except KeyError:
            raise AttributeError('scripts')

    @scripts.setter
    def scripts(self, value):
        if False:
            for i in range(10):
                print('nop')
        self['scripts'] = value