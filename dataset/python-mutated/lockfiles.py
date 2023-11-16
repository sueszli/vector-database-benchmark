import json
import numbers
import collections.abc as collections_abc
from .models import DataView, Meta, PackageCollection

class _LockFileEncoder(json.JSONEncoder):
    """A specilized JSON encoder to convert loaded data into a lock file.

    This adds a few characteristics to the encoder:

    * The JSON is always prettified with indents and spaces.
    * The output is always UTF-8-encoded text, never binary, even on Python 2.
    """

    def __init__(self):
        if False:
            return 10
        super(_LockFileEncoder, self).__init__(indent=4, separators=(',', ': '), sort_keys=True)

    def encode(self, obj):
        if False:
            for i in range(10):
                print('nop')
        content = super(_LockFileEncoder, self).encode(obj)
        if not isinstance(content, str):
            content = content.decode('utf-8')
        content += '\n'
        return content

    def iterencode(self, obj):
        if False:
            print('Hello World!')
        for chunk in super(_LockFileEncoder, self).iterencode(obj):
            if not isinstance(chunk, str):
                chunk = chunk.decode('utf-8')
            yield chunk
        yield '\n'
PIPFILE_SPEC_CURRENT = 6

def _copy_jsonsafe(value):
    if False:
        for i in range(10):
            print('nop')
    'Deep-copy a value into JSON-safe types.\n    '
    if isinstance(value, (str, numbers.Number)):
        return value
    if isinstance(value, collections_abc.Mapping):
        return {str(k): _copy_jsonsafe(v) for (k, v) in value.items()}
    if isinstance(value, collections_abc.Iterable):
        return [_copy_jsonsafe(v) for v in value]
    if value is None:
        return None
    return str(value)

class Lockfile(DataView):
    """Representation of a Pipfile.lock.
    """
    __SCHEMA__ = {'_meta': {'type': 'dict', 'required': True}, 'default': {'type': 'dict', 'required': True}, 'develop': {'type': 'dict', 'required': True}}

    @classmethod
    def validate(cls, data):
        if False:
            for i in range(10):
                print('nop')
        super(Lockfile, cls).validate(data)
        for (key, value) in data.items():
            if key == '_meta':
                Meta.validate(value)
            else:
                PackageCollection.validate(value)

    @classmethod
    def load(cls, f, encoding=None):
        if False:
            print('Hello World!')
        if encoding is None:
            data = json.load(f)
        else:
            data = json.loads(f.read().decode(encoding))
        return cls(data)

    @classmethod
    def with_meta_from(cls, pipfile, categories=None):
        if False:
            while True:
                i = 10
        data = {'_meta': {'hash': _copy_jsonsafe(pipfile.get_hash()._data), 'pipfile-spec': PIPFILE_SPEC_CURRENT, 'requires': _copy_jsonsafe(pipfile._data.get('requires', {})), 'sources': _copy_jsonsafe(pipfile.sources._data)}}
        if categories is None:
            data['default'] = _copy_jsonsafe(pipfile._data.get('packages', {}))
            data['develop'] = _copy_jsonsafe(pipfile._data.get('dev-packages', {}))
        else:
            for category in categories:
                if category == 'default' or category == 'packages':
                    data['default'] = _copy_jsonsafe(pipfile._data.get('packages', {}))
                elif category == 'develop' or category == 'dev-packages':
                    data['develop'] = _copy_jsonsafe(pipfile._data.get('dev-packages', {}))
                else:
                    data[category] = _copy_jsonsafe(pipfile._data.get(category, {}))
        if 'default' not in data:
            data['default'] = {}
        if 'develop' not in data:
            data['develop'] = {}
        return cls(data)

    def __getitem__(self, key):
        if False:
            return 10
        value = self._data[key]
        try:
            if key == '_meta':
                return Meta(value)
            else:
                return PackageCollection(value)
        except KeyError:
            return value

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, DataView):
            self._data[key] = value._data
        else:
            self._data[key] = value

    def is_up_to_date(self, pipfile):
        if False:
            for i in range(10):
                print('nop')
        return self.meta.hash == pipfile.get_hash()

    def dump(self, f, encoding=None):
        if False:
            return 10
        encoder = _LockFileEncoder()
        if encoding is None:
            for chunk in encoder.iterencode(self._data):
                f.write(chunk)
        else:
            content = encoder.encode(self._data)
            f.write(content.encode(encoding))

    @property
    def meta(self):
        if False:
            i = 10
            return i + 15
        try:
            return self['_meta']
        except KeyError:
            raise AttributeError('meta')

    @meta.setter
    def meta(self, value):
        if False:
            i = 10
            return i + 15
        self['_meta'] = value

    @property
    def _meta(self):
        if False:
            while True:
                i = 10
        try:
            return self['_meta']
        except KeyError:
            raise AttributeError('meta')

    @_meta.setter
    def _meta(self, value):
        if False:
            while True:
                i = 10
        self['_meta'] = value

    @property
    def default(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self['default']
        except KeyError:
            raise AttributeError('default')

    @default.setter
    def default(self, value):
        if False:
            while True:
                i = 10
        self['default'] = value

    @property
    def develop(self):
        if False:
            while True:
                i = 10
        try:
            return self['develop']
        except KeyError:
            raise AttributeError('develop')

    @develop.setter
    def develop(self, value):
        if False:
            while True:
                i = 10
        self['develop'] = value