from __future__ import annotations
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass

class _EventSource:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._handlers = set()

    def __iadd__(self, handler):
        if False:
            i = 10
            return i + 15
        if not callable(handler):
            raise ValueError('handler must be callable')
        self._handlers.add(handler)
        return self

    def __isub__(self, handler):
        if False:
            print('Hello World!')
        try:
            self._handlers.remove(handler)
        except KeyError:
            pass
        return self

    def _on_exception(self, handler, exc, *args, **kwargs):
        if False:
            return 10
        return True

    def fire(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        for h in self._handlers:
            try:
                h(*args, **kwargs)
            except Exception as ex:
                if self._on_exception(h, ex, *args, **kwargs):
                    raise

class _AnsibleCollectionConfig(type):

    def __init__(cls, meta, name, bases):
        if False:
            while True:
                i = 10
        cls._collection_finder = None
        cls._default_collection = None
        cls._on_collection_load = _EventSource()

    @property
    def collection_finder(cls):
        if False:
            print('Hello World!')
        return cls._collection_finder

    @collection_finder.setter
    def collection_finder(cls, value):
        if False:
            return 10
        if cls._collection_finder:
            raise ValueError('an AnsibleCollectionFinder has already been configured')
        cls._collection_finder = value

    @property
    def collection_paths(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._require_finder()
        return [to_text(p) for p in cls._collection_finder._n_collection_paths]

    @property
    def default_collection(cls):
        if False:
            i = 10
            return i + 15
        return cls._default_collection

    @default_collection.setter
    def default_collection(cls, value):
        if False:
            while True:
                i = 10
        cls._default_collection = value

    @property
    def on_collection_load(cls):
        if False:
            i = 10
            return i + 15
        return cls._on_collection_load

    @on_collection_load.setter
    def on_collection_load(cls, value):
        if False:
            print('Hello World!')
        if value is not cls._on_collection_load:
            raise ValueError('on_collection_load is not directly settable (use +=)')

    @property
    def playbook_paths(cls):
        if False:
            while True:
                i = 10
        cls._require_finder()
        return [to_text(p) for p in cls._collection_finder._n_playbook_paths]

    @playbook_paths.setter
    def playbook_paths(cls, value):
        if False:
            while True:
                i = 10
        cls._require_finder()
        cls._collection_finder.set_playbook_paths(value)

    def _require_finder(cls):
        if False:
            return 10
        if not cls._collection_finder:
            raise NotImplementedError('an AnsibleCollectionFinder has not been installed in this process')

@add_metaclass(_AnsibleCollectionConfig)
class AnsibleCollectionConfig(object):
    pass