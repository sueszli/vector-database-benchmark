from werobot.session import SessionStorage
from werobot.utils import json_loads, json_dumps
from djangoblog.utils import cache

class MemcacheStorage(SessionStorage):

    def __init__(self, prefix='ws_'):
        if False:
            print('Hello World!')
        self.prefix = prefix
        self.cache = cache

    @property
    def is_available(self):
        if False:
            print('Hello World!')
        value = '1'
        self.set('checkavaliable', value=value)
        return value == self.get('checkavaliable')

    def key_name(self, s):
        if False:
            i = 10
            return i + 15
        return '{prefix}{s}'.format(prefix=self.prefix, s=s)

    def get(self, id):
        if False:
            while True:
                i = 10
        id = self.key_name(id)
        session_json = self.cache.get(id) or '{}'
        return json_loads(session_json)

    def set(self, id, value):
        if False:
            while True:
                i = 10
        id = self.key_name(id)
        self.cache.set(id, json_dumps(value))

    def delete(self, id):
        if False:
            while True:
                i = 10
        id = self.key_name(id)
        self.cache.delete(id)