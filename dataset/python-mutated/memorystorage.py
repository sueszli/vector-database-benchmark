from wechatpy.session import SessionStorage

class MemoryStorage(SessionStorage):

    def __init__(self):
        if False:
            return 10
        self._data = {}

    def get(self, key, default=None):
        if False:
            while True:
                i = 10
        return self._data.get(key, default)

    def set(self, key, value, ttl=None):
        if False:
            i = 10
            return i + 15
        if value is None:
            return
        self._data[key] = value

    def delete(self, key):
        if False:
            i = 10
            return i + 15
        self._data.pop(key, None)