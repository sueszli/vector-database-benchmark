import json

class Buffer:

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        self._data = data
        self._i = 0

    def readinto(self, buf):
        if False:
            i = 10
            return i + 15
        end = self._i + len(buf)
        remaining = len(self._data) - self._i
        end = min(end, len(self._data))
        l = min(len(buf), remaining)
        buf[:l] = self._data[self._i:end]
        self._i += l
        return l
print(json.load(Buffer(b'null')))
print(json.load(Buffer(b'"abc\\u0064e"')))
print(json.load(Buffer(b'[false, true, 1, -2]')))
print(json.load(Buffer(b'{"a":true}')))