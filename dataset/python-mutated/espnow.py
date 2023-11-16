from _espnow import *

class ESPNow(ESPNowBase):
    _data = [None, bytearray(MAX_DATA_LEN)]
    _none_tuple = (None, None)

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def irecv(self, timeout_ms=None):
        if False:
            print('Hello World!')
        n = self.recvinto(self._data, timeout_ms)
        return self._data if n else self._none_tuple

    def recv(self, timeout_ms=None):
        if False:
            while True:
                i = 10
        n = self.recvinto(self._data, timeout_ms)
        return [bytes(x) for x in self._data] if n else self._none_tuple

    def irq(self, callback):
        if False:
            for i in range(10):
                print('nop')
        super().irq(callback, self)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return self.irecv()