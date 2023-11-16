import json
from . import Device
from websocket import create_connection

class Minitouch:

    def __init__(self, d: Device):
        if False:
            print('Hello World!')
        self._d = d
        self._prepare()

    def _prepare(self):
        if False:
            print('Hello World!')
        (self._w, self._h) = self._d.window_size()
        uri = self._d.path2url('/minitouch').replace('http:', 'ws:')
        self._ws = create_connection(uri)

    def down(self, x, y, index: int=0):
        if False:
            return 10
        px = x / self._w
        py = y / self._h
        self._ws_send({'operation': 'd', 'index': index, 'xP': px, 'yP': py, 'pressure': 0.5})
        self._commit()

    def move(self, x, y, index: int=0):
        if False:
            i = 10
            return i + 15
        px = x / self._w
        py = y / self._h
        self._ws_send({'operation': 'm', 'index': index, 'xP': px, 'yP': py, 'pressure': 0.5})

    def up(self, x, y, index: int=0):
        if False:
            return 10
        self._ws_send({'operation': 'u', 'index': index})
        self._commit()

    def click(self, x, y):
        if False:
            i = 10
            return i + 15
        self.down(x, y)
        self.up(x, y)

    def pinch_in(self, x, y, radius: int, steps: int=10):
        if False:
            while True:
                i = 10
        '\n        Args:\n            x, y: center point\n        '
        pass

    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._ws_send({'operation': 'r'})

    def _commit(self):
        if False:
            print('Hello World!')
        self._ws_send({'operation': 'c'})

    def _ws_send(self, payload: dict):
        if False:
            return 10
        from pprint import pprint
        pprint(payload)
        self._ws.send(json.dumps(payload), opcode=1)