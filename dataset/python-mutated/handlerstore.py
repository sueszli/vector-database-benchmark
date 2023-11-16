from itertools import chain
from robot.errors import DataError
from robot.utils import NormalizedDict, seq2str
from .usererrorhandler import UserErrorHandler

class HandlerStore:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._normal = NormalizedDict(ignore='_')
        self._embedded = []

    def add(self, handler, embedded=False):
        if False:
            while True:
                i = 10
        if embedded:
            self._embedded.append(handler)
        elif handler.name not in self._normal:
            self._normal[handler.name] = handler
        else:
            error = DataError('Keyword with same name defined multiple times.')
            self._normal[handler.name] = UserErrorHandler(error, handler.name, handler.owner)
            raise error

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return chain(self._normal.values(), self._embedded)

    def __len__(self):
        if False:
            return 10
        return len(self._normal) + len(self._embedded)

    def __contains__(self, name):
        if False:
            while True:
                i = 10
        if name in self._normal:
            return True
        if not self._embedded:
            return False
        return any((template.matches(name) for template in self._embedded))

    def __getitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        handlers = self.get_handlers(name)
        if len(handlers) == 1:
            return handlers[0]
        if not handlers:
            raise ValueError(f"No handler with name '{name}' found.")
        names = seq2str([handler.name for handler in handlers])
        raise ValueError(f"Multiple handlers matching name '{name}' found: {names}")

    def get_handlers(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in self._normal:
            return [self._normal[name]]
        return [template for template in self._embedded if template.matches(name)]