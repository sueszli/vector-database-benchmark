import threading
from pyramid.registry import global_registry

class ThreadLocalManager(threading.local):

    def __init__(self, default=None):
        if False:
            i = 10
            return i + 15
        self.stack = []
        self.default = default

    def push(self, info):
        if False:
            while True:
                i = 10
        self.stack.append(info)
    set = push

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        if self.stack:
            return self.stack.pop()

    def get(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.stack[-1]
        except IndexError:
            return self.default()

    def clear(self):
        if False:
            while True:
                i = 10
        self.stack[:] = []

def defaults():
    if False:
        print('Hello World!')
    return {'request': None, 'registry': global_registry}
manager = ThreadLocalManager(default=defaults)

def get_current_request():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the currently active request or ``None`` if no request\n    is currently active.\n\n    This function should be used *extremely sparingly*, usually only\n    in unit testing code.  It's almost always usually a mistake to use\n    ``get_current_request`` outside a testing context because its\n    usage makes it possible to write code that can be neither easily\n    tested nor scripted.\n\n    "
    return manager.get()['request']

def get_current_registry(context=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the currently active :term:`application registry` or the\n    global application registry if no request is currently active.\n\n    This function should be used *extremely sparingly*, usually only\n    in unit testing code.  It's almost always usually a mistake to use\n    ``get_current_registry`` outside a testing context because its\n    usage makes it possible to write code that can be neither easily\n    tested nor scripted.\n\n    "
    return manager.get()['registry']

class RequestContext:

    def __init__(self, request):
        if False:
            for i in range(10):
                print('nop')
        self.request = request

    def begin(self):
        if False:
            print('Hello World!')
        request = self.request
        registry = request.registry
        manager.push({'registry': registry, 'request': request})
        return request

    def end(self):
        if False:
            i = 10
            return i + 15
        manager.pop()

    def __enter__(self):
        if False:
            return 10
        return self.begin()

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.end()