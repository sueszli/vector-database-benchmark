import threading
from . import shared_abstract

class SimpleSharedDict(shared_abstract.SharedDict):
    __slots__ = ('d',)

    def __init__(self):
        if False:
            return 10
        self.d = {}

    def get_int(self, key):
        if False:
            print('Hello World!')
        return self.d.get(key, None)

    def set_int(self, key, value):
        if False:
            print('Hello World!')
        self.d[key] = value

    def get_str(self, key):
        if False:
            i = 10
            return i + 15
        return self.d.get(key, None)

    def set_str(self, key, value):
        if False:
            while True:
                i = 10
        self.d[key] = value

def schedule(delay, func, *args):
    if False:
        i = 10
        return i + 15

    def call_later():
        if False:
            print('Hello World!')
        t = threading.Timer(delay, wrapper)
        t.daemon = True
        t.start()

    def wrapper():
        if False:
            i = 10
            return i + 15
        call_later()
        func(*args)
    call_later()
    return True