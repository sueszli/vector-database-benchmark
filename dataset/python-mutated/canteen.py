import time
from contextlib import contextmanager
from ctypes import Array, Structure, c_bool, c_byte, c_int

class Buffer(Array):
    _length_ = 1024 * 1024
    _type_ = c_byte

class Canteen(Structure):
    _fields_ = [('initialized', c_bool), ('last_position', c_int), ('paths', Buffer)]

def canteen_add(canteen, path):
    if False:
        i = 10
        return i + 15
    lo = canteen.last_position
    hi = canteen.last_position + len(path) + 1
    if hi > len(canteen.paths):
        raise RuntimeError('canteen is full')
    canteen.paths[lo:hi] = path.encode('utf-8') + b';'
    canteen.last_position = hi

def canteen_get(canteen, timeout=1):
    if False:
        print('Hello World!')
    if not wait(canteen, timeout):
        return []
    data = bytes(canteen.paths[:canteen.last_position])
    return data.decode('utf-8').split(';')[:-1]

@contextmanager
def canteen_try_init(cv):
    if False:
        while True:
            i = 10
    if cv.initialized:
        yield False
        return
    with cv.get_lock():
        if cv.initialized:
            yield False
            return
        yield True
        cv.initialized = True

def wait(canteen, timeout):
    if False:
        i = 10
        return i + 15
    deadline = time.monotonic() + timeout
    while not canteen.initialized:
        if time.monotonic() > deadline:
            return False
        time.sleep(0)
    return True