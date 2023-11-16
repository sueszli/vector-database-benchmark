import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
_ForkingPickler = reduction.ForkingPickler
__all__ = ['RawValue', 'RawArray', 'Value', 'Array', 'copy', 'synchronized']
typecode_to_type = {'c': ctypes.c_char, 'u': ctypes.c_wchar, 'b': ctypes.c_byte, 'B': ctypes.c_ubyte, 'h': ctypes.c_short, 'H': ctypes.c_ushort, 'i': ctypes.c_int, 'I': ctypes.c_uint, 'l': ctypes.c_long, 'L': ctypes.c_ulong, 'q': ctypes.c_longlong, 'Q': ctypes.c_ulonglong, 'f': ctypes.c_float, 'd': ctypes.c_double}

def _new_value(type_):
    if False:
        i = 10
        return i + 15
    size = ctypes.sizeof(type_)
    wrapper = heap.BufferWrapper(size)
    return rebuild_ctype(type_, wrapper, None)

def RawValue(typecode_or_type, *args):
    if False:
        while True:
            i = 10
    '\n    Returns a ctypes object allocated from shared memory\n    '
    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    obj = _new_value(type_)
    ctypes.memset(ctypes.addressof(obj), 0, ctypes.sizeof(obj))
    obj.__init__(*args)
    return obj

def RawArray(typecode_or_type, size_or_initializer):
    if False:
        while True:
            i = 10
    '\n    Returns a ctypes array allocated from shared memory\n    '
    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
        obj = _new_value(type_)
        ctypes.memset(ctypes.addressof(obj), 0, ctypes.sizeof(obj))
        return obj
    else:
        type_ = type_ * len(size_or_initializer)
        result = _new_value(type_)
        result.__init__(*size_or_initializer)
        return result

def Value(typecode_or_type, *args, lock=True, ctx=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a synchronization wrapper for a Value\n    '
    obj = RawValue(typecode_or_type, *args)
    if lock is False:
        return obj
    if lock in (True, None):
        ctx = ctx or get_context()
        lock = ctx.RLock()
    if not hasattr(lock, 'acquire'):
        raise AttributeError("%r has no method 'acquire'" % lock)
    return synchronized(obj, lock, ctx=ctx)

def Array(typecode_or_type, size_or_initializer, *, lock=True, ctx=None):
    if False:
        while True:
            i = 10
    '\n    Return a synchronization wrapper for a RawArray\n    '
    obj = RawArray(typecode_or_type, size_or_initializer)
    if lock is False:
        return obj
    if lock in (True, None):
        ctx = ctx or get_context()
        lock = ctx.RLock()
    if not hasattr(lock, 'acquire'):
        raise AttributeError("%r has no method 'acquire'" % lock)
    return synchronized(obj, lock, ctx=ctx)

def copy(obj):
    if False:
        for i in range(10):
            print('nop')
    new_obj = _new_value(type(obj))
    ctypes.pointer(new_obj)[0] = obj
    return new_obj

def synchronized(obj, lock=None, ctx=None):
    if False:
        return 10
    assert not isinstance(obj, SynchronizedBase), 'object already synchronized'
    ctx = ctx or get_context()
    if isinstance(obj, ctypes._SimpleCData):
        return Synchronized(obj, lock, ctx)
    elif isinstance(obj, ctypes.Array):
        if obj._type_ is ctypes.c_char:
            return SynchronizedString(obj, lock, ctx)
        return SynchronizedArray(obj, lock, ctx)
    else:
        cls = type(obj)
        try:
            scls = class_cache[cls]
        except KeyError:
            names = [field[0] for field in cls._fields_]
            d = {name: make_property(name) for name in names}
            classname = 'Synchronized' + cls.__name__
            scls = class_cache[cls] = type(classname, (SynchronizedBase,), d)
        return scls(obj, lock, ctx)

def reduce_ctype(obj):
    if False:
        print('Hello World!')
    assert_spawning(obj)
    if isinstance(obj, ctypes.Array):
        return (rebuild_ctype, (obj._type_, obj._wrapper, obj._length_))
    else:
        return (rebuild_ctype, (type(obj), obj._wrapper, None))

def rebuild_ctype(type_, wrapper, length):
    if False:
        while True:
            i = 10
    if length is not None:
        type_ = type_ * length
    _ForkingPickler.register(type_, reduce_ctype)
    buf = wrapper.create_memoryview()
    obj = type_.from_buffer(buf)
    obj._wrapper = wrapper
    return obj

def make_property(name):
    if False:
        for i in range(10):
            print('nop')
    try:
        return prop_cache[name]
    except KeyError:
        d = {}
        exec(template % ((name,) * 7), d)
        prop_cache[name] = d[name]
        return d[name]
template = '\ndef get%s(self):\n    self.acquire()\n    try:\n        return self._obj.%s\n    finally:\n        self.release()\ndef set%s(self, value):\n    self.acquire()\n    try:\n        self._obj.%s = value\n    finally:\n        self.release()\n%s = property(get%s, set%s)\n'
prop_cache = {}
class_cache = weakref.WeakKeyDictionary()

class SynchronizedBase(object):

    def __init__(self, obj, lock=None, ctx=None):
        if False:
            print('Hello World!')
        self._obj = obj
        if lock:
            self._lock = lock
        else:
            ctx = ctx or get_context(force=True)
            self._lock = ctx.RLock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def __enter__(self):
        if False:
            print('Hello World!')
        return self._lock.__enter__()

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._lock.__exit__(*args)

    def __reduce__(self):
        if False:
            return 10
        assert_spawning(self)
        return (synchronized, (self._obj, self._lock))

    def get_obj(self):
        if False:
            while True:
                i = 10
        return self._obj

    def get_lock(self):
        if False:
            return 10
        return self._lock

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s wrapper for %s>' % (type(self).__name__, self._obj)

class Synchronized(SynchronizedBase):
    value = make_property('value')

class SynchronizedArray(SynchronizedBase):

    def __len__(self):
        if False:
            return 10
        return len(self._obj)

    def __getitem__(self, i):
        if False:
            return 10
        with self:
            return self._obj[i]

    def __setitem__(self, i, value):
        if False:
            for i in range(10):
                print('nop')
        with self:
            self._obj[i] = value

    def __getslice__(self, start, stop):
        if False:
            while True:
                i = 10
        with self:
            return self._obj[start:stop]

    def __setslice__(self, start, stop, values):
        if False:
            for i in range(10):
                print('nop')
        with self:
            self._obj[start:stop] = values

class SynchronizedString(SynchronizedArray):
    value = make_property('value')
    raw = make_property('raw')