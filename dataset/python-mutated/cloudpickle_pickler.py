"""Pickler for values, functions, and classes.

For internal use only. No backwards compatibility guarantees.

Uses the cloudpickle library to pickle data, functions, lambdas
and classes.

dump_session and load_session are no-ops.
"""
import base64
import bz2
import io
import threading
import zlib
import cloudpickle
try:
    from absl import flags
except (ImportError, ModuleNotFoundError):
    pass
_pickle_lock = threading.RLock()
RLOCK_TYPE = type(_pickle_lock)

def dumps(o, enable_trace=True, use_zlib=False):
    if False:
        print('Hello World!')
    'For internal use only; no backwards-compatibility guarantees.'
    with _pickle_lock:
        with io.BytesIO() as file:
            pickler = cloudpickle.CloudPickler(file)
            try:
                pickler.dispatch_table[type(flags.FLAGS)] = _pickle_absl_flags
            except NameError:
                pass
            try:
                pickler.dispatch_table[RLOCK_TYPE] = _pickle_rlock
            except NameError:
                pass
            pickler.dump(o)
            s = file.getvalue()
    if use_zlib:
        c = zlib.compress(s, 9)
    else:
        c = bz2.compress(s, compresslevel=9)
    del s
    return base64.b64encode(c)

def loads(encoded, enable_trace=True, use_zlib=False):
    if False:
        i = 10
        return i + 15
    'For internal use only; no backwards-compatibility guarantees.'
    c = base64.b64decode(encoded)
    if use_zlib:
        s = zlib.decompress(c)
    else:
        s = bz2.decompress(c)
    del c
    with _pickle_lock:
        unpickled = cloudpickle.loads(s)
        return unpickled

def _pickle_absl_flags(obj):
    if False:
        print('Hello World!')
    return (_create_absl_flags, tuple([]))

def _create_absl_flags():
    if False:
        i = 10
        return i + 15
    return flags.FLAGS

def _pickle_rlock(obj):
    if False:
        while True:
            i = 10
    return (RLOCK_TYPE, tuple([]))

def dump_session(file_path):
    if False:
        while True:
            i = 10
    pass

def load_session(file_path):
    if False:
        while True:
            i = 10
    pass