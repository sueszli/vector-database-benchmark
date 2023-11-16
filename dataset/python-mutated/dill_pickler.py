"""Pickler for values, functions, and classes.

For internal use only. No backwards compatibility guarantees.

Pickles created by the pickling library contain non-ASCII characters, so
we base64-encode the results so that we can put them in a JSON objects.
The pickler is used to embed FlatMap callable objects into the workflow JSON
description.

The pickler module should be used to pickle functions and modules; for values,
the coders.*PickleCoder classes should be used instead.
"""
import base64
import bz2
import logging
import sys
import threading
import traceback
import types
import zlib
from typing import Any
from typing import Dict
from typing import Tuple
import dill
settings = {'dill_byref': None}
if sys.version_info >= (3, 10) and dill.__version__ == '0.3.1.1':
    from types import CodeType

    @dill.register(CodeType)
    def save_code(pickler, obj):
        if False:
            print('Hello World!')
        if hasattr(obj, 'co_endlinetable'):
            args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname, obj.co_firstlineno, obj.co_linetable, obj.co_endlinetable, obj.co_columntable, obj.co_exceptiontable, obj.co_freevars, obj.co_cellvars)
        elif hasattr(obj, 'co_exceptiontable'):
            args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname, obj.co_firstlineno, obj.co_linetable, obj.co_exceptiontable, obj.co_freevars, obj.co_cellvars)
        elif hasattr(obj, 'co_linetable'):
            args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_linetable, obj.co_freevars, obj.co_cellvars)
        elif hasattr(obj, 'co_posonlyargcount'):
            args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
        else:
            args = (obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
        pickler.save_reduce(CodeType, args, obj=obj)
    dill._dill.save_code = save_code

class _NoOpContextManager(object):

    def __enter__(self):
        if False:
            return 10
        pass

    def __exit__(self, *unused_exc_info):
        if False:
            while True:
                i = 10
        pass
_pickle_lock = threading.RLock()
if not getattr(dill, 'dill', None):
    dill.dill = dill._dill
    sys.modules['dill.dill'] = dill._dill
if not getattr(dill, '_dill', None):
    dill._dill = dill.dill
    sys.modules['dill._dill'] = dill.dill
dill_log = getattr(dill.dill, 'log', None)
if not dill_log:
    dill_log = getattr(dill.dill, 'logger')

def _is_nested_class(cls):
    if False:
        return 10
    'Returns true if argument is a class object that appears to be nested.'
    return isinstance(cls, type) and cls.__module__ is not None and (cls.__module__ != 'builtins') and (cls.__name__ not in sys.modules[cls.__module__].__dict__)

def _find_containing_class(nested_class):
    if False:
        print('Hello World!')
    'Finds containing class of a nested class passed as argument.'
    seen = set()

    def _find_containing_class_inner(outer):
        if False:
            print('Hello World!')
        if outer in seen:
            return None
        seen.add(outer)
        for (k, v) in outer.__dict__.items():
            if v is nested_class:
                return (outer, k)
            elif isinstance(v, type) and hasattr(v, '__dict__'):
                res = _find_containing_class_inner(v)
                if res:
                    return res
    return _find_containing_class_inner(sys.modules[nested_class.__module__])

def _dict_from_mappingproxy(mp):
    if False:
        for i in range(10):
            print('nop')
    d = mp.copy()
    d.pop('__dict__', None)
    d.pop('__prepare__', None)
    d.pop('__weakref__', None)
    return d

def _nested_type_wrapper(fun):
    if False:
        for i in range(10):
            print('nop')
    'A wrapper for the standard pickler handler for class objects.\n\n  Args:\n    fun: Original pickler handler for type objects.\n\n  Returns:\n    A wrapper for type objects that handles nested classes.\n\n  The wrapper detects if an object being pickled is a nested class object.\n  For nested class object only it will save the containing class object so\n  the nested structure is recreated during unpickle.\n  '

    def wrapper(pickler, obj):
        if False:
            while True:
                i = 10
        if _is_nested_class(obj) and obj.__module__ != '__main__':
            containing_class_and_name = _find_containing_class(obj)
            if containing_class_and_name is not None:
                return pickler.save_reduce(getattr, containing_class_and_name, obj=obj)
        try:
            return fun(pickler, obj)
        except dill.dill.PicklingError:
            return pickler.save_reduce(dill.dill._create_type, (type(obj), obj.__name__, obj.__bases__, _dict_from_mappingproxy(obj.__dict__)), obj=obj)
    return wrapper
dill.dill.Pickler.dispatch[type] = _nested_type_wrapper(dill.dill.Pickler.dispatch[type])

def _reject_generators(unused_pickler, unused_obj):
    if False:
        while True:
            i = 10
    raise TypeError("can't (safely) pickle generator objects")
dill.dill.Pickler.dispatch[types.GeneratorType] = _reject_generators
if 'save_module' in dir(dill.dill):
    old_save_module = dill.dill.save_module

    @dill.dill.register(dill.dill.ModuleType)
    def save_module(pickler, obj):
        if False:
            while True:
                i = 10
        if dill.dill.is_dill(pickler) and obj is pickler._main:
            return old_save_module(pickler, obj)
        else:
            dill_log.info('M2: %s' % obj)
            pickler.save_reduce(dill.dill._import_module, (obj.__name__,), obj=obj)
            dill_log.info('# M2')
    old_save_module_dict = dill.dill.save_module_dict
    known_module_dicts = {}

    @dill.dill.register(dict)
    def new_save_module_dict(pickler, obj):
        if False:
            while True:
                i = 10
        obj_id = id(obj)
        if not known_module_dicts or '__file__' in obj or '__package__' in obj:
            if obj_id not in known_module_dicts:
                for m in list(sys.modules.values()):
                    try:
                        _ = m.__dict__
                    except AttributeError:
                        pass
                for m in list(sys.modules.values()):
                    try:
                        if m and m.__name__ != '__main__' and isinstance(m, dill.dill.ModuleType):
                            d = m.__dict__
                            known_module_dicts[id(d)] = (m, d)
                    except AttributeError:
                        pass
        if obj_id in known_module_dicts and dill.dill.is_dill(pickler):
            m = known_module_dicts[obj_id][0]
            try:
                dill.dill._import_module(m.__name__)
                return pickler.save_reduce(getattr, (known_module_dicts[obj_id][0], '__dict__'), obj=obj)
            except (ImportError, AttributeError):
                return old_save_module_dict(pickler, obj)
        else:
            return old_save_module_dict(pickler, obj)
    dill.dill.save_module_dict = new_save_module_dict

    def _nest_dill_logging():
        if False:
            for i in range(10):
                print('nop')
        'Prefix all dill logging with its depth in the callstack.\n\n    Useful for debugging pickling of deeply nested structures.\n    '
        old_log_info = dill_log.info

        def new_log_info(msg, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            old_log_info(('1 2 3 4 5 6 7 8 9 0 ' * 10)[:len(traceback.extract_stack())] + msg, *args, **kwargs)
        dill_log.info = new_log_info
logging.getLogger('dill').setLevel(logging.WARN)

def dumps(o, enable_trace=True, use_zlib=False):
    if False:
        while True:
            i = 10
    'For internal use only; no backwards-compatibility guarantees.'
    with _pickle_lock:
        try:
            s = dill.dumps(o, byref=settings['dill_byref'])
        except Exception:
            if enable_trace:
                dill.dill._trace(True)
                s = dill.dumps(o, byref=settings['dill_byref'])
            else:
                raise
        finally:
            dill.dill._trace(False)
    if use_zlib:
        c = zlib.compress(s, 9)
    else:
        c = bz2.compress(s, compresslevel=9)
    del s
    return base64.b64encode(c)

def loads(encoded, enable_trace=True, use_zlib=False):
    if False:
        while True:
            i = 10
    'For internal use only; no backwards-compatibility guarantees.'
    c = base64.b64decode(encoded)
    if use_zlib:
        s = zlib.decompress(c)
    else:
        s = bz2.decompress(c)
    del c
    with _pickle_lock:
        try:
            return dill.loads(s)
        except Exception:
            if enable_trace:
                dill.dill._trace(True)
                return dill.loads(s)
            else:
                raise
        finally:
            dill.dill._trace(False)

def dump_session(file_path):
    if False:
        i = 10
        return i + 15
    'For internal use only; no backwards-compatibility guarantees.\n\n  Pickle the current python session to be used in the worker.\n\n  Note: Due to the inconsistency in the first dump of dill dump_session we\n  create and load the dump twice to have consistent results in the worker and\n  the running session. Check: https://github.com/uqfoundation/dill/issues/195\n  '
    with _pickle_lock:
        dill.dump_session(file_path)
        dill.load_session(file_path)
        return dill.dump_session(file_path)

def load_session(file_path):
    if False:
        return 10
    with _pickle_lock:
        return dill.load_session(file_path)

def override_pickler_hooks(extend=True):
    if False:
        for i in range(10):
            print('nop')
    ' Extends the dill library hooks into that of the standard pickler library.\n\n  If false all hooks that dill overrides will be removed.\n  If true dill hooks will be injected into the pickler library dispatch_table.\n  '
    dill.extend(extend)