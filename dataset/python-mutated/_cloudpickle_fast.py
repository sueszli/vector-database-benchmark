"""
This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
'\nNew, fast version of the CloudPickler.\n\nThis new CloudPickler class can now extend the fast C Pickler instead of the\nprevious Python implementation of the Pickler class. Because this functionality\nis only available for Python versions 3.8+, a lot of backward-compatibility\ncode is also removed.\n\nNote that the C Pickler sublassing API is CPython-specific. Therefore, some\nguards present in cloudpickle.py that were written to handle PyPy specificities\nare not present in cloudpickle_fast.py\n'
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap
from ._compat import pickle, Pickler
from ._cloudpickle import _extract_code_globals, _BUILTIN_TYPE_NAMES, DEFAULT_PROTOCOL, _find_imported_submodules, _get_cell_contents, _is_importable, _builtin_type, _get_or_create_tracker_id, _make_skeleton_class, _make_skeleton_enum, _extract_class_dict, dynamic_subimport, subimport, _typevar_reduce, _get_bases, _make_cell, _make_empty_cell, CellType, _is_parametrized_type_hint, PYPY, cell_set, parametrized_type_hint_getinitargs, _create_parametrized_type_hint, builtin_code_type
if pickle.HIGHEST_PROTOCOL >= 5 and (not PYPY):

    def dump(obj, file, protocol=None, buffer_callback=None):
        if False:
            return 10
        'Serialize obj as bytes streamed into file\n\n        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to\n        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication\n        speed between processes running the same Python version.\n\n        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure\n        compatibility with older versions of Python.\n        '
        CloudPickler(file, protocol=protocol, buffer_callback=buffer_callback).dump(obj)

    def dumps(obj, protocol=None, buffer_callback=None):
        if False:
            return 10
        'Serialize obj as a string of bytes allocated in memory\n\n        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to\n        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication\n        speed between processes running the same Python version.\n\n        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure\n        compatibility with older versions of Python.\n        '
        with io.BytesIO() as file:
            cp = CloudPickler(file, protocol=protocol, buffer_callback=buffer_callback)
            cp.dump(obj)
            return file.getvalue()
else:

    def dump(obj, file, protocol=None):
        if False:
            i = 10
            return i + 15
        'Serialize obj as bytes streamed into file\n\n        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to\n        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication\n        speed between processes running the same Python version.\n\n        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure\n        compatibility with older versions of Python.\n        '
        CloudPickler(file, protocol=protocol).dump(obj)

    def dumps(obj, protocol=None):
        if False:
            i = 10
            return i + 15
        'Serialize obj as a string of bytes allocated in memory\n\n        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to\n        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication\n        speed between processes running the same Python version.\n\n        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure\n        compatibility with older versions of Python.\n        '
        with io.BytesIO() as file:
            cp = CloudPickler(file, protocol=protocol)
            cp.dump(obj)
            return file.getvalue()
(load, loads) = (pickle.load, pickle.loads)

def _class_getnewargs(obj):
    if False:
        i = 10
        return i + 15
    type_kwargs = {}
    if '__slots__' in obj.__dict__:
        type_kwargs['__slots__'] = obj.__slots__
    __dict__ = obj.__dict__.get('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__
    return (type(obj), obj.__name__, _get_bases(obj), type_kwargs, _get_or_create_tracker_id(obj), None)

def _enum_getnewargs(obj):
    if False:
        while True:
            i = 10
    members = dict(((e.name, e.value) for e in obj))
    return (obj.__bases__, obj.__name__, obj.__qualname__, members, obj.__module__, _get_or_create_tracker_id(obj), None)

def _file_reconstructor(retval):
    if False:
        i = 10
        return i + 15
    return retval

def _function_getstate(func):
    if False:
        for i in range(10):
            print('nop')
    slotstate = {'__name__': func.__name__, '__qualname__': func.__qualname__, '__annotations__': func.__annotations__, '__kwdefaults__': func.__kwdefaults__, '__defaults__': func.__defaults__, '__module__': func.__module__, '__doc__': func.__doc__, '__closure__': func.__closure__}
    f_globals_ref = _extract_code_globals(func.__code__)
    f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in func.__globals__}
    closure_values = list(map(_get_cell_contents, func.__closure__)) if func.__closure__ is not None else ()
    slotstate['_cloudpickle_submodules'] = _find_imported_submodules(func.__code__, itertools.chain(f_globals.values(), closure_values))
    slotstate['__globals__'] = f_globals
    state = func.__dict__
    return (state, slotstate)

def _class_getstate(obj):
    if False:
        while True:
            i = 10
    clsdict = _extract_class_dict(obj)
    clsdict.pop('__weakref__', None)
    if issubclass(type(obj), abc.ABCMeta):
        clsdict.pop('_abc_cache', None)
        clsdict.pop('_abc_negative_cache', None)
        clsdict.pop('_abc_negative_cache_version', None)
        registry = clsdict.pop('_abc_registry', None)
        if registry is None:
            clsdict.pop('_abc_impl', None)
            (registry, _, _, _) = abc._get_dump(obj)
            clsdict['_abc_impl'] = [subclass_weakref() for subclass_weakref in registry]
        else:
            clsdict['_abc_impl'] = [type_ for type_ in registry]
    if '__slots__' in clsdict:
        if isinstance(obj.__slots__, str):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)
    clsdict.pop('__dict__', None)
    return (clsdict, {})

def _enum_getstate(obj):
    if False:
        i = 10
        return i + 15
    (clsdict, slotstate) = _class_getstate(obj)
    members = dict(((e.name, e.value) for e in obj))
    for attrname in ['_generate_next_value_', '_member_names_', '_member_map_', '_member_type_', '_value2member_map_']:
        clsdict.pop(attrname, None)
    for member in members:
        clsdict.pop(member)
    return (clsdict, slotstate)

def _code_reduce(obj):
    if False:
        return 10
    'codeobject reducer'
    if hasattr(obj, 'co_posonlyargcount'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    else:
        args = (obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    return (types.CodeType, args)

def _cell_reduce(obj):
    if False:
        return 10
    "Cell (containing values of a function's free variables) reducer"
    try:
        obj.cell_contents
    except ValueError:
        return (_make_empty_cell, ())
    else:
        return (_make_cell, (obj.cell_contents,))

def _classmethod_reduce(obj):
    if False:
        while True:
            i = 10
    orig_func = obj.__func__
    return (type(obj), (orig_func,))

def _file_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    'Save a file'
    import io
    if not hasattr(obj, 'name') or not hasattr(obj, 'mode'):
        raise pickle.PicklingError('Cannot pickle files that do not map to an actual file')
    if obj is sys.stdout:
        return (getattr, (sys, 'stdout'))
    if obj is sys.stderr:
        return (getattr, (sys, 'stderr'))
    if obj is sys.stdin:
        raise pickle.PicklingError('Cannot pickle standard input')
    if obj.closed:
        raise pickle.PicklingError('Cannot pickle closed files')
    if hasattr(obj, 'isatty') and obj.isatty():
        raise pickle.PicklingError('Cannot pickle files that map to tty objects')
    if 'r' not in obj.mode and '+' not in obj.mode:
        raise pickle.PicklingError('Cannot pickle files that are not opened for reading: %s' % obj.mode)
    name = obj.name
    retval = io.StringIO()
    try:
        curloc = obj.tell()
        obj.seek(0)
        contents = obj.read()
        obj.seek(curloc)
    except IOError as e:
        raise pickle.PicklingError('Cannot pickle file %s as it cannot be read' % name) from e
    retval.write(contents)
    retval.seek(curloc)
    retval.name = name
    return (_file_reconstructor, (retval,))

def _getset_descriptor_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    return (getattr, (obj.__objclass__, obj.__name__))

def _mappingproxy_reduce(obj):
    if False:
        print('Hello World!')
    return (types.MappingProxyType, (dict(obj),))

def _memoryview_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    return (bytes, (obj.tobytes(),))

def _module_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    if _is_importable(obj):
        return (subimport, (obj.__name__,))
    else:
        obj.__dict__.pop('__builtins__', None)
        return (dynamic_subimport, (obj.__name__, vars(obj)))

def _method_reduce(obj):
    if False:
        i = 10
        return i + 15
    return (types.MethodType, (obj.__func__, obj.__self__))

def _logger_reduce(obj):
    if False:
        return 10
    return (logging.getLogger, (obj.name,))

def _root_logger_reduce(obj):
    if False:
        i = 10
        return i + 15
    return (logging.getLogger, ())

def _property_reduce(obj):
    if False:
        return 10
    return (property, (obj.fget, obj.fset, obj.fdel, obj.__doc__))

def _weakset_reduce(obj):
    if False:
        while True:
            i = 10
    return (weakref.WeakSet, (list(obj),))

def _dynamic_class_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    "\n    Save a class that can't be stored as module global.\n\n    This method is used to serialize classes that are defined inside\n    functions, or that otherwise can't be serialized as attribute lookups\n    from global modules.\n    "
    if Enum is not None and issubclass(obj, Enum):
        return (_make_skeleton_enum, _enum_getnewargs(obj), _enum_getstate(obj), None, None, _class_setstate)
    else:
        return (_make_skeleton_class, _class_getnewargs(obj), _class_getstate(obj), None, None, _class_setstate)

def _class_reduce(obj):
    if False:
        i = 10
        return i + 15
    'Select the reducer depending on the dynamic nature of the class obj'
    if obj is type(None):
        return (type, (None,))
    elif obj is type(Ellipsis):
        return (type, (Ellipsis,))
    elif obj is type(NotImplemented):
        return (type, (NotImplemented,))
    elif obj in _BUILTIN_TYPE_NAMES:
        return (_builtin_type, (_BUILTIN_TYPE_NAMES[obj],))
    elif not _is_importable(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented

def _function_setstate(obj, state):
    if False:
        print('Hello World!')
    'Update the state of a dynaamic function.\n\n    As __closure__ and __globals__ are readonly attributes of a function, we\n    cannot rely on the native setstate routine of pickle.load_build, that calls\n    setattr on items of the slotstate. Instead, we have to modify them inplace.\n    '
    (state, slotstate) = state
    obj.__dict__.update(state)
    obj_globals = slotstate.pop('__globals__')
    obj_closure = slotstate.pop('__closure__')
    slotstate.pop('_cloudpickle_submodules')
    obj.__globals__.update(obj_globals)
    obj.__globals__['__builtins__'] = __builtins__
    if obj_closure is not None:
        for (i, cell) in enumerate(obj_closure):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            cell_set(obj.__closure__[i], value)
    for (k, v) in slotstate.items():
        setattr(obj, k, v)

def _class_setstate(obj, state):
    if False:
        return 10
    (state, slotstate) = state
    registry = None
    for (attrname, attr) in state.items():
        if attrname == '_abc_impl':
            registry = attr
        else:
            setattr(obj, attrname, attr)
    if registry is not None:
        for subclass in registry:
            obj.register(subclass)
    return obj

class CloudPickler(Pickler):
    _dispatch_table = {}
    _dispatch_table[classmethod] = _classmethod_reduce
    _dispatch_table[io.TextIOWrapper] = _file_reduce
    _dispatch_table[logging.Logger] = _logger_reduce
    _dispatch_table[logging.RootLogger] = _root_logger_reduce
    _dispatch_table[memoryview] = _memoryview_reduce
    _dispatch_table[property] = _property_reduce
    _dispatch_table[staticmethod] = _classmethod_reduce
    _dispatch_table[CellType] = _cell_reduce
    _dispatch_table[types.CodeType] = _code_reduce
    _dispatch_table[types.GetSetDescriptorType] = _getset_descriptor_reduce
    _dispatch_table[types.ModuleType] = _module_reduce
    _dispatch_table[types.MethodType] = _method_reduce
    _dispatch_table[types.MappingProxyType] = _mappingproxy_reduce
    _dispatch_table[weakref.WeakSet] = _weakset_reduce
    _dispatch_table[typing.TypeVar] = _typevar_reduce
    dispatch_table = ChainMap(_dispatch_table, copyreg.dispatch_table)

    def _dynamic_function_reduce(self, func):
        if False:
            return 10
        'Reduce a function that is not pickleable via attribute lookup.'
        newargs = self._function_getnewargs(func)
        state = _function_getstate(func)
        return (types.FunctionType, newargs, state, None, None, _function_setstate)

    def _function_reduce(self, obj):
        if False:
            while True:
                i = 10
        'Reducer for function objects.\n\n        If obj is a top-level attribute of a file-backed module, this\n        reducer returns NotImplemented, making the CloudPickler fallback to\n        traditional _pickle.Pickler routines to save obj. Otherwise, it reduces\n        obj using a custom cloudpickle reducer designed specifically to handle\n        dynamic functions.\n\n        As opposed to cloudpickle.py, There no special handling for builtin\n        pypy functions because cloudpickle_fast is CPython-specific.\n        '
        if _is_importable(obj):
            return NotImplemented
        else:
            return self._dynamic_function_reduce(obj)

    def _function_getnewargs(self, func):
        if False:
            for i in range(10):
                print('nop')
        code = func.__code__
        base_globals = self.globals_ref.setdefault(id(func.__globals__), {})
        if base_globals == {}:
            for k in ['__package__', '__name__', '__path__', '__file__']:
                if k in func.__globals__:
                    base_globals[k] = func.__globals__[k]
        if func.__closure__ is None:
            closure = None
        else:
            closure = tuple((_make_empty_cell() for _ in range(len(code.co_freevars))))
        return (code, base_globals, None, None, closure)

    def dump(self, obj):
        if False:
            while True:
                i = 10
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = 'Could not pickle object as excessively deep recursion required.'
                raise pickle.PicklingError(msg) from e
            else:
                raise
    if pickle.HIGHEST_PROTOCOL >= 5:
        dispatch = dispatch_table

        def __init__(self, file, protocol=None, buffer_callback=None):
            if False:
                return 10
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(self, file, protocol=protocol, buffer_callback=buffer_callback)
            self.globals_ref = {}
            self.proto = int(protocol)

        def reducer_override(self, obj):
            if False:
                while True:
                    i = 10
            "Type-agnostic reducing callback for function and classes.\n\n            For performance reasons, subclasses of the C _pickle.Pickler class\n            cannot register custom reducers for functions and classes in the\n            dispatch_table. Reducer for such types must instead implemented in\n            the special reducer_override method.\n\n            Note that method will be called for any object except a few\n            builtin-types (int, lists, dicts etc.), which differs from reducers\n            in the Pickler's dispatch_table, each of them being invoked for\n            objects of a specific type only.\n\n            This property comes in handy for classes: although most classes are\n            instances of the ``type`` metaclass, some of them can be instances\n            of other custom metaclasses (such as enum.EnumMeta for example). In\n            particular, the metaclass will likely not be known in advance, and\n            thus cannot be special-cased using an entry in the dispatch_table.\n            reducer_override, among other things, allows us to register a\n            reducer that will be called for any class, independently of its\n            type.\n\n\n            Notes:\n\n            * reducer_override has the priority over dispatch_table-registered\n            reducers.\n            * reducer_override can be used to fix other limitations of\n              cloudpickle for other types that suffered from type-specific\n              reducers, such as Exceptions. See\n              https://github.com/cloudpipe/cloudpickle/issues/248\n            "
            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):
                return (_create_parametrized_type_hint, parametrized_type_hint_getinitargs(obj))
            t = type(obj)
            try:
                is_anyclass = issubclass(t, type)
            except TypeError:
                is_anyclass = False
            if is_anyclass:
                return _class_reduce(obj)
            elif isinstance(obj, types.FunctionType):
                return self._function_reduce(obj)
            else:
                return NotImplemented
    else:
        dispatch = Pickler.dispatch.copy()

        def __init__(self, file, protocol=None):
            if False:
                return 10
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(self, file, protocol=protocol)
            self.globals_ref = {}
            assert hasattr(self, 'proto')

        def _save_reduce_pickle5(self, func, args, state=None, listitems=None, dictitems=None, state_setter=None, obj=None):
            if False:
                print('Hello World!')
            save = self.save
            write = self.write
            self.save_reduce(func, args, state=None, listitems=listitems, dictitems=dictitems, obj=obj)
            save(state_setter)
            save(obj)
            save(state)
            write(pickle.TUPLE2)
            write(pickle.REDUCE)
            write(pickle.POP)

        def save_global(self, obj, name=None, pack=struct.pack):
            if False:
                i = 10
                return i + 15
            '\n            Save a "global".\n\n            The name of this method is somewhat misleading: all types get\n            dispatched here.\n            '
            if obj is type(None):
                return self.save_reduce(type, (None,), obj=obj)
            elif obj is type(Ellipsis):
                return self.save_reduce(type, (Ellipsis,), obj=obj)
            elif obj is type(NotImplemented):
                return self.save_reduce(type, (NotImplemented,), obj=obj)
            elif obj in _BUILTIN_TYPE_NAMES:
                return self.save_reduce(_builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)
            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):
                self.save_reduce(_create_parametrized_type_hint, parametrized_type_hint_getinitargs(obj), obj=obj)
            elif name is not None:
                Pickler.save_global(self, obj, name=name)
            elif not _is_importable(obj, name=name):
                self._save_reduce_pickle5(*_dynamic_class_reduce(obj), obj=obj)
            else:
                Pickler.save_global(self, obj, name=name)
        dispatch[type] = save_global

        def save_function(self, obj, name=None):
            if False:
                return 10
            ' Registered with the dispatch to handle all function types.\n\n            Determines what kind of function obj is (e.g. lambda, defined at\n            interactive prompt, etc) and handles the pickling appropriately.\n            '
            if _is_importable(obj, name=name):
                return Pickler.save_global(self, obj, name=name)
            elif PYPY and isinstance(obj.__code__, builtin_code_type):
                return self.save_pypy_builtin_func(obj)
            else:
                return self._save_reduce_pickle5(*self._dynamic_function_reduce(obj), obj=obj)

        def save_pypy_builtin_func(self, obj):
            if False:
                print('Hello World!')
            'Save pypy equivalent of builtin functions.\n            PyPy does not have the concept of builtin-functions. Instead,\n            builtin-functions are simple function instances, but with a\n            builtin-code attribute.\n            Most of the time, builtin functions should be pickled by attribute.\n            But PyPy has flaky support for __qualname__, so some builtin\n            functions such as float.__new__ will be classified as dynamic. For\n            this reason only, we created this special routine. Because\n            builtin-functions are not expected to have closure or globals,\n            there is no additional hack (compared the one already implemented\n            in pickle) to protect ourselves from reference cycles. A simple\n            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note\n            also that PyPy improved their support for __qualname__ in v3.6, so\n            this routing should be removed when cloudpickle supports only PyPy\n            3.6 and later.\n            '
            rv = (types.FunctionType, (obj.__code__, {}, obj.__name__, obj.__defaults__, obj.__closure__), obj.__dict__)
            self.save_reduce(*rv, obj=obj)
        dispatch[types.FunctionType] = save_function