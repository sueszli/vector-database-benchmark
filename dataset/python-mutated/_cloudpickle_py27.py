"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

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
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import dis
from functools import partial
import imp
import io
import itertools
import logging
import opcode
import operator
import pickle
import struct
import sys
import traceback
import types
import weakref
if sys.version_info.major == 2:
    from pickle import Pickler
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
    from types import ClassType as _ClassType
    _class_type = _ClassType
else:
    _class_type = type
    from pickle import _Pickler as Pickler
    from io import BytesIO as StringIO
    PY3 = True

def _make_cell_set_template_code():
    if False:
        for i in range(10):
            print('nop')
    "Get the Python compiler to emit LOAD_FAST(arg); STORE_DEREF\n\n    Notes\n    -----\n    In Python 3, we could use an easier function:\n\n    .. code-block:: python\n\n       def f():\n           cell = None\n\n           def _stub(value):\n               nonlocal cell\n               cell = value\n\n           return _stub\n\n        _cell_set_template_code = f()\n\n    This function is _only_ a LOAD_FAST(arg); STORE_DEREF, but that is\n    invalid syntax on Python 2. If we use this function we also don't need\n    to do the weird freevars/cellvars swap below\n    "

    def inner(value):
        if False:
            i = 10
            return i + 15
        lambda : cell
        cell = value
    co = inner.__code__
    if not PY3:
        return types.CodeType(co.co_argcount, co.co_nlocals, co.co_stacksize, co.co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_cellvars, ())
    else:
        return types.CodeType(co.co_argcount, co.co_kwonlyargcount, co.co_nlocals, co.co_stacksize, co.co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_cellvars, ())
_cell_set_template_code = _make_cell_set_template_code()

def cell_set(cell, value):
    if False:
        for i in range(10):
            print('nop')
    'Set the value of a closure cell.\n    '
    return types.FunctionType(_cell_set_template_code, {}, '_cell_set_inner', (), (cell,))(value)
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG

def islambda(func):
    if False:
        return 10
    return getattr(func, '__name__') == '<lambda>'
_BUILTIN_TYPE_NAMES = {}
for (k, v) in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k

def _builtin_type(name):
    if False:
        return 10
    return getattr(types, name)
if sys.version_info < (3, 4):

    def _walk_global_ops(code):
        if False:
            print('Hello World!')
        '\n        Yield (opcode, argument number) tuples for all\n        global-referencing instructions in *code*.\n        '
        code = getattr(code, 'co_code', b'')
        if not PY3:
            code = map(ord, code)
        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            op = code[i]
            i += 1
            if op >= HAVE_ARGUMENT:
                oparg = code[i] + code[i + 1] * 256 + extended_arg
                extended_arg = 0
                i += 2
                if op == EXTENDED_ARG:
                    extended_arg = oparg * 65536
                if op in GLOBAL_OPS:
                    yield (op, oparg)
else:

    def _walk_global_ops(code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Yield (opcode, argument number) tuples for all\n        global-referencing instructions in *code*.\n        '
        for instr in dis.get_instructions(code):
            op = instr.opcode
            if op in GLOBAL_OPS:
                yield (op, instr.arg)

class CloudPickler(Pickler):
    dispatch = Pickler.dispatch.copy()

    def __init__(self, file, protocol=None):
        if False:
            for i in range(10):
                print('nop')
        Pickler.__init__(self, file, protocol)
        self.modules = set()
        self.globals_ref = {}

    def dump(self, obj):
        if False:
            while True:
                i = 10
        self.inject_addons()
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = 'Could not pickle object as excessively deep recursion required.'
                raise pickle.PicklingError(msg)
        except pickle.PickleError:
            raise
        except Exception as e:
            emsg = e
            if hasattr(e, 'message'):
                emsg = e.message
            else:
                emsg = str(e)
            if "'i' format requires" in emsg:
                msg = 'Object too large to serialize: %s' % emsg
            else:
                msg = 'Could not serialize object: %s: %s' % (e.__class__.__name__, emsg)
            print_exec(sys.stderr)
            raise pickle.PicklingError(msg)

    def save_memoryview(self, obj):
        if False:
            print('Hello World!')
        'Fallback to save_string'
        Pickler.save_string(self, str(obj))

    def save_buffer(self, obj):
        if False:
            return 10
        'Fallback to save_string'
        Pickler.save_string(self, str(obj))
    if PY3:
        dispatch[memoryview] = save_memoryview
    else:
        dispatch[buffer] = save_buffer

    def save_unsupported(self, obj):
        if False:
            for i in range(10):
                print('nop')
        raise pickle.PicklingError('Cannot pickle objects of type %s' % type(obj))
    dispatch[types.GeneratorType] = save_unsupported
    for v in itertools.__dict__.values():
        if type(v) is type:
            dispatch[v] = save_unsupported

    def save_module(self, obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save a module as an import\n        '
        mod_name = obj.__name__
        if hasattr(obj, '__file__'):
            is_dynamic = False
        else:
            try:
                _find_module(mod_name)
                is_dynamic = False
            except ImportError:
                is_dynamic = True
        self.modules.add(obj)
        if is_dynamic:
            self.save_reduce(dynamic_subimport, (obj.__name__, vars(obj)), obj=obj)
        else:
            self.save_reduce(subimport, (obj.__name__,), obj=obj)
    dispatch[types.ModuleType] = save_module

    def save_codeobject(self, obj):
        if False:
            print('Hello World!')
        '\n        Save a code object\n        '
        if PY3:
            args = (obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
        else:
            args = (obj.co_argcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
        self.save_reduce(types.CodeType, args, obj=obj)
    dispatch[types.CodeType] = save_codeobject

    def save_function(self, obj, name=None):
        if False:
            return 10
        ' Registered with the dispatch to handle all function types.\n\n        Determines what kind of function obj is (e.g. lambda, defined at\n        interactive prompt, etc) and handles the pickling appropriately.\n        '
        write = self.write
        if name is None:
            name = obj.__name__
        try:
            modname = pickle.whichmodule(obj, name)
        except Exception:
            modname = None
        try:
            themodule = sys.modules[modname]
        except KeyError:
            modname = '__main__'
        if modname == '__main__':
            themodule = None
        if themodule:
            self.modules.add(themodule)
            if getattr(themodule, name, None) is obj:
                return self.save_global(obj, name)
        if not hasattr(obj, '__code__'):
            if PY3:
                if sys.version_info < (3, 4):
                    raise pickle.PicklingError("Can't pickle %r" % obj)
                else:
                    rv = obj.__reduce_ex__(self.proto)
            elif hasattr(obj, '__self__'):
                rv = (getattr, (obj.__self__, name))
            else:
                raise pickle.PicklingError("Can't pickle %r" % obj)
            return Pickler.save_reduce(self, *rv, obj=obj)
        if islambda(obj) or getattr(obj.__code__, 'co_filename', None) == '<stdin>' or themodule is None:
            self.save_function_tuple(obj)
            return
        else:
            klass = getattr(themodule, name, None)
            if klass is None or klass is not obj:
                self.save_function_tuple(obj)
                return
        if obj.__dict__:
            self.save(_restore_attr)
            write(pickle.MARK + pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
            self.save(obj.__dict__)
            write(pickle.TUPLE + pickle.REDUCE)
        else:
            write(pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
    dispatch[types.FunctionType] = save_function

    def _save_subimports(self, code, top_level_dependencies):
        if False:
            return 10
        '\n        Ensure de-pickler imports any package child-modules that\n        are needed by the function\n        '
        for x in top_level_dependencies:
            if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
                prefix = x.__name__ + '.'
                for (name, module) in sys.modules.items():
                    if name is not None and name.startswith(prefix):
                        tokens = set(name[len(prefix):].split('.'))
                        if not tokens - set(code.co_names):
                            self.save(module)
                            self.write(pickle.POP)

    def save_dynamic_class(self, obj):
        if False:
            for i in range(10):
                print('nop')
        "\n        Save a class that can't be stored as module global.\n\n        This method is used to serialize classes that are defined inside\n        functions, or that otherwise can't be serialized as attribute lookups\n        from global modules.\n        "
        clsdict = dict(obj.__dict__)
        if not isinstance(clsdict.get('__dict__', None), property):
            clsdict.pop('__dict__', None)
            clsdict.pop('__weakref__', None)
        new_override = clsdict.get('__new__', None)
        if new_override:
            clsdict['__new__'] = obj.__new__
        if getattr(obj, '_is_namedtuple_', False):
            self.save_reduce(_load_namedtuple, (obj.__name__, obj._fields))
            return
        save = self.save
        write = self.write
        save(_rehydrate_skeleton_class)
        write(pickle.MARK)
        doc_dict = {'__doc__': clsdict.pop('__doc__', None)}
        save(type(obj))
        save((obj.__name__, obj.__bases__, doc_dict))
        write(pickle.REDUCE)
        self.memoize(obj)
        save(clsdict)
        write(pickle.TUPLE)
        write(pickle.REDUCE)

    def save_function_tuple(self, func):
        if False:
            i = 10
            return i + 15
        "  Pickles an actual func object.\n\n        A func comprises: code, globals, defaults, closure, and dict.  We\n        extract and save these, injecting reducing functions at certain points\n        to recreate the func object.  Keep in mind that some of these pieces\n        can contain a ref to the func itself.  Thus, a naive save on these\n        pieces could trigger an infinite loop of save's.  To get around that,\n        we first create a skeleton func object using just the code (this is\n        safe, since this won't contain a ref to the func), and memoize it as\n        soon as it's created.  The other stuff can then be filled in later.\n        "
        if is_tornado_coroutine(func):
            self.save_reduce(_rebuild_tornado_coroutine, (func.__wrapped__,), obj=func)
            return
        save = self.save
        write = self.write
        (code, f_globals, defaults, closure_values, dct, base_globals) = self.extract_func_data(func)
        save(_fill_function)
        write(pickle.MARK)
        self._save_subimports(code, itertools.chain(f_globals.values(), closure_values or ()))
        save(_make_skel_func)
        save((code, len(closure_values) if closure_values is not None else -1, base_globals))
        write(pickle.REDUCE)
        self.memoize(func)
        save(f_globals)
        save(defaults)
        save(dct)
        save(func.__module__)
        save(closure_values)
        write(pickle.TUPLE)
        write(pickle.REDUCE)
    _extract_code_globals_cache = weakref.WeakKeyDictionary() if sys.version_info >= (2, 7) and (not hasattr(sys, 'pypy_version_info')) else {}

    @classmethod
    def extract_code_globals(cls, co):
        if False:
            print('Hello World!')
        '\n        Find all globals names read or written to by codeblock co\n        '
        out_names = cls._extract_code_globals_cache.get(co)
        if out_names is None:
            try:
                names = co.co_names
            except AttributeError:
                out_names = set()
            else:
                out_names = set((names[oparg] for (op, oparg) in _walk_global_ops(co)))
                if co.co_consts:
                    for const in co.co_consts:
                        if type(const) is types.CodeType:
                            out_names |= cls.extract_code_globals(const)
            cls._extract_code_globals_cache[co] = out_names
        return out_names

    def extract_func_data(self, func):
        if False:
            i = 10
            return i + 15
        '\n        Turn the function into a tuple of data necessary to recreate it:\n            code, globals, defaults, closure_values, dict\n        '
        code = func.__code__
        func_global_refs = self.extract_code_globals(code)
        f_globals = {}
        for var in func_global_refs:
            if var in func.__globals__:
                f_globals[var] = func.__globals__[var]
        defaults = func.__defaults__
        closure = list(map(_get_cell_contents, func.__closure__)) if func.__closure__ is not None else None
        dct = func.__dict__
        base_globals = self.globals_ref.get(id(func.__globals__), {})
        self.globals_ref[id(func.__globals__)] = base_globals
        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_builtin_function(self, obj):
        if False:
            print('Hello World!')
        if obj.__module__ == '__builtin__':
            return self.save_global(obj)
        return self.save_function(obj)
    dispatch[types.BuiltinFunctionType] = save_builtin_function

    def save_global(self, obj, name=None, pack=struct.pack):
        if False:
            while True:
                i = 10
        '\n        Save a "global".\n\n        The name of this method is somewhat misleading: all types get\n        dispatched here.\n        '
        if obj.__module__ == '__builtin__' or obj.__module__ == 'builtins':
            if obj in _BUILTIN_TYPE_NAMES:
                return self.save_reduce(_builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)
        if name is None:
            name = obj.__name__
        modname = getattr(obj, '__module__', None)
        if modname is None:
            try:
                modname = pickle.whichmodule(obj, name)
            except Exception:
                modname = '__main__'
        if modname == '__main__':
            themodule = None
        else:
            __import__(modname)
            themodule = sys.modules[modname]
            self.modules.add(themodule)
        if hasattr(themodule, name) and getattr(themodule, name) is obj:
            return Pickler.save_global(self, obj, name)
        typ = type(obj)
        if typ is not obj and isinstance(obj, (type, _class_type)):
            self.save_dynamic_class(obj)
        else:
            raise pickle.PicklingError("Can't pickle %r" % obj)
    dispatch[type] = save_global
    dispatch[_class_type] = save_global

    def save_instancemethod(self, obj):
        if False:
            return 10
        if obj.__self__ is None:
            self.save_reduce(getattr, (obj.__self__.__class__, obj.__name__))
        elif PY3:
            self.save_reduce(types.MethodType, (obj.__func__, obj.__self__), obj=obj)
        else:
            self.save_reduce(types.MethodType, (obj.__func__, obj.__self__, obj.__self__.__class__), obj=obj)
    dispatch[types.MethodType] = save_instancemethod

    def save_inst(self, obj):
        if False:
            return 10
        'Inner logic to save instance. Based off pickle.save_inst\n        Supports __transient__'
        cls = obj.__class__
        f = self.dispatch.get(cls)
        if f:
            f(self, obj)
            return
        memo = self.memo
        write = self.write
        save = self.save
        if hasattr(obj, '__getinitargs__'):
            args = obj.__getinitargs__()
            len(args)
            pickle._keep_alive(args, memo)
        else:
            args = ()
        write(pickle.MARK)
        if self.bin:
            save(cls)
            for arg in args:
                save(arg)
            write(pickle.OBJ)
        else:
            for arg in args:
                save(arg)
            write(pickle.INST + cls.__module__ + '\n' + cls.__name__ + '\n')
        self.memoize(obj)
        try:
            getstate = obj.__getstate__
        except AttributeError:
            stuff = obj.__dict__
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                stuff = stuff.copy()
                for k in list(stuff.keys()):
                    if k in transient:
                        del stuff[k]
        else:
            stuff = getstate()
            pickle._keep_alive(stuff, memo)
        save(stuff)
        write(pickle.BUILD)
    if not PY3:
        dispatch[types.InstanceType] = save_inst

    def save_property(self, obj):
        if False:
            return 10
        self.save_reduce(property, (obj.fget, obj.fset, obj.fdel, obj.__doc__), obj=obj)
    dispatch[property] = save_property

    def save_classmethod(self, obj):
        if False:
            while True:
                i = 10
        try:
            orig_func = obj.__func__
        except AttributeError:
            orig_func = obj.__get__(None, object)
            if isinstance(obj, classmethod):
                orig_func = orig_func.__func__
        self.save_reduce(type(obj), (orig_func,), obj=obj)
    dispatch[classmethod] = save_classmethod
    dispatch[staticmethod] = save_classmethod

    def save_itemgetter(self, obj):
        if False:
            print('Hello World!')
        'itemgetter serializer (needed for namedtuple support)'

        class Dummy:

            def __getitem__(self, item):
                if False:
                    while True:
                        i = 10
                return item
        items = obj(Dummy())
        if not isinstance(items, tuple):
            items = (items,)
        return self.save_reduce(operator.itemgetter, items)
    if type(operator.itemgetter) is type:
        dispatch[operator.itemgetter] = save_itemgetter

    def save_attrgetter(self, obj):
        if False:
            print('Hello World!')
        'attrgetter serializer'

        class Dummy(object):

            def __init__(self, attrs, index=None):
                if False:
                    while True:
                        i = 10
                self.attrs = attrs
                self.index = index

            def __getattribute__(self, item):
                if False:
                    print('Hello World!')
                attrs = object.__getattribute__(self, 'attrs')
                index = object.__getattribute__(self, 'index')
                if index is None:
                    index = len(attrs)
                    attrs.append(item)
                else:
                    attrs[index] = '.'.join([attrs[index], item])
                return type(self)(attrs, index)
        attrs = []
        obj(Dummy(attrs))
        return self.save_reduce(operator.attrgetter, tuple(attrs))
    if type(operator.attrgetter) is type:
        dispatch[operator.attrgetter] = save_attrgetter

    def save_reduce(self, func, args, state=None, listitems=None, dictitems=None, obj=None):
        if False:
            return 10
        'Modified to support __transient__ on new objects\n        Change only affects protocol level 2 (which is always used by PiCloud'
        if not isinstance(args, tuple):
            raise pickle.PicklingError('args from reduce() should be a tuple')
        if not hasattr(func, '__call__'):
            raise pickle.PicklingError('func from reduce should be callable')
        save = self.save
        write = self.write
        if self.proto >= 2 and getattr(func, '__name__', '') == '__newobj__':
            cls = args[0]
            if not hasattr(cls, '__new__'):
                raise pickle.PicklingError('args[0] from __newobj__ args has no __new__')
            if obj is not None and cls is not obj.__class__:
                raise pickle.PicklingError('args[0] from __newobj__ args has the wrong class')
            args = args[1:]
            save(cls)
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                state = state.copy()
                for k in list(state.keys()):
                    if k in transient:
                        del state[k]
            save(args)
            write(pickle.NEWOBJ)
        else:
            save(func)
            save(args)
            write(pickle.REDUCE)
        if obj is not None:
            self.memoize(obj)
        if listitems is not None:
            self._batch_appends(listitems)
        if dictitems is not None:
            self._batch_setitems(dictitems)
        if state is not None:
            save(state)
            write(pickle.BUILD)

    def save_partial(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Partial objects do not serialize correctly in python2.x -- this fixes the bugs'
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))
    if sys.version_info < (2, 7):
        dispatch[partial] = save_partial

    def save_file(self, obj):
        if False:
            print('Hello World!')
        'Save a file'
        try:
            import StringIO as pystringIO
        except ImportError:
            import io as pystringIO
        if not hasattr(obj, 'name') or not hasattr(obj, 'mode'):
            raise pickle.PicklingError('Cannot pickle files that do not map to an actual file')
        if obj is sys.stdout:
            return self.save_reduce(getattr, (sys, 'stdout'), obj=obj)
        if obj is sys.stderr:
            return self.save_reduce(getattr, (sys, 'stderr'), obj=obj)
        if obj is sys.stdin:
            raise pickle.PicklingError('Cannot pickle standard input')
        if obj.closed:
            raise pickle.PicklingError('Cannot pickle closed files')
        if hasattr(obj, 'isatty') and obj.isatty():
            raise pickle.PicklingError('Cannot pickle files that map to tty objects')
        if 'r' not in obj.mode and '+' not in obj.mode:
            raise pickle.PicklingError('Cannot pickle files that are not opened for reading: %s' % obj.mode)
        name = obj.name
        retval = pystringIO.StringIO()
        try:
            curloc = obj.tell()
            obj.seek(0)
            contents = obj.read()
            obj.seek(curloc)
        except IOError:
            raise pickle.PicklingError('Cannot pickle file %s as it cannot be read' % name)
        retval.write(contents)
        retval.seek(curloc)
        retval.name = name
        self.save(retval)
        self.memoize(obj)

    def save_ellipsis(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self.save_reduce(_gen_ellipsis, ())

    def save_not_implemented(self, obj):
        if False:
            print('Hello World!')
        self.save_reduce(_gen_not_implemented, ())
    if PY3:
        dispatch[io.TextIOWrapper] = save_file
    else:
        dispatch[file] = save_file
    dispatch[type(Ellipsis)] = save_ellipsis
    dispatch[type(NotImplemented)] = save_not_implemented
    if hasattr(weakref, 'WeakSet'):

        def save_weakset(self, obj):
            if False:
                return 10
            self.save_reduce(weakref.WeakSet, (list(obj),))
        dispatch[weakref.WeakSet] = save_weakset

    def inject_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        numpy = sys.modules.get('numpy')
        if not numpy or not hasattr(numpy, 'ufunc'):
            return
        self.dispatch[numpy.ufunc] = self.__class__.save_ufunc

    def save_ufunc(self, obj):
        if False:
            return 10
        'Hack function for saving numpy ufunc objects'
        name = obj.__name__
        numpy_tst_mods = ['numpy', 'scipy.special']
        for tst_mod_name in numpy_tst_mods:
            tst_mod = sys.modules.get(tst_mod_name, None)
            if tst_mod and name in tst_mod.__dict__:
                return self.save_reduce(_getobject, (tst_mod_name, name))
        raise pickle.PicklingError('cannot save %s. Cannot resolve what module it is defined in' % str(obj))

    def inject_unity_proxy(self):
        if False:
            i = 10
            return i + 15
        from turicreate.toolkits._model import Model
        tc = __import__(__name__.split('.')[0])
        if not tc:
            return
        self.dispatch[tc.SArray] = self.__class__.save_unsupported
        self.dispatch[tc.SFrame] = self.__class__.save_unsupported
        self.dispatch[tc.SGraph] = self.__class__.save_unsupported
        self.dispatch[tc.Sketch] = self.__class__.save_unsupported
        self.dispatch[Model] = self.__class__.save_unsupported
        self.dispatch[tc._cython.cy_sarray.UnitySArrayProxy] = self.__class__.save_unsupported
        self.dispatch[tc._cython.cy_sframe.UnitySFrameProxy] = self.__class__.save_unsupported
        self.dispatch[tc._cython.cy_sketch.UnitySketchProxy] = self.__class__.save_unsupported
        self.dispatch[tc._cython.cy_graph.UnityGraphProxy] = self.__class__.save_unsupported
        self.dispatch[tc._cython.cy_model.UnityModel] = self.__class__.save_unsupported
    'Special functions for Add-on libraries'

    def inject_addons(self):
        if False:
            return 10
        'Plug in system. Register additional pickling functions if modules already loaded'
        self.inject_numpy()
        self.inject_unity_proxy()

    def save_logger(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self.save_reduce(logging.getLogger, (obj.name,), obj=obj)
    dispatch[logging.Logger] = save_logger

def is_tornado_coroutine(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return whether *func* is a Tornado coroutine function.\n    Running coroutines are not supported.\n    '
    if 'tornado.gen' not in sys.modules:
        return False
    gen = sys.modules['tornado.gen']
    if not hasattr(gen, 'is_coroutine_function'):
        return False
    return gen.is_coroutine_function(func)

def _rebuild_tornado_coroutine(func):
    if False:
        i = 10
        return i + 15
    from tornado import gen
    return gen.coroutine(func)

def dump(obj, file, protocol=2):
    if False:
        i = 10
        return i + 15
    CloudPickler(file, protocol).dump(obj)

def dumps(obj, protocol=2):
    if False:
        print('Hello World!')
    file = StringIO()
    cp = CloudPickler(file, protocol)
    cp.dump(obj)
    return file.getvalue()

def subimport(name):
    if False:
        for i in range(10):
            print('nop')
    __import__(name)
    return sys.modules[name]

def dynamic_subimport(name, vars):
    if False:
        i = 10
        return i + 15
    mod = imp.new_module(name)
    mod.__dict__.update(vars)
    sys.modules[name] = mod
    return mod

def _restore_attr(obj, attr):
    if False:
        return 10
    for (key, val) in attr.items():
        setattr(obj, key, val)
    return obj

def _get_module_builtins():
    if False:
        for i in range(10):
            print('nop')
    return pickle.__builtins__

def print_exec(stream):
    if False:
        i = 10
        return i + 15
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)

def _modules_to_main(modList):
    if False:
        for i in range(10):
            print('nop')
    'Force every module in modList to be placed into main'
    if not modList:
        return
    main = sys.modules['__main__']
    for modname in modList:
        if type(modname) is str:
            try:
                mod = __import__(modname)
            except Exception:
                sys.stderr.write('warning: could not import %s\n.  Your function may unexpectedly error due to this import failing;A version mismatch is likely.  Specific error was:\n' % modname)
                print_exec(sys.stderr)
            else:
                setattr(main, mod.__name__, mod)

def _genpartial(func, args, kwds):
    if False:
        return 10
    if not args:
        args = ()
    if not kwds:
        kwds = {}
    return partial(func, *args, **kwds)

def _gen_ellipsis():
    if False:
        while True:
            i = 10
    return Ellipsis

def _gen_not_implemented():
    if False:
        print('Hello World!')
    return NotImplemented

def _get_cell_contents(cell):
    if False:
        for i in range(10):
            print('nop')
    try:
        return cell.cell_contents
    except ValueError:
        return _empty_cell_value

def instance(cls):
    if False:
        while True:
            i = 10
    'Create a new instance of a class.\n\n    Parameters\n    ----------\n    cls : type\n        The class to create an instance of.\n\n    Returns\n    -------\n    instance : cls\n        A new instance of ``cls``.\n    '
    return cls()

@instance
class _empty_cell_value(object):
    """sentinel for empty closures
    """

    @classmethod
    def __reduce__(cls):
        if False:
            while True:
                i = 10
        return cls.__name__

def _fill_function(func, globals, defaults, dict, module, closure_values):
    if False:
        while True:
            i = 10
    ' Fills in the rest of function data into the skeleton function object\n        that were created via _make_skel_func().\n    '
    func.__globals__.update(globals)
    func.__defaults__ = defaults
    func.__dict__ = dict
    func.__module__ = module
    cells = func.__closure__
    if cells is not None:
        for (cell, value) in zip(cells, closure_values):
            if value is not _empty_cell_value:
                cell_set(cell, value)
    return func

def _make_empty_cell():
    if False:
        i = 10
        return i + 15
    if False:
        cell = None
        raise AssertionError('this route should not be executed')
    return (lambda : cell).__closure__[0]

def _make_skel_func(code, cell_count, base_globals=None):
    if False:
        i = 10
        return i + 15
    ' Creates a skeleton function object that contains just the provided\n        code and the correct number of cells in func_closure.  All other\n        func attributes (e.g. func_globals) are empty.\n    '
    if base_globals is None:
        base_globals = {}
    base_globals['__builtins__'] = __builtins__
    closure = tuple((_make_empty_cell() for _ in range(cell_count))) if cell_count >= 0 else None
    return types.FunctionType(code, base_globals, None, None, closure)

def _rehydrate_skeleton_class(skeleton_class, class_dict):
    if False:
        print('Hello World!')
    'Put attributes from `class_dict` back on `skeleton_class`.\n\n    See CloudPickler.save_dynamic_class for more info.\n    '
    for (attrname, attr) in class_dict.items():
        setattr(skeleton_class, attrname, attr)
    return skeleton_class

def _find_module(mod_name):
    if False:
        return 10
    '\n    Iterate over each part instead of calling imp.find_module directly.\n    This function is able to find submodules (e.g. sickit.tree)\n    '
    path = None
    for part in mod_name.split('.'):
        if path is not None:
            path = [path]
        (file, path, description) = imp.find_module(part, path)
        if file is not None:
            file.close()
    return (path, description)

def _load_namedtuple(name, fields):
    if False:
        while True:
            i = 10
    '\n    Loads a class generated by namedtuple\n    '
    from collections import namedtuple
    return namedtuple(name, fields)
'Constructors for 3rd party libraries\nNote: These can never be renamed due to client compatibility issues'

def _getobject(modname, attribute):
    if False:
        while True:
            i = 10
    mod = __import__(modname, fromlist=[attribute])
    return mod.__dict__[attribute]
' Use copy_reg to extend global pickle definitions '
if sys.version_info < (3, 4):
    method_descriptor = type(str.upper)

    def _reduce_method_descriptor(obj):
        if False:
            i = 10
            return i + 15
        return (getattr, (obj.__objclass__, obj.__name__))
    try:
        import copy_reg as copyreg
    except ImportError:
        import copyreg
    copyreg.pickle(method_descriptor, _reduce_method_descriptor)