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
Copyright (c) 2009 `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
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
import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
try:
    import typing_extensions as _typing_extensions
    from typing_extensions import Literal, Final
except ImportError:
    _typing_extensions = Literal = Final = None
if sys.version_info >= (3, 8):
    from types import CellType
else:

    def f():
        if False:
            for i in range(10):
                print('nop')
        a = 1

        def g():
            if False:
                print('Hello World!')
            return a
        return g
    CellType = type(f().__closure__[0])
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
_PICKLE_BY_VALUE_MODULES = set()
_DYNAMIC_CLASS_TRACKER_BY_CLASS = weakref.WeakKeyDictionary()
_DYNAMIC_CLASS_TRACKER_BY_ID = weakref.WeakValueDictionary()
_DYNAMIC_CLASS_TRACKER_LOCK = threading.Lock()
PYPY = platform.python_implementation() == 'PyPy'
builtin_code_type = None
if PYPY:
    builtin_code_type = type(float.__new__.__code__)
_extract_code_globals_cache = weakref.WeakKeyDictionary()

def _get_or_create_tracker_id(class_def):
    if False:
        i = 10
        return i + 15
    with _DYNAMIC_CLASS_TRACKER_LOCK:
        class_tracker_id = _DYNAMIC_CLASS_TRACKER_BY_CLASS.get(class_def)
        if class_tracker_id is None:
            class_tracker_id = uuid.uuid4().hex
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
            _DYNAMIC_CLASS_TRACKER_BY_ID[class_tracker_id] = class_def
    return class_tracker_id

def _lookup_class_or_track(class_tracker_id, class_def):
    if False:
        return 10
    if class_tracker_id is not None:
        with _DYNAMIC_CLASS_TRACKER_LOCK:
            class_def = _DYNAMIC_CLASS_TRACKER_BY_ID.setdefault(class_tracker_id, class_def)
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
    return class_def

def register_pickle_by_value(module):
    if False:
        for i in range(10):
            print('nop')
    'Register a module to make it functions and classes picklable by value.\n\n    By default, functions and classes that are attributes of an importable\n    module are to be pickled by reference, that is relying on re-importing\n    the attribute from the module at load time.\n\n    If `register_pickle_by_value(module)` is called, all its functions and\n    classes are subsequently to be pickled by value, meaning that they can\n    be loaded in Python processes where the module is not importable.\n\n    This is especially useful when developing a module in a distributed\n    execution environment: restarting the client Python process with the new\n    source code is enough: there is no need to re-install the new version\n    of the module on all the worker nodes nor to restart the workers.\n\n    Note: this feature is considered experimental. See the cloudpickle\n    README.md file for more details and limitations.\n    '
    if not isinstance(module, types.ModuleType):
        raise ValueError(f'Input should be a module object, got {str(module)} instead')
    if module.__name__ not in sys.modules:
        raise ValueError(f'{module} was not imported correctly, have you used an `import` statement to access it?')
    _PICKLE_BY_VALUE_MODULES.add(module.__name__)

def unregister_pickle_by_value(module):
    if False:
        return 10
    'Unregister that the input module should be pickled by value.'
    if not isinstance(module, types.ModuleType):
        raise ValueError(f'Input should be a module object, got {str(module)} instead')
    if module.__name__ not in _PICKLE_BY_VALUE_MODULES:
        raise ValueError(f'{module} is not registered for pickle by value')
    else:
        _PICKLE_BY_VALUE_MODULES.remove(module.__name__)

def list_registry_pickle_by_value():
    if False:
        return 10
    return _PICKLE_BY_VALUE_MODULES.copy()

def _is_registered_pickle_by_value(module):
    if False:
        while True:
            i = 10
    module_name = module.__name__
    if module_name in _PICKLE_BY_VALUE_MODULES:
        return True
    while True:
        parent_name = module_name.rsplit('.', 1)[0]
        if parent_name == module_name:
            break
        if parent_name in _PICKLE_BY_VALUE_MODULES:
            return True
        module_name = parent_name
    return False

def _whichmodule(obj, name):
    if False:
        while True:
            i = 10
    "Find the module an object belongs to.\n\n    This function differs from ``pickle.whichmodule`` in two ways:\n    - it does not mangle the cases where obj's module is __main__ and obj was\n      not found in any module.\n    - Errors arising during module introspection are ignored, as those errors\n      are considered unwanted side effects.\n    "
    if sys.version_info[:2] < (3, 7) and isinstance(obj, typing.TypeVar):
        if name is not None and getattr(typing, name, None) is obj:
            return 'typing'
        else:
            module_name = None
    else:
        module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        return module_name
    for (module_name, module) in sys.modules.copy().items():
        if module_name == '__main__' or module is None or (not isinstance(module, types.ModuleType)):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None

def _should_pickle_by_reference(obj, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Test whether an function or a class should be pickled by reference\n\n    Pickling by reference means by that the object (typically a function or a\n    class) is an attribute of a module that is assumed to be importable in the\n    target Python environment. Loading will therefore rely on importing the\n    module and then calling `getattr` on it to access the function or class.\n\n    Pickling by reference is the only option to pickle functions and classes\n    in the standard library. In cloudpickle the alternative option is to\n    pickle by value (for instance for interactively or locally defined\n    functions and classes or for attributes of modules that have been\n    explicitly registered to be pickled by value.\n    '
    if isinstance(obj, types.FunctionType) or issubclass(type(obj), type):
        module_and_name = _lookup_module_and_qualname(obj, name=name)
        if module_and_name is None:
            return False
        (module, name) = module_and_name
        return not _is_registered_pickle_by_value(module)
    elif isinstance(obj, types.ModuleType):
        if _is_registered_pickle_by_value(obj):
            return False
        return obj.__name__ in sys.modules
    else:
        raise TypeError('cannot check importability of {} instances'.format(type(obj).__name__))

def _lookup_module_and_qualname(obj, name=None):
    if False:
        i = 10
        return i + 15
    if name is None:
        name = getattr(obj, '__qualname__', None)
    if name is None:
        name = getattr(obj, '__name__', None)
    module_name = _whichmodule(obj, name)
    if module_name is None:
        return None
    if module_name == '__main__':
        return None
    module = sys.modules.get(module_name, None)
    if module is None:
        return None
    try:
        (obj2, parent) = _getattribute(module, name)
    except AttributeError:
        return None
    if obj2 is not obj:
        return None
    return (module, name)

def _extract_code_globals(co):
    if False:
        i = 10
        return i + 15
    '\n    Find all globals names read or written to by codeblock co\n    '
    out_names = _extract_code_globals_cache.get(co)
    if out_names is None:
        out_names = {name: None for name in _walk_global_ops(co)}
        if co.co_consts:
            for const in co.co_consts:
                if isinstance(const, types.CodeType):
                    out_names.update(_extract_code_globals(const))
        _extract_code_globals_cache[co] = out_names
    return out_names

def _find_imported_submodules(code, top_level_dependencies):
    if False:
        while True:
            i = 10
    "\n    Find currently imported submodules used by a function.\n\n    Submodules used by a function need to be detected and referenced for the\n    function to work correctly at depickling time. Because submodules can be\n    referenced as attribute of their parent package (``package.submodule``), we\n    need a special introspection technique that does not rely on GLOBAL-related\n    opcodes to find references of them in a code object.\n\n    Example:\n    ```\n    import concurrent.futures\n    import cloudpickle\n    def func():\n        x = concurrent.futures.ThreadPoolExecutor\n    if __name__ == '__main__':\n        cloudpickle.dumps(func)\n    ```\n    The globals extracted by cloudpickle in the function's state include the\n    concurrent package, but not its submodule (here, concurrent.futures), which\n    is the module used by func. Find_imported_submodules will detect the usage\n    of concurrent.futures. Saving this module alongside with func will ensure\n    that calling func once depickled does not fail due to concurrent.futures\n    not being imported\n    "
    subimports = []
    for x in top_level_dependencies:
        if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
            prefix = x.__name__ + '.'
            for name in list(sys.modules):
                if name is not None and name.startswith(prefix):
                    tokens = set(name[len(prefix):].split('.'))
                    if not tokens - set(code.co_names):
                        subimports.append(sys.modules[name])
    return subimports

def cell_set(cell, value):
    if False:
        print('Hello World!')
    "Set the value of a closure cell.\n\n    The point of this function is to set the cell_contents attribute of a cell\n    after its creation. This operation is necessary in case the cell contains a\n    reference to the function the cell belongs to, as when calling the\n    function's constructor\n    ``f = types.FunctionType(code, globals, name, argdefs, closure)``,\n    closure will not be able to contain the yet-to-be-created f.\n\n    In Python3.7, cell_contents is writeable, so setting the contents of a cell\n    can be done simply using\n    >>> cell.cell_contents = value\n\n    In earlier Python3 versions, the cell_contents attribute of a cell is read\n    only, but this limitation can be worked around by leveraging the Python 3\n    ``nonlocal`` keyword.\n\n    In Python2 however, this attribute is read only, and there is no\n    ``nonlocal`` keyword. For this reason, we need to come up with more\n    complicated hacks to set this attribute.\n\n    The chosen approach is to create a function with a STORE_DEREF opcode,\n    which sets the content of a closure variable. Typically:\n\n    >>> def inner(value):\n    ...     lambda: cell  # the lambda makes cell a closure\n    ...     cell = value  # cell is a closure, so this triggers a STORE_DEREF\n\n    (Note that in Python2, A STORE_DEREF can never be triggered from an inner\n    function. The function g for example here\n    >>> def f(var):\n    ...     def g():\n    ...         var += 1\n    ...     return g\n\n    will not modify the closure variable ``var```inplace, but instead try to\n    load a local variable var and increment it. As g does not assign the local\n    variable ``var`` any initial value, calling f(1)() will fail at runtime.)\n\n    Our objective is to set the value of a given cell ``cell``. So we need to\n    somewhat reference our ``cell`` object into the ``inner`` function so that\n    this object (and not the smoke cell of the lambda function) gets affected\n    by the STORE_DEREF operation.\n\n    In inner, ``cell`` is referenced as a cell variable (an enclosing variable\n    that is referenced by the inner function). If we create a new function\n    cell_set with the exact same code as ``inner``, but with ``cell`` marked as\n    a free variable instead, the STORE_DEREF will be applied on its closure -\n    ``cell``, which we can specify explicitly during construction! The new\n    cell_set variable thus actually sets the contents of a specified cell!\n\n    Note: we do not make use of the ``nonlocal`` keyword to set the contents of\n    a cell in early python3 versions to limit possible syntax errors in case\n    test and checker libraries decide to parse the whole file.\n    "
    if sys.version_info[:2] >= (3, 7):
        cell.cell_contents = value
    else:
        _cell_set = types.FunctionType(_cell_set_template_code, {}, '_cell_set', (), (cell,))
        _cell_set(value)

def _make_cell_set_template_code():
    if False:
        i = 10
        return i + 15

    def _cell_set_factory(value):
        if False:
            while True:
                i = 10
        lambda : cell
        cell = value
    co = _cell_set_factory.__code__
    _cell_set_template_code = types.CodeType(co.co_argcount, co.co_kwonlyargcount, co.co_nlocals, co.co_stacksize, co.co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_cellvars, ())
    return _cell_set_template_code
if sys.version_info[:2] < (3, 7):
    _cell_set_template_code = _make_cell_set_template_code()
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG
_BUILTIN_TYPE_NAMES = {}
for (k, v) in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k

def _builtin_type(name):
    if False:
        return 10
    if name == 'ClassType':
        return type
    return getattr(types, name)

def _walk_global_ops(code):
    if False:
        for i in range(10):
            print('nop')
    '\n    Yield referenced name for all global-referencing instructions in *code*.\n    '
    for instr in dis.get_instructions(code):
        op = instr.opcode
        if op in GLOBAL_OPS:
            yield instr.argval

def _extract_class_dict(cls):
    if False:
        return 10
    'Retrieve a copy of the dict of a class without the inherited methods'
    clsdict = dict(cls.__dict__)
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)
    to_remove = []
    for (name, value) in clsdict.items():
        try:
            base_value = inherited_dict[name]
            if value is base_value:
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        clsdict.pop(name)
    return clsdict
if sys.version_info[:2] < (3, 7):

    def _is_parametrized_type_hint(obj):
        if False:
            while True:
                i = 10
        type_module = getattr(type(obj), '__module__', None)
        from_typing_extensions = type_module == 'typing_extensions'
        from_typing = type_module == 'typing'
        is_typing = getattr(obj, '__origin__', None) is not None
        is_literal = getattr(obj, '__values__', None) is not None and from_typing_extensions
        is_final = getattr(obj, '__type__', None) is not None and from_typing_extensions
        is_classvar = getattr(obj, '__type__', None) is not None and from_typing
        is_union = getattr(obj, '__union_params__', None) is not None
        is_tuple = getattr(obj, '__tuple_params__', None) is not None
        is_callable = getattr(obj, '__result__', None) is not None and getattr(obj, '__args__', None) is not None
        return any((is_typing, is_literal, is_final, is_classvar, is_union, is_tuple, is_callable))

    def _create_parametrized_type_hint(origin, args):
        if False:
            return 10
        return origin[args]
else:
    _is_parametrized_type_hint = None
    _create_parametrized_type_hint = None

def parametrized_type_hint_getinitargs(obj):
    if False:
        for i in range(10):
            print('nop')
    if type(obj) is type(Literal):
        initargs = (Literal, obj.__values__)
    elif type(obj) is type(Final):
        initargs = (Final, obj.__type__)
    elif type(obj) is type(ClassVar):
        initargs = (ClassVar, obj.__type__)
    elif type(obj) is type(Generic):
        initargs = (obj.__origin__, obj.__args__)
    elif type(obj) is type(Union):
        initargs = (Union, obj.__args__)
    elif type(obj) is type(Tuple):
        initargs = (Tuple, obj.__args__)
    elif type(obj) is type(Callable):
        (*args, result) = obj.__args__
        if len(args) == 1 and args[0] is Ellipsis:
            args = Ellipsis
        else:
            args = list(args)
        initargs = (Callable, (args, result))
    else:
        raise pickle.PicklingError(f'Cloudpickle Error: Unknown type {type(obj)}')
    return initargs

def is_tornado_coroutine(func):
    if False:
        return 10
    '\n    Return whether *func* is a Tornado coroutine function.\n    Running coroutines are not supported.\n    '
    if 'tornado.gen' not in sys.modules:
        return False
    gen = sys.modules['tornado.gen']
    if not hasattr(gen, 'is_coroutine_function'):
        return False
    return gen.is_coroutine_function(func)

def _rebuild_tornado_coroutine(func):
    if False:
        while True:
            i = 10
    from tornado import gen
    return gen.coroutine(func)
load = pickle.load
loads = pickle.loads

def subimport(name):
    if False:
        i = 10
        return i + 15
    __import__(name)
    return sys.modules[name]

def dynamic_subimport(name, vars):
    if False:
        for i in range(10):
            print('nop')
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    mod.__dict__['__builtins__'] = builtins.__dict__
    return mod

def _gen_ellipsis():
    if False:
        i = 10
        return i + 15
    return Ellipsis

def _gen_not_implemented():
    if False:
        for i in range(10):
            print('nop')
    return NotImplemented

def _get_cell_contents(cell):
    if False:
        i = 10
        return i + 15
    try:
        return cell.cell_contents
    except ValueError:
        return _empty_cell_value

def instance(cls):
    if False:
        return 10
    'Create a new instance of a class.\n\n    Parameters\n    ----------\n    cls : type\n        The class to create an instance of.\n\n    Returns\n    -------\n    instance : cls\n        A new instance of ``cls``.\n    '
    return cls()

@instance
class _empty_cell_value:
    """sentinel for empty closures"""

    @classmethod
    def __reduce__(cls):
        if False:
            return 10
        return cls.__name__

def _fill_function(*args):
    if False:
        i = 10
        return i + 15
    'Fills in the rest of function data into the skeleton function object\n\n    The skeleton itself is create by _make_skel_func().\n    '
    if len(args) == 2:
        func = args[0]
        state = args[1]
    elif len(args) == 5:
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'closure_values']
        state = dict(zip(keys, args[1:]))
    elif len(args) == 6:
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'module', 'closure_values']
        state = dict(zip(keys, args[1:]))
    else:
        raise ValueError(f'Unexpected _fill_value arguments: {args!r}')
    func.__globals__.update(state['globals'])
    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    if 'annotations' in state:
        func.__annotations__ = state['annotations']
    if 'doc' in state:
        func.__doc__ = state['doc']
    if 'name' in state:
        func.__name__ = state['name']
    if 'module' in state:
        func.__module__ = state['module']
    if 'qualname' in state:
        func.__qualname__ = state['qualname']
    if 'kwdefaults' in state:
        func.__kwdefaults__ = state['kwdefaults']
    if '_cloudpickle_submodules' in state:
        state.pop('_cloudpickle_submodules')
    cells = func.__closure__
    if cells is not None:
        for (cell, value) in zip(cells, state['closure_values']):
            if value is not _empty_cell_value:
                cell_set(cell, value)
    return func

def _make_function(code, globals, name, argdefs, closure):
    if False:
        print('Hello World!')
    globals['__builtins__'] = __builtins__
    return types.FunctionType(code, globals, name, argdefs, closure)

def _make_empty_cell():
    if False:
        print('Hello World!')
    if False:
        cell = None
        raise AssertionError('this route should not be executed')
    return (lambda : cell).__closure__[0]

def _make_cell(value=_empty_cell_value):
    if False:
        i = 10
        return i + 15
    cell = _make_empty_cell()
    if value is not _empty_cell_value:
        cell_set(cell, value)
    return cell

def _make_skel_func(code, cell_count, base_globals=None):
    if False:
        for i in range(10):
            print('nop')
    'Creates a skeleton function object that contains just the provided\n    code and the correct number of cells in func_closure.  All other\n    func attributes (e.g. func_globals) are empty.\n    '
    warnings.warn('A pickle file created using an old (<=1.4.1) version of cloudpickle is currently being loaded. This is not supported by cloudpickle and will break in cloudpickle 1.7', category=UserWarning)
    if base_globals is None or isinstance(base_globals, str):
        base_globals = {}
    base_globals['__builtins__'] = __builtins__
    closure = tuple((_make_empty_cell() for _ in range(cell_count))) if cell_count >= 0 else None
    return types.FunctionType(code, base_globals, None, None, closure)

def _make_skeleton_class(type_constructor, name, bases, type_kwargs, class_tracker_id, extra):
    if False:
        while True:
            i = 10
    'Build dynamic class with an empty __dict__ to be filled once memoized\n\n    If class_tracker_id is not None, try to lookup an existing class definition\n    matching that id. If none is found, track a newly reconstructed class\n    definition under that id so that other instances stemming from the same\n    class id will also reuse this class definition.\n\n    The "extra" variable is meant to be a dict (or None) that can be used for\n    forward compatibility shall the need arise.\n    '
    skeleton_class = types.new_class(name, bases, {'metaclass': type_constructor}, lambda ns: ns.update(type_kwargs))
    return _lookup_class_or_track(class_tracker_id, skeleton_class)

def _rehydrate_skeleton_class(skeleton_class, class_dict):
    if False:
        i = 10
        return i + 15
    'Put attributes from `class_dict` back on `skeleton_class`.\n\n    See CloudPickler.save_dynamic_class for more info.\n    '
    registry = None
    for (attrname, attr) in class_dict.items():
        if attrname == '_abc_impl':
            registry = attr
        else:
            setattr(skeleton_class, attrname, attr)
    if registry is not None:
        for subclass in registry:
            skeleton_class.register(subclass)
    return skeleton_class

def _make_skeleton_enum(bases, name, qualname, members, module, class_tracker_id, extra):
    if False:
        print('Hello World!')
    'Build dynamic enum with an empty __dict__ to be filled once memoized\n\n    The creation of the enum class is inspired by the code of\n    EnumMeta._create_.\n\n    If class_tracker_id is not None, try to lookup an existing enum definition\n    matching that id. If none is found, track a newly reconstructed enum\n    definition under that id so that other instances stemming from the same\n    class id will also reuse this enum definition.\n\n    The "extra" variable is meant to be a dict (or None) that can be used for\n    forward compatibility shall the need arise.\n    '
    enum_base = bases[-1]
    metacls = enum_base.__class__
    classdict = metacls.__prepare__(name, bases)
    for (member_name, member_value) in members.items():
        classdict[member_name] = member_value
    enum_class = metacls.__new__(metacls, name, bases, classdict)
    enum_class.__module__ = module
    enum_class.__qualname__ = qualname
    return _lookup_class_or_track(class_tracker_id, enum_class)

def _make_typevar(name, bound, constraints, covariant, contravariant, class_tracker_id):
    if False:
        while True:
            i = 10
    tv = typing.TypeVar(name, *constraints, bound=bound, covariant=covariant, contravariant=contravariant)
    if class_tracker_id is not None:
        return _lookup_class_or_track(class_tracker_id, tv)
    else:
        return tv

def _decompose_typevar(obj):
    if False:
        return 10
    return (obj.__name__, obj.__bound__, obj.__constraints__, obj.__covariant__, obj.__contravariant__, _get_or_create_tracker_id(obj))

def _typevar_reduce(obj):
    if False:
        for i in range(10):
            print('nop')
    module_and_name = _lookup_module_and_qualname(obj, name=obj.__name__)
    if module_and_name is None:
        return (_make_typevar, _decompose_typevar(obj))
    elif _is_registered_pickle_by_value(module_and_name[0]):
        return (_make_typevar, _decompose_typevar(obj))
    return (getattr, module_and_name)

def _get_bases(typ):
    if False:
        for i in range(10):
            print('nop')
    if '__orig_bases__' in getattr(typ, '__dict__', {}):
        bases_attr = '__orig_bases__'
    else:
        bases_attr = '__bases__'
    return getattr(typ, bases_attr)

def _make_dict_keys(obj, is_ordered=False):
    if False:
        while True:
            i = 10
    if is_ordered:
        return OrderedDict.fromkeys(obj).keys()
    else:
        return dict.fromkeys(obj).keys()

def _make_dict_values(obj, is_ordered=False):
    if False:
        print('Hello World!')
    if is_ordered:
        return OrderedDict(((i, _) for (i, _) in enumerate(obj))).values()
    else:
        return {i: _ for (i, _) in enumerate(obj)}.values()

def _make_dict_items(obj, is_ordered=False):
    if False:
        while True:
            i = 10
    if is_ordered:
        return OrderedDict(obj).items()
    else:
        return obj.items()