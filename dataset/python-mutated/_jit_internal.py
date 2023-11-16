"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""
import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import Any, Callable, Dict, Final, ForwardRef, Generic, get_args, get_origin, List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
IS_PY39_PLUS: Final[bool] = sys.version_info >= (3, 9)
IS_PY310_PLUS: Final[bool] = sys.version_info >= (3, 10)
BuiltinUnionType: Union[Type, Tuple[Type, ...]]
if sys.version_info >= (3, 10):
    BuiltinUnionType = types.UnionType
else:
    BuiltinUnionType = ()
LockType: Type
try:
    import _thread
    LockType = _thread.LockType
except ImportError:
    import _dummy_thread
    LockType = _dummy_thread.LockType
boolean_dispatched: 'weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]' = weakref.WeakKeyDictionary()
FAKE_FILENAME_PREFIX = '__torch_jit_dataclass'

class SourceLoader:

    def __init__(self):
        if False:
            return 10
        self.content = {}

    def cache(self, fn, source):
        if False:
            while True:
                i = 10
        self.content[fn] = source

    def get_source(self, fn):
        if False:
            for i in range(10):
                print('nop')
        return self.content.get(fn)
loader = SourceLoader()

def createResolutionCallbackFromEnv(lookup_base):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a resolution callback that will look up qualified names in an\n    environment, starting with `lookup_base` for the base of any qualified\n    names, then proceeding down the lookup chain with the resolved object.\n\n    You should not use this directly, it should only be used from the other\n    createResolutionCallbackFrom* functions.\n    '

    def lookupInModule(qualified_name, module):
        if False:
            return 10
        if '.' in qualified_name:
            parts = qualified_name.split('.')
            base = parts[0]
            remaining_pieces = '.'.join(parts[1:])
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

    def parseNestedExpr(expr, module) -> Tuple[Any, int]:
        if False:
            i = 10
            return i + 15
        i = 0
        while i < len(expr) and expr[i] not in (',', '[', ']'):
            i += 1
        if expr[:i] == '()':
            return ((), i)
        base = lookupInModule(expr[:i].strip(), module)
        assert base is not None, f'Unresolvable type {expr[:i]}'
        if i == len(expr) or expr[i] != '[':
            return (base, i)
        assert expr[i] == '['
        parts = []
        while expr[i] != ']':
            part_len = 0
            i += 1
            (part, part_len) = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        if len(parts) > 1:
            return (base[tuple(parts)], i + 1)
        else:
            return (base[parts[0]], i + 1)

    def parseExpr(expr, module):
        if False:
            return 10
        try:
            (value, len_parsed) = parseNestedExpr(expr, module)
            assert len_parsed == len(expr), 'whole expression was not parsed, falling back to c++ parser'
            return value
        except Exception:
            '\n            The python resolver fails in several cases in known unit tests, and is intended\n            to fall back gracefully to the c++ resolver in general.  For example, python 2 style\n            annotations which are frequent in our unit tests often fail with types e.g. int not\n            resolvable from the calling frame.\n            '
            return None
    return lambda expr: parseExpr(expr, lookup_base)

def createResolutionCallbackFromFrame(frames_up: int=0):
    if False:
        print('Hello World!')
    '\n    Creates a function which, given a string variable name,\n    returns the value of the variable in the scope of the caller of\n    the function which called createResolutionCallbackFromFrame (by default).\n\n    This is used to enable access in-scope Python variables inside\n    TorchScript fragments.\n\n    frames_up is number of additional frames to go up on the stack.\n    The default value is 0, which correspond to the frame of the caller\n    of createResolutionCallbackFromFrame. Also for example, if frames_up is set\n    to 1, then the frame of the caller\'s caller of createResolutionCallbackFromFrame\n    will be taken.\n\n    For example, the following program prints 2::\n\n        def bar():\n            cb = createResolutionCallbackFromFrame(1)\n            print(cb("foo"))\n\n        def baz():\n            foo = 2\n            bar()\n\n        baz()\n    '
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        assert frame is not None
        frame = frame.f_back
        i += 1
    assert frame is not None
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    class env:

        def __getattr__(self, key):
            if False:
                print('Hello World!')
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)
    return createResolutionCallbackFromEnv(env())

def get_closure(fn):
    if False:
        print('Hello World!')
    '\n    Get a dictionary of closed over variables from a function\n    '
    captures = {}
    captures.update(fn.__globals__)
    for (index, captured_name) in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents
    return captures

def createResolutionCallbackFromClosure(fn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a resolutionCallback by introspecting the function instead of\n    looking up the stack for the enclosing scope\n    '
    closure = get_closure(fn)

    class closure_lookup:

        def __getattr__(self, key):
            if False:
                while True:
                    i = 10
            if key in closure:
                return closure[key]
            elif hasattr(typing, key):
                return getattr(typing, key)
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            return None
    return createResolutionCallbackFromEnv(closure_lookup())

def can_compile_class(cls) -> bool:
    if False:
        i = 10
        return i + 15
    if is_ignored_fn(cls):
        return False
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False
    names = cls.__dict__
    fns = [getattr(cls, name) for name in names if inspect.isroutine(getattr(cls, name, None))]
    has_code = [hasattr(fn, '__code__') for fn in fns]
    return all(has_code)

def get_callable_argument_names(fn) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.\n    Returns an empty list when other types of arguments are present.\n\n    This is used by `torch.jit.trace` to assign meaningful argument names to\n    traced functions and modules.\n\n    Args:\n        fn: A callable.\n    Returns:\n        Argument names: List[str]\n    '
    try:
        callable_signature = inspect.signature(fn)
    except Exception:
        return []
    argument_names = []
    for (name, param) in callable_signature.parameters.items():
        if not param.kind == param.POSITIONAL_OR_KEYWORD:
            continue
        argument_names.append(name)
    return argument_names

def get_annotation_str(annotation):
    if False:
        while True:
            i = 10
    '\n    Convert an AST node containing a type annotation to the string present in the source\n    that represents the same annotation.\n    '
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return '.'.join([get_annotation_str(annotation.value), annotation.attr])
    elif isinstance(annotation, ast.Subscript):
        subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice.value
        return f'{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]'
    elif isinstance(annotation, ast.Tuple):
        return ','.join([get_annotation_str(elt) for elt in annotation.elts])
    elif isinstance(annotation, (ast.Constant, ast.NameConstant)):
        return f'{annotation.value}'
    return None

def get_type_hint_captures(fn):
    if False:
        i = 10
        return i + 15
    "\n    Get a dictionary containing type resolution mappings necessary to resolve types\n    for the literal annotations on 'fn'. These are not considered to be closed-over by fn\n    and must be obtained separately (e.g. using this function).\n\n    Args:\n        fn: A callable.\n    Returns:\n        A Dict[str, Any] containing a mapping from the literal annotations used on\n        fn to the Python objects they refer to.\n    "
    src = loader.get_source(fn)
    if src is None:
        src = inspect.getsource(fn)
    signature = inspect.signature(fn)
    name_to_type = {name: parameter.annotation for (name, parameter) in signature.parameters.items() if parameter.annotation is not inspect.Parameter.empty and (not isinstance(parameter.annotation, str))}
    a = ast.parse(dedent(src))
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f'Expected {fn} to be a function')
    f = a.body[0]
    annotation_to_type = {}
    for arg in f.args.args:
        arg_annotation_str = get_annotation_str(arg.annotation) if arg.annotation else None
        if arg_annotation_str is None:
            continue
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = return_annotation is not inspect.Parameter.empty and (not isinstance(return_annotation, str))
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation
    return annotation_to_type

def createResolutionCallbackForClassMethods(cls):
    if False:
        return 10
    '\n    This looks at all the methods defined in a class and pulls their closed-over\n    variables into a dictionary and uses that to resolve variables.\n    '
    fns = [getattr(cls, name) for name in cls.__dict__ if inspect.isroutine(getattr(cls, name))]
    fns = [fn for fn in fns if not inspect.isbuiltin(fn) and hasattr(fn, '__globals__')]
    captures = {}
    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    def lookup_in_class(key):
        if False:
            i = 10
            return i + 15
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)
    return lookup_in_class

def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Dispatches to either of 2 script functions based on a boolean argument.\n    In TorchScript, the boolean argument must be constant so that the correct\n    function to use can be determined at compile time.\n    '

    def fn(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        dispatch_flag = default
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]
        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)
    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        doc = None
    else:
        raise RuntimeError('only one function can have a docstring')
    fn.__doc__ = doc
    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name
    boolean_dispatched[fn] = {'if_true': if_true, 'if_false': if_false, 'index': arg_index, 'default': default, 'arg_name': arg_name}
    return fn

class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED = 'unused (ignored and replaced with raising of an exception)'
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = 'export (compile this function even if nothing calls it)'
    DEFAULT = 'default (compile if called from a exported function / forward)'
    COPY_TO_SCRIPT_WRAPPER = 'if this method is not scripted, copy the python method onto the scripted model'
    _DROP = '_drop (function is fully ignored, declaration can be unscriptable)'

def export(fn):
    if False:
        return 10
    "\n    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a\n    :class:`ScriptModule` and should be compiled.\n\n    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.\n    Functions and methods called from ``forward`` are compiled as they are seen\n    by the compiler, so they do not need this decorator either.\n\n    Example (using ``@torch.jit.export`` on a method):\n\n    .. testcode::\n\n        import torch\n        import torch.nn as nn\n\n        class MyModule(nn.Module):\n            def implicitly_compiled_method(self, x):\n                return x + 99\n\n            # `forward` is implicitly decorated with `@torch.jit.export`,\n            # so adding it here would have no effect\n            def forward(self, x):\n                return x + 10\n\n            @torch.jit.export\n            def another_forward(self, x):\n                # When the compiler sees this call, it will compile\n                # `implicitly_compiled_method`\n                return self.implicitly_compiled_method(x)\n\n            def unused_method(self, x):\n                return x - 20\n\n        # `m` will contain compiled methods:\n        #     `forward`\n        #     `another_forward`\n        #     `implicitly_compiled_method`\n        # `unused_method` will not be compiled since it was not called from\n        # any compiled methods and wasn't decorated with `@torch.jit.export`\n        m = torch.jit.script(MyModule())\n    "
    fn._torchscript_modifier = FunctionModifiers.EXPORT
    return fn

def unused(fn):
    if False:
        while True:
            i = 10
    '\n    This decorator indicates to the compiler that a function or method should\n    be ignored and replaced with the raising of an exception. This allows you\n    to leave code in your model that is not yet TorchScript compatible and still\n    export your model.\n\n        Example (using ``@torch.jit.unused`` on a method)::\n\n            import torch\n            import torch.nn as nn\n\n            class MyModule(nn.Module):\n                def __init__(self, use_memory_efficient):\n                    super().__init__()\n                    self.use_memory_efficient = use_memory_efficient\n\n                @torch.jit.unused\n                def memory_efficient(self, x):\n                    import pdb\n                    pdb.set_trace()\n                    return x + 10\n\n                def forward(self, x):\n                    # Use not-yet-scriptable memory efficient mode\n                    if self.use_memory_efficient:\n                        return self.memory_efficient(x)\n                    else:\n                        return x + 10\n\n            m = torch.jit.script(MyModule(use_memory_efficient=False))\n            m.save("m.pt")\n\n            m = torch.jit.script(MyModule(use_memory_efficient=True))\n            # exception raised\n            m(torch.rand(100))\n    '
    if isinstance(fn, property):
        prop = fn
        setattr(prop.fget, '_torchscript_modifier', FunctionModifiers.UNUSED)
        if prop.fset:
            setattr(prop.fset, '_torchscript_modifier', FunctionModifiers.UNUSED)
        return prop
    fn._torchscript_modifier = FunctionModifiers.UNUSED
    return fn

class _IgnoreContextManager(contextlib.AbstractContextManager):

    def __init__(self, **kwargs):
        if False:
            return 10
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            return 10
        pass

def ignore(drop=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    This decorator indicates to the compiler that a function or method should\n    be ignored and left as a Python function. This allows you to leave code in\n    your model that is not yet TorchScript compatible. If called from TorchScript,\n    ignored functions will dispatch the call to the Python interpreter. Models with ignored\n    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.\n\n    Example (using ``@torch.jit.ignore`` on a method)::\n\n        import torch\n        import torch.nn as nn\n\n        class MyModule(nn.Module):\n            @torch.jit.ignore\n            def debugger(self, x):\n                import pdb\n                pdb.set_trace()\n\n            def forward(self, x):\n                x += 10\n                # The compiler would normally try to compile `debugger`,\n                # but since it is `@ignore`d, it will be left as a call\n                # to Python\n                self.debugger(x)\n                return x\n\n        m = torch.jit.script(MyModule())\n\n        # Error! The call `debugger` cannot be saved since it calls into Python\n        m.save("m.pt")\n\n    Example (using ``@torch.jit.ignore(drop=True)`` on a method):\n\n    .. testcode::\n\n        import torch\n        import torch.nn as nn\n\n        class MyModule(nn.Module):\n            @torch.jit.ignore(drop=True)\n            def training_method(self, x):\n                import pdb\n                pdb.set_trace()\n\n            def forward(self, x):\n                if self.training:\n                    self.training_method(x)\n                return x\n\n        m = torch.jit.script(MyModule())\n\n        # This is OK since `training_method` is not saved, the call is replaced\n        # with a `raise`.\n        m.save("m.pt")\n\n    .. testcleanup::\n\n        import os\n        os.remove(\'m.pt\')\n    '
    if callable(drop):
        fn = drop
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn
    if not isinstance(drop, bool):
        raise RuntimeError(f'Argument to @torch.jit.ignore must be a bool or a function but got {drop}')
    drop_on_export = kwargs.pop('drop_on_export', None)
    if drop_on_export:
        warnings.warn('ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function call on compilation. Use torch.jit.unused now. {}', category=FutureWarning)
        drop = drop_on_export
    elif drop:
        warnings.warn('ignore(True) has been deprecated. TorchScript will now drop the function call on compilation. Use torch.jit.unused now. {}', category=FutureWarning)

    def decorator(fn):
        if False:
            while True:
                i = 10
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn
    return decorator

def _drop(fn):
    if False:
        for i in range(10):
            print('nop')
    fn._torchscript_modifier = FunctionModifiers._DROP
    return fn

def _copy_to_script_wrapper(fn):
    if False:
        print('Hello World!')
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn

def module_has_exports(mod):
    if False:
        print('Hello World!')
    for name in dir(mod):
        if hasattr(mod, name):
            item = getattr(mod, name)
            if callable(item):
                if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                    return True
    return False

def should_drop(fn) -> bool:
    if False:
        while True:
            i = 10
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED or attr is FunctionModifiers._DROP

def is_ignored_fn(fn) -> bool:
    if False:
        for i in range(10):
            print('nop')
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers.UNUSED or mod is FunctionModifiers.IGNORE or mod is FunctionModifiers._DROP

def _is_drop_fn(fn) -> bool:
    if False:
        for i in range(10):
            print('nop')
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers._DROP

def is_static_fn(cls, fn) -> bool:
    if False:
        print('Hello World!')
    return isinstance(inspect.getattr_static(cls, fn, default=None), staticmethod)

def get_static_fn(cls, fn):
    if False:
        print('Hello World!')
    return inspect.getattr_static(cls, fn).__func__

def get_torchscript_modifier(fn):
    if False:
        print('Hello World!')
    if not callable(fn):
        return None
    if hasattr(fn, '__func__'):
        fn = fn.__func__
    return getattr(fn, '_torchscript_modifier', FunctionModifiers.DEFAULT)

def copy_torchscript_modifier(orig, new) -> None:
    if False:
        return 10
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr
_overloaded_fns: Dict[str, List[Callable]] = {}
_OVERLOAD_EXAMPLE = '\nExample usage of overload function:\n@torch.jit._overload\ndef my_function(x: type0) -> type0: # decl 1\n    pass\n\n@torch.jit._overload\ndef my_function(x: type1) -> type1: # decl 2\n    pass\n\ndef my_function(x):                 # implementation\n    if isinstance(x, type0):\n        return x\n    elif isinstance(x, type1):\n        return x\n'

def get_overload_no_implementation_error_message(kind, obj):
    if False:
        return 10
    (sourcelines, file_lineno, filename) = get_source_lines_and_file(obj)
    return f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make sure a definition is provided and defined after all overload declarations.\nFile "{filename}", line {file_lineno}:\n' + ''.join(sourcelines) + '\n' + _OVERLOAD_EXAMPLE

def _check_overload_body(func):
    if False:
        print('Hello World!')
    try:
        parsed_def = parse_def(func)
    except OSError as e:
        warnings.warn(f'Unable to retrieve source for @torch.jit._overload function: {func}.')
        return
    body = parsed_def.ast.body[0].body

    def is_pass(x):
        if False:
            return 10
        return isinstance(x, ast.Pass)

    def is_ellipsis(x):
        if False:
            print('Hello World!')
        return isinstance(x, ast.Expr) and isinstance(x.value, ast.Ellipsis)
    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = 'Only `pass` statement or `...` can be the body of overload declaration:\n'
        msg += '\n'.join(parsed_def.source.split('\n')[:3])
        msg += ' <- Expecting `pass` or `...` here!\n' + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)

def _overload(func):
    if False:
        print('Hello World!')
    _check_overload_body(func)
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func

def _get_fn_overloads(qual_name):
    if False:
        print('Hello World!')
    return _overloaded_fns.get(qual_name)

def _clear_fn_overloads(qual_name) -> None:
    if False:
        i = 10
        return i + 15
    del _overloaded_fns[qual_name]

def get_class_name_lineno(method) -> Tuple[str, int]:
    if False:
        return 10
    current_frame = inspect.currentframe()
    for i in range(2):
        assert current_frame is not None
        current_frame = current_frame.f_back
    assert current_frame is not None
    class_name = current_frame.f_code.co_name
    line_no = current_frame.f_code.co_firstlineno
    return (class_name, line_no)
_overloaded_methods: Dict[str, Dict[str, List[Callable]]] = {}
_overloaded_method_class_fileno = {}

def _overload_method(func):
    if False:
        while True:
            i = 10
    _check_overload_body(func)
    qual_name = _qualified_name(func)
    global _overloaded_methods
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        class_name_map = {}
        _overloaded_methods[qual_name] = class_name_map
    (class_name, line_no) = get_class_name_lineno(func)
    method_overloads = class_name_map.get(class_name, None)
    if method_overloads is None:
        method_overloads = []
        class_name_map[class_name] = method_overloads
        _overloaded_method_class_fileno[qual_name, class_name] = line_no
    else:
        existing_lineno = _overloaded_method_class_fileno[qual_name, class_name]
        if existing_lineno != line_no:
            raise RuntimeError('Cannot currently overload the same method name in two different classes with the same name in the same module')
    method_overloads.append(func)
    return func

def _get_overloaded_methods(method, mod_class):
    if False:
        while True:
            i = 10
    if not hasattr(method, '__name__'):
        return None
    qual_name = _qualified_name(method)
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        return None
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None
    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise Exception('Overloads are not useable when a module is redeclared within the same file: ' + str(method))
    return overloads

def is_tuple(ann) -> bool:
    if False:
        print('Hello World!')
    if ann is Tuple:
        raise_error_container_parameter_missing('Tuple')
    if not hasattr(ann, '__module__'):
        return False
    ann_origin = get_origin(ann)
    if IS_PY39_PLUS and ann.__module__ == 'builtins' and (ann_origin is tuple):
        return True
    return ann.__module__ == 'typing' and (ann_origin is Tuple or ann_origin is tuple)

def is_list(ann) -> bool:
    if False:
        return 10
    if ann is List:
        raise_error_container_parameter_missing('List')
    if not hasattr(ann, '__module__'):
        return False
    ann_origin = get_origin(ann)
    if IS_PY39_PLUS and ann.__module__ == 'builtins' and (ann_origin is list):
        return True
    return ann.__module__ == 'typing' and (ann_origin is List or ann_origin is list)

def is_dict(ann) -> bool:
    if False:
        i = 10
        return i + 15
    if ann is Dict:
        raise_error_container_parameter_missing('Dict')
    if not hasattr(ann, '__module__'):
        return False
    ann_origin = get_origin(ann)
    if IS_PY39_PLUS and ann.__module__ == 'builtins' and (ann_origin is dict):
        return True
    return ann.__module__ == 'typing' and (ann_origin is Dict or ann_origin is dict)

def is_union(ann):
    if False:
        print('Hello World!')
    if ann is Union:
        raise_error_container_parameter_missing('Union')
    return isinstance(ann, BuiltinUnionType) or (hasattr(ann, '__module__') and ann.__module__ == 'typing' and (get_origin(ann) is Union))

def is_optional(ann):
    if False:
        while True:
            i = 10
    if ann is Optional:
        raise_error_container_parameter_missing('Optional')

    def is_optional_as_optional(ann):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(ann, '__module__') and ann.__module__ == 'typing' and (get_origin(ann) is Optional)

    def is_union_as_optional(ann):
        if False:
            print('Hello World!')
        ann_args = get_args(ann)
        return len(ann_args) == 2 and (None in ann_args or type(None) in ann_args)
    return is_optional_as_optional(ann) or (is_union(ann) and is_union_as_optional(ann))

def is_future(ann) -> bool:
    if False:
        i = 10
        return i + 15
    if ann is Future:
        raise RuntimeError('Attempted to use Future without a contained type. Please add a contained type, e.g. Future[int]')
    return get_origin(ann) is Future

def is_await(ann) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if ann is _Await:
        return True
    return get_origin(ann) is _Await
if torch.distributed.rpc.is_available():
    from torch._C._distributed_rpc import PyRRef
    from torch.distributed.rpc import RRef

    def is_rref(ann) -> bool:
        if False:
            while True:
                i = 10
        if ann is RRef:
            raise RuntimeError('Attempted to use RRef without a contained type. Please add a contained type, e.g. RRef[int]')
        return get_origin(ann) is RRef

    def is_rref_instance(obj) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(obj, PyRRef)
else:

    def is_rref_instance(obj) -> bool:
        if False:
            i = 10
            return i + 15
        return False

def is_final(ann) -> bool:
    if False:
        i = 10
        return i + 15
    return ann.__module__ in {'typing', 'typing_extensions'} and (get_origin(ann) is Final or isinstance(ann, type(Final)))

class BroadcastingListCls:

    def __getitem__(self, types):
        if False:
            while True:
                i = 10
        return
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f'BroadcastingList{i}'] = BroadcastingList1

def is_scripting() -> bool:
    if False:
        print('Hello World!')
    '\n    Function that returns True when in compilation and False otherwise. This\n    is useful especially with the @unused decorator to leave code in your\n    model that is not yet TorchScript compatible.\n    .. testcode::\n\n        import torch\n\n        @torch.jit.unused\n        def unsupported_linear_op(x):\n            return x\n\n        def linear(x):\n           if torch.jit.is_scripting():\n              return torch.linear(x)\n           else:\n              return unsupported_linear_op(x)\n    '
    return False

def _qualified_name(obj, mangle_name=True) -> str:
    if False:
        i = 10
        return i + 15
    if hasattr(obj, '_jit_override_qualname'):
        return obj._jit_override_qualname
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name
    if getattr(obj, '__name__', None):
        name = obj.__name__
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError('Could not get name of python class object')
    if name == '<lambda>':
        name = '_lambda'
    module_name = obj.__module__
    if module_name == 'torch._classes':
        return obj.qualified_name
    if module_name is None:
        raise RuntimeError(f"Could not get qualified name for class '{name}': __module__ can't be None.")
    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace('<', '_')
        module_name = module_name.replace('>', '_')
    if mangle_name:
        if module_name == '__main__':
            module_name = '__torch__'
        else:
            module_name = '__torch__.' + module_name
    if '.' in name:
        raise RuntimeError(f"Could not get qualified name for class '{name}': '{name}' is not a valid identifier")
    return module_name + '.' + name

def _try_get_dispatched_fn(fn):
    if False:
        while True:
            i = 10
    if not callable(fn):
        return None
    return boolean_dispatched.get(fn)

def _get_named_tuple_properties(obj, loc: Optional[torch._C._jit_tree_views.SourceRange]=None, rcb=None):
    if False:
        return 10
    if loc is None:
        loc = fake_range()
    assert issubclass(obj, tuple) and hasattr(obj, '_fields')
    if hasattr(obj, '_field_defaults'):
        defaults = [obj._field_defaults[field] for field in obj._fields if field in obj._field_defaults]
    else:
        defaults = []
    if sys.version_info[:2] < (3, 10):
        obj_annotations = getattr(obj, '__annotations__', {})
    else:
        obj_annotations = inspect.get_annotations(obj)
        if len(obj_annotations) == 0 and hasattr(obj, '__base__'):
            obj_annotations = inspect.get_annotations(obj.__base__)
    annotations = []
    for field in obj._fields:
        if field in obj_annotations:
            field_type = obj_annotations[field]
            if isinstance(field_type, ForwardRef) and rcb is not None:
                rcb_type = rcb(field_type.__forward_arg__)
                if rcb_type is None:
                    raise ValueError(f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}. Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858. Issue occurred at {loc.highlight()}")
                field_type = rcb_type
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.getInferred())
    return (type(obj).__name__, obj._fields, annotations, defaults)

def _create_named_tuple(t, unqual_name: str, field_names: List[str], defaults: Tuple[Any, ...]):
    if False:
        i = 10
        return i + 15
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)
    return TupleType(*t)

@contextlib.contextmanager
def _disable_emit_hooks():
    if False:
        print('Hello World!')
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    try:
        yield
    finally:
        torch._C._jit_set_emit_hooks(hooks[0], hooks[1])

def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None:
    if False:
        i = 10
        return i + 15

    def __enter__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hooks = torch._C._jit_get_emit_hooks()
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args) -> None:
        if False:
            return 10
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])

def _is_exception(obj) -> bool:
    if False:
        print('Hello World!')
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, Exception)

def raise_error_container_parameter_missing(target_type) -> None:
    if False:
        print('Hello World!')
    if target_type == 'Dict':
        raise RuntimeError('Attempted to use Dict without contained types. Please add contained type, e.g. Dict[int, int]')
    raise RuntimeError(f'Attempted to use {target_type} without a contained type. Please add a contained type, e.g. {target_type}[int]')

def check_args_exist(target_type) -> None:
    if False:
        i = 10
        return i + 15
    if target_type is List or target_type is list:
        raise_error_container_parameter_missing('List')
    elif target_type is Tuple or target_type is tuple:
        raise_error_container_parameter_missing('Tuple')
    elif target_type is Dict or target_type is dict:
        raise_error_container_parameter_missing('Dict')
    elif target_type is None or target_type is Optional:
        raise_error_container_parameter_missing('Optional')

def check_empty_containers(obj) -> None:
    if False:
        for i in range(10):
            print('nop')
    if obj == [] or obj == {} or obj == ():
        warnings.warn('The inner type of a container is lost when calling torch.jit.isinstance in eager mode. For example, List[int] would become list and therefore falsely return True for List[float] or List[str].')

def container_checker(obj, target_type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        return False
    elif origin_type is list or origin_type is List:
        check_empty_containers(obj)
        if not isinstance(obj, list):
            return False
        arg_type = get_args(target_type)[0]
        arg_origin = get_origin(arg_type)
        for el in obj:
            if arg_origin:
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
        return True
    elif origin_type is Dict or origin_type is dict:
        check_empty_containers(obj)
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for (key, val) in obj.items():
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
        return True
    elif origin_type is Tuple or origin_type is tuple:
        check_empty_containers(obj)
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for (el, el_type) in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
        return True
    elif origin_type is Union or issubclass(origin_type, BuiltinUnionType):
        if obj is None:
            return True
        inner_types = get_args(target_type)
        for t in inner_types:
            t_origin = get_origin(t)
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                return True
    return False

def _isinstance(obj, target_type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(target_type, collections.abc.Container):
        if not isinstance(target_type, tuple):
            raise RuntimeError('The second argument to `torch.jit.isinstance` must be a type or a tuple of types')
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False
    origin_type = get_origin(target_type)
    if origin_type:
        return container_checker(obj, target_type)
    check_args_exist(target_type)
    return isinstance(obj, target_type)

class _TensorExtractor(pickle.Pickler):

    def __init__(self, *args, tensors: List[torch.Tensor], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        if False:
            print('Hello World!')
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ''
        if isinstance(obj, LockType):
            return ''
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ''
        if isinstance(obj, CAwait):
            return ''
        if isinstance(obj, torch.cuda.Event):
            return ''
        if isinstance(obj, threading.Thread):
            return ''
        return None

def _extract_tensors(obj):
    if False:
        i = 10
        return i + 15
    '\n    This function is exclusively called from C++.\n    See ``torch/csrc/jit/python/python_ivalue.h``.\n\n    It extracts the tensors contained in the given object, through pickling.\n    '
    tensors: List[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors
if sys.version_info > (3, 10):
    _drop(enum.Enum.__new__)
    _drop(enum.Enum.__format__)
    _drop(enum.Enum.__repr__)
    _drop(enum.Enum.__str__)