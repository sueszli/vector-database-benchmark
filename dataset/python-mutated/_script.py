"""TorchScript.

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import JitTypeTraceConfig, JitTypeTraceStore, monkeytype_trace
from torch.jit._recursive import _compile_and_register_class, infer_methods_to_compile, ScriptMethodStub, wrap_cpp_module
from torch.jit._state import _enabled, _set_jit_function_cache, _set_jit_overload_cache, _try_get_jit_cached_function, _try_get_jit_cached_overloads
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import has_torch_function, has_torch_function_unary, has_torch_function_variadic
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
type_trace_db = JitTypeTraceStore()
torch._C.ScriptMethod.graph_for = _script_method_graph_for
torch._C.ScriptFunction.graph_for = _graph_for
ScriptFunction = torch._C.ScriptFunction
ScriptFunction.__doc__ = '\nFunctionally equivalent to a :class:`ScriptModule`, but represents a single\nfunction and does not have any attributes or Parameters.\n'
set_module(ScriptFunction, 'torch.jit')

def _reduce(cls):
    if False:
        for i in range(10):
            print('nop')
    raise pickle.PickleError('ScriptFunction cannot be pickled')
ScriptFunction.__reduce__ = _reduce
if _enabled:
    Attribute = collections.namedtuple('Attribute', ['value', 'type'])
else:

    def Attribute(value, type):
        if False:
            while True:
                i = 10
        return value
Attribute.__doc__ = '\n    This method is a pass-through function that returns `value`, mostly\n    used to indicate to the TorchScript compiler that the left-hand side\n    expression is a class instance attribute with type of `type`. Note that\n    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`\n    subclasses.\n\n    Though TorchScript can infer correct type for most Python expressions, there are some cases where\n    type inference can be wrong, including:\n\n    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`\n    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume\n      it is type `T` rather than `Optional[T]`\n\n    In eager mode, it is simply a pass-through function that returns `value`\n    without other implications.\n\n    Example:\n\n    .. testcode::\n\n        import torch\n        from typing import Dict\n\n        class AttributeModule(torch.jit.ScriptModule):\n            def __init__(self):\n                super().__init__()\n                self.foo = torch.jit.Attribute(0.1, float)\n\n                # we should be able to use self.foo as a float here\n                assert 0.0 < self.foo\n\n                self.names_ages = torch.jit.Attribute({}, Dict[str, int])\n                self.names_ages["someone"] = 20\n                assert isinstance(self.names_ages["someone"], int)\n\n        m = AttributeModule()\n        # m will contain two attributes\n        # 1. foo of type float\n        # 2. names_ages of type Dict[str, int]\n\n    .. testcleanup::\n\n        del AttributeModule\n        del m\n\n    Note: it\'s now preferred to instead use type annotations instead of `torch.jit.Attribute`:\n\n    .. testcode::\n\n        import torch\n        from typing import Dict\n\n        class AttributeModule(torch.nn.Module):\n            names: Dict[str, int]\n\n            def __init__(self):\n                super().__init__()\n                self.names = {}\n\n        m = AttributeModule()\n\n    .. testcleanup::\n\n        del AttributeModule\n        del m\n\n    Args:\n        value: An initial value to be assigned to attribute.\n        type: A Python type\n\n    Returns:\n        Returns `value`\n'

def _get_type_trace_db():
    if False:
        return 10
    return type_trace_db

def _get_function_from_type(cls, name):
    if False:
        print('Hello World!')
    return getattr(cls, name, None)

def _is_new_style_class(cls):
    if False:
        i = 10
        return i + 15
    if hasattr(cls, '__class__'):
        return '__dict__' in dir(cls) or hasattr(cls, '__slots__')

class OrderedDictWrapper:

    def __init__(self, _c):
        if False:
            i = 10
            return i + 15
        self._c = _c

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return [k for (k, v) in self.items()]

    def values(self):
        if False:
            return 10
        return [v for (k, v) in self.items()]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.values())

    def __delitem__(self, k):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('cannot delete methods or parameters of a script module')

    def items(self):
        if False:
            i = 10
            return i + 15
        return self._c.items()

    def __setitem__(self, k, v):
        if False:
            i = 10
            return i + 15
        if k not in self:
            raise RuntimeError(f"Can't add a new parameter after ScriptModule construction. Tried to add '{k}")
        self._c.setattr(k, v)

    def __contains__(self, k):
        if False:
            print('Hello World!')
        return self._c.contains(k)

    def __getitem__(self, k):
        if False:
            print('Hello World!')
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)

class OrderedModuleDict(OrderedDictWrapper):

    def __init__(self, module, python_dict):
        if False:
            print('Hello World!')
        super().__init__(torch._C.ModuleDict(module))
        self._python_modules = python_dict

    def items(self):
        if False:
            print('Hello World!')
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        if False:
            for i in range(10):
                print('nop')
        return k in self._python_modules

    def __setitem__(self, k, v):
        if False:
            return 10
        if isinstance(v, ScriptModule):
            self._c.setattr(k, v)
            self._python_modules[k] = v
        else:
            raise RuntimeError(f"Cannot re-assign modules in a ScriptModule with non-scripted module, tried to replace existing module '{k}': {v}")

    def __getitem__(self, k):
        if False:
            for i in range(10):
                print('nop')
        return self._python_modules[k]

class ScriptMeta(type):

    def __init__(cls, name, bases, attrs):
        if False:
            for i in range(10):
                print('nop')
        cls._methods: Dict[str, Any] = {}
        cls._constants_set = set(getattr(cls, '__constants__', ()))
        for base in reversed(bases):
            for (k, v) in getattr(base, '_methods', {}).items():
                cls._methods[k] = v
            base_constants: Set = getattr(base, '_constants_set', set())
            cls._constants_set = cls._constants_set.union(base_constants)
        for (k, v) in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v
        if getattr(cls, '_disable_script_meta', False):
            return super().__init__(name, bases, attrs)
        original_init = getattr(cls, '__init__', lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            num_methods = len(cls._methods)
            original_init(self, *args, **kwargs)
            added_methods_in_init = len(cls._methods) > num_methods
            if type(self) == cls:

                def make_stubs(module):
                    if False:
                        print('Hello World!')
                    cls = type(module)
                    if hasattr(cls, '_methods'):
                        return [v for (k, v) in sorted(cls._methods.items())]
                    else:
                        return infer_methods_to_compile(module)
                self.__dict__['_actual_script_module'] = torch.jit._recursive.create_script_module(self, make_stubs, share_types=not added_methods_in_init)
                concrete_type = self._actual_script_module._concrete_type
                for name in concrete_type.get_attributes():
                    delattr(self, name)
                for (name, _) in concrete_type.get_modules():
                    delattr(self, name)
                for name in ('_parameters', '_buffers', '_modules'):
                    delattr(self, name)
        cls.__init__ = init_then_script
        super().__init__(name, bases, attrs)

class _CachedForward:

    def __get__(self, obj, cls):
        if False:
            for i in range(10):
                print('nop')
        return self.__getattr__('forward')

class ScriptWarning(Warning):
    pass

def script_method(fn):
    if False:
        while True:
            i = 10
    if not _enabled:
        return fn
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name='ScriptModule')
    return ScriptMethodStub(_rcb, ast, fn)

class ConstMap:

    def __init__(self, const_mapping):
        if False:
            return 10
        self.const_mapping = const_mapping

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        return self.const_mapping[attr]

def unpackage_script_module(importer: PackageImporter, script_module_id: str) -> torch.nn.Module:
    if False:
        print('Hello World!')
    "\n    Call by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.\n\n    Performs work of loading and returning a ScriptModule from a ``torch.package`` archive.\n    "
    if not isinstance(importer.zip_reader, torch._C.PyTorchFileReader):
        raise RuntimeError('Loading ScriptObjects from a PackageImporter created from a directory is not supported. Use a package archive file instead.')
    cu = torch._C.CompilationUnit()
    cpp_module = torch._C._import_ir_module_from_package(cu, importer.zip_reader, importer.storage_context, validate_map_location(importer.last_map_location), script_module_id)
    return wrap_cpp_module(cpp_module)
if _enabled:
    _magic_methods = ['__iter__', '__len__', '__neg__', '__mul__', '__contains__', '__add__', '__sub__', '__pow__', '__truediv__', '__mod__', '__ne__', '__eq__', '__lt__', '__gt__', '__le__', '__ge__', '__and__', '__or__', '__xor__', '__getitem__', '__setitem__', '__call__', '__int__', '__float__', '__bool__', '__str__', '__enter__', '__exit__']

    class RecursiveScriptClass:
        """Wrapper for a TorchScript class instance for use in Python.

        An analogue of RecursiveScriptModule for regular objects that are not modules.
        This class is a wrapper around a torch._C.ScriptObject that represents an instance
        of a TorchScript class and allows it to be used in Python.

        Attributes:
            _c [torch._C.ScriptObject]: The C++ object to which attribute lookups and method
                calls are forwarded.
            _props [Dict[str, property]]: A dictionary of properties fetched from self._c and
                exposed on this wrppaer.
        """

        def __init__(self, cpp_class):
            if False:
                return 10
            super().__init__()
            self.__dict__['_initializing'] = True
            self._c = cpp_class
            self._props = {prop.name: property(prop.getter, prop.setter) for prop in self._c._properties()}
            self.__dict__['_initializing'] = False

        def __getattr__(self, attr):
            if False:
                while True:
                    i = 10
            if '_initializing' in self.__dict__ and self.__dict__['_initializing']:
                return super().__getattr__(attr)
            if attr in self._props:
                return self._props[attr].fget()
            return getattr(self._c, attr)

        def __setattr__(self, attr, value):
            if False:
                for i in range(10):
                    print('nop')
            if '_initializing' in self.__dict__ and self.__dict__['_initializing']:
                return super().__setattr__(attr, value)
            if attr in self._props:
                return self._props[attr].fset(value)
            setattr(self._c, attr, value)

        def forward_magic_method(self, method_name, *args, **kwargs):
            if False:
                print('Hello World!')
            if not self._c._has_method(method_name):
                raise TypeError()
            self_method = self.__getattr__(method_name)
            return self_method(*args, **kwargs)

        def __getstate__(self):
            if False:
                return 10
            raise pickle.PickleError('ScriptClasses cannot be pickled')

        def __iadd__(self, other):
            if False:
                i = 10
                return i + 15
            if self._c._has_method('__iadd__'):
                return self.forward_magic_method('__iadd__', other)
            else:
                return self.forward_magic_method('__add__', other)
    for method_name in _magic_methods:

        def method_template(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return self.forward_magic_method(method_name, *args, **kwargs)
        setattr(RecursiveScriptClass, method_name, method_template)

    class ScriptModule(Module, metaclass=ScriptMeta):
        """Wrapper for C++ torch::jit::Module with methods, attributes, and parameters.

        A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\\s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same way as on a normal ``nn.Module``.
        """
        __jit_unused_properties__ = ['code', 'code_with_constants', 'graph', 'inlined_graph', 'original_name']

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
        forward: Callable[..., Any] = _CachedForward()

        def __getattr__(self, attr):
            if False:
                print('Hello World!')
            if '_actual_script_module' not in self.__dict__:
                return super().__getattr__(attr)
            return getattr(self._actual_script_module, attr)

        def __setattr__(self, attr, value):
            if False:
                for i in range(10):
                    print('nop')
            if '_actual_script_module' not in self.__dict__:
                if isinstance(value, Attribute):
                    if '__annotations__' not in self.__class__.__dict__:
                        self.__class__.__annotations__ = {}
                    self.__annotations__[attr] = value.type
                    value = value.value
                return super().__setattr__(attr, value)
            setattr(self._actual_script_module, attr, value)

        def define(self, src):
            if False:
                for i in range(10):
                    print('nop')
            if '_actual_script_module' in self.__dict__:
                return self._actual_script_module.define(src)
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            ast = torch._C._parse_source_def(src)
            self._methods[ast.name().name] = ScriptMethodStub(rcb, ast, None)

        def _replicate_for_data_parallel(self):
            if False:
                print('Hello World!')
            return self._actual_script_module._replicate_for_data_parallel()

        def __reduce_package__(self, exporter: PackageExporter):
            if False:
                print('Hello World!')
            "Save a ScriptModule inside of a ``torch.package`` archive.\n\n            Called by ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` when\n            saving TorchScript objects. Performs act of saving a ScriptModule inside of\n            a ``torch.package`` archive.\n\n            Returns method to load the ScriptModule from a ``torch.package.PackageImporter``'s\n            Pickler's ``persistent_load`` function.\n            "
            script_module_id = exporter.get_unique_id()
            exporter.script_module_serializer.serialize(self._c, int(script_module_id))
            return (unpackage_script_module, (script_module_id,))

    class RecursiveScriptModule(ScriptModule):
        """Retain the existing isinstance(ScriptModule) behavior.

        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\\s methods are implemented
        as Python functions, but in ``ScriptModule``\\s methods are implemented as
        TorchScript functions, a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\\s code to run without the need for a Python interpreter.

        ``ScriptModule``\\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
        _disable_script_meta = True

        def __init__(self, cpp_module):
            if False:
                while True:
                    i = 10
            self.__dict__['_initializing'] = True
            self._c = cpp_module
            super().__init__()
            delattr(self, 'training')

        @staticmethod
        def _construct(cpp_module, init_fn):
            if False:
                return 10
            "\n            Construct a RecursiveScriptModule that's ready for use.\n\n            PyTorch code should use this to construct a RecursiveScriptModule instead\n            of instead of calling `__init__` directly, as it makes sure the\n            object is properly finalized (and in the future, we may take\n            control of how the RecursiveScriptModule instance is created).\n\n            Args:\n                cpp_module:  The C++ Module that will hold the actual state of\n                             this RecursiveScriptModule instance.\n                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.\n            "
            script_module = RecursiveScriptModule(cpp_module)
            init_fn(script_module)
            RecursiveScriptModule._finalize_scriptmodule(script_module)
            return script_module

        @staticmethod
        def _finalize_scriptmodule(script_module):
            if False:
                return 10
            script_module._parameters = OrderedDictWrapper(torch._C.ParameterDict(script_module._c))
            script_module._buffers = OrderedDictWrapper(torch._C.BufferDict(script_module._c))
            script_module._modules = OrderedModuleDict(script_module._c, script_module._modules)
            script_module._initializing = False

        def _reconstruct(self, cpp_module):
            if False:
                while True:
                    i = 10
            '\n            Re-construct an instance of RecursiveScriptModule using an instance of a C++ module.\n\n            Args:\n                cpp_module: The C++ module that this RecursiveScriptModule will be rebuilt around.\n            '
            self.__init__(cpp_module)
            self._concrete_type = torch._C.ConcreteModuleType.from_jit_type(self._c._type())
            modules = {}
            for (name, cpp_module) in torch._C.ModuleDict(self._c).items():
                modules[name] = wrap_cpp_module(cpp_module)
            self._modules = OrderedModuleDict(self._c, modules)
            self._parameters = OrderedDictWrapper(torch._C.ParameterDict(self._c))
            self._buffers = OrderedDictWrapper(torch._C.BufferDict(self._c))
            self.__dict__ = {k: v for (k, v) in self.__dict__.items() if not isinstance(v, torch._C.ScriptMethod)}
            self.__dict__['_initializing'] = False

        @property
        def graph(self):
            if False:
                for i in range(10):
                    print('nop')
            'Return a string representation of the internal graph for the ``forward`` method.\n\n            See :ref:`interpreting-graphs` for details.\n            '
            return self._c._get_method('forward').graph

        @property
        def inlined_graph(self):
            if False:
                return 10
            '\n            Return a string representation of the internal graph for the ``forward`` method.\n\n            This graph will be preprocessed to inline all function and method calls.\n            See :ref:`interpreting-graphs` for details.\n            '
            return self.forward.inlined_graph

        @property
        def code(self):
            if False:
                i = 10
                return i + 15
            '\n            Return a pretty-printed representation (as valid Python syntax) of the internal graph for the ``forward`` method.\n\n            See :ref:`inspecting-code` for details.\n            '
            return self.forward.code

        @property
        def code_with_constants(self):
            if False:
                print('Hello World!')
            "Return a tuple.\n\n            Returns a tuple of:\n\n            [0] a pretty-printed representation (as valid Python syntax) of\n            the internal graph for the ``forward`` method. See `code`.\n            [1] a ConstMap following the CONSTANT.cN format of the output in [0].\n            The indices in the [0] output are keys to the underlying constant's values.\n\n            See :ref:`inspecting-code` for details.\n            "
            r = self.forward.code_with_constants
            return (r[0], ConstMap(r[1]))

        def save(self, f, **kwargs):
            if False:
                i = 10
                return i + 15
            "Save with a file-like object.\n\n            save(f, _extra_files={})\n\n            See :func:`torch.jit.save <torch.jit.save>` witch accepts a file-like object.\n            This function, torch.save(), converts the object to a string, treating it as a path.\n            DO NOT confuse these two functions when it comes to the 'f' parameter functionality.\n            "
            return self._c.save(str(f), **kwargs)

        def _save_for_lite_interpreter(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            "Add (or update) the bytecode session to the script model.\n\n            _save_for_lite_interpreter(f)\n\n            The updated model is used\n            in lite interpreter for mobile applications.\n\n            Args:\n                f: a string containing a file name.\n                _extra_files: Map from filename to contents which will be stored as part of 'f'.\n\n            "
            return self._c._save_for_mobile(*args, **kwargs)

        def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return self._c._save_to_buffer_for_mobile(*args, **kwargs)

        def save_to_buffer(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return self._c.save_to_buffer(*args, **kwargs)

        def get_debug_state(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return self._c.get_debug_state()

        def extra_repr(self):
            if False:
                return 10
            return f'original_name={self.original_name}'

        def graph_for(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return self.forward.graph_for(self, *args, **kwargs)

        @property
        def original_name(self):
            if False:
                return 10
            if type(self) == str(self._c._type().name()):
                return ''
            return str(self._c._type().name())

        def define(self, src):
            if False:
                print('Hello World!')
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            self._c._define(self._concrete_type, src, rcb)

        def __getattr__(self, attr):
            if False:
                print('Hello World!')
            if '_initializing' not in self.__dict__:
                raise RuntimeError("ScriptModule has not been initialized, did you forget to call super's init?")
            if self._initializing:
                return super().__getattr__(attr)
            if attr in self._modules:
                return self._modules[attr]
            elif self._c.hasattr(attr):
                return self._c.getattr(attr)
            elif self._c._has_method(attr):
                script_method = self._c._get_method(attr)
                self.__dict__[attr] = script_method
                return script_method
            return super().__getattr__(attr)

        def __setattr__(self, attr, value):
            if False:
                print('Hello World!')
            if self._initializing:
                return super().__setattr__(attr, value)
            if attr in self._modules:
                self._modules[attr] = value
            elif self._c.hasattr(attr):
                self._c.setattr(attr, value)
            elif hasattr(self, '_concrete_type') and attr in self._concrete_type.get_constants().keys():
                raise AttributeError(f"Cannot mutate TorchScript constant value: '{attr}'. Value: '{value}'")
            else:
                return super().__setattr__(attr, value)

        def __copy__(self):
            if False:
                return 10
            return torch.jit._recursive.wrap_cpp_module(copy.copy(self._c))

        def __deepcopy__(self, memo):
            if False:
                for i in range(10):
                    print('nop')
            return torch.jit._recursive.wrap_cpp_module(copy.deepcopy(self._c, memo))

        def forward_magic_method(self, method_name, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            self_method = getattr(self, method_name)
            if getattr(self_method, '__func__', None) == getattr(RecursiveScriptModule, method_name):
                raise NotImplementedError()
            return self_method(*args, **kwargs)

        def __iter__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.forward_magic_method('__iter__')

        def __getitem__(self, idx):
            if False:
                return 10
            return self.forward_magic_method('__getitem__', idx)

        def __len__(self):
            if False:
                while True:
                    i = 10
            return self.forward_magic_method('__len__')

        def __contains__(self, key):
            if False:
                i = 10
                return i + 15
            return self.forward_magic_method('__contains__', key)

        def __dir__(self):
            if False:
                return 10
            self_method = self.__dir__
            if self_method.__func__ == _get_function_from_type(RecursiveScriptModule, '__dir__'):
                return super().__dir__()
            return self_method()

        def __bool__(self):
            if False:
                for i in range(10):
                    print('nop')
            self_method = self.__bool__
            if self_method.__func__ == _get_function_from_type(RecursiveScriptModule, '__bool__'):
                return True
            return self_method()

        def _replicate_for_data_parallel(self):
            if False:
                print('Hello World!')

            def init_fn(script_module):
                if False:
                    for i in range(10):
                        print('nop')
                return
            return RecursiveScriptModule._construct(self._c._replicate_for_data_parallel(), init_fn)
    for (name, item) in RecursiveScriptModule.__dict__.items():
        if not callable(item) and (not isinstance(item, property)):
            continue
        if name.startswith('__') or hasattr(ScriptModule, name):
            continue
        setattr(ScriptModule, name, item)

    def _get_methods(cls):
        if False:
            i = 10
            return i + 15
        import inspect
        return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
    _compiled_methods_allowlist = {'forward', 'register_buffer', 'register_parameter', 'register_module', 'add_module', '_apply', 'apply', 'cuda', 'cpu', 'to', 'type', 'float', 'double', 'half', 'state_dict', '_save_to_state_dict', 'load_state_dict', '_load_from_state_dict', '_named_members', 'parameters', 'named_parameters', 'buffers', 'named_buffers', 'children', 'named_children', 'modules', 'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr', '_slow_forward', '_tracing_name', 'eval', 'train', 'get_extra_state', 'set_extra_state'}

    def _make_fail(name):
        if False:
            print('Hello World!')

        def fail(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise RuntimeError(name + ' is not supported on ScriptModules')
        return fail
    for (name, method) in _get_methods(torch.nn.Module):
        if name.startswith('__') or name.endswith('_call_impl'):
            continue
        if name not in RecursiveScriptModule.__dict__ and name not in _compiled_methods_allowlist:
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))
else:

    class RecursiveScriptClass:
        pass

    class ScriptModule(torch.nn.Module):

        def __init__(self, arg=None):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()

    class RecursiveScriptModule(ScriptModule):

        def __init__(self, arg=None):
            if False:
                return 10
            super().__init__()

def call_prepare_scriptable_func_impl(obj, memo):
    if False:
        i = 10
        return i + 15
    if not isinstance(obj, torch.nn.Module):
        return obj
    obj_id = id(obj)
    if obj_id in memo:
        return memo[id(obj)]
    obj = obj.__prepare_scriptable__() if hasattr(obj, '__prepare_scriptable__') else obj
    memo[obj_id] = obj
    new_obj_dict = {}
    for (name, sub_module) in obj.__dict__.items():
        if name == '_modules':
            for (k, v) in sub_module.items():
                sub_module[k] = call_prepare_scriptable_func_impl(v, memo)
            new_obj_dict[name] = sub_module
        elif isinstance(sub_module, torch.nn.Module) and (not isinstance(sub_module, ScriptModule)):
            new_obj_dict[name] = call_prepare_scriptable_func_impl(sub_module, memo)
        else:
            new_obj_dict[name] = sub_module
    for (k, v) in new_obj_dict.items():
        obj.__dict__[name] = v
    return obj

def call_prepare_scriptable_func(obj):
    if False:
        while True:
            i = 10
    memo: Dict[int, torch.nn.Module] = {}
    return call_prepare_scriptable_func_impl(obj, memo)

def create_script_dict(obj):
    if False:
        while True:
            i = 10
    '\n    Create a ``torch._C.ScriptDict`` instance with the data from ``obj``.\n\n    Args:\n        obj (dict): The Python dictionary that is used to initialize the ``ScriptDict``\n                    returned by this function.\n\n    Returns:\n        An instance of ``torch._C.ScriptDict`` that has the same data as ``obj``\n        and can be passed between Python and TorchScript with reference semantics and\n        zero copy overhead.\n    '
    return torch._C.ScriptDict(obj)

def create_script_list(obj, type_hint=None):
    if False:
        print('Hello World!')
    '\n    Create a ``torch._C.ScriptList`` instance with the data from ``obj``.\n\n    Args:\n        obj (dict): The Python list that is used to initialize the ``ScriptList``\n                    returned by this function.\n    Returns:\n        An instance of ``torch._C.ScriptList`` that has the same data as ``obj``\n        and can be passed between Python and TorchScript with reference semantics and\n        zero copy overhead.\n    '
    return torch._C.ScriptList(obj)

def script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None]=None):
    if False:
        print('Hello World!')
    "Script the function.\n\n    Scripting a function or ``nn.Module`` will inspect the source code, compile\n    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or\n    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all\n    features in Python work, but we provide enough functionality to compute on\n    tensors and do control-dependent operations. For a complete guide, see the\n    :ref:`language-reference`.\n\n    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be\n    subsequently passed by reference between Python and TorchScript with zero copy overhead.\n\n    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists\n     and as a decorator ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.\n\n    Args:\n        obj (Callable, class, or nn.Module):  The ``nn.Module``, function, class type,\n                                                  dictionary, or list to compile.\n        example_inputs (Union[List[Tuple], Dict[Callable, List[Tuple]], None]): Provide example inputs\n            to annotate the arguments for a function or ``nn.Module``.\n\n    Returns:\n        If ``obj`` is ``nn.Module``, ``script`` returns\n        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will\n        have the same set of sub-modules and parameters as the\n        original ``nn.Module``. If ``obj`` is a standalone function,\n        a :class:`ScriptFunction` will be returned. If ``obj`` is a ``dict``, then\n        ``script`` returns an instance of `torch._C.ScriptDict`. If ``obj`` is a ``list``,\n        then ``script`` returns an instance of `torch._C.ScriptList`.\n\n    **Scripting a function**\n        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`\n        by compiling the body of the function.\n\n        Example (scripting a function):\n\n        .. testcode::\n\n            import torch\n\n            @torch.jit.script\n            def foo(x, y):\n                if x.max() > y.max():\n                    r = x\n                else:\n                    r = y\n                return r\n\n            print(type(foo))  # torch.jit.ScriptFunction\n\n            # See the compiled graph as Python code\n            print(foo.code)\n\n            # Call the function using the TorchScript interpreter\n            foo(torch.ones(2, 2), torch.ones(2, 2))\n\n        .. testoutput::\n            :hide:\n\n            ...\n\n    ****Scripting a function using example_inputs**\n        Example inputs can be used to annotate a function arguments.\n\n        Example (annotating a function before scripting):\n\n        .. testcode::\n\n            import torch\n\n            def test_sum(a, b):\n                return a + b\n\n            # Annotate the arguments to be int\n            scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])\n\n            print(type(scripted_fn))  # torch.jit.ScriptFunction\n\n            # See the compiled graph as Python code\n            print(scripted_fn.code)\n\n            # Call the function using the TorchScript interpreter\n            scripted_fn(20, 100)\n\n        .. testoutput::\n            :hide:\n\n            ...\n\n    **Scripting an nn.Module**\n        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively\n        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses\n        features supported in TorchScript, no changes to the original module code should be necessary. ``script``\n        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of\n        the original module.\n\n        Example (scripting a simple module with a Parameter):\n\n        .. testcode::\n\n            import torch\n\n            class MyModule(torch.nn.Module):\n                def __init__(self, N, M):\n                    super().__init__()\n                    # This parameter will be copied to the new ScriptModule\n                    self.weight = torch.nn.Parameter(torch.rand(N, M))\n\n                    # When this submodule is used, it will be compiled\n                    self.linear = torch.nn.Linear(N, M)\n\n                def forward(self, input):\n                    output = self.weight.mv(input)\n\n                    # This calls the `forward` method of the `nn.Linear` module, which will\n                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here\n                    output = self.linear(output)\n                    return output\n\n            scripted_module = torch.jit.script(MyModule(2, 3))\n\n        Example (scripting a module with traced submodules):\n\n        .. testcode::\n\n            import torch\n            import torch.nn as nn\n            import torch.nn.functional as F\n\n            class MyModule(nn.Module):\n                def __init__(self):\n                    super().__init__()\n                    # torch.jit.trace produces a ScriptModule's conv1 and conv2\n                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))\n                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))\n\n                def forward(self, input):\n                    input = F.relu(self.conv1(input))\n                    input = F.relu(self.conv2(input))\n                    return input\n\n            scripted_module = torch.jit.script(MyModule())\n\n        To compile a method other than ``forward`` (and recursively compile anything it calls), add\n        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation\n        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.\n\n        Example (an exported and ignored method in a module)::\n\n            import torch\n            import torch.nn as nn\n\n            class MyModule(nn.Module):\n                def __init__(self):\n                    super().__init__()\n\n                @torch.jit.export\n                def some_entry_point(self, input):\n                    return input + 10\n\n                @torch.jit.ignore\n                def python_only_fn(self, input):\n                    # This function won't be compiled, so any\n                    # Python APIs can be used\n                    import pdb\n                    pdb.set_trace()\n\n                def forward(self, input):\n                    if self.training:\n                        self.python_only_fn(input)\n                    return input * 99\n\n            scripted_module = torch.jit.script(MyModule())\n            print(scripted_module.some_entry_point(torch.randn(2, 2)))\n            print(scripted_module(torch.randn(2, 2)))\n\n        Example ( Annotating forward of nn.Module using example_inputs)::\n\n            import torch\n            import torch.nn as nn\n            from typing import NamedTuple\n\n            class MyModule(NamedTuple):\n            result: List[int]\n\n            class TestNNModule(torch.nn.Module):\n                def forward(self, a) -> MyModule:\n                    result = MyModule(result=a)\n                    return result\n\n            pdt_model = TestNNModule()\n\n            # Runs the pdt_model in eager model with the inputs provided and annotates the arguments of forward\n            scripted_model = torch.jit.script(pdt_model, example_inputs={pdt_model: [([10, 20, ], ), ], })\n\n            # Run the scripted_model with actual inputs\n            print(scripted_model([20]))\n    "
    global type_trace_db
    if not _enabled:
        return obj
    if optimize is not None:
        warnings.warn('`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead')
    if isinstance(obj, RecursiveScriptClass):
        return obj
    if isinstance(obj, ScriptModule):
        return obj
    if isinstance(obj, ScriptFunction):
        return obj
    if example_inputs:
        type_trace_db = JitTypeTraceStore()
        if monkeytype_trace:
            monkeytype_config = JitTypeTraceConfig(type_trace_db)
            with monkeytype_trace(monkeytype_config):
                if isinstance(example_inputs, Dict):
                    for (module, example_input) in example_inputs.items():
                        for example in example_input:
                            module(*example)
                elif isinstance(example_inputs, List):
                    for examples in example_inputs:
                        obj(*examples)
                else:
                    raise ValueError('Error: Unable to infer types. Please format the inputs to type `List[Tuple]` or `Dict[Callable, List[Tuple]]` to be run with MonkeyType.')
        else:
            warnings.warn('Warning: monkeytype is not installed. Please install https://github.com/Instagram/MonkeyType to enable Profile-Directed Typing in TorchScript. Refer to https://github.com/Instagram/MonkeyType/blob/master/README.rst to install MonkeyType. ')
    if isinstance(obj, torch.nn.Module):
        obj = call_prepare_scriptable_func(obj)
        return torch.jit._recursive.create_script_module(obj, torch.jit._recursive.infer_methods_to_compile)
    else:
        obj = obj.__prepare_scriptable__() if hasattr(obj, '__prepare_scriptable__') else obj
    if isinstance(obj, dict):
        return create_script_dict(obj)
    if isinstance(obj, list):
        return create_script_list(obj)
    if inspect.isclass(obj):
        qualified_name = _qualified_name(obj)
        if issubclass(obj, torch.nn.Module):
            raise RuntimeError(f"Type '{obj}' cannot be compiled since it inherits from nn.Module, pass an instance instead")
        if issubclass(obj, enum.Enum):
            return obj
        if not _is_new_style_class(obj):
            raise RuntimeError("TorchScript classes must be new-style classes. Please inherit from 'object'.")
        if len(obj.mro()) > 2:
            raise RuntimeError("TorchScript classes does not support inheritance yet. Please directly inherit from 'object'.")
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
        qualified_name = _qualified_name(obj)
        if hasattr(obj, '__script_if_tracing_wrapper'):
            obj = obj.__original_fn
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        if hasattr(obj, '__script_unsupported'):
            raise RuntimeError('TorchScript error: ' + obj.__script_unsupported)
        _check_directly_compile_overloaded(obj)
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            return maybe_already_compiled_fn
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
        fn.__doc__ = obj.__doc__
        fn._torchdynamo_inline = obj
        _set_jit_function_cache(obj, fn)
        return fn
    else:
        return torch.jit._recursive.create_script_class(obj)

def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    if False:
        i = 10
        return i + 15
    for (name, overload_value) in overload_defaults.items():
        if name not in impl_defaults or impl_defaults[name] != overload_value:
            raise torch.jit.frontend.FrontendError(loc, f'Default parameters on overloads do not affect the runtime so they must equal to the default parameter on the implementation function. Found on parameter {name}')

def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    if False:
        print('Hello World!')
    overload_decl = get_jit_def(overload_fn, overload_fn.__name__).decl()
    overload_signature = torch.jit.annotations.get_signature(overload_fn, None, None, inspect.ismethod(overload_fn))
    impl_ast = get_jit_def(impl_fn, impl_fn.__name__)
    overload_defaults = get_default_args(overload_fn)
    implementation_defaults = get_default_args(impl_fn)
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    _check_overload_defaults(implementation_defaults, overload_defaults, overload_decl.range())
    fn = torch._C._jit_script_compile_overload(qual_name, overload_decl, impl_ast, _rcb, implementation_defaults, overload_signature)
    return fn

def _get_overloads(obj):
    if False:
        while True:
            i = 10
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    qual_name = _qualified_name(obj)
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns
    if obj in uncompiled_overloads:
        raise RuntimeError(_jit_internal.get_overload_no_implementation_error_message('function', obj))
    compiled_fns = []
    for overload_fn in uncompiled_overloads:
        compiled_fns.append(_compile_function_with_overload(overload_fn, qual_name, obj))
    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns

def _check_directly_compile_overloaded(obj):
    if False:
        return 10
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(f'Function {qual_name} cannot be directly compiled because it is overloaded. It must be used in a context of a function where its inputs can determine which overload to call.')

def interface(obj):
    if False:
        return 10
    'Decorate to annotate classes or modules of different types.\n\n    This decorator can be used to define an interface that can be used to annotate\n    classes or modules of different types. This can be used for to annotate a submodule\n    or attribute class that could have different types that implement the same\n    interface, or which could be swapped at runtime; or to store a list of modules or\n    classes of varying types.\n\n    It is sometimes used to implement "Callables" - functions or modules that implement\n    an interface but whose implementations differ and which can be swapped out.\n\n    Example:\n    .. testcode::\n\n        import torch\n        from typing import List\n\n        @torch.jit.interface\n        class InterfaceType:\n            def run(self, x: torch.Tensor) -> torch.Tensor:\n                pass\n\n        # implements InterfaceType\n        @torch.jit.script\n        class Impl1:\n            def run(self, x: torch.Tensor) -> torch.Tensor:\n                return x.relu()\n\n        class Impl2(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.val = torch.rand(())\n\n            @torch.jit.export\n            def run(self, x: torch.Tensor) -> torch.Tensor:\n                return x + self.val\n\n        def user_fn(impls: List[InterfaceType], idx: int, val: torch.Tensor) -> torch.Tensor:\n            return impls[idx].run(val)\n\n        user_fn_jit = torch.jit.script(user_fn)\n\n        impls = [Impl1(), torch.jit.script(Impl2())]\n        val = torch.rand(4, 4)\n        user_fn_jit(impls, 0, val)\n        user_fn_jit(impls, 1, val)\n    '
    if not inspect.isclass(obj):
        raise RuntimeError('interface must be applied to a class')
    if not _is_new_style_class(obj):
        raise RuntimeError("TorchScript interfaces must inherit from 'object'")
    is_module_interface = issubclass(obj, torch.nn.Module) and len(obj.mro()) == 3
    if not is_module_interface and len(obj.mro()) > 2:
        raise RuntimeError("TorchScript interface does not support inheritance yet. Please directly inherit from 'object' or 'nn.Module'.")
    qualified_name = _qualified_name(obj)
    rcb = _jit_internal.createResolutionCallbackFromFrame(1)
    ast = get_jit_class_def(obj, obj.__name__)
    mangled_classname = torch._C._jit_script_interface_compile(qualified_name, ast, rcb, is_module_interface)
    obj.__torch_script_interface__ = mangled_classname
    return obj

def _recursive_compile_class(obj, loc):
    if False:
        print('Hello World!')
    _qual_name = _qualified_name(obj)
    error_stack = torch._C.CallStack(_qual_name, loc)
    rcb = _jit_internal.createResolutionCallbackForClassMethods(obj)
    return _compile_and_register_class(obj, rcb, _qual_name)
CompilationUnit = torch._C.CompilationUnit
set_module(CompilationUnit, 'torch.jit')

def pad(s: str, padding: int, offset: int=0, char: str=' '):
    if False:
        while True:
            i = 10
    if padding >= len(s):
        padding -= len(s)
    return ''.join([char for _ in range(padding + offset)]) + s

class _ScriptProfileColumn:

    def __init__(self, header: str, alignment: int=4, offset: int=0):
        if False:
            return 10
        self.header = header
        self.alignment = alignment
        self.offset = offset
        self.rows: Dict[int, Any] = {}

    def add_row(self, lineno: int, value: Any):
        if False:
            return 10
        self.rows[lineno] = value

    def materialize(self):
        if False:
            print('Hello World!')
        max_length = len(self.header)
        rows: List[Tuple[int, str]] = []
        for (key, value) in self.rows.items():
            cell = str(value)
            rows.append((key, cell))
            max_length = max(len(cell), max_length)
        if self.alignment > 0:
            padding = max_length + self.alignment
            padding -= padding % self.alignment
        else:
            padding = 0
        rows = [(key, pad(cell, padding, self.offset)) for (key, cell) in rows]
        return (pad(self.header, padding, self.offset), rows)

class _ScriptProfileTable:

    def __init__(self, cols: List[_ScriptProfileColumn], source_range: List[int]):
        if False:
            while True:
                i = 10
        self.cols = cols
        self.source_range = source_range

    def dump_string(self):
        if False:
            for i in range(10):
                print('nop')
        outputs: List[str] = []
        cells: List[Tuple[str, Dict[int, str]]] = []
        header_buffer = ''
        for col in self.cols:
            (header, rows) = col.materialize()
            header_buffer += header
            cells.append((header, dict(rows)))
        outputs.append(header_buffer)
        outputs.append(pad('', len(header_buffer), 0, '='))
        for line in self.source_range:
            row_buffer = ''
            for (header, rows) in cells:
                cell = rows.get(line)
                if cell is None:
                    row_buffer += pad('', len(header))
                else:
                    row_buffer += cell
            outputs.append(row_buffer)
        return '\n'.join(outputs)

class _ScriptProfile:

    def __init__(self):
        if False:
            return 10
        self.profile = classes.profiling._ScriptProfile()

    def enable(self):
        if False:
            for i in range(10):
                print('nop')
        self.profile.enable()

    def disable(self):
        if False:
            i = 10
            return i + 15
        self.profile.disable()

    def dump_string(self) -> str:
        if False:
            i = 10
            return i + 15
        outputs: List[str] = []
        for source_stats in self.profile._dump_stats():
            source_ref = source_stats.source()
            source_lines = source_ref.text().splitlines()
            dedent = min([len(line) - len(line.lstrip(' ')) for line in source_lines])
            source_lines = [line[dedent:] for line in source_lines]
            start_line = source_ref.starting_lineno()
            end_line = start_line + len(source_lines)
            source_range = range(start_line, end_line)
            lineno = _ScriptProfileColumn('Line #')
            hits = _ScriptProfileColumn('Hits')
            time_ns = _ScriptProfileColumn('Time (ns)')
            line_contents = _ScriptProfileColumn('Line Contents', 0, 1)
            stats = source_stats.line_map()
            for line in source_range:
                lineno.add_row(line, line)
                line_contents.add_row(line, source_lines[line - start_line])
                stat = stats.get(line)
                if stat is not None:
                    hits.add_row(line, stat.count())
                    time_ns.add_row(line, stat.duration_ns())
            table = _ScriptProfileTable([lineno, hits, time_ns, line_contents], list(source_range))
            outputs.append(table.dump_string())
        return '\n\n'.join(outputs)

    def dump(self):
        if False:
            i = 10
            return i + 15
        print(self.dump_string())

def _unwrap_optional(x):
    if False:
        for i in range(10):
            print('nop')
    assert x is not None, 'Unwrapping null optional'
    return x
_register_builtin(_unwrap_optional, 'aten::_unwrap_optional')
_register_builtin(_jit_internal.is_scripting, 'aten::is_scripting')
_register_builtin(has_torch_function, 'aten::has_torch_function')
_register_builtin(has_torch_function_unary, 'aten::has_torch_function')
_register_builtin(has_torch_function_variadic, 'aten::has_torch_function')