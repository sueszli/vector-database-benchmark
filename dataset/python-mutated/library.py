from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
__all__ = ['Library', 'impl', 'define', 'fallthrough_kernel', 'impl_abstract', 'get_ctx']
_impls: Set[str] = set()
_defs: Set[str] = set()
_reserved_namespaces = ['prim']

def fallthrough_kernel():
    if False:
        print('Hello World!')
    '\n    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.\n    '
    raise NotImplementedError('fallthrough_kernel() should never be called.')

class Library:
    """
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    To create a library to override operators in an existing library (with name ns), set the kind to "IMPL".
    To create a new library (with name ns) to register new operators, set the kind to "DEF".
    To create a fragment of a possibly existing library to register operators (and bypass
    the limitation that there is only one library for a given namespace), set the kind to
    "FRAGMENT".

    Args:
        ns: library name
        kind: "DEF", "IMPL" (default: "IMPL"), "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    """

    def __init__(self, ns, kind, dispatch_key=''):
        if False:
            print('Hello World!')
        if kind not in ('IMPL', 'DEF', 'FRAGMENT'):
            raise ValueError('Unsupported kind: ', kind)
        if ns in _reserved_namespaces and (kind == 'DEF' or kind == 'FRAGMENT'):
            raise ValueError(ns, ' is a reserved namespace. Please try creating a library with another name.')
        frame = traceback.extract_stack(limit=3)[0]
        (filename, lineno) = (frame.filename, frame.lineno)
        self.m: Optional[Any] = torch._C._dispatch_library(kind, ns, dispatch_key, filename, lineno)
        self.ns = ns
        self._op_defs: Set[str] = set()
        self._op_impls: Set[str] = set()
        self._registration_handles: List['torch._library.utils.RegistrationHandle'] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        weakref.finalize(self, _del_library, _impls, self._op_impls, _defs, self._op_defs, self._registration_handles)

    def __repr__(self):
        if False:
            return 10
        return f'Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>'

    def define(self, schema, alias_analysis='', *, tags=()):
        if False:
            print('Hello World!')
        'Defines a new operator and its semantics in the ns namespace.\n\n        Args:\n            schema: function schema to define a new operator.\n            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be\n                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").\n            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this\n                                       operator. Tagging an operator changes the operator\'s behavior\n                                       under various PyTorch subsystems; please read the docs for the\n                                       torch.Tag carefully before applying it.\n\n        Returns:\n            name of the operator as inferred from the schema.\n\n        Example::\n            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LIBRARY)\n            >>> my_lib = Library("foo", "DEF")\n            >>> my_lib.define("sum(Tensor self) -> Tensor")\n        '
        if alias_analysis not in ['', 'FROM_SCHEMA', 'CONSERVATIVE']:
            raise RuntimeError(f'Invalid alias_analysis type {alias_analysis}')
        assert self.m is not None
        if isinstance(tags, torch.Tag):
            tags = (tags,)
        result = self.m.define(schema, alias_analysis, tuple(tags))
        qualname = self.ns + '::' + schema.split('(')[0]
        self._op_defs.add(qualname)
        _defs.add(qualname)
        return result

    def impl(self, op_name, fn, dispatch_key=''):
        if False:
            for i in range(10):
                print('nop')
        'Registers the function implementation for an operator defined in the library.\n\n        Args:\n            op_name: operator name (along with the overload) or OpOverload object.\n            fn: function that\'s the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`\n                to register a fallthrough.\n            dispatch_key: dispatch key that the input function should be registered for. By default, it uses\n                          the dispatch key that the library was created with.\n\n        Example::\n            >>> my_lib = Library("aten", "IMPL")\n            >>> def div_cpu(self, other):\n            >>>     return self * (1 / other)\n            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")\n        '
        if not callable(fn):
            raise TypeError(f'Input function is required to be a callable but found type {type(fn)}')
        if dispatch_key == '':
            dispatch_key = self.dispatch_key
        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != '':
                name = name + '.' + overload_name
        else:
            raise RuntimeError('impl should be passed either a name or an OpOverload object as the first argument')
        key = self.ns + '/' + name.split('::')[-1] + '/' + dispatch_key
        if key in _impls:
            raise RuntimeError("This is not allowed since there's already a kernel registered from python overriding {}'s behavior for {} dispatch key and {} namespace.".format(name.split('::')[-1], dispatch_key, self.ns))
        if dispatch_key == 'Meta':
            dispatcher_op_name = name
            if '::' not in dispatcher_op_name:
                dispatcher_op_name = f'{self.ns}::{dispatcher_op_name}'
            if torch._C._dispatch_has_kernel_for_dispatch_key(dispatcher_op_name, 'CompositeImplicitAutograd'):
                raise RuntimeError(f"We should not register a meta kernel directly to the operator '{name}', because it has a CompositeImplicitAutograd kernel in core. Instead we should let the operator decompose, and ensure that we have meta kernels for the base ops that it decomposes into.")
        assert self.m is not None
        self.m.impl(name, dispatch_key if dispatch_key != '' else 'CompositeImplicitAutograd', fn)
        _impls.add(key)
        self._op_impls.add(key)

    def _destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.m = None
        for handle in self._registration_handles:
            handle.destroy()
        self._registration_handles.clear()

def _del_library(captured_impls, op_impls, captured_defs, op_defs, registration_handles):
    if False:
        print('Hello World!')
    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()
_keep_alive = []
NAMELESS_SCHEMA = re.compile('\\(.*\\) -> .*')

@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()):
    if False:
        print('Hello World!')
    'Defines a new operator.\n\n    In PyTorch, defining an op (short for "operator") is a two step-process:\n    - we need to define the op (by providing an operator name and schema)\n    - we need to implement behavior for how the operator interacts with\n    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.\n\n    This entrypoint defines the custom operator (the first step)\n    you must then perform the second step by calling various\n    ``impl_*`` APIs, like :func:`torch.library.impl` or\n    :func:`torch.library.impl_abstract`.\n\n    Args:\n        qualname (str): The qualified name for the operator. Should be\n            a string that looks like "namespace::name", e.g. "aten::sin".\n            Operators in PyTorch need a namespace to\n            avoid name collisions; a given operator may only be created once.\n            If you are writing a Python library, we recommend the namespace to\n            be the name of your top-level module.\n        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"\n            for an op that accepts one Tensor and returns one Tensor. It does\n            not contain the operator name (that is passed in ``qualname``).\n        lib (Optional[Library]): If provided, the lifetime of this operator\n            will be tied to the lifetime of the Library object.\n        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this\n            operator. Tagging an operator changes the operator\'s behavior\n            under various PyTorch subsystems; please read the docs for the\n            torch.Tag carefully before applying it.\n\n    Example::\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LIBRARY)\n        >>> import torch\n        >>> import numpy as np\n        >>>\n        >>> # Define the operator\n        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")\n        >>>\n        >>> # Add implementations for the operator\n        >>> @torch.library.impl("mylibrary::sin", "cpu")\n        >>> def f(x):\n        >>>     return torch.from_numpy(np.sin(x.numpy()))\n        >>>\n        >>> # Call the new operator from torch.ops.\n        >>> x = torch.randn(3)\n        >>> y = torch.ops.mylib.sin(x)\n        >>> assert torch.allclose(y, x)\n\n    '
    if not isinstance(qualname, str):
        raise ValueError(f'define(qualname, schema): expected qualname to be instance of str, got {type(qualname)}')
    (namespace, name) = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, 'FRAGMENT')
        _keep_alive.append(lib)
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(f'define(qualname, schema, ...): expected schema to look like e.g. "(Tensor x) -> Tensor" but got "{schema}"')
    lib.define(name + schema, alias_analysis='', tags=tags)

@define.register
def _(lib: Library, schema, alias_analysis=''):
    if False:
        for i in range(10):
            print('nop')
    "The old torch.library.define.\n    We're keeping this around for BC reasons\n    "

    def wrap(f):
        if False:
            i = 10
            return i + 15
        name = lib.define(schema, alias_analysis)
        lib.impl(name, f)
        return f
    return wrap

@functools.singledispatch
def impl(qualname, types, func=None, *, lib=None):
    if False:
        for i in range(10):
            print('nop')
    'Register an implementation for a device type for this operator.\n\n    You may pass "default" for ``types`` to register this implementation as the\n    default implementation for ALL device types.\n    Please only use this if the implementation truly supports all device types;\n    for example, this is true if it is a composition of built-in PyTorch operators.\n\n    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".\n\n    Args:\n        qualname (str): Should be a string that looks like "namespace::operator_name".\n        types (str | Sequence[str]): The device types to register an impl to.\n        lib (Optional[Library]): If provided, the lifetime of this registration\n            will be tied to the lifetime of the Library object.\n\n    Examples:\n        >>> import torch\n        >>> import numpy as np\n        >>>\n        >>> # Define the operator\n        >>> torch.library.define("mylibrary::sin", "(Tensor x) -> Tensor")\n        >>>\n        >>> # Add implementations for the cpu device\n        >>> @torch.library.impl("mylibrary::sin", "cpu")\n        >>> def f(x):\n        >>>     return torch.from_numpy(np.sin(x.numpy()))\n        >>>\n        >>> x = torch.randn(3)\n        >>> y = torch.ops.mylibrary.sin(x)\n        >>> assert torch.allclose(y, x.sin())\n    '
    if isinstance(types, str):
        types = (types,)
    keys = set({})
    for typ in types:
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    def register(func):
        if False:
            while True:
                i = 10
        (namespace, _) = torch._library.utils.parse_namespace(qualname)
        if lib is None:
            use_lib = Library(namespace, 'FRAGMENT')
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        for key in keys:
            use_lib.impl(qualname, func, key)
    if func is None:
        return register
    else:
        register(func)

def _device_type_to_key(device_type: str) -> str:
    if False:
        while True:
            i = 10
    if device_type == 'default':
        return 'CompositeExplicitAutograd'
    return torch._C._dispatch_key_for_device(device_type)

@impl.register
def _(lib: Library, name, dispatch_key=''):
    if False:
        return 10
    'Legacy torch.library.impl API. Kept around for BC'

    def wrap(f):
        if False:
            return 10
        lib.impl(name, f, dispatch_key)
        return f
    return wrap

def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    if False:
        i = 10
        return i + 15
    'Register an abstract implementation for this operator.\n\n    An "abstract implementation" specifies the behavior of this operator on\n    Tensors that carry no data. Given some input Tensors with certain properties\n    (sizes/strides/storage_offset/device), it specifies what the properties of\n    the output Tensors are.\n\n    The abstract implementation has the same signature as the operator.\n    It is run for both FakeTensors and meta tensors. To write an abstract\n    implementation, assume that all Tensor inputs to the operator are\n    regular CPU/CUDA/Meta tensors, but they do not have storage, and\n    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.\n    The abstract implementation must consist of only PyTorch operations\n    (and may not directly access the storage or data of any input or\n    intermediate Tensors).\n\n    This API may be used as a decorator (see examples).\n\n    For a detailed guide on custom ops, please see\n    https://docs.google.com/document/d/1W--T6wz8IY8fOI0Vm8BF44PdBgs283QvpelJZWieQWQ/edit\n\n    Examples:\n        >>> import torch\n        >>> import numpy as np\n        >>> from torch import Tensor\n        >>>\n        >>> # Example 1: an operator without data-dependent output shape\n        >>> torch.library.define(\n        >>>     "mylib::custom_linear",\n        >>>     "(Tensor x, Tensor weight, Tensor bias) -> Tensor")\n        >>>\n        >>> @torch.library.impl_abstract("mylib::custom_linear")\n        >>> def custom_linear_abstract(x, weight):\n        >>>     assert x.dim() == 2\n        >>>     assert weight.dim() == 2\n        >>>     assert bias.dim() == 1\n        >>>     assert x.shape[1] == weight.shape[1]\n        >>>     assert weight.shape[0] == bias.shape[0]\n        >>>     assert x.device == weight.device\n        >>>\n        >>>     return (x @ weight.t()) + bias\n        >>>\n        >>> # Example 2: an operator with data-dependent output shape\n        >>> torch.library.define("mylib::custom_nonzero", "(Tensor x) -> Tensor")\n        >>>\n        >>> @torch.library.impl_abstract("mylib::custom_nonzero")\n        >>> def custom_nonzero_abstract(x):\n        >>>     # Number of nonzero-elements is data-dependent.\n        >>>     # Since we cannot peek at the data in an abstract impl,\n        >>>     # we use the ctx object to construct a new symint that\n        >>>     # represents the data-dependent size.\n        >>>     ctx = torch.library.get_ctx()\n        >>>     nnz = ctx.new_dynamic_size()\n        >>>     shape = [nnz, x.dim()]\n        >>>     result = x.new_empty(shape, dtype=torch.int64)\n        >>>     return result\n        >>>\n        >>> @torch.library.impl("mylib::custom_nonzero", "cpu")\n        >>> def custom_nonzero_cpu(x):\n        >>>     x_np = x.numpy()\n        >>>     res = np.stack(np.nonzero(x_np), axis=1)\n        >>>     return torch.tensor(res, device=x.device)\n\n    '
    source = torch._library.utils.get_source(_stacklevel + 1)
    frame = inspect.stack()[_stacklevel]
    caller_module = inspect.getmodule(frame[0])
    caller_module_name = None if caller_module is None else caller_module.__name__
    if caller_module_name is not None and caller_module_name.startswith('torchvision.'):
        caller_module_name = None

    def inner(func):
        if False:
            print('Hello World!')
        entry = torch._library.simple_registry.singleton.find(qualname)
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(func, qualname, caller_module_name)
        else:
            func_to_register = func
        handle = entry.abstract_impl.register(func_to_register, source)
        if lib is not None:
            lib._registration_handles.append(handle)
        return func
    if func is None:
        return inner
    return inner(func)

def _check_pystubs_once(func, qualname, actual_module_name):
    if False:
        while True:
            i = 10
    checked = False

    def inner(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        nonlocal checked
        if checked:
            return func(*args, **kwargs)
        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            checked = True
            return func(*args, **kwargs)
        maybe_pystub = torch._C._dispatch_pystub(op._schema.name, op._schema.overload_name)
        if not maybe_pystub:
            raise RuntimeError(f'''Operator '{qualname}' was defined in C++ and has a Python abstract impl. In this situation, it is required to have a C++ `m.impl_abstract_pystub` call, but we could not find one.Please add a call to `m.impl_abstract_pystub("{actual_module_name}");` to the C++ TORCH_LIBRARY block the operator was defined in.''')
        pystub_module = maybe_pystub[0]
        if actual_module_name != pystub_module:
            raise RuntimeError(f"Operator '{qualname}' specified that its python abstract impl is in the Python module '{pystub_module}' but it was actually found in '{actual_module_name}'. Please either move the abstract impl or correct the m.impl_abstract_pystub call.")
        checked = True
        return func(*args, **kwargs)
    return inner

def get_ctx() -> 'torch._library.abstract_impl.AbstractImplCtx':
    if False:
        while True:
            i = 10
    'get_ctx() returns the current AbstractImplCtx object.\n\n    Calling ``get_ctx()`` is only valid inside of an abstract impl\n    (see :func:`torch.library.impl_abstract` for more usage details.\n    '
    return torch._library.abstract_impl.global_ctx_getter()