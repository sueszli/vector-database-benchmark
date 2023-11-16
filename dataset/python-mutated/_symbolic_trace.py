import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type, Union
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_proxyable_classes: Dict[Type, None] = {}
_is_fx_tracing_flag = False

def is_fx_tracing():
    if False:
        for i in range(10):
            print('nop')
    return _is_fx_tracing_flag

@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x : TensorPair, y : torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)

        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """

    def __init__(cls, name, bases, attrs):
        if False:
            i = 10
            return i + 15
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        instance = cls.__new__(cls)
        if not is_fx_tracing():
            cls.__init__(instance, *args, **kwargs)
            return instance
        found_proxies = []

        def check_proxy(a):
            if False:
                while True:
                    i = 10
            if isinstance(a, Proxy):
                found_proxies.append(a)
        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)
        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy('call_function', cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)
            return instance

def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    if False:
        return 10
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    if hasattr(co, 'co_qualname'):
        co_args = (nargs, 0, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_qualname, co.co_firstlineno, co.co_lnotab, co.co_exceptiontable, co.co_freevars, co.co_cellvars)
    elif hasattr(co, 'co_posonlyargcount'):
        co_args = (nargs, 0, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_freevars, co.co_cellvars)
    else:
        co_args = (nargs, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

@compatibility(is_backward_compatible=False)
class PHBase:
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'PH'
PH = PHBase()

@compatibility(is_backward_compatible=False)
class PHWithMeta(PHBase):
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __init__(self, ph_key: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.ph_key = ph_key

@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules: Tuple[ModuleType]=(math,), autowrap_functions: Tuple[Callable, ...]=(), param_shapes_constant: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a Tracer object.\n\n        Args:\n\n            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,\n                Python modules whose functions should be wrapped automatically\n                without needing to use fx.wrap(). Backward-compatibility for\n                this parameter is guaranteed.\n\n            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,\n                Python functions that should be wrapped automatically without\n                needing to use fx.wrap(). Backward compatibility for this\n                parameter is guaranteed.\n\n            param_shapes_constant (bool): When this flag is set,  calls to shape,\n                size and a few other shape like attributes of a module's parameter\n                will be evaluated directly, rather than returning a new Proxy value\n                for an attribute access. Backward compatibility for this parameter\n                is guaranteed.\n        "
        super().__init__()
        self._autowrap_function_ids: Set[int] = {id(value) for (name, value) in chain(*[m.__dict__.items() for m in autowrap_modules]) if not name.startswith('_') and callable(value)}
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
        self.param_shapes_constant = param_shapes_constant
        self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None
        self.root_module_name: str = ''
        self.scope = Scope('', None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> 'Argument':
        if False:
            print('Hello World!')
        '\n        A method to specify the behavior of tracing when preparing values to\n        be used as arguments to nodes in the ``Graph``.\n\n        By default, the behavior includes:\n\n        #. Iterate through collection types (e.g. tuple, list, dict) and recursively\n           call ``create_args`` on the elements.\n        #. Given a Proxy object, return a reference to the underlying IR ``Node``\n        #. Given a non-Proxy Tensor object, emit IR for various cases:\n\n            * For a Parameter, emit a ``get_attr`` node referring to that Parameter\n            * For a non-Parameter Tensor, store the Tensor away in a special\n              attribute referring to that attribute.\n\n        This method can be overridden to support more types.\n\n        Args:\n\n            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.\n\n\n        Returns:\n\n            The value ``a`` converted into the appropriate ``Argument``\n        '
        if isinstance(a, torch.nn.Parameter):
            for (n, p) in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, torch.Tensor):
            for (n_, p_) in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for (n_, p_) in self.root.named_modules():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple((self.create_arg(elem) for elem in a))
            return self.create_node('call_function', a.__class__, args, {})
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)
            if not qualname:
                i = 0
                while True:
                    qualname = f'_tensor_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        if type(a) in _proxyable_classes:
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if False:
            print('Hello World!')
        '\n        A method to specify whether a given ``nn.Module`` is a "leaf" module.\n\n        Leaf modules are the atomic units that appear in\n        the IR, referenced by ``call_module`` calls. By default,\n        Modules in the PyTorch standard library namespace (torch.nn)\n        are leaf modules. All other modules are traced through and\n        their constituent ops are recorded, unless specified otherwise\n        via this parameter.\n\n        Args:\n\n            m (Module): The module being queried about\n            module_qualified_name (str): The path to root of this module. For example,\n                if you have a module hierarchy where submodule ``foo`` contains\n                submodule ``bar``, which contains submodule ``baz``, that module will\n                appear with the qualified name ``foo.bar.baz`` here.\n        '
        return (m.__module__.startswith('torch.nn') or m.__module__.startswith('torch.ao.nn')) and (not isinstance(m, torch.nn.Sequential))

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        if False:
            while True:
                i = 10
        '\n        Helper method to find the qualified name of ``mod`` in the Module hierarchy\n        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has\n        a submodule named ``bar``, passing ``bar`` into this function will return\n        the string "foo.bar".\n\n        Args:\n\n            mod (str): The ``Module`` to retrieve the qualified name for.\n        '
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError('module is not installed as a submodule')
            assert isinstance(path, str)
            return path
        else:
            for (n, p) in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError('module is not installed as a submodule')

    @compatibility(is_backward_compatible=True)
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        '\n        Method that specifies the behavior of this ``Tracer`` when it encounters\n        a call to an ``nn.Module`` instance.\n\n        By default, the behavior is to check if the called module is a leaf module\n        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to\n        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through\n        the operations in its ``forward`` function.\n\n        This method can be overridden to--for example--create nested traced\n        GraphModules, or any other behavior you would want while tracing across\n        ``Module`` boundaries.\n\n        Args:\n\n            m (Module): The module for which a call is being emitted\n            forward (Callable): The forward() method of the ``Module`` to be invoked\n            args (Tuple): args of the module callsite\n            kwargs (Dict): kwargs of the module callsite\n\n        Return:\n\n            The return value from the Module call. In the case that a ``call_module``\n            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever\n            value was returned from the ``Module`` invocation.\n        '
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))) as _scope:
            self.module_stack[_scope.module_path] = _scope.module_type
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy('call_module', module_qualified_name, args, kwargs)
            (key, _) = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f' Unexpected key {key}'
        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        if False:
            return 10
        '\n        Method that specifies the behavior of this ``Tracer`` when we call getattr\n        on a call to an ``nn.Module`` instance.\n\n        By default, the behavior is to return a proxy value for the attribute. It\n        also stores the proxy value in the ``parameter_proxy_cache``, so that future\n        calls will reuse the proxy rather than creating a new one.\n\n        This method can be overridden to --for example-- not return proxies when\n        querying parameters.\n\n        Args:\n\n            attr (str): The name of the attribute being queried\n            attr_val (Any): The value of the attribute\n            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies\n\n        Return:\n\n            The return value from the getattr call.\n        '

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            if False:
                i = 10
                return i + 15
            for (n, p) in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if 'proxy_factory_fn' in inspect.signature(self.create_proxy).parameters:
                            kwargs['proxy_factory_fn'] = None if not self.param_shapes_constant else lambda node: ParameterProxy(self, node, n, attr_val)
                        val_proxy = self.create_proxy('get_attr', n, (), {}, **kwargs)
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None
        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(), parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy
        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy
        return attr_val

    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        if False:
            print('Hello World!')
        "\n        Create ``placeholder`` nodes corresponding to the signature of the ``root``\n        Module. This method introspects root's signature and emits those\n        nodes accordingly, also supporting ``*args`` and ``**kwargs``.\n        "
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)
            args.append(self.root)
        sig = inspect.signature(fn_for_analysis)

        def proxy_placeholder(name: str):
            if False:
                for i in range(10):
                    print('nop')
            if concrete_args is not None and name in concrete_args:
                cnt = 0

                def replace_ph(x):
                    if False:
                        i = 10
                        return i + 15
                    nonlocal cnt
                    cnt += 1
                    param = sig.parameters[name]
                    default = () if param.default is inspect.Parameter.empty else (param.default,)
                    out = self.create_proxy('placeholder', f'{name}_{str(cnt)}', default, {})
                    if isinstance(x, PHBase):

                        def transfer_attrs(fr, to):
                            if False:
                                for i in range(10):
                                    print('nop')
                            for attr_name in dir(fr):
                                attr_val = getattr(fr, attr_name)
                                if not callable(attr_val) and (not attr_name.startswith('__')) and (not hasattr(to, attr_name)):
                                    setattr(to, attr_name, attr_val)
                        if x != PH:
                            transfer_attrs(fr=x, to=out.node)
                        return out
                    if type(x) == bool or (type(x) in base_types and type(x) != torch.Tensor):
                        torch._assert(out == x, f'{name} has been specialized to have value {x} but got another value')
                    elif type(x) == type(None):
                        args = (out, f'{name} has been specialized to have value None but got another value')
                        self.create_proxy('call_function', _assert_is_none, args, {})
                    else:
                        warnings.warn(f'Was not able to add assertion to guarantee correct input {name} to specialized function. It is up to the user to make sure that your inputs match the inputs you specialized the function with.')
                    return x
                return pytree.tree_map(replace_ph, concrete_args[name])
            if name[0] == '*':
                default = ()
            else:
                param = sig.parameters[name]
                default = () if param.default is inspect.Parameter.empty else (param.default,)
            return self.create_proxy('placeholder', name, default, {}, type_expr=fn_for_analysis.__annotations__.get(name, None))
        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(f'Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments')
            concrete_args = dict(zip(arg_names, concrete_args))
        args.extend((proxy_placeholder(names) for names in arg_names))
        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder('*' + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder('**' + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))
        (flat_args, in_spec) = pytree.tree_flatten(tuple(args))
        if any((not isinstance(i, pytree.LeafSpec) for i in in_spec.children_specs)):
            self.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(orig_args[:total_args], in_spec, None))

            def flatten_fn(*args):
                if False:
                    print('Hello World!')
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                (out_args, out_spec) = pytree.tree_flatten(tree_out)
                assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                self.graph._codegen.pytree_info = self.graph._codegen.pytree_info._replace(out_spec=out_spec)
                return out_args
            return (flatten_fn, flat_args)
        return (root_fn, args)

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]]=None) -> Graph:
        if False:
            print('Hello World!')
        '\n        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``\n        can either be an ``nn.Module`` instance or a Python callable.\n\n        Note that after this call, ``self.root`` may be different from the ``root`` passed\n        in here. For example, when a free function is passed to ``trace()``, we will\n        create an ``nn.Module`` instance to use as the root and add embedded constants\n        to.\n\n\n        Args:\n\n            root (Union[Module, Callable]): Either a ``Module`` or a function to be\n                traced through. Backwards-compatibility for this parameter is\n                guaranteed.\n            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should\n                not be treated as Proxies. This parameter is experimental and\n                its backwards-compatibility is *NOT* guaranteed.\n\n        Returns:\n\n            A ``Graph`` representing the semantics of the passed-in ``root``.\n        '
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                self.root = root
                assert hasattr(type(root), self.traced_func_name), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"
                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for (name, mod) in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root
            tracer_cls: Optional[Type[Tracer]] = getattr(self, '__class__', None)
            self.graph = Graph(tracer_cls=tracer_cls)
            if hasattr(fn, '__code__'):
                code = fn.__code__
                self.graph._co_fields = {'co_name': code.co_name, 'co_filename': code.co_filename, 'co_firstlineno': code.co_firstlineno}
            self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                if False:
                    i = 10
                    return i + 15
                for (k, v) in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
                for (k, v) in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])
            collect_tensor_attrs(self.root, [])
            assert isinstance(fn, FunctionType)
            fn_globals = fn.__globals__
            (fn, args) = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
            parameter_proxy_cache: Dict[str, Proxy] = {}

            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                if False:
                    print('Hello World!')
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15

                def forward(*args, **kwargs):
                    if False:
                        return 10
                    return _orig_module_call(mod, *args, **kwargs)
                _autowrap_check(patcher, getattr(getattr(mod, 'forward', mod), '__globals__', {}), self._autowrap_function_ids)
                return self.call_module(mod, forward, args, kwargs)
            with _Patcher() as patcher:
                patcher.patch_method(torch.nn.Module, '__getattr__', module_getattr_wrapper, deduplicate=False)
                patcher.patch_method(torch.nn.Module, '__call__', module_call_wrapper, deduplicate=False)
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)
                self.create_node('output', 'output', (self.create_arg(fn(*args)),), {}, type_expr=fn.__annotations__.get('return', None))
            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph

    def __deepcopy__(self, memo):
        if False:
            return 10
        new_tracer = Tracer.__new__(Tracer)
        for (k, v) in self.__dict__.items():
            if k in {'_autowrap_search'}:
                new_obj = copy.copy(v)
            else:
                new_obj = copy.deepcopy(v, memo)
            new_tracer.__dict__[k] = new_obj
        return new_tracer
_wrapped_fns_to_patch: Dict[Tuple[int, str], dict] = {}
_wrapped_methods_to_patch: List[Tuple[type, str]] = []
if os.environ.get('FX_PATCH_GETITEM') == '1':
    _wrapped_methods_to_patch.append((torch.Tensor, '__getitem__'))

def _find_proxy(*objects_to_search):
    if False:
        i = 10
        return i + 15
    '\n    Recursively search a data structure for a Proxy() and return it,\n    return None if not found.\n    '
    proxy = None

    def find_proxy(x):
        if False:
            while True:
                i = 10
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x
    map_aggregate(objects_to_search, find_proxy)
    return proxy

def _create_wrapped_func(orig_fn):
    if False:
        print('Hello World!')

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Given an closed-over ``orig_function`` to invoke, search the args and kwargs for\n        a Proxy object. If there is one, emit a ``call_function`` node to preserve the\n        call to this leaf function directly. Otherwise, just return the results of\n        this function call, as this function is not being traced.\n        '
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy('call_function', orig_fn, args, kwargs)
            return_proxy.node.meta['is_wrapped'] = True
            return return_proxy
        return orig_fn(*args, **kwargs)
    return wrapped

def _create_wrapped_method(cls, name):
    if False:
        for i in range(10):
            print('nop')
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        if False:
            return 10
        '\n        Search the args and kwargs for a Proxy object. If there is one,\n        emit a ``call_method`` node to preserve the call to this method\n        directly. Otherwise, just return the results of this function\n        call, as this function is not being traced.\n        '
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy('call_method', name, args, kwargs)
        return orig_fn(*args, **kwargs)
    return wrapped

class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class _PatchedFnSetItem(_PatchedFn):

    def revert(self):
        if False:
            while True:
                i = 10
        self.frame_dict[self.fn_name] = self.orig_fn

class _PatchedFnDel(_PatchedFn):

    def revert(self):
        if False:
            while True:
                i = 10
        del self.frame_dict[self.fn_name]

class _PatchedFnSetAttr(_PatchedFn):

    def revert(self):
        if False:
            while True:
                i = 10
        setattr(self.frame_dict, self.fn_name, self.orig_fn)

class _Patcher:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.patches_made: List[_PatchedFn] = []
        self.visited: Set[int] = set()

    def patch(self, frame_dict: Dict[str, Any], name: str, new_fn: Callable, deduplicate: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        Replace frame_dict[name] with new_fn until we exit the context manager.\n        '
        new_fn.__fx_already_patched = deduplicate
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], '__fx_already_patched', False):
            return
        else:
            self.patches_made.append(_PatchedFnSetItem(frame_dict, name, frame_dict[name]))
        frame_dict[name] = new_fn

    def patch_method(self, cls: type, name: str, new_fn: Callable, deduplicate: bool=True):
        if False:
            while True:
                i = 10
        '\n        Replace object_or_dict.name with new_fn until we exit the context manager.\n        '
        new_fn.__fx_already_patched = deduplicate
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, '__fx_already_patched', False):
            return
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        if False:
            i = 10
            return i + 15
        'Return True on the first call to with thing, otherwise false'
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        '\n        Undo all the changes made via self.patch() and self.patch_method()\n        '
        while self.patches_made:
            self.patches_made.pop().revert()
        self.visited.clear()

def _patch_wrapped_functions(patcher: _Patcher):
    if False:
        while True:
            i = 10
    '\n    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap\n    the listed global functions in the `_create_wrapped_func` wrapper.\n    '
    for ((_, name), frame_dict) in _wrapped_fns_to_patch.copy().items():
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))
    for (cls, name) in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))

def _autowrap_check(patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]):
    if False:
        while True:
            i = 10
    '\n    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.\n    This method searches a scope for them and patches them if found.\n    '
    if patcher.visit_once(frame_dict):
        for (name, value) in frame_dict.items():
            if not name.startswith('_') and callable(value) and (id(value) in function_ids):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))

@compatibility(is_backward_compatible=True)
def wrap(fn_or_name: Union[str, Callable]):
    if False:
        return 10
    '\n    This function can be called at module-level scope to register fn_or_name as a "leaf function".\n    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being\n    traced through::\n\n        # foo/bar/baz.py\n        def my_custom_function(x, y):\n            return x * x + y * y\n\n        torch.fx.wrap(\'my_custom_function\')\n\n        def fn_to_be_traced(x, y):\n            # When symbolic tracing, the below call to my_custom_function will be inserted into\n            # the graph rather than tracing it.\n            return my_custom_function(x, y)\n\n    This function can also equivalently be used as a decorator::\n\n        # foo/bar/baz.py\n        @torch.fx.wrap\n        def my_custom_function(x, y):\n            return x * x + y * y\n\n    A wrapped function can be thought of a "leaf function", analogous to the concept of\n    "leaf modules", that is, they are functions that are left as calls in the FX trace\n    rather than traced through.\n\n    Args:\n\n        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the\n            graph when it\'s called\n    '
    if not callable(fn_or_name) and (not isinstance(fn_or_name, str)):
        raise RuntimeError('Unsupported type for global function! Must be either a callable or string name')
    if callable(fn_or_name):
        assert not isinstance(fn_or_name, str)
        fn_name = fn_or_name.__name__
    else:
        assert isinstance(fn_or_name, str), 'fn_or_name must be a global function or string name'
        fn_name = fn_or_name
    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != '<module>':
        raise NotImplementedError('wrap must be called at the top level of a module')
    _wrapped_fns_to_patch[id(f.f_globals), fn_name] = f.f_globals
    return fn_or_name

@compatibility(is_backward_compatible=True)
def symbolic_trace(root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]]=None) -> GraphModule:
    if False:
        print('Hello World!')
    "\n    Symbolic tracing API\n\n    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``\n    constructed by recording operations seen while tracing through ``root``.\n\n    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.\n\n    For example::\n\n        def f(a, b):\n            if b == True:\n                return a\n            else:\n                return a*2\n\n    FX can typically not trace through this due to the presence of control\n    flow. However, we can use `concrete_args` to specialize on the value of\n    `b` to trace through this::\n\n        f = fx.symbolic_trace(f, concrete_args={'b': False})\n        assert f(3, False)  == 6\n\n    Note that although you can still pass in different values of `b`, they will be ignored.\n\n    We can also use `concrete_args` to eliminate data-structure handling from\n    our function. This will use pytrees to flatten your input. To avoid\n    overspecializing, pass in `fx.PH` for values that shouldn't be\n    specialized. For example::\n\n        def f(x):\n            out = 0\n            for v in x.values():\n                out += v\n            return out\n        f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})\n        assert f({'a': 1, 'b': 2, 'c': 4}) == 7\n\n\n    Args:\n        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted\n            into a Graph representation.\n        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized\n\n    Returns:\n        GraphModule: a Module created from the recorded operations from ``root``.\n    "
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)

@wrap
def _assert_is_none(value, msg):
    if False:
        i = 10
        return i + 15
    assert value is None, msg