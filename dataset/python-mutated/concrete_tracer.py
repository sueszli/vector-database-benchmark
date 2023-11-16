from __future__ import annotations
import collections
import copy
import sys
import inspect
import logging
import operator
import functools
import builtins
from itertools import chain
from types import BuiltinMethodType, FunctionType, MethodDescriptorType, MethodType, MethodWrapperType, ModuleType
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple, Type, List, Callable, Union
from contextlib import contextmanager
import torch
from torch._C import ScriptObject
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
from torch.utils._pytree import tree_map
import torch.fx
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import _Patcher, _proxyable_classes
from torch.fx.graph import Graph
from torch.fx.node import Target, Node, Argument, _side_effectful_functions
from torch.fx.proxy import TracerBase
from torch.fx.operator_schemas import check_for_mutable_operation
try:
    from torch.fx.proxy import Scope
except ImportError:

    @compatibility(is_backward_compatible=False)
    class Scope:

        def __init__(self, module_path: str, module_type: Any):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.module_path = module_path
            self.module_type = module_type
try:
    from torch.fx.proxy import ScopeContextManager
except ImportError:

    @compatibility(is_backward_compatible=False)
    class ScopeContextManager:
        """ A context manager to track the Scope of Node during symbolic tracing.
        When entering a forward function of a Module, we'll update the scope information of
        the current module, and when we exit, we'll restore the previous scope information.
        """

        def __init__(self, scope: Scope, current_scope: Scope):
            if False:
                return 10
            super().__init__()
            self._prev_scope = copy.copy(scope)
            scope.module_path = current_scope.module_path
            scope.module_type = current_scope.module_type
            self._scope = scope

        def __enter__(self):
            if False:
                print('Hello World!')
            return self._scope

        def __exit__(self, *args):
            if False:
                print('Hello World!')
            self._scope.module_path = self._prev_scope.module_path
            self._scope.module_type = self._prev_scope.module_type
            return
from . import concrete_proxy as ep
from .operator_patcher import OperatorPatcherContext
from .utils import _orig_module_call, _orig_module_getattr, _orig_module_getattribute, _orig_agfunc_apply, _orig_torch_assert, _orig_type, _orig_isinstance, _orig_issubclass, _orig_getattr, _orig_range, _orig_int, _orig_bool, _orig_tuple, _orig_list, _orig_set, _orig_frozenset, _orig_dict, _orig_map, _orig_zip, _orig_enumerate, _orig_slice, _orig_reversed, _orig_torch_size, _orig_torch_finfo, _orig_len, _orig_not, _orig_is, _orig_is_not, _orig_contains, _orig_index, _orig_all, _orig_min, _orig_max, _orig_node_is_impure
extra_side_effectful_functions = {operator.setitem, builtins.next}
_side_effectful_functions = _side_effectful_functions.union(extra_side_effectful_functions)
_logger = logging.getLogger(__name__)
HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

@compatibility(is_backward_compatible=True)
class ConcreteTracer(TracerBase):
    """
    A model tracer similar to _symbolic_trace.Tracer, but with concrete execution and real value so we can pass complex conditions
    and go into correct brunches.
    """
    default_module_getattr = ('training',)
    default_autowrap_modules = ('math',)
    default_autowrap_leaf_function: Dict[Any, Tuple[List[Tuple[Union[ModuleType, Type], str]], bool, Optional[Callable]]] = {_orig_len: ([], False, None), _orig_not: ([], False, None), _orig_is: ([], False, None), _orig_is_not: ([], False, None), _orig_contains: ([], False, None), _orig_index: ([], False, None), _orig_all: ((), False, None), _orig_min: ((), False, None), _orig_max: ((), False, None), torch.arange: ([], True, None), torch.empty: ([], True, None), torch.eye: ([], True, None), torch.full: ([], True, None), torch.linspace: ([], True, None), torch.logspace: ([], True, None), torch.ones: ([], True, None), torch.rand: ([], True, None), torch.randint: ([], True, None), torch.randn: ([], True, None), torch.randperm: ([], True, None), torch.tensor: ([], True, None), torch.zeros: ([], True, None), Sequential.__getitem__: ([], False, operator.getitem), Sequential.__len__: ([], False, _orig_len), Sequential.__iter__: ([], False, iter), ModuleList.__getitem__: ([], False, operator.getitem), ModuleList.__len__: ([], False, _orig_len), ModuleList.__iter__: ([], False, iter), ModuleDict.__getitem__: ([], False, operator.getitem), ModuleDict.__len__: ([], False, _orig_len), ModuleDict.__iter__: ([], False, iter), ModuleDict.__contains__: ([], False, _orig_contains), ParameterList.__getitem__: ([], False, operator.getitem), ParameterList.__len__: ([], False, _orig_len), ParameterList.__iter__: ([], False, iter), ParameterDict.__getitem__: ([], False, operator.getitem), ParameterDict.__len__: ([], False, _orig_len), ParameterDict.__iter__: ([], False, iter), ParameterDict.__contains__: ([], False, _orig_contains)}
    nn_functional = getattr(torch.nn, 'functional')
    for name in torch.functional.__all__:
        attr = getattr(torch.functional, name)
        if attr not in default_autowrap_leaf_function:
            default_autowrap_leaf_function[attr] = ([], False, attr)
    for name in dir(nn_functional):
        attr = getattr(nn_functional, name)
        if callable(attr) and (not _orig_isinstance(attr, Type)) and (not name.startswith('__')) and (getattr(attr, '__module__', None) not in ('typing', 'torch.nn.modules.utils')):
            if attr not in default_autowrap_leaf_function:
                default_autowrap_leaf_function[attr] = ([], False, getattr(torch.functional, name, None))
            if hasattr(attr, '__module__') and attr.__module__ != 'torch.nn.functional':
                default_autowrap_leaf_function[attr][0].append((nn_functional, name))
    for name in dir(torch._C._VariableFunctions):
        attr = getattr(torch._C._VariableFunctions, name)
        if callable(attr) and (not _orig_isinstance(attr, Type)) and (not name.startswith('__')):
            if attr not in default_autowrap_leaf_function:
                default_autowrap_leaf_function[attr] = ([], False, getattr(torch.functional, name, None))
    for name in dir(torch._C._nn):
        attr = getattr(torch._C._nn, name)
        if callable(attr) and (not _orig_isinstance(attr, Type)) and (not name.startswith('__')):
            if attr not in default_autowrap_leaf_function:
                default_autowrap_leaf_function[attr] = ([], False, getattr(torch.functional, name, None))
            if hasattr(attr, '__module__') and attr.__module__ != 'torch._C._nn':
                default_autowrap_leaf_function[attr][0].append((torch._C._nn, name))
    for name in dir(torch._C._TensorBase):
        attr = getattr(torch._C._TensorBase, name)
        if callable(attr) and (not _orig_isinstance(attr, Type)) and (not name.startswith('__')):
            if attr not in default_autowrap_leaf_function:
                to_func = getattr(torch.Tensor, name, None)
                to_func = None if to_func == attr else to_func
                default_autowrap_leaf_function[attr] = ([], False, to_func)
    default_autowrap_leaf_class: Dict[Type, Tuple[List[Tuple[Union[ModuleType, Type], str]], bool]] = {_orig_bool: ([], False), _orig_zip: ([], False), _orig_int: ([], False), _orig_tuple: ([], True), _orig_list: ([], True), _orig_set: ([], True), _orig_frozenset: ([], True), _orig_dict: ([], True), _orig_reversed: ((), False), _orig_torch_size: ((), False), _orig_torch_finfo: ((), False)}

    @compatibility(is_backward_compatible=True)
    def __init__(self, cpu_offload=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        similar to _symbolic_trace.Tracer.__init__.\n        remove the 'param_shapes_constant' because we can get real shape when executing.\n        "
        super().__init__()
        self.scope = Scope('', None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}
        self.cpu_offload = cpu_offload

    @contextmanager
    def do_temp_disable(self, call=False, attr=False, agfunc_apply=False):
        if False:
            while True:
                i = 10
        assert call | attr | agfunc_apply
        (temp_disable_call, temp_disable_attr, temp_disable_agfunc_apply) = (False, False, False)
        if call:
            self.temp_disable_call_level += 1
            temp_disable_call = self.temp_disable_call
            self.temp_disable_call = True
        if attr:
            self.temp_disable_attr_level += 1
            temp_disable_attr = self.temp_disable_attr
            self.temp_disable_attr = True
        if agfunc_apply:
            self.temp_disable_agfunc_apply_level += 1
            temp_disable_agfunc_apply = self.temp_disable_agfunc_apply
            self.temp_disable_agfunc_apply = True
        try:
            yield
        finally:
            if agfunc_apply:
                self.temp_disable_agfunc_apply = temp_disable_agfunc_apply
                self.temp_disable_agfunc_apply_level -= 1
            if attr:
                self.temp_disable_attr = temp_disable_attr
                self.temp_disable_attr_level -= 1
            if call:
                self.temp_disable_call = temp_disable_call
                self.temp_disable_call_level -= 1

    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Any:
        if False:
            i = 10
            return i + 15
        "\n        to get the attr in self.root. only for execution of 'call_module' nodes.\n        "
        with self.do_temp_disable(attr=True):
            target_atoms = target.split('.')
            attr_itr = self.root
            for (i, atom) in _orig_enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target '{'.'.join(target_atoms[:i])}'")
                attr_itr = _orig_getattr(attr_itr, atom)
            return attr_itr

    def run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        if False:
            return 10
        '\n        actually execute the code.\n        apply the patcher, and the _autowrap_check to the target function.\n        '
        if kind == 'output':
            return args[0]
        elif kind == 'placeholder':
            return self.placeholder_dict[target]
        to_cpu = lambda t: t.cpu() if _orig_isinstance(t, torch.Tensor) else t
        to_cuda = lambda t: t.cuda() if _orig_isinstance(t, torch.Tensor) else t

        def run(kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            if False:
                while True:
                    i = 10
            if self.cpu_offload:
                args = tree_map(to_cuda, args)
                kwargs = tree_map(to_cuda, kwargs)
            if kind == 'call_function':
                assert isinstance(target, Callable)
                fn = target
                if _orig_getattr(fn, '__module__', None) != 'nni.common.concrete_trace_utils.concrete_tracer' and hasattr(fn, '__globals__'):
                    _autowrap_check(self, fn.__globals__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                return OperatorPatcherContext.patch_run(fn, *args, **kwargs)
            elif kind == 'call_method':
                (self_obj, *args_tail) = args
                fn = _orig_getattr(self_obj, target)
                if _orig_getattr(fn, '__module__', None) != 'nni.common.concrete_trace_utils.concrete_tracer' and hasattr(fn, '__globals__'):
                    _autowrap_check(self, fn.__globals__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                result = fn(*args_tail, **kwargs)
            elif kind == 'call_module':
                assert isinstance(target, str)
                mod = self.fetch_attr(target)
                if self.cpu_offload:
                    mod.cuda()
                if _orig_getattr(mod, '__module__', None) != 'nni.common.concrete_trace_utils.concrete_tracer' and hasattr(mod, '__globals__'):
                    _autowrap_check(self, mod.__globals__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                result = OperatorPatcherContext.patch_run(mod, *args, **kwargs)
                if self.cpu_offload:
                    mod.cpu()
            elif kind == 'get_attr':
                assert isinstance(target, str)
                return self.fetch_attr(target)
            else:
                raise RuntimeError()
            return result
        with self.do_temp_disable(call=True):
            result = run(kind, target, args, kwargs)
            if self.cpu_offload:
                if isinstance(result, torch.Tensor):
                    result = result.cpu()
                elif isinstance(result, (list, dict, tuple)):
                    result = tree_map(to_cpu, result)
                else:
                    _logger.warning(f'result of target {target} is {type(result)}, which is not a common behavior.')
                torch.cuda.empty_cache()
        self.temp_disable_call = False
        return result

    @compatibility(is_backward_compatible=True)
    def create_node(self, kind: str, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: Optional[str]=None, type_expr: Optional[Any]=None) -> Node:
        if False:
            for i in range(10):
                print('nop')
        "\n        This method is almost the same as the one in `TracerBase` class of Pytorch2.0.\n        Add it here because this method of Pytorch1.13 and older version\n        doesn't have the part related to `module_stack` and `node_name_to_scope`.\n        If we don't add it here, we can not use these two attributes in Pytorch1.13 and older version.\n        "
        if kind == 'call_function' and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)
        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
        if self.module_stack:
            node.meta['nn_module_stack'] = copy.copy(self.module_stack)
        else:
            node.meta['nn_module_stack'] = collections.OrderedDict()
        return node

    @compatibility(is_backward_compatible=True)
    def proxy(self, value: Any, node: Node) -> ep.ConcreteProxy:
        if False:
            return 10
        "\n        overloaded to use custom 'proxy'.\n        "
        return ep.ConcreteProxy(node, value, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None, proxy_factory_fn: Optional[Callable[[Node], Any]]=None):
        if False:
            print('Hello World!')
        "\n        similar to _symbolic_trace.Tracer.create_proxy.\n        use the 'run_target' to actually execute the code, and store the value in 'value' field.\n        "

        def upwrapper(obj: Any):
            if False:
                return 10
            while _orig_isinstance(obj, ep.ConcreteProxy):
                obj = obj.value
            return obj
        args_unwrapped = ep.map_aggregate_not_proxy(args, upwrapper)
        kwargs_unwrapped = ep.map_aggregate_not_proxy(kwargs, upwrapper)
        value_unwrapped = self.run_target(kind, target, args_unwrapped, kwargs_unwrapped)
        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)
        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)
        proxy = self.proxy(value_unwrapped, node)
        return proxy

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Union[Node, Any]:
        if False:
            return 10
        "\n        similar to _symbolic_trace.Tracer.create_arg\n        move the base case to the top in case the wrapping of the function 'isinstance'\n        "
        if isinstance(a, ep.ConcreteProxy):
            return a.node
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
        if isinstance(a, slice):
            start = self.create_arg(a.start)
            stop = self.create_arg(a.stop)
            step = self.create_arg(a.step)
            if _orig_isinstance(start, Node) or _orig_isinstance(stop, Node) or _orig_isinstance(step, Node):
                return self.create_node('call_function', _orig_slice, (start, stop, step), {})
            else:
                return a
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
        if _orig_type(a) in _proxyable_classes:
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        if isinstance(a, (torch.autograd.function.Function, torch.autograd.function.FunctionMeta)):
            return a
        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        similar to _symbolic_trace.Tracer.is_leaf_module\n        '
        return m.__module__.startswith('torch.nn') and (not _orig_isinstance(m, (Sequential, ModuleList, ModuleDict))) or _orig_isinstance(m, self.leaf_module)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        if False:
            print('Hello World!')
        '\n        similar to _symbolic_trace.Tracer.path_of_module\n        '
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                if not hasattr(self.root, '_module_constants'):
                    self.root._module_constants = torch.nn.ModuleList()
                module_constants = self.root._module_constants
                assert isinstance(module_constants, torch.nn.ModuleList)
                if hasattr(mod, 'extra_repr'):
                    sub_path = _orig_type(mod).__name__ + mod.extra_repr()
                else:
                    sub_path = str(_orig_len(module_constants))
                if not hasattr(module_constants, sub_path):
                    module_constants.add_module(sub_path, mod)
                path = '_module_constants.%s' % sub_path
                self.submodule_paths[mod] = path
                return path
            assert isinstance(path, str)
            return path
        else:
            for (n, p) in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError('module is not installed as a submodule')

    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args: Union[Dict[str, Any], Tuple]) -> Tuple[Any, list, Any, Any]:
        if False:
            return 10
        '\n        for wrapping all the parameters of the function with dummy_input.\n        in concrete tracer, we need all the parameters input by users.\n\n        todo: this function should be refactored after the same function in torch.fx be refactored.\n        '
        fn_for_analysis = inspect.unwrap(root_fn)
        default_value_list = fn_for_analysis.__defaults__
        if default_value_list is None:
            default_value_list = tuple()
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        more_args = []
        kwargs = {}
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)
            args.append(self.root)
        cnt = 0
        self.placeholder_dict = {}
        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        diff_len = _orig_len(arg_names) - _orig_len(default_value_list)
        default_args = {arg_names[idx + diff_len]: default_value_list[idx] for idx in range(len(default_value_list))}
        if isinstance(concrete_args, tuple):
            if _orig_len(arg_names) != _orig_len(concrete_args):
                raise RuntimeError(f'Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments')
            concrete_args = {name: val for (name, val) in zip(arg_names, concrete_args)}

        def proxy_placeholder(name: str):
            if False:
                i = 10
                return i + 15
            nonlocal cnt
            cnt += 1
            default_arg = ()
            if name in default_args and (not name.startswith('*')):
                default_arg = (default_args[name],)
            if name in concrete_args:
                self.placeholder_dict[name] = concrete_args[name]
            else:
                assert name in default_args
                self.placeholder_dict[name] = default_args[name]
            return self.create_proxy('placeholder', name, default_arg, {})
        args.extend((proxy_placeholder(names) for names in arg_names))
        if hasattr(co, 'co_kwonlyargcount') and (co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF):
            if co.co_flags & inspect.CO_VARARGS:
                name = '*' + next(names_iter)
                default_args[name] = ()
                more_args = proxy_placeholder(name)
            if co.co_flags & inspect.CO_VARKEYWORDS:
                name = '**' + next(names_iter)
                default_args[name] = {}
                kwargs = proxy_placeholder(name)
        return (root_fn, args, more_args, kwargs)

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], *, autowrap_modules: Tuple[str] | None=None, autowrap_leaf_function=None, autowrap_leaf_class=None, leaf_module=None, fake_middle_class=None, concrete_args: Union[Dict[str, Any], Tuple], use_operator_patch: bool=True, operator_patch_backlist: List[str] | None=None, forward_function_name: str='forward') -> Graph:
        if False:
            i = 10
            return i + 15
        "\n        similar to _symbolic_trace.Tracer.trace\n        different args:\n            use_operator_patch:\n                the operators 'not/is/is not/in/not in' cannot be wrapped after\n                    compiled. so we re-parse the functions, replace these operators\n                    with functions 'operator.not_/is_/is_not/contains', then we\n                    could wrap and trace these.\n                for example: in ``if x is None:``, if x is a proxy, the tracer will\n                    never go into the branch, even x is a proxy with value 'None'.\n                values:\n                true: before executing a func, the func will be patched if the func\n                    is not in operator_patch_backlist\n                false: before executing a func, the func will be patched if the func\n                    is in operator_patch_backlist\n\n            operator_patch_backlist:\n                such as '__main__.FooModel' or '__main__.bar_func'. the namespace is\n                always needed.\n        "
        args = inspect.getfullargspec(root.forward).args[1:]
        defaults = inspect.getfullargspec(root.forward).defaults
        defaults = tuple() if defaults is None else defaults
        if isinstance(concrete_args, (tuple, list)):
            concrete_args = (*concrete_args, *defaults[len(concrete_args) + len(defaults) - len(args):])
        else:
            kv_default = {k: v for (k, v) in zip(args[-len(defaults):], defaults)}
            concrete_args = {**concrete_args, **{n: kv_default[n] for n in args if n not in concrete_args}}
        autowrap_modules = autowrap_modules if autowrap_modules is not None else tuple()
        autowrap_leaf_function = autowrap_leaf_function if autowrap_leaf_function is not None else {}
        autowrap_leaf_class = autowrap_leaf_class if autowrap_leaf_class is not None else {}
        leaf_module = leaf_module if leaf_module is not None else ()
        fake_middle_class = fake_middle_class if fake_middle_class is not None else ()
        operator_patch_backlist = operator_patch_backlist if operator_patch_backlist is not None else []
        self._autowrap_search: List[ModuleType] = list((sys.modules[m] for m in (*autowrap_modules, *ConcreteTracer.default_autowrap_modules)))
        self._autowrap_function_ids: Set[int] = {id(value) for (name, value) in chain(*[m.__dict__.items() for m in self._autowrap_search]) if not name.startswith('_') and callable(value)}
        self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None
        self.autowrap_leaf_function = {**autowrap_leaf_function, **ConcreteTracer.default_autowrap_leaf_function}
        self.autowrap_leaf_class = {**autowrap_leaf_class, **ConcreteTracer.default_autowrap_leaf_class}
        self.leaf_module = leaf_module
        self.fake_middle_class = fake_middle_class
        if isinstance(root, torch.nn.Module):
            self.root = root
            assert hasattr(root, forward_function_name), f"traced_func_name={forward_function_name} doesn't exist in {_orig_type(root).__name__}"
            fn = getattr(root, forward_function_name)
            self.submodule_paths = {mod: name for (name, mod) in root.named_modules()}
        else:
            self.root = torch.nn.Module()
            fn = root
        tracer_cls = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            if False:
                return 10
            for (k, v) in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for (k, v) in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])
        collect_tensor_attrs(self.root, [])
        if isinstance(fn, MethodType):
            fn = fn.__func__
        assert isinstance(fn, FunctionType)
        fn_globals = fn.__globals__
        (fn, args, more_args, kwargs) = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
        self.the_path_of_parameter = {id(v): k for (k, v) in self.root.named_parameters()}
        self.the_path_of_buffer = {id(v): k for (k, v) in self.root.named_buffers()}

        def get_middle_class(node, memo=set(), prefix=''):
            if False:
                i = 10
                return i + 15
            if node not in memo:
                memo.add(node)
                yield (prefix, node)
                if isinstance(node, torch.nn.Module):
                    items = (*((k, v) for (k, v) in node.__dict__.items() if not k.startswith('_')), *node._modules.items())
                else:
                    items = ((k, v) for (k, v) in node.__dict__.items() if not k.startswith('_'))
                for (name, subfield) in items:
                    if isinstance(subfield, (torch.nn.Module, self.fake_middle_class)):
                        submodule_prefix = prefix + ('.' if prefix else '') + name
                        for m in get_middle_class(subfield, memo, submodule_prefix):
                            yield m
        self.the_path_of_middle_class = {id(v): k for (k, v) in get_middle_class(self.root)}

        @functools.wraps(_orig_module_getattribute)
        def module_getattribute_wrapper(mod, attr):
            if False:
                print('Hello World!')
            if self.temp_disable_call | self.temp_disable_attr:
                try:
                    return _orig_module_getattribute(mod, attr)
                except AttributeError:
                    return _orig_module_getattr(mod, attr)
            with self.do_temp_disable(attr=True):
                try:
                    attr_val = _orig_module_getattribute(mod, attr)
                except AttributeError:
                    attr_val = _orig_module_getattr(mod, attr)
            if callable(attr_val):
                if attr_val in self.wrapped_leaf:
                    return self.wrapped_leaf[attr_val][1]
                return attr_val
            elif attr in self.default_module_getattr:
                path = self.the_path_of_middle_class[id(mod)]
                path = path + '.' if path else ''
                return self.create_proxy('get_attr', f'{path + attr}', (), {})
            elif _orig_isinstance(attr_val, (_orig_tuple, _orig_list)):
                if self.the_path_of_middle_class[id(mod)] == '':
                    return self.create_proxy('get_attr', f'{attr}', (), {})
                else:
                    return self.create_proxy('get_attr', f'{self.the_path_of_middle_class[id(mod)]}.{attr}', (), {})
            elif id(attr_val) in self.the_path_of_parameter:
                return self.create_proxy('get_attr', self.the_path_of_parameter[id(attr_val)], (), {})
            elif id(attr_val) in self.the_path_of_buffer:
                return self.create_proxy('get_attr', self.the_path_of_buffer[id(attr_val)], (), {})
            return attr_val

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            if False:
                print('Hello World!')
            if self.temp_disable_call:
                return _orig_module_call(mod, *args, **kwargs)
            else:
                module_qualified_name = self.path_of_module(mod)
                with ScopeContextManager(self.scope, Scope(module_qualified_name, type(mod))) as _scope:
                    self.module_stack[_scope.module_path] = _scope.module_type
                    if not self.is_leaf_module(mod, module_qualified_name):
                        _autowrap_check(self, mod.forward.__globals__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                        _autowrap_check(self, mod.__dict__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                        ret_val = _orig_module_call(mod, *args, **kwargs)
                    else:
                        ret_val = self.create_proxy('call_module', module_qualified_name, args, kwargs)
                    (key, _) = self.module_stack.popitem(last=True)
                    assert key == _scope.module_path, f' Unexpected key {key}'
                return ret_val

        class map_wrapper_clz:

            @functools.wraps(_orig_map)
            def __call__(self, the_func, *iterables: Any):
                if False:
                    return 10
                tracers = _orig_set()
                for one_iter in iterables:
                    if _orig_isinstance(one_iter, ep.Proxy):
                        tracers.add(one_iter.tracer)
                if _orig_len(tracers) > 1:
                    raise Exception('more than 1 tracer detected. please report the issue')
                elif _orig_len(tracers) == 1:
                    results = _orig_list()
                    for args in _orig_zip(*iterables):
                        results.append(the_func(*args))
                    return next(iter(tracers)).create_proxy('call_function', _orig_tuple, (results,), {})
                iterables = _orig_list((_orig_list(it) for it in iterables))
                for it in iterables:
                    for arg in it:
                        if _orig_isinstance(arg, ep.Proxy):
                            tracers.add(arg.tracer)
                if _orig_len(tracers) > 1:
                    raise Exception('more than 1 tracer detected. please report the issue')
                elif _orig_len(tracers) == 1:
                    results = _orig_list()
                    for args in _orig_zip(*iterables):
                        results.append(the_func(*args))
                    return next(iter(tracers)).create_proxy('call_function', _orig_tuple, (results,), {})
                return _orig_map(the_func, *iterables)

            def __eq__(self, __o: object) -> bool:
                if False:
                    i = 10
                    return i + 15
                return id(__o) in (id(self), id(_orig_map))

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return id(self)
        map_wrapper = map_wrapper_clz()

        class range_wrapper_clz:

            @functools.wraps(_orig_range)
            def __call__(self, *args):
                if False:
                    i = 10
                    return i + 15
                assert 1 <= _orig_len(args) <= 3
                args = (arg.value if _orig_isinstance(arg, ep.ConcreteProxy) else arg for arg in args)
                return _orig_range(*args)

            def __eq__(self, __o: object) -> bool:
                if False:
                    return 10
                return id(__o) in (id(self), id(_orig_range))

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return id(self)
        range_wrapper = range_wrapper_clz()

        class enumerate_wrapper_clz:

            @functools.wraps(_orig_enumerate)
            def __call__(self, iterable, start=0):
                if False:
                    return 10
                count = start
                for elem in iterable:
                    if _orig_isinstance(elem, ep.ConcreteProxy) and _orig_isinstance(elem.value, (_orig_int, str)):
                        yield (count, elem.value)
                    else:
                        yield (count, elem)
                    count += 1

            def __eq__(self, __o: object) -> bool:
                if False:
                    i = 10
                    return i + 15
                return id(__o) in (id(self), id(_orig_enumerate))

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return id(self)
        enumerate_wrapper = enumerate_wrapper_clz()

        class type_wrapper_clz:

            @functools.wraps(_orig_type)
            def __call__(self, instance):
                if False:
                    return 10
                orig_type = _orig_type(instance)
                if orig_type in (ep.ConcreteProxy, ep.ConcreteAttrProxy, ep.ConcreteUnpackIterProxy):
                    return _orig_type(instance.value)
                else:
                    return orig_type

            def __eq__(self, __o: object) -> bool:
                if False:
                    print('Hello World!')
                return id(__o) in (id(self), id(_orig_enumerate))

            def __hash__(self):
                if False:
                    return 10
                return id(self)
        type_wrapper = type_wrapper_clz()

        @classmethod
        @functools.wraps(_orig_agfunc_apply)
        def agfunc_apply_wrapper(clz, *args, **kwargs):
            if False:
                print('Hello World!')
            if clz not in self.agfunc_dict:
                self.agfunc_dict[clz] = torch._C._FunctionBase.__dict__['apply'].__get__(None, clz)
            if self.temp_disable_agfunc_apply or self.temp_disable_call:
                return self.agfunc_dict[clz](*args, **kwargs)
            tracers = _orig_set()

            def unwrap_detect_tracers(obj):
                if False:
                    print('Hello World!')
                if isinstance(obj, ep.ConcreteProxy):
                    tracers.add(obj.tracer)
            ep.map_aggregate_not_proxy(args, unwrap_detect_tracers)
            ep.map_aggregate_not_proxy(kwargs, unwrap_detect_tracers)
            if _orig_len(tracers) == 0:
                return self.agfunc_dict[clz](*args, **kwargs)
            elif _orig_len(tracers) == 1 and next(iter(tracers)) == self:
                return self.create_proxy('call_function', self.agfunc_dict[clz], args, kwargs)
            else:
                raise Exception('more than 1 tracer detected. please report the issue')

        @functools.wraps(_orig_torch_assert)
        def torch_assert_wrapper(condition, message):
            if False:
                i = 10
                return i + 15
            while _orig_isinstance(condition, ep.ConcreteProxy):
                condition = condition.value
            return _orig_torch_assert(condition, message)
        self.agfunc_dict: dict[Type, Any] = {}
        self.autowrap_leaf_pairs = {id(_orig_torch_assert): torch_assert_wrapper}
        self.wrapped_leaf = dict()
        for (func, (positions, is_force_trace, to_func)) in self.autowrap_leaf_function.items():
            if _orig_isinstance(func, BuiltinMethodType) and getattr(func, '__name__', None) == 'apply' and _orig_isinstance(getattr(func, '__self__', None), Type) and issubclass(func.__self__, torch.autograd.Function):
                assert to_func == None, '<subclass of torch.autograd.Function>.apply should set to_func to None!'
                if func.__self__ not in self.agfunc_dict:
                    self.agfunc_dict[func.__self__] = _create_wrapped_leaf_func(self, func, func)
                wrapped = self.agfunc_dict[func.__self__]
            elif func.__qualname__.startswith('_TensorBase'):
                positions = (*positions, (torch.Tensor, func.__name__))
                wrapped = _create_wrapped_leaf_method(self, getattr(torch.Tensor, func.__name__), func.__name__, to_func)
            elif func.__qualname__.startswith('_VariableFunctionsClass'):
                if hasattr(torch, func.__name__):
                    positions = (*positions, (torch, func.__name__))
                if is_force_trace:
                    wrapped = _create_wrapped_leaf_func(self, func, to_func, (self,))
                else:
                    wrapped = _create_wrapped_leaf_func(self, func, to_func)
            elif _orig_isinstance(func, (MethodDescriptorType, MethodWrapperType)):
                wrapped = _create_wrapped_leaf_method(self, func, func.__name__, to_func)
            elif func.__name__ != func.__qualname__ and func.__qualname__ != 'boolean_dispatch.<locals>.fn':
                if func.__module__.startswith('_') and func.__module__ != '__main__':
                    path = sys.modules[func.__module__[1:]]
                else:
                    path = sys.modules[func.__module__]
                path = getattr(path, func.__qualname__.split('.')[0])
                positions = (*positions, (path, func.__name__))
                wrapped = _create_wrapped_leaf_method(self, func, func.__name__, to_func)
            else:
                if func.__module__.startswith('_') and func.__module__ != '__main__':
                    path = sys.modules[func.__module__[1:]]
                else:
                    path = sys.modules[func.__module__]
                positions = (*positions, (path, func.__name__))
                if is_force_trace:
                    wrapped = _create_wrapped_leaf_func(self, func, to_func, (self,))
                else:
                    wrapped = _create_wrapped_leaf_func(self, func, to_func)
            self.wrapped_leaf[func] = (positions, wrapped)
        self.clz_wrapper_map: Dict[Any, Type] = {map_wrapper: _orig_map, enumerate_wrapper: _orig_enumerate, range_wrapper: _orig_range, type_wrapper: _orig_type}
        for (clz, (positions, is_iterable)) in self.autowrap_leaf_class.items():
            if clz.__module__.startswith('_') and clz.__module__ != '__main__':
                path = sys.modules[clz.__module__[1:]]
            else:
                path = sys.modules[clz.__module__]
            if is_iterable:
                wrapped = _create_wrapped_leaf_iterable_class(self, clz)
            else:
                wrapped = _create_wrapped_leaf_class(self, clz)
            positions = (*positions, (path, clz.__name__))
            self.wrapped_leaf[clz] = (positions, wrapped)
            self.clz_wrapper_map[wrapped] = clz
        for clz in self.fake_middle_class:
            wrapped = _create_wrapped_attr_for_middle_class(self, clz, self.the_path_of_middle_class)
            self.wrapped_leaf[clz.__getattribute__] = (((clz, '__getattribute__'),), wrapped)

        @functools.wraps(_orig_isinstance)
        def isinstance_wrapper(instance, clz):
            if False:
                while True:
                    i = 10
            if _orig_type(clz) in (slice, tuple, list, _orig_slice, _orig_tuple, _orig_list):
                clz_wrapped = []
                for (wrapped_type, orig_type) in self.clz_wrapper_map.items():
                    if wrapped_type in clz:
                        clz_wrapped.append(orig_type)
                clz = (*clz_wrapped, *(aclz for aclz in clz if aclz not in self.clz_wrapper_map))
                for cls in (object, ep.ConcreteProxy, ep.ConcreteAttrProxy, ep.ConcreteUnpackIterProxy):
                    if cls in clz and _orig_isinstance(instance, cls):
                        return True
                if _orig_isinstance(instance, ep.ConcreteProxy):
                    return _orig_isinstance(instance.value, clz)
                else:
                    return _orig_isinstance(instance, clz)
            else:
                if clz in (object, ep.ConcreteProxy, ep.ConcreteAttrProxy, ep.ConcreteUnpackIterProxy):
                    return _orig_isinstance(instance, clz)
                if clz in self.clz_wrapper_map:
                    clz = self.clz_wrapper_map[clz]
                if _orig_isinstance(instance, ep.ConcreteProxy):
                    instance = instance.value
                return _orig_isinstance(instance, clz)

        @functools.wraps(_orig_issubclass)
        def issubclass_wrapper(subclass, clz):
            if False:
                i = 10
                return i + 15
            if _orig_type(clz) in (slice, tuple, list, _orig_slice, _orig_tuple, _orig_list):
                clz_wrapped = []
                for (wrapped_type, orig_type) in self.clz_wrapper_map.items():
                    if wrapped_type in clz:
                        clz_wrapped.append(orig_type)
                clz = (*clz_wrapped, *(aclz for aclz in clz if aclz not in self.clz_wrapper_map))
                return _orig_issubclass(subclass, clz)
            else:
                if clz in self.clz_wrapper_map:
                    clz = self.clz_wrapper_map[clz]
                return _orig_issubclass(subclass, clz)

        @functools.wraps(_orig_getattr)
        def getattr_wrapper(obj, *args):
            if False:
                i = 10
                return i + 15
            if not 1 <= _orig_len(args) <= 2:
                raise Exception()
            args = _orig_list(args)
            if _orig_isinstance(args[0], ep.ConcreteProxy):
                args[0] = args[0].value
            return _orig_getattr(obj, *args)
        self.temp_disable_call = False
        self.temp_disable_attr = False
        self.temp_disable_agfunc_apply = False
        self.temp_disable_call_level = 0
        self.temp_disable_attr_level = 0
        self.temp_disable_agfunc_apply_level = 0
        try:
            with _Patcher() as self.patcher:
                self.patcher.patch_method(torch.nn.Module, '__getattribute__', module_getattribute_wrapper, deduplicate=False)
                self.patcher.patch_method(torch.nn.Module, '__call__', module_call_wrapper, deduplicate=False)
                self.patcher.patch_method(torch.autograd.Function, 'apply', agfunc_apply_wrapper, deduplicate=False)
                self.patcher.patch_method(torch, '_assert', torch_assert_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'map', map_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'enumerate', enumerate_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'range', range_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'type', type_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'isinstance', isinstance_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'issubclass', issubclass_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, 'getattr', getattr_wrapper, deduplicate=False)
                for (obj, (positions, wrapped)) in self.wrapped_leaf.items():
                    for (path, name) in positions:
                        self.patcher.patch_method(path, name, wrapped, deduplicate=False)
                    self.autowrap_leaf_pairs[id(obj)] = wrapped
                _patch_wrapped_functions(self.patcher)
                _autowrap_check(self, fn_globals, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                for module in self._autowrap_search:
                    _autowrap_check(self, module.__dict__, self._autowrap_function_ids, self.autowrap_leaf_pairs, self.agfunc_dict)
                with OperatorPatcherContext(self, use_operator_patch, operator_patch_backlist):
                    self.create_node('output', 'output', (self.create_arg(OperatorPatcherContext.patch_run(fn, *args, *more_args, **kwargs)),), {}, type_expr=fn.__annotations__.get('return', None))
        finally:
            delattr(torch.autograd.Function, 'apply')
            _retain_weight_consistency(self.root)
            pass
        self.submodule_paths = None
        return self.graph
_wrapped_fns_to_patch: List[Tuple[dict, str]] = []
_wrapped_methods_to_patch: List[Tuple[type, str]] = []

def _find_proxy(*objects_to_search):
    if False:
        i = 10
        return i + 15
    '\n    Recursively search a data structure for a Proxy() and return it,\n    return None if not found.\n    '
    proxy = None

    def find_proxy(x):
        if False:
            print('Hello World!')
        nonlocal proxy
        if isinstance(x, ep.ConcreteProxy):
            proxy = x
    ep.map_aggregate_not_proxy(objects_to_search, find_proxy)
    return proxy

def _create_wrapped_func(orig_fn):
    if False:
        return 10

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        if False:
            return 10
        '\n        Given an closed-over ``orig_function`` to invoke, search the args and kwargs for\n        a Proxy object. If there is one, emit a ``call_function`` node to preserve the\n        call to this leaf function directly. Otherwise, just return the results of\n        this function call, as this function is not being traced.\n        '
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy('call_function', orig_fn, args, kwargs)
            return_proxy.node.meta['is_wrapped'] = True
            return return_proxy
        return orig_fn(*args, **kwargs)
    return wrapped

def _patch_wrapped_functions(patcher: _Patcher):
    if False:
        print('Hello World!')
    '\n    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap\n    the listed global functions in the `_create_wrapped_func` wrapper.\n    '
    for (frame_dict, name) in _wrapped_fns_to_patch:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = _orig_getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))
    for (cls, name) in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))

def _autowrap_check(tracer: ConcreteTracer, frame_dict: Dict[str, Any], function_ids: Set[int], function_pairs: Dict[int, Callable], agfunc_dict: dict[Type, Any]):
    if False:
        while True:
            i = 10
    '\n    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.\n    This method searches a scope for them and patches them if found.\n    '
    patcher = tracer.patcher
    if patcher.visit_once(frame_dict):
        for (name, value) in frame_dict.items():
            if callable(value) and (not name.startswith('__')) and (not name.startswith('_orig_')):
                if id(value) in function_ids:
                    patcher.patch(frame_dict, name, _create_wrapped_func(value))
                elif id(value) in function_pairs:
                    patcher.patch(frame_dict, name, function_pairs[id(value)])
                elif _orig_isinstance(value, BuiltinMethodType) and getattr(value, '__name__', None) == 'apply' and _orig_isinstance(getattr(value, '__self__', None), Type) and issubclass(value.__self__, torch.autograd.Function):
                    if value.__self__ not in agfunc_dict:
                        agfunc_dict[value.__self__] = _create_wrapped_leaf_func(tracer, value, value)
                    patcher.patch(frame_dict, name, agfunc_dict[value.__self__])

def _create_wrapped_method(cls, name):
    if False:
        while True:
            i = 10
    orig_fn = _orig_getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Search the args and kwargs for a Proxy object. If there is one,\n        emit a ``call_method`` node to preserve the call to this method\n        directly. Otherwise, just return the results of this function\n        call, as this function is not being traced.\n        '
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy('call_method', name, args, kwargs)
        return orig_fn(*args, **kwargs)
    return wrapped

@compatibility(is_backward_compatible=True)
class GraphAppendingConcreteTracer(ConcreteTracer):

    def __init__(self, graph: Graph):
        if False:
            return 10
        super().__init__()
        self.graph = graph

class MagicMethodPatcher:
    from torch.fx import graph as fx_graph
    from torch.fx import graph_module as fx_graph_module
    from torch.fx import node as fx_node
    magic_methods_ori = fx_graph.magic_methods
    magic_methods_new = {**fx_graph.magic_methods, 'not_': 'not {}', 'is_': '{} is {}', 'is_not': '{} is not {}', 'contains': '{1} in {0}'}
    copy_attr_ori: Any = fx_graph_module._copy_attr
    find_module_of_method_ori: Any = fx_node._find_module_of_method
    format_import_statement_ori: Any = fx_graph_module._format_import_statement

    @staticmethod
    def copy_attr_new(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
        if False:
            return 10
        (*prefix, field) = target.split('.')
        for item in prefix:
            f = getattr(from_module, item)
            t = getattr(to_module, item, None)
            if f is t:
                return
            if t is None:
                if isinstance(f, Sequential):
                    t = Sequential()
                elif isinstance(f, ModuleList):
                    t = ModuleList()
                elif isinstance(f, ModuleDict):
                    t = ModuleDict()
                else:
                    t = torch.nn.Module()
                if hasattr(f, '_get_name'):
                    t._get_name = f._get_name
                to_module.add_module(item, t)
            (from_module, to_module) = (f, t)
        orig = getattr(from_module, field)
        if isinstance(orig, torch.Tensor) and (not isinstance(orig, torch.nn.Parameter)):
            to_module.register_buffer(field, orig)
        else:
            setattr(to_module, field, orig)

    @staticmethod
    def find_module_of_method_new(orig_method: Callable[..., Any]) -> str:
        if False:
            print('Hello World!')
        name = orig_method.__name__
        module = orig_method.__module__
        if module is not None:
            return module
        elif hasattr(orig_method, '__qualname__') and isinstance(orig_method.__qualname__, str) and orig_method.__qualname__.startswith('_VariableFunctionsClass.'):
            return 'torch._C._VariableFunctions'
        elif hasattr(orig_method, '__self__') and isinstance(orig_method.__self__, Type) and issubclass(orig_method.__self__, torch.autograd.Function):
            return f'{orig_method.__self__.__module__}.{orig_method.__self__.__name__}'
        for guess in [torch, getattr(torch.nn, 'functional')]:
            if getattr(guess, name, None) is orig_method:
                return guess.__name__
        raise RuntimeError(f'cannot find module for {orig_method}')

    @staticmethod
    def format_import_statement_new(name: str, obj: Any, importer) -> str:
        if False:
            print('Hello World!')
        if isinstance(obj, BuiltinMethodType) and getattr(obj, '__name__', None) == 'apply' and isinstance(getattr(obj, '__self__', None), Type) and issubclass(obj.__self__, torch.autograd.Function):
            return MagicMethodPatcher.format_import_statement_ori(name, obj.__self__, importer) + f'\n{name} = {name}.apply'
        return MagicMethodPatcher.format_import_statement_ori(name, obj, importer)

    def __enter__(self):
        if False:
            while True:
                i = 10
        MagicMethodPatcher.fx_graph.magic_methods = self.magic_methods_new
        MagicMethodPatcher.fx_graph_module._copy_attr = self.copy_attr_new
        MagicMethodPatcher.fx_node._find_module_of_method = self.find_module_of_method_new
        MagicMethodPatcher.fx_graph_module._format_import_statement = self.format_import_statement_new
        MagicMethodPatcher.available = True

    def __exit__(self, exc_type, exc_value, tb):
        if False:
            i = 10
            return i + 15
        MagicMethodPatcher.fx_graph.magic_methods = MagicMethodPatcher.magic_methods_ori
        MagicMethodPatcher.fx_graph_module._copy_attr = MagicMethodPatcher.copy_attr_ori
        MagicMethodPatcher.fx_node._find_module_of_method = MagicMethodPatcher.find_module_of_method_ori
        MagicMethodPatcher.fx_graph_module._format_import_statement = MagicMethodPatcher.format_import_statement_ori
        MagicMethodPatcher.available = False
        return exc_type is None

def _create_wrapped_leaf_func(tracer: ConcreteTracer, func: Callable, to_func: Optional[Callable], init_tracers=()):
    if False:
        return 10
    if to_func is None:
        to_func = func

    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        if tracer.temp_disable_call:
            return func(*args, **kwargs)
        tracers = _orig_set(init_tracers)

        def unwrap_detect_tracers(obj):
            if False:
                return 10
            if isinstance(obj, ep.ConcreteProxy):
                tracers.add(obj.tracer)
        ep.map_aggregate_not_proxy(args, unwrap_detect_tracers)
        ep.map_aggregate_not_proxy(kwargs, unwrap_detect_tracers)
        if _orig_len(tracers) == 0:
            return to_func(*args, **kwargs)
        elif _orig_len(tracers) == 1 and next(iter(tracers)) == tracer:
            return tracer.create_proxy('call_function', to_func, args, kwargs)
        else:
            raise Exception('more than 1 tracer detected. please report the issue')
    return func_wrapper

def _create_wrapped_leaf_method(tracer: ConcreteTracer, method, name: str, to_func: Optional[Callable]):
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(method)
    def method_wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        if tracer.temp_disable_call:
            return method(*args, **kwargs)
        tracers = _orig_set()

        def unwrap_detect_tracers(obj):
            if False:
                print('Hello World!')
            if isinstance(obj, ep.ConcreteProxy):
                tracers.add(obj.tracer)
        ep.map_aggregate_not_proxy(args, unwrap_detect_tracers)
        ep.map_aggregate_not_proxy(kwargs, unwrap_detect_tracers)
        if _orig_len(tracers) == 0:
            return method(*args, **kwargs)
        elif _orig_len(tracers) == 1 and next(iter(tracers)) == tracer:
            if to_func is not None:
                return tracer.create_proxy('call_function', to_func, args, kwargs)
            else:
                return tracer.create_proxy('call_method', name, args, kwargs)
        else:
            raise Exception('more than 1 tracer detected. please report the issue')
    return method_wrapper

def _create_wrapped_leaf_class(tracer: ConcreteTracer, clz):
    if False:
        for i in range(10):
            print('nop')

    class clz_wrapper_clz:

        @functools.wraps(clz)
        def __call__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if tracer.temp_disable_call:
                return clz(*args, **kwargs)
            tracers = _orig_set()

            def unwrap_detect_tracers(obj):
                if False:
                    return 10
                if isinstance(obj, ep.ConcreteProxy):
                    tracers.add(obj.tracer)
            ep.map_aggregate_not_proxy(args, unwrap_detect_tracers)
            ep.map_aggregate_not_proxy(kwargs, unwrap_detect_tracers)
            if _orig_len(tracers) == 0:
                return clz(*args, **kwargs)
            elif _orig_len(tracers) == 1 and next(iter(tracers)) == tracer:
                return tracer.create_proxy('call_function', clz, args, kwargs)
            else:
                raise Exception('more than 1 tracer detected. please report the issue')

        def __eq__(self, __o: object) -> bool:
            if False:
                i = 10
                return i + 15
            return id(__o) in (id(self), id(clz))

        def __hash__(self):
            if False:
                while True:
                    i = 10
            return id(self)
    return clz_wrapper_clz()

def _create_wrapped_leaf_iterable_class(tracer: ConcreteTracer, clz):
    if False:
        return 10

    class clz_wrapper_clz:

        @functools.wraps(clz)
        def __call__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if tracer.temp_disable_call:
                return clz(*args, **kwargs)
            tracers = _orig_set()
            if _orig_len(args) != 0:
                if _orig_isinstance(args[0], ep.Proxy):
                    tracers.add(args[0].tracer)
                if _orig_isinstance(args[0], Iterator):
                    args = (clz(args[0]), *args[1:])
                if _orig_isinstance(args[0], Iterable):
                    for item in args[0]:
                        if _orig_isinstance(item, ep.Proxy):
                            tracers.add(item.tracer)
            if _orig_len(tracers) == 0:
                return clz(*args, **kwargs)
            elif _orig_len(tracers) == 1 and next(iter(tracers)) == tracer:
                return tracer.create_proxy('call_function', clz, args, kwargs)
            else:
                raise Exception('more than 1 tracer detected. please report the issue')

        def __eq__(self, __o: object) -> bool:
            if False:
                return 10
            return id(__o) in (id(self), id(clz))

        def __hash__(self):
            if False:
                print('Hello World!')
            return id(self)
    clz_wrapper = clz_wrapper_clz()
    for name in dir(clz):
        attr = _orig_getattr(clz, name)
        if not name.startswith('_') or name in ('__getitem__', '__setitem__', '__iter__', '__len__'):
            if _orig_isinstance(attr, Callable):
                setattr(clz_wrapper, name, _create_wrapped_leaf_method(tracer, attr, name, None))
            else:
                setattr(clz_wrapper, name, attr)
    return clz_wrapper

def _create_wrapped_attr_for_middle_class(tracer: ConcreteTracer, clz, the_path_of_middle_class):
    if False:
        return 10
    _orig_clz_getattribute = clz.__getattribute__
    if hasattr(clz, '__getattr__'):
        _orig_clz_getattr = clz.__getattr__
    else:
        _orig_clz_getattr = None

    @functools.wraps(_orig_clz_getattribute)
    def clz_getattr_wrapper(obj, attr):
        if False:
            print('Hello World!')
        if tracer.temp_disable_call | tracer.temp_disable_attr:
            if _orig_clz_getattr == None:
                return _orig_clz_getattribute(obj, attr)
            else:
                try:
                    return _orig_clz_getattribute(obj, attr)
                except AttributeError:
                    return _orig_clz_getattr(obj, attr)
        else:
            return tracer.create_proxy('get_attr', f'{the_path_of_middle_class[id(obj)]}.{attr}', (), {})
    return clz_getattr_wrapper

def _retain_weight_consistency(root: torch.nn.Module):
    if False:
        print('Hello World!')
    _flag = 0
    for module in root.modules():
        for (name, param) in module.named_parameters():
            if _orig_isinstance(param, ep.ConcreteProxy):
                param: ep.ConcreteProxy
                _logger.warning(f'Parameter {name} of {module} is a ConcreteProxy. Some weight may be modified inplace within forward().')
                setattr(module, name, param.value)
                _flag |= 1
        for (name, buffer) in module.named_buffers():
            if _orig_isinstance(buffer, ep.ConcreteProxy):
                buffer: ep.ConcreteProxy
                _logger.warning(f'Buffer {name} of {module} is a ConcreteProxy. Some buffer may be modified inplace within forward().')
                setattr(module, name, buffer.value)
                _flag |= 1
    if _flag:
        _logger.warning('Some weight or buffer is modified inplace within forward(). This may cause unexpected behavior. ``concrete_trace`` may not guarantee the consistency of the traced graph.')
    return root

@functools.wraps(_orig_node_is_impure)
def node_is_impure_wrapper(node):
    if False:
        print('Hello World!')
    if node.op in {'placeholder', 'output'}:
        return True
    if node.op == 'call_function':
        return node.target in _side_effectful_functions
    if node.op == 'call_method':
        return node.target.endswith('_')
    if node.op == 'call_module':
        assert node.graph.owning_module is not None, 'self.graph.owning_module not set for purity check'
        target_mod = node.graph.owning_module.get_submodule(node.target)
        assert target_mod is not None, f'Did not find expected submodule target {node.target}'
        return getattr(target_mod, '_is_impure', False)
    return False

def concrete_trace(root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Union[Dict[str, Any], Tuple], *, use_operator_patch: bool=True, operator_patch_backlist: List[str] | None=None, forward_function_name: str='forward', check_args: Optional[Dict[str, Any]]=None, autowrap_leaf_function=None, autowrap_leaf_class=None, leaf_module: Tuple | None=None, fake_middle_class=None, dce=True, cpu_offload=False, trace_twice=False) -> GraphModule:
    if False:
        return 10
    "\n    Concrete tracing API\n\n    Given an ``nn.Module`` or function instance ``root`` and a dummy input `concrete_args`, this function will return a ``GraphModule``\n    constructed by recording operations seen while tracing through ``root``.\n\n    It has solved many problems compared to fx.symbolic_trace, and can execute on many third-party models.\n\n    For example::\n\n        def f(a, b):\n            return a + b\n\n        traced_f = concrete_trace(f, concrete_args={'a': 1, 'b': 2})\n        # or `traced_f = concrete_trace(f, (1, 2))`\n        assert traced_f(3, 4) == 7\n\n        def f(x):\n            out1, out2 = 0, 0\n            for k, v in x.items():\n                out1 += k\n                out2 += v\n            return out1, out2\n        traced_f = concrete_trace(f, ({1: 1, 2: 2}, ))\n        assert traced_f({2: 3, 4: 5}) == (6, 8)\n\n    Note that we can only record static structure, so all the branches such as if-else or loop will be flattened::\n\n        def f(x):\n            out1, out2 = 0, 0\n            for k, v in x.items():\n                out1 += k\n                out2 += v\n            return out1, out2\n        traced_f = concrete_trace(f, ({1: 1, 2: 2}, ))\n        assert traced_f({2: 3, 4: 5, 6:7}) == (6, 8) # not (12, 15)\n\n        # traced code like:\n        def traced_f(self, x):\n            out1, out2 = 0, 0\n            items = x.items()\n\n            # for loop\n            iter = iter(items)\n\n            # first loop content\n            items0 = next(iter)\n            out1 += items0[0]\n            out2 += items0[1]\n\n            # second loop content\n            items1 = next(iter)\n            out1 += items1[0]\n            out2 += items1[1]\n\n            return (out1, out2)\n\n    If you want to trace 'is', 'is not', 'in' or 'not in' in your module, you can set use_function_patch to True::\n\n        def f(x, y):\n            if x is None:\n                return y\n            else:\n                return x - y\n        # traced_f = concrete_trace(f, (None, 1)) # bad\n        traced_f = concrete_trace(f, (None, 1), use_function_patch=True) # f should exist in a file.\n\n    If you have a function/method that should be treated as a leaf function but not trace into it, use autowrap_leaf_function to mark it::\n\n        def leaf_op(x, y, z):\n            # if not treated as a leaf function, then only 1 branch will exist.\n            if x > 0:\n                return y + z\n            else:\n                return y - z\n\n        def f(x):\n            return leaf_op(x, 3, 2)\n\n        traced_f = concrete_trace(f, (1, ), autowrap_leaf_function = {\n            leaf_op: ([], False, None), **ConcreteTracer.default_autowrap_leaf_function})\n        assert traced_f(1) == 5 and traced_f(-1) == 1\n\n    If you have a class that should be treated as a leaf class, use autowrap_leaf_class to mark it::\n\n        class leaf_clz:\n            def __init__(self, a, b):\n                self.c = a + b\n\n        def f(x, y):\n            return leaf_clz(x, y)\n\n        traced_f = concrete_trace(f, (1, 2), autowrap_leaf_class = {\n            leaf_clz: ([], False), **ConcreteTracer.default_autowrap_leaf_class})\n        assert isinstance(traced_f(3, 4), leaf_clz) and traced_f(3, 4).c == 7\n\n    Args:\n        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted into a Graph representation.\n        concrete_args (Union[Dict[str, Any], Tuple]): Dummy inputs to do concrete trace.\n\n        use_function_patch (bool): Use operator patcher recursively on function calls. Operator patcher will re-compile the function and\n            translate '{} is {}' into 'operator.is_({}, {})', then we can treat 'is', 'is not', 'in' and 'not in' as function calls.\n\n        operator_patch_backlist (List[str]): Blacklist of the operator patcher.\n\n        autowrap_leaf_function (Dict[Any, Tuple[List[Tuple[Union[ModuleType, Type], str]], bool, Optional[Callable]]]): Leaf function dict,\n            such as 'add' or 'torch.xxx'. You can add your own leaf functions.\n\n            The struct of dict is: leaf_function: ([(module_path, module_name)], force_to_trace, replace_to_function).\n                (module_path, module_name): The place the function exists. Such as torch.meshgrid, there are `torch.meshgrid`,\n                    'torch.functional.meshgrid', 'torch._C._VariableFunctions.meshgrid', we should wrap them all.\n                force_to_trace: If set to false, the function will only be traced if input relates to concrete_args.\n                    Such as 'torch.rand', we should trace it even if it doesn't relate to concrete_args.\n                replace_to_function: If not `None`, we will use it to replace the original function in traced code.\n                    Such as ModuleList.__getitem__, we can use operator.getitem to replace it.\n\n        default_autowrap_leaf_class (Dict[Type, Tuple[List[Tuple[Union[ModuleType, Type], str]], bool]]): Leaf class dict, such as 'int',\n            'range' or 'zip'. You can add your own leaf functions such as 'torch.finfo' or 'modeling_outputs.SequenceClassifierOutput'.\n\n            The struct of dict is: leaf_class: ([(module_path, module_name)], is_iterator_class).\n                is_iterator_class: Is the class init from an iterator. Only 'tuple', 'list', 'set' or 'dict' needs to set it to True.\n\n        cpu_offload (bool): Whether to offload the module to CPU during tracing. If set to True, the traced code will be executed on GPU,\n            but is offloaded to CPU afterward. This is useful for reducing memory usage during tracing, but may cause performance issues.\n            If set to False, there will be no offloading during tracing, but the traced code will be executed on default device.\n\n    Returns:\n        fx.GraphModule: a Module created from the recorded operations from ``root``.\n    "
    tracer = ConcreteTracer(cpu_offload=cpu_offload)
    is_training = root.training
    root.eval()
    graph = tracer.trace(root, autowrap_leaf_function=autowrap_leaf_function, autowrap_leaf_class=autowrap_leaf_class, leaf_module=leaf_module, fake_middle_class=fake_middle_class, concrete_args=concrete_args, use_operator_patch=use_operator_patch, operator_patch_backlist=operator_patch_backlist, forward_function_name=forward_function_name)
    if trace_twice:
        graph_check = tracer.trace(root, autowrap_leaf_function=autowrap_leaf_function, autowrap_leaf_class=autowrap_leaf_class, leaf_module=leaf_module, fake_middle_class=fake_middle_class, concrete_args=concrete_args, use_operator_patch=use_operator_patch, operator_patch_backlist=operator_patch_backlist, forward_function_name=forward_function_name)
        assert len(graph.nodes) == len(graph_check.nodes), f'number nodes: {len(graph.nodes)} vs {len(graph_check.nodes)}'
        for (node_a, node_b) in zip(graph.nodes, graph_check.nodes):
            node_a: Node
            node_b: Node
            target_a = node_a.target
            target_b = node_b.target
            if node_a.op == 'get_attr' and node_a.name.startswith('_tensor_constant'):
                assert node_b.op == 'get_attr' and node_b.name.startswith('_tensor_constant')
                assert torch.equal(getattr(root, node_a.name), getattr(root, node_b.name))
            elif node_a.op == 'call_function' and isinstance(target_a, Callable) and (target_a.__name__ == 'apply') and hasattr(target_a, '__self__') and issubclass(target_a.__self__, torch.autograd.Function):
                assert node_b.op == 'call_function' and isinstance(target_b, Callable) and (target_b.__name__ == 'apply') and hasattr(target_b, '__self__') and issubclass(target_b.__self__, torch.autograd.Function)
            else:
                assert node_a.op == node_b.op and target_a == target_b, f'op: {node_a.op} vs {node_b.op}, target: {target_a} vs {target_b}'
    with MagicMethodPatcher():
        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        traced = GraphModule(tracer.root, graph, name)
        if dce:
            with _Patcher() as patcher:
                patcher.patch_method(Node, 'is_impure', node_is_impure_wrapper, deduplicate=False)
                traced.graph.eliminate_dead_code()
            traced.recompile()
    if check_args is not None:
        assert root(**check_args) == traced(**check_args)
    if is_training:
        root.train()
    return traced