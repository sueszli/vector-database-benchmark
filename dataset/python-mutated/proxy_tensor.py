import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher, enable_pre_dispatch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
from torch.utils._stats import count
import logging
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode, _push_mode
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary
__all__ = ['PythonKeyTracer', 'dispatch_trace', 'make_fx', 'DecompositionInterpreter', 'py_sym_types', 'get_innermost_proxy_mode']
aten = torch.ops.aten
prim = torch.ops.prim
log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, 'not_implemented')
CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OperatorBase, Callable] = {}
CONSTANT_NUMEL_LIMIT = 1
pytree._register_pytree_node(torch.Size, lambda x: (list(x), None), lambda xs, _: tuple(xs))

def fake_signature(fn, nargs):
    if False:
        print('Hello World!')
    'FX gets confused by varargs, de-confuse it'
    argnames = ','.join((f'arg{i}' for i in range(nargs)))
    return eval(f'lambda {argnames}: fn({argnames})', {'fn': fn})

@contextmanager
def decompose(decomposition_table):
    if False:
        for i in range(10):
            print('nop')
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table
proxy_slot = object()
no_default = object()
py_sym_types = (SymInt, SymFloat, SymBool)

def is_sym_node(node):
    if False:
        i = 10
        return i + 15
    assert hasattr(node, 'meta'), 'All nodes traced with proxy_tensor should have meta'
    return 'val' in node.meta and isinstance(node.meta['val'], py_sym_types)

def set_proxy_slot(obj, tracer, proxy):
    if False:
        print('Hello World!')
    if isinstance(obj, torch.Tensor):
        tracer.tensor_tracker[obj] = proxy
    else:
        assert isinstance(obj, SymNode), type(obj)
        if obj not in tracer.symnode_tracker:
            tracer.symnode_tracker[obj] = proxy

def has_proxy_slot(obj, tracer):
    if False:
        print('Hello World!')
    assert isinstance(obj, (torch.Tensor, SymNode)), type(obj)
    return get_proxy_slot(obj, tracer, False, lambda _: True)

def get_proxy_slot(obj, tracer, default=no_default, transform=lambda x: x):
    if False:
        return 10
    if isinstance(obj, torch.Tensor):
        tracker = tracer.tensor_tracker
    else:
        assert isinstance(obj, SymNode), type(obj)
        tracker = tracer.symnode_tracker
    if obj not in tracker:
        if default is no_default:
            raise RuntimeError(f'{obj} is not tracked with proxy for {tracer}')
        return default
    return transform(tracker[obj])

def snapshot_fake(val):
    if False:
        i = 10
        return i + 15
    return val.detach()

def extract_val(val):
    if False:
        print('Hello World!')
    if is_fake(val):
        return snapshot_fake(val)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, torch.Tensor):
        if not val.is_sparse:
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(val.shape, val.stride(), device=val.device, dtype=val.dtype)
        else:
            return None

def set_meta(proxy, val):
    if False:
        i = 10
        return i + 15
    proxy.node.meta['val'] = extract_val(val)
    if is_fake(val):
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    elif isinstance(val, torch.Tensor) and (not val.is_sparse):
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    return proxy

def thunkify(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Delays computation of f until it's called again\n    Also caches the result\n    "
    return functools.lru_cache(1)(functools.partial(f, *args, **kwargs))

def track_tensor(tensor, proxy, *, constant, tracer):
    if False:
        for i in range(10):
            print('nop')

    def try_set_proxy_slot(outer_s, proxy_callable, *args):
        if False:
            return 10
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            inner_s = outer_s.node
            set_proxy_slot(inner_s, tracer, thunkify(proxy_callable, outer_s, *args))
    for (i, s) in enumerate(tensor.shape):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_size.int(proxy, i), x), i)
    for (i, s) in enumerate(tensor.stride()):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_stride.int(proxy, i), x), i)
    try_set_proxy_slot(tensor.numel(), lambda x: set_meta(torch.ops.aten.sym_numel.default(proxy), x))
    try_set_proxy_slot(tensor.storage_offset(), lambda x: set_meta(torch.ops.aten.sym_storage_offset.default(proxy), x))
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))

def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):
    if False:
        for i in range(10):
            print('nop')

    def wrap_with_proxy(e, proxy, constant):
        if False:
            return 10
        if isinstance(e, torch.Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            set_meta(proxy, e)
            set_proxy_slot(e.node, tracer, lambda : proxy)
        elif isinstance(e, (tuple, list)):
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)
            for (idx, ee) in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], get_constant(idx))
        elif isinstance(e, dict):
            assert constant is None
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)
            for (key, val) in e.items():
                wrap_with_proxy(val, proxy[key], None)
        else:
            pass

    def get_constant(idx):
        if False:
            print('Hello World!')
        if constant is None:
            return None
        else:
            return constant[idx]
    wrap_with_proxy(inner_res, proxy_res, constant)
    return inner_res

def maybe_disable_fake_tensor_mode():
    if False:
        print('Hello World!')
    return unset_fake_temporarily()

@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[torch.Tensor]

def fetch_sym_proxy(tracer):
    if False:
        for i in range(10):
            print('nop')

    def inner(e):
        if False:
            print('Hello World!')
        n = e.node
        if n.constant is not None:
            return n.constant
        else:
            return get_proxy_slot(n, tracer)()
    return inner

def fetch_tensor_proxy(tracer):
    if False:
        for i in range(10):
            print('nop')
    return lambda t: get_proxy_slot(t, tracer, t)
HANDLED_TYPES = (torch.Tensor, torch.nn.Parameter, FakeTensor)

def proxy_call(proxy_mode, func, pre_dispatch, args, kwargs):
    if False:
        return 10
    unrecognized_types = []

    def can_handle_tensor(x):
        if False:
            return 10
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) in (torch._subclasses.FakeTensor,)
        if not r:
            unrecognized_types.append(type(x))
        return r
    if not pytree.tree_all_only(torch.Tensor, can_handle_tensor, (args, kwargs)):
        not_implemented_log.debug('ProxyTensorMode tensors without proxy had unrecognized subclasses: %s', unrecognized_types)
        return NotImplemented
    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        return r
    if not pre_dispatch and func not in [torch.ops.aten.size.default, torch.ops.aten.stride.default, torch.ops.aten.storage_offset.default]:
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r
    tracer = proxy_mode.tracer
    (f_args, f_kwargs) = pytree.tree_map_only(torch.Tensor, fetch_tensor_proxy(tracer), (args, kwargs))
    all_constant = pytree.tree_all_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs)) and pytree.tree_all_only((SymInt, SymFloat, SymBool), lambda _: False, (args, kwargs))
    if torch.Tag.data_dependent_output in func.tags:
        if all_constant:
            (const_args, const_kwargs) = pytree.tree_map_only(_ProxyTensor, lambda t: t.constant, (f_args, f_kwargs))
            with maybe_disable_fake_tensor_mode():
                return func(*const_args, **const_kwargs)
        if pytree.tree_all_only(torch.Tensor, lambda t: not is_fake(t), (args, kwargs)):
            raise RuntimeError(f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! It's likely that this is caused by data-dependent control flow or similar.  It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' in your make_fx call.")
    (proxy_args, proxy_kwargs) = pytree.tree_map_only((SymInt, SymFloat, SymBool), fetch_sym_proxy(proxy_mode.tracer), pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (f_args, f_kwargs)))
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default
    proxy_out = proxy_mode.tracer.create_proxy('call_function', func, proxy_args, proxy_kwargs, name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__))
    if func.overloadpacket.__name__[-1] == '_' and func.overloadpacket.__name__[0] != '_':
        if isinstance(args[0], List):
            for (i, a) in enumerate(args[0]):
                a.proxy = proxy_out[0][i]
        else:
            args[0].proxy = proxy_out
    out = func(*args, **kwargs)
    any_constant = pytree.tree_any_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))
    constant = None
    if func is torch.ops.aten.lift_fresh_copy.default and out.numel() <= CONSTANT_NUMEL_LIMIT:
        with maybe_disable_fake_tensor_mode():
            constant = args[0].clone()
    elif torch.Tag.nondeterministic_seeded not in func.tags and all_constant and any_constant and pytree.tree_all_only(torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out):
        with maybe_disable_fake_tensor_mode():
            (const_args, const_kwargs) = pytree.tree_map_only(_ProxyTensor, lambda t: t.constant, (f_args, f_kwargs))
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None
    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    return out

class PythonKeyTracer(Tracer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(autowrap_modules=())
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = weakref.WeakKeyDictionary()

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        return forward(*args, **kwargs)

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if False:
            print('Hello World!')
        return attr_val

    def create_arg(self, a: Any):
        if False:
            return 10
        if isinstance(a, torch.nn.Parameter):
            for (n, p) in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None
            if not qualname:
                i = 0
                while True:
                    qualname = f'_param_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        elif isinstance(a, (SymInt, SymFloat, SymBool)):
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)

    def unwrap_proxy(self, e):
        if False:
            print('Hello World!')
        if isinstance(e, torch.Tensor):
            return get_proxy_slot(e, self, e, lambda e: e.proxy)
        elif isinstance(e, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return get_proxy_slot(e.node, self, e, lambda e: e())
        else:
            return e

@torch._disable_dynamo
def dispatch_trace(root: Union[torch.nn.Module, Callable], tracer: Tracer, concrete_args: Optional[Tuple[Any, ...]]=None) -> GraphModule:
    if False:
        print('Hello World!')
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)

@contextlib.contextmanager
def _pop_proxy_mode_temporarily(dk):
    if False:
        return 10
    if dk is not None:
        old = _pop_mode(dk)
        try:
            yield old
        finally:
            _push_mode(old, dk)
    else:
        old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
        try:
            yield old
        finally:
            torch._C._set_dispatch_mode(old)

def wrap_key(f, tensors, tracer, pre_dispatch: bool):
    if False:
        print('Hello World!')
    (flat_tensors, tensors_spec) = pytree.tree_flatten(tensors)
    dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None

    @functools.wraps(f)
    def wrapped(*proxies):
        if False:
            return 10
        (flat_proxies, proxies_spec) = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        with _pop_proxy_mode_temporarily(dk) as m:
            assert isinstance(m, ProxyTorchDispatchMode)
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)
        out = f(*tensors)
        out = pytree.tree_map_only(torch.Tensor, lambda t: get_proxy_slot(t, tracer, t, lambda x: x.proxy), out)
        out = pytree.tree_map_only((SymInt, SymFloat, SymBool), lambda t: get_proxy_slot(t.node, tracer)(), out)
        return out
    return wrapped
ORIGINAL_ATEN = None

@contextmanager
def set_original_aten_op(func):
    if False:
        while True:
            i = 10
    global ORIGINAL_ATEN
    if ORIGINAL_ATEN is None and fx_traceback.has_preserved_node_meta():
        ORIGINAL_ATEN = func
        fx_traceback.current_meta['original_aten'] = func
        try:
            yield
        finally:
            ORIGINAL_ATEN = None
            fx_traceback.current_meta['original_aten'] = None
    else:
        yield

class PreDispatchTorchFunctionMode(TorchFunctionMode):

    def __init__(self, tracer):
        if False:
            for i in range(10):
                print('nop')
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        kwargs = kwargs or {}
        pre_dispatch_ops = [torch._C._set_grad_enabled, torch.amp._enter_autocast, torch.amp._exit_autocast]
        if func in pre_dispatch_ops:
            return self.tracer.create_node('call_function', func, args, {})
        return func(*args, **kwargs)

class ProxyTorchDispatchMode(TorchDispatchMode):

    def __init__(self, tracer, tracing_mode, pre_dispatch=False, _allow_fake_constant=False):
        if False:
            return 10
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.enable_tracing = True
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self.sym_mode = ProxySymDispatchMode(tracer)
        self.trace_state = {}
        self._managers = []
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        self.enter_stack: List[Optional[ProxyTorchDispatchMode]] = []

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            print('Hello World!')
        with self.sym_mode.enable(False), set_original_aten_op(func):
            return self.inner_torch_dispatch(func, types, args, kwargs)

    def __enter__(self):
        if False:
            print('Hello World!')
        m = self.sym_mode.enable(True)
        self._managers.append(m)
        m.__enter__()
        maybe_prev_proxy_mode = torch._C._unset_dispatch_mode(self._mode_key)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        m = self._managers.pop()
        b = super().__exit__(exc_type, exc_value, traceback)
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            torch._C._set_dispatch_mode(mb_previous_proxy_mode)
        if not b:
            return m.__exit__(exc_type, exc_value, traceback)
        else:
            return m.__exit__(None, None, None)

    def inner_torch_dispatch(self, func, types, args=(), kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.enable_tracing:
            return func(*args, **kwargs)
        if func in [prim.device.default]:
            return func(*args, **kwargs)
        return proxy_call(self, func, self.pre_dispatch, args, kwargs)

class ProxySymDispatchMode(SymDispatchMode):

    def __init__(self, tracer):
        if False:
            while True:
                i = 10
        super().__init__()
        self.tracer = tracer
        self.enable_tracing = True

    @contextmanager
    def enable(self, b):
        if False:
            while True:
                i = 10
        old = self.enable_tracing
        self.enable_tracing = b
        try:
            yield
        finally:
            self.enable_tracing = old

    def _compute_proxy(self, func, args, out: Union[SymInt, SymFloat, SymBool]):
        if False:
            while True:
                i = 10
        n_args = tuple((get_proxy_slot(a.node, self.tracer)().node if isinstance(a, py_sym_types) else a for a in args))
        n_out = self.tracer.create_node('call_function', func, n_args, {})
        p_out = fx.Proxy(n_out, self.tracer)
        set_meta(p_out, out)
        return p_out

    def __sym_dispatch__(self, func, types, args, kwargs):
        if False:
            return 10
        if not self.enable_tracing:
            return func(*args, **kwargs)
        if func == operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]
        assert not kwargs
        out = func(*args, **kwargs)
        if isinstance(out, py_sym_types):
            p_out_thunk = thunkify(self._compute_proxy, func=func, args=args, out=out)
            set_proxy_slot(out.node, self.tracer, p_out_thunk)
        return out

class DecompositionInterpreter(torch.fx.Interpreter):

    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        self.tracer.tensor_tracker = WeakTensorKeyDictionary()
        self.tracer.symnode_tracker = weakref.WeakKeyDictionary()
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode='real')

    def placeholder(self, target, args, kwargs):
        if False:
            while True:
                i = 10
        out = super().placeholder(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    def get_attr(self, target, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        out = super().get_attr(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    def output(self, target, args, kwargs):
        if False:
            print('Hello World!')
        out = super().output(target, args, kwargs)

        def unwrap(e):
            if False:
                i = 10
                return i + 15
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)
        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)

def wrapper_and_args_for_make_fx(func, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    (flat_args, spec) = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args):
        if False:
            print('Hello World!')
        (fn_args, fn_kwargs) = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)
    return (wrapped, flat_args)

@contextmanager
def disable_autocast_cache():
    if False:
        print('Hello World!')
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)

def make_fx(f, decomposition_table=None, tracing_mode='real', _allow_non_fake_inputs=False, *, pre_dispatch=False, _allow_fake_constant=False):
    if False:
        while True:
            i = 10
    assert tracing_mode in ['real', 'fake', 'symbolic']
    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        if False:
            while True:
                i = 10
        from .symbolic_shapes import ShapeEnv
        phs = pytree.tree_map(lambda _: fx.PH, args)
        fx_tracer = PythonKeyTracer()
        fake_tensor_mode: Any = nullcontext()
        if tracing_mode == 'real':
            fake_tensor_mode = nullcontext()
        elif tracing_mode == 'fake':
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True, allow_non_fake_inputs=_allow_non_fake_inputs, shape_env=ShapeEnv(), static_shapes=True)
        elif tracing_mode == 'symbolic':
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                shape_env = ShapeEnv()
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=_allow_non_fake_inputs, shape_env=shape_env)
            else:
                shape_env = fake_tensor_mode.shape_env
                assert shape_env is not None, "shape_env should be set if tracing with 'symbolic'"
        else:
            raise AssertionError(f'Unexpected tracing type: {tracing_mode}')
        python_dispatcher_mode: Any = nullcontext()
        pre_dispatch_mode: Any = nullcontext()
        if tracing_mode == 'symbolic' or pre_dispatch:
            python_dispatcher_mode = enable_python_dispatcher()
        if pre_dispatch:
            pre_dispatch_mode = enable_pre_dispatch()
        proxy_function_mode: Any = nullcontext()
        if pre_dispatch:
            proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)
        proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode, pre_dispatch=pre_dispatch, _allow_fake_constant=_allow_fake_constant)
        arg_count = 0

        def wrap_fake(x):
            if False:
                while True:
                    i = 10
            nonlocal arg_count
            if isinstance(x, torch.Tensor):
                from torch._dynamo.source import ConstantSource
                source = ConstantSource(f'input{arg_count}')
                arg_count += 1
                return fake_tensor_mode.from_tensor(x, source=source)
            return x
        sym_mode = proxy_mode.sym_mode
        wrap_fn_map = {'real': lambda x: x, 'fake': wrap_fake, 'symbolic': wrap_fake}
        args = pytree.tree_map(wrap_fn_map[tracing_mode], args)
        if not hasattr(inspect.unwrap(f), '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
            func = fake_signature(f, len(phs))
        else:
            func = f
        with decompose(decomposition_table), fake_tensor_mode, python_dispatcher_mode, pre_dispatch_mode, proxy_function_mode, sym_mode, proxy_mode, disable_autocast_cache():
            t = dispatch_trace(wrap_key(func, args, fx_tracer, pre_dispatch), tracer=fx_tracer, concrete_args=tuple(phs))
        if tracing_mode == 'symbolic':
            t.shape_env = shape_env
        return t
    return wrapped

def get_torch_dispatch_modes():
    if False:
        i = 10
        return i + 15
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()

def get_innermost_proxy_mode():
    if False:
        return 10
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)

@contextlib.contextmanager
def disable_proxy_modes_tracing(enable_current=False):
    if False:
        for i in range(10):
            print('nop')
    maybe_old = None
    if not enable_current:
        maybe_old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    try:
        yield
    finally:
        if maybe_old is not None:
            torch._C._set_dispatch_mode(maybe_old)

def maybe_handle_decomp(proxy_mode, op, args, kwargs):
    if False:
        return 10
    if op in CURRENT_DECOMPOSITION_TABLE:
        with proxy_mode:
            return CURRENT_DECOMPOSITION_TABLE[op](*args, **kwargs)
    return NotImplemented

def get_isolated_graphmodule(func, args, kwargs, tracing_mode='real'):
    if False:
        while True:
            i = 10
    "A helper function used to get the GraphModule for the given func.\n\n    It's expected to be used in the ProxyTensor tracing context.\n    It detaches the args and kwargs from the current tracer so that the trace of\n    the current graph module can be created without any side-effects.\n    "
    (wrapped, all_args) = wrapper_and_args_for_make_fx(func, args, kwargs)
    with disable_proxy_modes_tracing():
        gm = make_fx(wrapped, tracing_mode=tracing_mode)(all_args)
    return gm