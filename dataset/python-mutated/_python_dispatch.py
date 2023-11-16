import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at, _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack, DispatchKey

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    def __init__(self, _dispatch_key=None):
        if False:
            print('Hello World!')
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__['_dispatch_key'] = _dispatch_key

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def __enter__(self):
        if False:
            while True:
                i = 10
        _push_mode(self, self.__dict__.get('_dispatch_key', None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        mb_dk_or_mode_key = self.__dict__.get('_dispatch_key', None)
        if mb_dk_or_mode_key is None:
            mb_dk_or_mode_key = self.__dict__.get('_mode_key', None)
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    def push(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        warnings.warn('`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`')
        instance = cls(*args, **kwargs)
        return instance

def _get_current_dispatch_mode():
    if False:
        i = 10
        return i + 15
    stack_len = _len_torch_dispatch_stack()
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None

def _get_current_dispatch_mode_stack():
    if False:
        for i in range(10):
            print('nop')
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]

def _push_mode(mode, k: Optional[DispatchKey]=None):
    if False:
        for i in range(10):
            print('nop')
    if k is not None:
        from torch._ops import push_mode_for_key, get_cached_ops
        ks = torch._C._functionality_to_backend_keys(k)
        for op in get_cached_ops():
            for key in ks:
                op._uncache_dispatch(key)
        push_mode_for_key(k, mode)
    else:
        _push_on_torch_dispatch_stack(mode)

def _pop_mode(k: Optional[Union[DispatchKey, torch._C._TorchDispatchModeKey]]=None):
    if False:
        i = 10
        return i + 15
    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)
    from torch._ops import pop_mode_for_key
    return pop_mode_for_key(k)

@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey]=None):
    if False:
        print('Hello World!')
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old, k)

@contextlib.contextmanager
def _disable_current_modes():
    if False:
        while True:
            i = 10
    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]
    try:
        yield old_modes
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)

class BaseTorchDispatchMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

def is_traceable_wrapper_subclass(t):
    if False:
        i = 10
        return i + 15
    "\n    Returns whether or not a tensor subclass that implements __torch_dispatch__\n    is 'traceable' with torch.compile.\n    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,\n    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.\n    It is also expected to obey some restrictions around traceability and aliasing\n    (TODO: add clear documentation around this.)\n    "
    is_subclass = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    return is_subclass and hasattr(t, '__tensor_flatten__') and hasattr(t, '__tensor_unflatten__')

def transform_subclass(t, callback):
    if False:
        i = 10
        return i + 15
    '\n    Given a traceable, wrapper tensor subclass ``t`` that implements\n    ``__torch_dispatch__`` and holds some inner tensors,\n    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,\n    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.\n    It will do so by grabbing each inner tensor attribute from the wrapper,\n    passing them into ``callback`` to get a transformed tensor,\n    and putting each transformed tensor into the fresh tensor subclass instance.\n\n    Note: this function will not handle ensuring that the fresh subclass\n    gets the same (autograd, and aliasing) metadata as the original tensor.\n    This is generally handled in other subsystems like AOTAutograd.\n    '
    (attrs, ctx) = t.__tensor_flatten__()
    transformed_tensors_dict = {}
    for attr in attrs:
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    return type(t).__tensor_unflatten__(transformed_tensors_dict, ctx)

def _correct_storage_aliasing(func, schema_info, args, outs):
    if False:
        while True:
            i = 10
    "\n    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),\n    and the inputs/outputs to the OpOverload,\n    this function checks to see if func is a view operator\n    (by checking if any of the outputs in the op's schema\n     are immutable aliases of inputs).\n    If so, this function manually aliases the storage of the output tensor\n    with its corresponding input tensor alias.\n    It does this by unsafely overwriting the storage field of the output tensor\n    to be the same storage as the input.\n    "
    assert isinstance(func, torch._ops.OpOverload)
    assert isinstance(args, tuple)
    assert isinstance(outs, (list, tuple))
    flat_outs = torch.utils._pytree.tree_leaves(outs)

    def alias_non_inplace_storage(arg, ret):
        if False:
            for i in range(10):
                print('nop')
        if is_traceable_wrapper_subclass(arg) or is_traceable_wrapper_subclass(ret):
            ret_list = ret if isinstance(ret, list) else [ret]
            for r in ret_list:
                assert type(arg) == type(r), f'Called {str(func)} with input of type {type(arg)}\nand output of type {type(ret)}. But expected types to match.'
        with torch.utils._mode_utils.no_dispatch():
            meta_in_tls = torch._C._meta_in_tls_dispatch_include()
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                if isinstance(ret, list):
                    for r in ret:
                        torch.ops.aten.set_.source_Storage_storage_offset(r, arg.untyped_storage(), r.storage_offset(), r.shape)
                else:
                    assert isinstance(ret, torch.Tensor), f'type: {type(ret)}'
                    torch.ops.aten.set_.source_Storage_storage_offset(ret, arg.untyped_storage(), ret.storage_offset(), ret.shape)
            finally:
                torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    def is_read_only_alias_match(arg, ret):
        if False:
            for i in range(10):
                print('nop')
        shared_aliases = arg.alias_set & ret.alias_set
        return len(shared_aliases) > 0 and (not arg.is_write)
    num_args = len(func._schema.arguments)
    num_returns = len(func._schema.returns)
    for arg_idx in range(num_args):
        for return_idx in range(num_returns):
            if is_read_only_alias_match(schema_info.args[arg_idx], schema_info.outs[return_idx]):
                alias_non_inplace_storage(args[arg_idx], outs[return_idx])

@dataclass
class AliasInfo:
    alias_set: Set[str]
    is_write: bool
    name: Optional[str]

@dataclass
class SchemaInfo:
    args: List[AliasInfo]
    outs: List[AliasInfo]
parsed_schema_map: Dict[Any, SchemaInfo] = {}

def get_alias_info(func) -> SchemaInfo:
    if False:
        while True:
            i = 10
    if func in parsed_schema_map:
        return parsed_schema_map[func]
    if func.namespace == 'aten':
        torchgen_schema_str = str(func._schema)
        assert torchgen_schema_str.startswith('aten::')
        torchgen_schema_str = torchgen_schema_str[6:]
        import re
        torchgen_schema_str = re.sub('=\\[[0, ]+\\]', '=0', torchgen_schema_str)
        torchgen_schema_str = re.sub('=\\[[1, ]+\\]', '=1', torchgen_schema_str)
        torchgen_schema_str = torchgen_schema_str.replace('=[0, 1]', '=[0,1]')
        torchgen_schema = torchgen.model.FunctionSchema.parse(torchgen_schema_str)
        arg_schemas = [AliasInfo(alias_set=set() if a.annotation is None else set(a.annotation.alias_set), is_write=a.annotation is not None and a.annotation.is_write, name=a.name) for a in torchgen_schema.arguments.flat_all]
        out_schemas = [AliasInfo(alias_set=set() if a.annotation is None else set(a.annotation.alias_set), is_write=a.annotation is not None and a.annotation.is_write, name=a.name) for a in torchgen_schema.returns]
    else:
        arg_schemas = [AliasInfo(alias_set=set() if a.alias_info is None else set(a.alias_info.before_set), is_write=a.alias_info is not None and a.alias_info.is_write, name=a.name) for a in func._schema.arguments]
        out_schemas = [AliasInfo(alias_set=set() if a.alias_info is None else set(a.alias_info.before_set), is_write=a.alias_info is not None and a.alias_info.is_write, name=a.name) for a in func._schema.returns]
    schema_info = SchemaInfo(args=arg_schemas, outs=out_schemas)
    parsed_schema_map[func] = schema_info
    return schema_info

def return_and_correct_aliasing(func, args, kwargs, out):
    if False:
        return 10
    '\n    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses\n    that would like to work with torch.compile. It ensures that the subclass\n    properly implements the aliasing behavior of every op,\n    which is needed for correctness in AOTAutograd.\n    This function will handle:\n\n        * When we see a view op, we will alias the storages of any\n          input and output tensor subclasses\n\n        * When we see an inplace or out= op, we will directly\n          return the corresponding input tensor, instead of returning\n          a (potentially) fresh output tensor.\n    '
    schema_info = get_alias_info(func)

    def get_write_alias(x):
        if False:
            i = 10
            return i + 15
        if len(x.alias_set) == 0:
            return None
        alias_set = list(x.alias_set)
        assert len(alias_set) == 1
        if x.is_write:
            return alias_set[0]
        return None

    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        (new_args, new_kwargs) = torch.fx.operator_schemas.normalize_function(func, args=args, kwargs=kwargs)
        arg_indices = [i for (i, a) in enumerate(schema_info.args) if output_alias in a.alias_set]
        assert len(arg_indices) == 1
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]
    _correct_storage_aliasing(func, schema_info, args, (out,) if not isinstance(out, tuple) else out)
    if torch.Tag.inplace_view in func.tags:
        mutated_args = [x for (i, x) in enumerate(args) if get_write_alias(schema_info.args[i]) is not None]
        assert len(mutated_args) == 1
        from torch._subclasses.functional_tensor import FunctionalTensor
        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
    if not any((get_write_alias(r) is not None for r in schema_info.outs)):
        return out
    if not all((get_write_alias(r) is not None for r in schema_info.outs)):
        raise RuntimeError('Unsupported schema: ' + str(func._schema))
    if len(func._schema.returns) == 1:
        return get_arg_from_alias(get_write_alias(schema_info.outs[0]), schema_info, args, kwargs)
    outs_to_return = type(out)([get_arg_from_alias(get_write_alias(schema_info.outs[i]), schema_info, args, kwargs) if get_write_alias(r) is not None else o for ((i, r), o) in zip(enumerate(schema_info.outs), out)])
    return outs_to_return