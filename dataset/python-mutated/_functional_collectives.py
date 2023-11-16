import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import get_innermost_proxy_mode
from torch._custom_ops import impl_abstract
try:
    from torch.utils._cxx_pytree import tree_map_only
except ImportError:
    from torch.utils._pytree import tree_map_only
if torch._running_with_deploy():

    def is_torchdynamo_compiling():
        if False:
            return 10
        "Can't import torchdynamo in torchdeploy builds currently."
        return False
else:
    try:
        from torch._dynamo.external_utils import is_compiling as is_torchdynamo_compiling
    except Exception:
        warnings.warn("Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly")

        def is_torchdynamo_compiling():
            if False:
                print('Hello World!')
            return False
"\nNew traceable, functional collectives.\nRFC: https://github.com/pytorch/pytorch/issues/93173\n\n  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.\n  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,\n         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to\n         a downstream op.\n\nIssues:\n* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files\n* Proper support for eager requires inplace ops. We should explore having it as an option for the API.\n"
"\nFunctional collectives are asynchronous only and we perform implicit stream synchronization\non behalf of the user.\n\nWe use AsyncCollectiveTensor to wrap the result tensor of a collective and it lets us witness\nfirst usage of the tensor and insert cross stream sync at the right place.\n\nThe above are the easy bits, the hard one is how we match the Work object returned by\nc10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective\nop implementation (see ``clone()`` call in ``_all_reduce``) and then it's handled by the\ndispatcher which might call other implementations that are allowed to change the returned\ntensor - even return a tensor with a different shape (see ``torch.vmap``).\n\nThis means the caller of our ops receives a Tensor that is not guaranteed to be the same\nallocated by our implementations and that makes pairing The AsyncTensor to the original\ntensor a lot harder. This pairing is needed so we can lookup the Work object to use.\n\nOriginally, we tried WeakKeyDictionary to map from Tensor to Work, but because Tensor's\nidentity is not stable across dispatch, the op caller would end up with a different Tensor\ninstance that would not match any in the dictionary.\n\nWith Tensor identity out of the question, we decided use the tensor data pointer, which\nshould be stable across all the Tensor changes done during dispatch.\n\nWe have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.\n\nWe use this dictionary when AsyncCollectiveTensor is used to invoke Work::wait()\n\nFinally, we setup a finalizer against the tensor wrapper to observe it getting collected so we\ncan clean up stale entries in the dictionary.\n\nTo eliminate the possibility of races we have a global version counter that is used by the finalizer.\n\nAs a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)\n\n"
'\nFunctional collectives can accept any of these types to describe the ranks participating in collectives.\n\nThe different types will be desugared to a canonical format\n'
RANK_TYPES = Union[List[int], List[List[int]], dist.ProcessGroup, 'dist._tensor.DeviceMesh', Tuple['dist._tensor.DeviceMesh', int]]
"\nUser facing APIs for functional collectives\n-------------------------------------------\n\nThese apis are called by user code and expected to work both in eager execution and compilation,\nbut there are significant differences to how the two modes are implemented underneath.\n\nEager execution is 'optimized' using a tensor subclass that schedules the synchronization (via wait_tensor() op)\njust before the tensor is first used.  Compiled tracing currently relies on the compiler to perform this optimization,\nand cannot yet correctly trace the AsyncTensor wrapper class.  In the future, these paths may be unified\nif sufficient subclass support is added in dynamo.\n\nExample: all_reduce is an entrypoint API, and other collectives follow a similar pattern.\n\nHere's how it works under torch.compile/dynamo:\nall_reduce(...)\n  |--> _expand_group(...)               - desugars processgroup into canonical/traceable format\n  |--> c10d_functional.all_reduce(...)  - dynamo captures this op call, doesn't trace deeper\n  |--> _maybe_wrap_tensor(...)          - wait_tensor() op is immediately called, no AsyncTensor subclass needed\n\nAnd under eager execution:\nall_reduce(...)\n  |--> _expand_group(...)               - same as above, but less critical for eager\n  |--> c10d_functional.all_reduce(...)  - dispatches to real kernel OR records op in trace\n  |--> _maybe_wrap_tensor(...)          - AsyncTensor wrapper applied to returned tensor,\n                                          which issues wait_tensor() at the time of first use\n"

def wait_tensor(tensor):
    if False:
        print('Hello World!')
    '\n    Wait on a tensor returned by the collectives ops.\n\n    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.\n    '
    return torch.ops.c10d_functional.wait_tensor(tensor)

def broadcast(self: torch.Tensor, src: int, group: RANK_TYPES, tag: str=''):
    if False:
        while True:
            i = 10
    '\n    Broadcasts the tensor to all processes in the given process group.\n\n    Args:\n        src (int): Source rank\n        group (ProcessGroup or List[int]): The process group to work on.\n        tag (str, optional): A unique identifier for the collective. Default: empty string\n    '
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.broadcast(self, src, tag, rankset, group_size)
    return _maybe_wrap_tensor(tensor)

def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str=''):
    if False:
        return 10
    "\n    Reduces the tensor data across all machines in such a way that all get\n    the final result.\n\n    The input tensor is left unmodified.\n\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_reduce(self, reduceOp, tag, rankset, group_size)
    return _maybe_wrap_tensor(tensor)

def all_gather_tensor(self: torch.Tensor, gather_dim: int, group: RANK_TYPES, tag: str=''):
    if False:
        while True:
            i = 10
    "\n    Gather tensor data across from all machines and concatenate over ``gather_dim``.\n\n    Note that it currently only supports gather_dim = 0.\n\n    The input tensor is left unmodified.\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    assert self.is_contiguous()
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(self, tag, rankset, group_size)
    res = _maybe_wrap_tensor(tensor)
    if gather_dim != 0:
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res

def reduce_scatter_tensor(self: torch.Tensor, reduceOp: str, scatter_dim: int, group: RANK_TYPES, tag: str=''):
    if False:
        return 10
    "\n    Reduces the tensor data across all machines in such a way that all get\n    the final result, then scatter the results to corresponding ranks.\n\n\n    The input tensor is left unmodified.\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    (tag, rankset, group_size) = _expand_group(group, tag)
    assert self.size(scatter_dim) % group_size == 0, f'input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size}'
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)
    tensor = torch.ops.c10d_functional.reduce_scatter_tensor(self, reduceOp, tag, rankset, group_size)
    res = _maybe_wrap_tensor(tensor)
    return res

def all_reduce_coalesced(self: List[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str='') -> List[torch.Tensor]:
    if False:
        return 10
    "\n    Reduces a list of tensors across all machines in such a way that all get\n    the final result.\n\n    The all tensors in the input list are left unmodified.\n\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor_list = torch.ops.c10d_functional.all_reduce_coalesced(self, reduceOp, tag, rankset, group_size)
    return list(map(_maybe_wrap_tensor, tensor_list))

def all_gather_into_tensor_coalesced(self: List[torch.Tensor], group: RANK_TYPES, tag: str='') -> List[torch.Tensor]:
    if False:
        while True:
            i = 10
    "\n    Gather a list of tensors across from all machines.\n\n    Note that it currently only supports gather_dim = 0.\n\n    The input tensor is left unmodified.\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(self, tag, rankset, group_size)
    return list(map(_maybe_wrap_tensor, tensor_list))

def reduce_scatter_tensor_coalesced(inputs: List[torch.Tensor], reduceOp: str, scatter_dim: List[int], group: RANK_TYPES, tag: str='') -> List[torch.Tensor]:
    if False:
        i = 10
        return i + 15
    "\n    Reduces a list of tensors across all machines in such a way that all get\n    the final result, then scatter the results to corresponding ranks.\n\n    The input tensors are left unmodified.\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    (tag, rankset, group_size) = _expand_group(group, tag)
    assert len(scatter_dim) == len(inputs)
    for (idx, (dim, tensor)) in enumerate(zip(scatter_dim, inputs)):
        assert tensor.size(dim) % group_size == 0, f'input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}'
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)
    tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced(inputs, reduceOp, tag, rankset, group_size)
    return list(map(_maybe_wrap_tensor, tensor_list))

def _is_view_op(tgt):
    if False:
        while True:
            i = 10
    assert isinstance(tgt, torch._ops.OpOverload)
    schema = tgt._schema
    if len(schema.arguments) > 0:
        first_arg = schema.arguments[0]
        return first_arg.alias_info is not None and (not first_arg.alias_info.is_write)

def all_to_all_single(self: torch.Tensor, output_split_sizes: Optional[List[int]], input_split_sizes: Optional[List[int]], group: RANK_TYPES, tag: str='') -> torch.Tensor:
    if False:
        print('Hello World!')
    "\n    Each process splits input tensor and then scatters the split list\n    to all processes in a group. Then concatenate the received tensors from all\n    the processes in the group and return single output tensor.\n\n    Group can be one of:\n        List[int]: ranks participating in the collective.\n        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.\n        ProcessGroup: Will perform a collective using the ranks and tag of the PG.\n        DeviceMesh: Do a SPMD collective over all ranks of the mesh\n        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh\n\n    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover\n    that information and perform collective algebraic optimization. Use other forms of input for that.\n    "
    if output_split_sizes is not None:
        assert all((isinstance(size, (int, torch.SymInt)) for size in output_split_sizes)), output_split_sizes
    if input_split_sizes is not None:
        assert all((isinstance(size, (int, torch.SymInt)) for size in input_split_sizes)), input_split_sizes
    (tag, rankset, group_size) = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_to_all_single(self, output_split_sizes, input_split_sizes, tag, rankset, group_size)
    return _maybe_wrap_tensor(tensor)

class AsyncCollectiveTensor(torch.Tensor):
    """
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """
    elem: torch.Tensor
    __slots__ = ['elem']
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        if False:
            while True:
                i = 10
        r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=False)
        r.elem = elem
        return r

    def __tensor_flatten__(self):
        if False:
            print('Hello World!')
        return (['elem'], None)

    def tolist(self):
        if False:
            i = 10
            return i + 15
        wait_tensor(self.elem)
        return self.elem.tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        if False:
            i = 10
            return i + 15
        assert meta is None
        elem = inner_tensors['elem']
        return AsyncCollectiveTensor(elem)

    def __repr__(self):
        if False:
            while True:
                i = 10
        wait_tensor(self.elem)
        return f'AsyncCollectiveTensor({self.elem})'

    def trigger_wait(self):
        if False:
            return 10
        wait_tensor(self.elem)
        return self

    def wait(self) -> torch.Tensor:
        if False:
            print('Hello World!')
        wait_tensor(self.elem)
        return self.elem

    def _get_acs_underlying_tensor(self):
        if False:
            return 10
        'This method enables  _functional_collectives_impl to test if a tensor is an ACS'
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            print('Hello World!')
        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            if False:
                print('Hello World!')
            if not is_view_op:
                wait_tensor(e.elem)
            return e.elem

        def wrap(e: torch.Tensor):
            if False:
                i = 10
                return i + 15
            assert not isinstance(e, AsyncCollectiveTensor)
            res = AsyncCollectiveTensor(e)
            _register_tensor_wrapper(res)
            return res
        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)
        out = func(*unwrapped_args, **unwrapped_kwargs)
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)
        return out

    def numpy(self):
        if False:
            i = 10
            return i + 15
        return self.wait().numpy()
'\nUtils and infrastructure for tracing support\n'

def _expand_group(group: RANK_TYPES, tag: str='') -> Tuple[str, List[int], int]:
    if False:
        print('Hello World!')
    '\n    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.\n\n    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside\n    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.\n    '
    import torch.distributed._tensor as dt
    if TYPE_CHECKING:

        def cast_listlistint(x):
            if False:
                for i in range(10):
                    print('nop')
            return cast(List[List[int]], x)

        def cast_listint(x):
            if False:
                while True:
                    i = 10
            return cast(List[int], x)
    else:

        def cast_listlistint(x):
            if False:
                print('Hello World!')
            return x

        def cast_listint(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
    rankset: List[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(f'group sizes must be identical found {group_size} and {len(rs)}')
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, dt.DeviceMesh):
        assert group.ndim == 1, 'Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D'
        (tag, rankset) = group._dim_group_infos[0]
        group_size = len(rankset)
    elif isinstance(group, tuple):
        if len(group) == 2 and isinstance(group[0], dt.DeviceMesh) and isinstance(group[1], int):
            dmesh = group[0]
            dim = group[1]
            (tag, rankset) = dmesh._dim_group_infos[dim]
            group_size = len(rankset)
        else:
            raise ValueError('Invalid tuple for group must be (DeviceMesh, int)')
    else:
        raise ValueError('Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int).')
    return (tag, rankset, group_size)

def _are_we_tracing() -> bool:
    if False:
        print('Hello World!')
    if is_torchdynamo_compiling():
        return True
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL) is not None:
        return True
    mode = get_innermost_proxy_mode()
    if mode is None:
        return False
    return mode.tracer is not None

def _maybe_wrap_tensor(self) -> torch.Tensor:
    if False:
        print('Hello World!')
    if _are_we_tracing():
        return wait_tensor(self)
    res = AsyncCollectiveTensor(self)
    _register_tensor_wrapper(res)
    return cast(torch.Tensor, res)

def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    if False:
        for i in range(10):
            print('nop')

    def mk_out_tensor(shard):
        if False:
            i = 10
            return i + 15
        out_size = list(shard.size())
        out_size[0] *= group_size
        out_tensor = shard.new_empty(out_size)
        return out_tensor
    return [mk_out_tensor(t) for t in self]

def _broadcast_meta(self, *args):
    if False:
        for i in range(10):
            print('nop')
    return torch.empty_like(self)

def _all_reduce_meta(self, *args):
    if False:
        for i in range(10):
            print('nop')
    return torch.empty_like(self)

def _wait_tensor_meta(self, *args):
    if False:
        return 10
    return torch.empty_like(self)

def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    if False:
        for i in range(10):
            print('nop')
    out_size = list(shard.size())
    out_size[0] *= group_size
    return shard.new_empty(out_size)

def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    if False:
        print('Hello World!')
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)

def _all_reduce_coalesced_meta(self, *args):
    if False:
        print('Hello World!')
    return [torch.empty_like(t) for t in self]

def _all_reduce__meta(inp, *args):
    if False:
        while True:
            i = 10
    return inp

def _all_reduce_coalesced__meta(inputs, *args):
    if False:
        print('Hello World!')
    return inputs

def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    if False:
        return 10

    def mk_out_tensor(input):
        if False:
            i = 10
            return i + 15
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        return out_tensor
    return [mk_out_tensor(t) for t in inputs]

def _all_to_all_single_meta(input, output_split_sizes, input_split_sizes, tag, rankset, group_size):
    if False:
        i = 10
        return i + 15
    if output_split_sizes is None:
        return input.new_empty(input.size())
    else:
        for s in output_split_sizes:
            torch._check_is_size(s)
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        return input.new_empty(out_size)

def _all_gather_into_tensor_native_meta(input, group_size, group_name):
    if False:
        i = 10
        return i + 15
    shape = list(input.size())
    shape[0] *= group_size
    return input.new_empty(shape)

def _all_gather_into_tensor_coalesced_native_meta(inputs, group_size, group_name):
    if False:
        i = 10
        return i + 15
    return [_all_gather_into_tensor_native_meta(input, group_size, group_name) for input in inputs]

def _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name):
    if False:
        for i in range(10):
            print('nop')
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)

def _reduce_scatter_tensor_coalesced_native_meta(inputs, reduce_op, group_size, group_name):
    if False:
        i = 10
        return i + 15
    return [_reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name) for inp in inputs]

def _register_ops():
    if False:
        print('Hello World!')
    ops_defs = ['broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor', 'all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor', 'all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]', 'wait_tensor(Tensor self) -> Tensor', 'all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor', 'all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]', 'reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor', 'reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]', 'all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor']
    my_module = sys.modules[__name__]
    for op_def in ops_defs:
        op_name = op_def[0:op_def.index('(')]
        backend_impl = getattr(fun_col_impl, f'_{op_name}')
        meta_impl = getattr(my_module, f'_{op_name}_meta')
        c10_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
        c10_lib_impl.impl(op_name, backend_impl, 'CompositeExplicitAutograd')
        impl_abstract(f'c10d_functional::{op_name}')(meta_impl)
if not torch._running_with_deploy():
    c10_lib = torch.library.Library('c10d_functional', 'DEF')
    c10_lib_impl = torch.library.Library('c10d_functional', 'IMPL')
    _register_ops()
    _c10_lib_impl = torch.library.Library('_c10d_functional', 'IMPL')
    _c10_lib_impl.impl('all_reduce', _all_reduce_meta, 'Meta')
    _c10_lib_impl.impl('all_reduce_', _all_reduce__meta, 'Meta')
    _c10_lib_impl.impl('all_reduce_coalesced', _all_reduce_coalesced_meta, 'Meta')
    _c10_lib_impl.impl('all_reduce_coalesced_', _all_reduce_coalesced__meta, 'Meta')
    _c10_lib_impl.impl('wait_tensor', _wait_tensor_meta, 'Meta')
    _c10_lib_impl.impl('all_gather_into_tensor', _all_gather_into_tensor_native_meta, 'Meta')
    _c10_lib_impl.impl('all_gather_into_tensor_coalesced', _all_gather_into_tensor_coalesced_native_meta, 'Meta')
    _c10_lib_impl.impl('reduce_scatter_tensor', _reduce_scatter_tensor_native_meta, 'Meta')
    _c10_lib_impl.impl('reduce_scatter_tensor_coalesced', _reduce_scatter_tensor_coalesced_native_meta, 'Meta')
else:
    warnings.warn('PyTorch Distributed functional collectives do not work with torch::deploy.')
'\nDynamo Remappings allow seamless translation from non-functional collectives of supportable form into\nfunctional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.\n\nWe implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via\nthe mapping dict below.\n\nThese schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from\n'

def all_gather_tensor_inplace(output: torch.Tensor, input: torch.Tensor, group, async_op: bool=False, tag: str='', gather_dim: int=0):
    if False:
        while True:
            i = 10
    assert not async_op, "Can't remap async version of inplace op to functional collective"
    return output.copy_(all_gather_tensor(input, gather_dim, group, tag))

def reduce_scatter_tensor_inplace(output: torch.Tensor, input: torch.Tensor, op: str='sum', group=None, async_op: bool=False, scatter_dim: int=0, tag: str=''):
    if False:
        return 10
    assert not async_op, "Can't remap async version of inplace op to functional collective"
    return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))
from torch.distributed.distributed_c10d import all_gather_into_tensor as legacy_allgather, reduce_scatter_tensor as legacy_reducescatter
traceable_collective_remaps = {legacy_allgather: all_gather_tensor_inplace, legacy_reducescatter: reduce_scatter_tensor_inplace}