from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import _ProxyTensor, fetch_tensor_proxy, get_innermost_proxy_mode, get_proxy_slot, set_proxy_slot, track_tensor_tree
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only

@dataclass
class _CommResult:
    _tensor: torch.Tensor
    _work: torch.distributed._Work

def _wait_comm(comm_result: _CommResult):
    if False:
        while True:
            i = 10
    comm_result._work.wait()
    return comm_result._tensor

def _wrap_comm_result(result: Tuple[Any, Any]) -> Tuple[Any, Any]:
    if False:
        print('Hello World!')

    def wrap(work, e):
        if False:
            print('Hello World!')
        assert isinstance(e, torch.Tensor), 'Excepting collection of tensors as the first element in the return value of communication operations.'
        return _CommResult(e, work)
    work = result[1]
    return (tree_map(partial(wrap, work), result[0]), work)

def _get_tracer() -> Optional[torch.fx.Tracer]:
    if False:
        return 10
    mode = get_innermost_proxy_mode()
    if mode is None:
        return None
    return mode.tracer

class CommTensor(torch.Tensor):
    """
    A Tensor subclass to wrap input tensors for collective communications.

    This Tensor subclass works for both eager and tracing mode.
    In eager mode, it will record whether the inplace collective communication
    has been launched using this Tensor and remember the corresponding work
    handle. If yes, it will explicitly call wait() in the ``__torch_dispatch__``
    function before subsequent operations consuming the value of the Tensor.

    In tracing mode, ``CommTensor`` inserts two node into the graph using the
    ``__torch_dispatch__`` function.
    1. The first node is inserted right after the
    communication, wrapping both the inplace output tensor and the returned
    work handle into a custom ``_CommResult`` type. We have to do this because
    ``ProxyTorchDispatchMode`` only handles ``torch.Tensor``, ``_ProxyTensor``,
    and ``torch.nn.Parameter`` objects and will treat the work handle
    as a constant and embed that into the graph. As a result, during execution,
    it will use the work handle created during tracing and will lead to wrong
    result. The solution in this test is to manually create a proxy on the
    return value of ``allreduce_`` which is ``([tensor], work)``, and wrap that
    to ``[(_CommResult(tensor, work)), work]``. In this way, subsequent nodes can
    directly consume ``_CommResult``.
    2. The second node is inserted right before any subsequent node reads from
    ``_CommResult``. It will call ``wait()`` on the stashed work handle to ensure
    that computation waits for communication.
    """
    _supported_comms: List[str] = ['_allgather_base_', '_reduce_scatter_base_', 'allreduce_', 'allgather_', 'alltoall_', 'broadcast_', 'reduce_scatter_', 'scatter_']
    _tensor: torch.Tensor
    _work: Optional[torch.distributed._Work]

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        if False:
            return 10
        t = tensor._tensor if isinstance(tensor, CommTensor) else tensor
        if get_innermost_proxy_mode() is None:
            return tensor
        r = torch.Tensor._make_subclass(cls, t, require_grad=t.requires_grad)
        r._tensor = tensor
        r._work = None
        return r

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'CommTensor({self._tensor}, work={self._work})'
    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def _is_supported(cls, op_name):
        if False:
            print('Hello World!')
        return any((comm in op_name for comm in cls._supported_comms))

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            return 10
        tracer: Optional[torch.fx.Tracer] = None
        work: Optional[torch.distributed._Work] = None

        def unwrap(e: Any):
            if False:
                while True:
                    i = 10
            if isinstance(e, CommTensor):
                nonlocal tracer, work
                work = e._work
                if not isinstance(e._tensor, CommTensor):
                    tracer = _get_tracer()
                if work is not None:
                    if tracer is not None:
                        proxy_res = tracer.create_proxy('call_function', _wait_comm, (get_proxy_slot(e._tensor, tracer).proxy,), {}, name='wait_comm')
                        set_proxy_slot(e._tensor, tracer, proxy_res)
                    work.wait()
                return e._tensor
            else:
                return e

        def wrap(e: Any):
            if False:
                while True:
                    i = 10
            return CommTensor(e) if isinstance(e, torch.Tensor) else e

        def set_work(work: torch.distributed._Work, e: Any):
            if False:
                while True:
                    i = 10
            if isinstance(e, CommTensor):
                e._work = work
            elif isinstance(e, torch.Tensor):
                raise RuntimeError('Type of output tensors from collective communication during tracing should always be CommTensor instead of torch.Tensor')
            return e
        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)
        if cls._is_supported(func.__name__):
            if tracer is not None:
                (proxy_args, proxy_kwargs) = tree_map_only(_ProxyTensor, lambda e: e.proxy, tree_map_only(torch.Tensor, fetch_tensor_proxy(tracer), (unwrapped_args, unwrapped_kwargs)))
                proxy_res = func(*proxy_args, **proxy_kwargs)
                assert isinstance(proxy_res, torch.fx.Proxy)
                comm_result_proxy = tracer.create_proxy('call_function', _wrap_comm_result, (proxy_res,), {}, name='comm_result')
                with no_dispatch():
                    out = func(*unwrapped_args, **unwrapped_kwargs)
                track_tensor_tree(out, comm_result_proxy, constant=None, tracer=tracer)
                pytree.tree_map_(partial(set_work, out[1]), args[0])
                (flat_args, args_spec) = tree_flatten(unwrapped_args[0])
                (flat_out, out_spec) = tree_flatten(out[0])
                for (a, o) in zip(flat_args, flat_out):
                    set_proxy_slot(a, tracer, get_proxy_slot(o, tracer))
                return out
            else:
                out = func(*unwrapped_args, **unwrapped_kwargs)
                pytree.tree_map_(partial(set_work, out[1]), args[0])
                return out
        elif work is not None:
            return func(*unwrapped_args, **unwrapped_kwargs)
        else:
            return tree_map(wrap, func(*unwrapped_args, **unwrapped_kwargs))