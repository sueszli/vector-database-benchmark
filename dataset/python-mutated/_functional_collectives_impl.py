import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
'\nMoved eager kernel implementations to a separate file partly for readability and partly as it is currently\neasier in dynamo to set tracing policy on a file-by-file level.\n\nDo not put code in this file that Dynamo is expected to trace into, as dynamo may disallow this whole file.\n\nDEBUG/TESTING HELPERS:\n\nThis module includes some helpers that are quite useful when debugging or testing functional collectives:\n\n_tensor_needs_wait\n_outstanding_wait_count\n_wait_all\n\n'
logger = logging.getLogger(__name__)
data_ptr_to_work = dict()
work_version = 0

class _WaitRegistration:

    def __init__(self, work):
        if False:
            while True:
                i = 10
        global work_version
        self.work = work
        self.version = work_version
        self.ptrs = set()
        self.ptr_alias_count = {}
        self.cleanup_count = 0
        work_version += 1

    def _register_tensor_ptr(self, data_ptr):
        if False:
            for i in range(10):
                print('nop')
        global data_ptr_to_work
        data_ptr_to_work[data_ptr] = self
        self.ptrs.add(data_ptr)

    def _record_wrapper(self, ptr):
        if False:
            while True:
                i = 10
        self._register_tensor_ptr(ptr)
        self.ptr_alias_count.setdefault(ptr, 0)
        self.ptr_alias_count[ptr] += 1
        self.cleanup_count += 1

    def wait(self):
        if False:
            return 10
        if self.work is not None:
            self.work.wait()
            self.work = None
        self.cleanup()

    def decrement_live_tensor(self, ptr):
        if False:
            for i in range(10):
                print('nop')
        self.cleanup_count -= 1
        if self.cleanup_count == 0:
            self.wait()
        else:
            self.ptr_alias_count[ptr] -= 1
            if self.ptr_alias_count[ptr] < 1 and data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]

    def cleanup(self):
        if False:
            return 10
        for ptr in self.ptrs:
            if data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]

def _register_tensor_work(tensor_or_list, work_or_list):
    if False:
        print('Hello World!')
    if not isinstance(tensor_or_list, list):
        tensor_or_list = [tensor_or_list]
    if not isinstance(work_or_list, list):
        reg = _WaitRegistration(work_or_list)
        for tensor in tensor_or_list:
            reg._register_tensor_ptr(tensor.data_ptr())
    else:
        for (tensor, work) in zip(tensor_or_list, work_or_list):
            reg = _WaitRegistration(work)
            reg._register_tensor_ptr(tensor.data_ptr())

def _wait_reg_dec(ptr, wait_reg):
    if False:
        i = 10
        return i + 15
    wait_reg.decrement_live_tensor(ptr)

def _register_tensor_wrapper(tensor) -> None:
    if False:
        return 10
    global data_ptr_to_work
    data_ptr = tensor.elem.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr, None)
    if wait_reg is None:
        warnings.warn('Trying to register finalizer to AsyncCollectiveTensor but the inner tensor is already gone')
    else:
        wait_reg._record_wrapper(data_ptr)
        weakref.finalize(tensor, _wait_reg_dec, data_ptr, wait_reg)

def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    global data_ptr_to_work
    data_ptr = tensor.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr)
    if wait_reg is not None:
        wait_reg.wait()
    return tensor

def _tensor_needs_wait(tensor: torch.Tensor) -> bool:
    if False:
        print('Hello World!')
    'Returns true if ```tensor``` needs to be waited. Works with ACS and inner tensors.'
    if hasattr(tensor, '_get_acs_underlying_tensor'):
        tensor = tensor._get_acs_underlying_tensor()
    data_ptr = tensor.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr)
    return wait_reg is not None and wait_reg.work is not None

def _outstanding_wait_count() -> int:
    if False:
        i = 10
        return i + 15
    ' Returns the number of outstanding work objects waiting to be waited (sic). '
    return len(data_ptr_to_work)

def _wait_all() -> None:
    if False:
        i = 10
        return i + 15
    ' Wait for all outstanding collectives. '
    for work_reg in list(data_ptr_to_work.values()):
        work_reg.wait()

def _str_to_reduce_op(reduceOp: str) -> dist.ReduceOp:
    if False:
        for i in range(10):
            print('nop')
    reduceOp = reduceOp.upper()
    op = dist.ReduceOp.RedOpType.__members__.get(reduceOp)
    if op is None:
        raise ValueError(f'Invalid reduce operation {reduceOp}')
    return cast(dist.ReduceOp, op)
'\nKernel implementations (for eager runtime only) - should never be traced by torch.compile\n\nThese functions should all be bound to dispatcher ops.  During tracing, the op itself should be\ncaptured in the graph and the backend should implement the op however it prefers.\n'

def _broadcast(self, src, tag, ranks, group_size):
    if False:
        return 10
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    inplace_tensor = self.clone(memory_format=torch.contiguous_format)
    work = dist.broadcast(inplace_tensor, src, group=group, async_op=True)
    _register_tensor_work(inplace_tensor, work)
    return inplace_tensor

def _all_reduce(self, reduceOp, tag, ranks, group_size):
    if False:
        print('Hello World!')
    op = _str_to_reduce_op(reduceOp)
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    inplace_tensor = self.clone(memory_format=torch.contiguous_format)
    work = dist.all_reduce(inplace_tensor, op=op, group=group, async_op=True)
    _register_tensor_work(inplace_tensor, work)
    return inplace_tensor

def _all_reduce_coalesced(self, reduceOp, tag, ranks, group_size):
    if False:
        return 10
    op = _str_to_reduce_op(reduceOp)
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    inplace_tensor_list = [t.clone(memory_format=torch.contiguous_format) for t in self]
    work = dist.all_reduce_coalesced(inplace_tensor_list, op=op, group=group, async_op=True)
    _register_tensor_work(inplace_tensor_list, work)
    return inplace_tensor_list

def _all_gather_into_tensor(shard, tag, ranks, group_size):
    if False:
        return 10
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    out_size = list(shard.size())
    out_size[0] *= group_size
    out_tensor = shard.new_empty(out_size)
    assert out_tensor.is_contiguous()
    if dist.get_backend(group) == dist.Backend.GLOO or shard.is_cpu:
        tensor_list = list(torch.chunk(out_tensor, group_size))
        work = dist.all_gather(tensor_list, shard, group=group, async_op=True)
    else:
        work = dist.all_gather_into_tensor(out_tensor, shard, group=group, async_op=True)
    _register_tensor_work(out_tensor, work)
    return out_tensor

def _all_gather_into_tensor_coalesced(self, tag, rankset, group_size):
    if False:
        print('Hello World!')
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, rankset, group_size)
    assert group is not None

    def mk_out_tensor(shard):
        if False:
            while True:
                i = 10
        out_size = list(shard.size())
        out_size[0] *= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor
    out_tensors = [mk_out_tensor(t) for t in self]
    work_list = _all_gather_into_tensor_coalesced_fallback(output_tensors=out_tensors, input_tensors=self, group=group, async_op=True)
    _register_tensor_work(out_tensors, work_list)
    return out_tensors

def _reduce_scatter_tensor(input: torch.Tensor, reduceOp: str, tag: str, ranks: List[int], group_size: int):
    if False:
        return 10
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduceOp)
    if dist.get_backend(group) == dist.Backend.GLOO or input.is_cpu:
        logger.warning('ProcessGroupGloo does not support reduce_scatter, falling back with all reduce!')
        reduction_input = input.clone()
        group_rank = dist.get_rank(group)
        work = dist.all_reduce(reduction_input, op=op, group=group, async_op=True)
        out_tensor = reduction_input.chunk(group_size, dim=0)[group_rank]
        _register_tensor_work(out_tensor, work)
    else:
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        work = dist.reduce_scatter_tensor(out_tensor, input, op=op, group=group, async_op=True)
        _register_tensor_work(out_tensor, work)
    return out_tensor

def _reduce_scatter_tensor_coalesced(inputs: List[torch.Tensor], reduce_op: str, tag: str, ranks: List[int], group_size: int):
    if False:
        while True:
            i = 10
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduce_op)

    def mk_out_tensor(shard):
        if False:
            i = 10
            return i + 15
        out_size = list(shard.size())
        out_size[0] //= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor
    out_tensors = [mk_out_tensor(t) for t in inputs]
    work_list = _reduce_scatter_tensor_coalesced_fallback(output_tensors=out_tensors, input_tensors=inputs, op=op, group=group, async_op=False)
    _register_tensor_work(out_tensors, work_list)
    return out_tensors

def _all_gather_into_tensor_coalesced_fallback(output_tensors, input_tensors, group, async_op=False):
    if False:
        for i in range(10):
            print('nop')
    if input_tensors[0].is_cpu or not async_op:
        work_list = []
        out_tensors_sliced = [list(torch.chunk(out_tensor, dist.get_world_size(group))) for out_tensor in output_tensors]
        for (shard, out_tensor) in zip(input_tensors, out_tensors_sliced):
            work = c10d.all_gather(out_tensor, shard, group=group, async_op=async_op)
            work_list.append(work)
        return work_list
    else:
        with c10d._coalescing_manager(group=group, async_ops=True) as cm:
            for (in_t, out_t) in zip(input_tensors, output_tensors):
                dist.all_gather_into_tensor(out_t, in_t, group=group, async_op=True)
        return cm

def _reduce_scatter_tensor_coalesced_fallback(output_tensors, input_tensors, op, group, async_op=False):
    if False:
        for i in range(10):
            print('nop')
    work_list = []
    for (shard, out_tensor) in zip(input_tensors, output_tensors):
        work = c10d.reduce_scatter_tensor(out_tensor, shard, op=op, group=group, async_op=async_op)
        work_list.append(work)
    return work_list

def _all_to_all_single(input: torch.Tensor, output_split_sizes: Optional[List[int]], input_split_sizes: Optional[List[int]], tag: str, ranks: List[int], group_size: int):
    if False:
        while True:
            i = 10
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    if output_split_sizes is not None:
        torch._check(input.dim() >= 1, lambda : f'Expected input to have at least 1 dim but got {input.dim()} dim')
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        out_tensor = input.new_empty(out_size)
    else:
        out_tensor = input.new_empty(input.size())
    work = c10d.all_to_all_single(out_tensor, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=True)
    _register_tensor_work(out_tensor, work)
    return out_tensor