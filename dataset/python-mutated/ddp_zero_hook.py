import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel.distributed import DistributedDataParallel
__all__ = ['hook_with_zero_step', 'hook_with_zero_step_interleaved']
_NO_PARAM_UPDATE = None

def _perform_local_step(bucket: dist.GradBucket, zero: ZeroRedundancyOptimizer, rank: int):
    if False:
        for i in range(10):
            print('nop')
    "\n    Performs a local optimizer step using the gradients provided by ``bucket``.\n\n    Arguments:\n        bucket (dist.GradBucket): the bucket providing the gradients.\n        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`\n            instance to perform the :meth:`_local_step`.\n        rank (int): the calling process's rank.\n\n    .. warning::\n        This function assumes that appropriate synchronization has taken place\n        so that the bucket's gradients can be used.\n    "
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    assert len(zero.optim.param_groups) == 1, 'Overlapping DDP with ZeRO only supports a single parameter group'
    num_local_optim_params = len(zero.optim.param_groups[0]['params'])
    gradients: List[Optional[torch.Tensor]] = [_NO_PARAM_UPDATE for _ in range(num_local_optim_params)]
    assert bucket_index in overlap_info.offsets, f'Bucket index {bucket_index} was not assigned to rank {rank}'
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset:bucket_offset + length]
    for (i, grad) in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad
    zero._local_step(gradients)

def _broadcast_bucket(bucket_index: int, zero: ZeroRedundancyOptimizer):
    if False:
        while True:
            i = 10
    "\n    Broadcasts a bucket's parameters.\n\n    Arguments:\n        bucket_index (int): the index of the bucket corresponding to the\n            parameters to broadcast.\n        zero (ZeroRedundancyOptimizer): the calling process's\n            :class:`ZeroRedundancyOptimizer` instance.\n    "
    overlap_info = zero._overlap_info
    assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, '`assigned_ranks_per_bucket` is not fully constructed'
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    assert len(assigned_ranks) > 0, f'Bucket {bucket_index} should be assigned to at least one rank'
    for assigned_rank in assigned_ranks:
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        if bucket_index in bucket_assignments:
            overlap_info.broadcast_handles.append(dist.broadcast(bucket_assignments[bucket_index].tensor, src=dist.get_global_rank(zero.process_group, assigned_rank), group=zero.process_group, async_op=True))

def _save_ddp_bucket_info(bucket: dist.GradBucket, zero: ZeroRedundancyOptimizer):
    if False:
        for i in range(10):
            print('nop')
    "\n    Saves :class:`DistributedDataParallel` gradient bucket information for the\n    :class:`ZeroRedundancyOptimizer` instance ``zero`` to use when overlapping.\n    In particular, this function is meant to be called upon seeing each\n    gradient bucket, meaning it does not save or compute any global\n    information.\n\n    Arguments:\n        bucket (dist.GradBucket): the current gradient bucket.\n        zero (ZeroRedundancyOptimizer): the calling process's\n            :class:`ZeroRedundancyOptimizer` instance.\n    "
    overlap_info = zero._overlap_info
    bucket_params = bucket.parameters()
    assert len(bucket_params) > 0, 'Empty bucket'
    overlap_info.params_per_bucket.append(bucket_params)
    if overlap_info.shard_buckets:
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()
        assert overlap_info.total_size is not None
        overlap_info.total_size += bucket_size

def _hook_with_zero_step_setup(ddp_ref: weakref.ReferenceType, zero: ZeroRedundancyOptimizer, bucket: dist.GradBucket):
    if False:
        print('Hello World!')
    "\n    Encapsulates the setup logic for :func:`hook_with_zero_step` and\n    :func:`hook_with_zero_step_interleaved`, meaning the logic to run in the\n    hook before the backward pass and optimizer step can actually be\n    overlapped. This is factored out since it is common to both\n    :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.\n\n    Arguments:\n        ddp_ref (weakref.ReferenceType): weak reference to the process's\n            :class:`DistributedDataParallel` instance.\n        zero (ZeroRedundancyOptimizer): the calling process's\n            :class:`ZeroRedundancyOptimizer` instance.\n        bucket (dist.GradBucket): the current gradient bucket.\n    "
    if not ddp_ref()._has_rebuilt_buckets:
        assert zero._overlap_info.status == _OverlapStatus.UNINITIALIZED
        return
    bucket_index = bucket.index()
    overlap_info = zero._overlap_info
    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS
    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            zero._init_zero_for_overlap()
        else:
            _save_ddp_bucket_info(bucket, zero)

def hook_with_zero_step(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool=False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    if False:
        while True:
            i = 10
    '\n    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`\n    optimizer step with the :class:`DistributedDataParallel` backward pass,\n    where the optimizer step computation begins after the last gradient bucket\n    computation has finished.\n\n    This approach overlaps the optimizer computation and communication with the\n    backward communication. In particular, the backward computation proceeds\n    contiguously, and the optimizer computation follows, overlapping with\n    outstanding backward communication (i.e. all-reduces) and possibly other\n    optimizer communication (i.e. broadcasts).\n\n    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`\n    if communication is relatively slow compared to computation.\n\n    Arguments:\n        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook\n            to modify.\n        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`\n            instance to use.\n        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`\n            instance to use.\n        shard_buckets (bool): if ``True``, then the assignment of each\n            :class:`DistributedDataParallel` bucket is partitioned across\n            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.\n            across possibly multiple ranks) to approximate uniformity; if\n            ``False``, then each bucket is wholly assigned to a single\n            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).\n\n    Returns:\n        The modified hook.\n\n    Raises:\n        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.\n        RuntimeError: if using any backend other than NCCL/HCCL since currently\n            Gloo may hang.\n\n    .. warning::\n        Given the way that overlapping :class:`DistributedDataParallel` with\n        :class:`ZeroRedundancyOptimizer` is currently implemented, the first\n        two or three training iterations do not perform parameter updates in\n        the optimizer step, depending on if ``static_graph=False`` or\n        ``static_graph=True``, respectively. This is because it needs\n        information about the gradient bucketing strategy used by\n        :class:`DistributedDataParallel`, which is not finalized until the\n        second forward pass if ``static_graph=False`` or until the third\n        forward pass if ``static_graph=True``.\n    '
    if not zero._overlap_with_ddp:
        raise ValueError('ZeroRedundancyOptimizer must be constructed with `overlap_with_ddp=True` to use this hook properly')
    ddp_ref = weakref.ref(ddp)
    pg = dist.get_backend(ddp_ref().process_group)
    if pg != dist.Backend.NCCL and pg != 'hccl':
        raise RuntimeError('Overlapping DDP with ZeRO using this approach currently requires NCCL/HCCL backend to avoid hangs')
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_fn(state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Returns a :class:`Future` that gives a gradient bucket tensor and\n        performs the equivalent of a :class:`ZeroRedundancyOptimizer`\n        :meth:`step` if ``bucket`` is the last gradient bucket.\n\n        The function performs additional computation on the iteration that\n        the :class:`DistributedDataParallel` buckets are rebuilt to collect\n        information used to implement the modified hook.\n\n        Arguments:\n            state (Any): any state for the hook.\n            bucket (dist.GradBucket): the :class:`DistributedDataParallel`\n                gradient bucket.\n        '
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut
        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank
        assert overlap_info.status == _OverlapStatus.INITIALIZED
        assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, '`assigned_ranks_per_bucket` is not fully constructed'
        assigned_to_bucket = rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut
        if len(overlap_info.bucket_indices_seen) > 0:
            assert overlap_info.bucket_indices_seen[-1] == bucket_index - 1, 'Bucket indices are not in incremental order'
        else:
            assert bucket_index == 0, 'Bucket indices do not start from 0'
        overlap_info.bucket_indices_seen.append(bucket_index)
        num_buckets = len(overlap_info.params_per_bucket)
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut
        for bucket_index in range(num_buckets):
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            if rank in assigned_ranks:
                assert bucket_index in overlap_info.bucket_index_to_future, f'All-reduce future for bucket {bucket_index} not saved on rank {rank}'
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)
            _broadcast_bucket(bucket_index, zero)
        overlap_info.wait_for_broadcasts()
        overlap_info.clear_per_iter_info()
        return fut
    return hook_with_zero_fn

def hook_with_zero_step_interleaved(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool=False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    if False:
        while True:
            i = 10
    "\n    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`\n    optimizer step with the :class:`DistributedDataParallel` backward pass,\n    where the optimizer step computation interleaves with the backward\n    computation.\n\n    This approach overlaps the optimizer computation and communication with the\n    backward computation and communication. In particular, once a bucket's\n    gradients have been computed, the optimizer computation using those\n    gradients is launched (though the actual computation must wait for the\n    bucket's all-reduce to complete). This yields an interleaving of all-\n    reduces and broadcasts in the communication stream.\n\n    This approach may be preferred over :meth:`hook_with_zero_step` if\n    communication is relatively fast compared to computation.\n\n    Arguments:\n        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to\n            modify.\n        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`\n            instance to use.\n        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`\n            instance to use.\n        shard_buckets (bool): if ``True``, then the assignment of each\n            :class:`DistributedDataParallel` bucket is partitioned across\n            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.\n            across possibly multiple ranks) to approximate uniformity; if\n            ``False``, then each bucket is wholly assigned to a single\n            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).\n\n    Returns:\n        The modified hook.\n\n    Raises:\n        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.\n        RuntimeError: if using any backend other than NCCL since currently\n            Gloo may hang.\n\n    .. warning::\n        Given the way that overlapping :class:`DistributedDataParallel` with\n        :class:`ZeroRedundancyOptimizer` is currently implemented, the first\n        two or three training iterations do not perform parameter updates in\n        the optimizer step, depending on if ``static_graph=False`` or\n        ``static_graph=True``, respectively. This is because it needs\n        information about the gradient bucketing strategy used by\n        :class:`DistributedDataParallel`, which is not finalized until the\n        second forward pass if ``static_graph=False`` or until the third\n        forward pass if ``static_graph=True``.\n    "
    if not zero._overlap_with_ddp:
        raise ValueError('ZeroRedundancyOptimizer must be constructed with `overlap_with_ddp=True` to use this hook properly')
    ddp_ref = weakref.ref(ddp)
    pg = dist.get_backend(ddp_ref().process_group)
    if pg != dist.Backend.NCCL and pg != 'hccl':
        raise RuntimeError('Overlapping DDP with ZeRO using this approach currently requires NCCL/HCCL backend to avoid hangs')
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_interleaved_fn(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Returns a :class:`Future` that gives a gradient bucket tensor and\n        performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step` using\n        the gradients in that bucket.\n        Arguments:\n            state: any state for the hook.\n            bucket (dist.GradBucket): the :class:`DistributedDataParallel`\n                gradient bucket.\n        '
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            if False:
                print('Hello World!')
            '\n            Performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step`\n            using the gradients in the given :class:`DistributedDataParallel`\n            gradient bucket.\n\n            Returns:\n                A :class:`torch.Tensor` representing the contents of the\n                gradient bucket.\n            '
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)
            _broadcast_bucket(bucket_index, zero)
            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()
            return bucket.buffer()
        return fut.then(zero_step)
    return hook_with_zero_interleaved_fn