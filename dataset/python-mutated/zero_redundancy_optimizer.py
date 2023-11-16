"""Zero Redundancy Optimizer."""
import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
logger = logging.getLogger(__name__)
__all__ = ['ZeroRedundancyOptimizer']

def _recursive_copy_to_device(value: Any, non_blocking: bool, device: torch.device) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Recursively searches lists, tuples, dicts and copies tensors to device if possible.\n\n    Non-tensor values are passed as-is in the result.\n\n    .. note:  These are all copies, so if there are two objects that reference\n    the same object, then after this call, there will be two different objects\n    referenced on the device.\n    '
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, (list, tuple)):
        values = [_recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for val in value]
        return values if isinstance(value, list) else tuple(values)
    if isinstance(value, collections.abc.Mapping):
        return {key: _recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for (key, val) in value.items()}
    return value

def _is_trainable(param: torch.Tensor) -> bool:
    if False:
        i = 10
        return i + 15
    'Return if a parameter is trainable, where trainability is equivalent to requiring a gradient.'
    return param.requires_grad

def _broadcast_object(obj: Any, src_rank: int, group: object=dist.group.WORLD, device: torch.device=torch.device('cpu')) -> Any:
    if False:
        i = 10
        return i + 15
    '\n    Broadcasts an object to the given group.\n\n    It will be sending the object if called from the source rank and receiving\n    the object otherwise.\n\n    Arguments:\n        obj: object to broadcast; only used if called on the source rank.\n        src_rank (int): source rank.\n        group (``ProcessGroup``, optional): group used for the broadcast\n            (default: ``dist.group.WORLD``).\n        device (``torch.device``, optional): device to send from or receive\n            to (default: ``torch.device("cpu")``).\n\n    Returns:\n        The broadcasted object.\n    '
    if dist.get_rank() == src_rank:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(device)
        data_send_tensor = torch.ByteTensor(data).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        length_tensor = torch.LongTensor([0]).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device)
    return obj

class _ZeROJoinHook(JoinHook):

    def __init__(self, zero):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(zero, ZeroRedundancyOptimizer), 'ZeRO join hook requires passing in a ZeroRedundancyOptimizer instance as the state'
        self.zero = zero
        super().__init__()

    def main_hook(self):
        if False:
            while True:
                i = 10
        "\n        Perform an optimizer step.\n\n        This step updates the joined process's shard of\n        the parameters and broadcasts those parameters.\n        "
        self.zero.step()

class _DDPBucketAssignment:
    """
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[torch.Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device (torch.device): device on which the parameters are stored.
        tensor (torch.Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """

    def __init__(self, bucket_index: int, parameters: List[torch.Tensor], offset: int):
        if False:
            return 10
        self.bucket_index = bucket_index
        self.parameters = parameters
        self.offset = offset
        if len(self.parameters) == 0:
            raise ValueError('Empty bucket assignment')
        self.device: torch.device = self.parameters[0].device
        self.tensor: Optional[torch.Tensor] = None

class _OverlapStatus(enum.IntEnum):
    """
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """
    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2

class _OverlapInfo:
    """
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity following
            a threshold given by the total parameter size divided by the world
            size; if ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank);
            this should be set to the value passed into the hook constructor.
        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        num_bucket_assignments (int): total number of bucket assignments across
            all ranks; this is equal to the number of
            :class:`DistributedDataParallel` gradient buckets if
            ``shard_buckets=False`` and possibly greater otherwise.
        total_size (int, optional): total size of all buckets (i.e. sum of
            ``param.numel()`` for all ``param`` across all buckets) if
            ``shard_buckets=True``; otherwise, ``None``.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """

    def __init__(self, world_size) -> None:
        if False:
            return 10
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        self.shard_buckets: bool = False
        self.params_per_bucket: List[List[torch.Tensor]] = []
        self.params_per_rank: List[List[torch.Tensor]] = [[] for _ in range(world_size)]
        self.offsets: Dict[int, int] = {}
        self.assigned_ranks_per_bucket: List[Set[int]] = []
        self.num_bucket_assignments: int = 0
        self.total_size: Optional[int] = None
        self.broadcast_handles: List[Any] = []
        self.bucket_indices_seen: List[int] = []
        self.bucket_index_to_future: Dict[int, torch.futures.Future] = {}
        self.bucket_index_to_bucket: Dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        if False:
            return 10
        '\n        Wait for all parameter broadcasts.\n\n        This function should be called once all broadcasts have been scheduled,\n        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``\n        in preparation for the next iteration.\n        '
        assert len(self.broadcast_handles) == self.num_bucket_assignments, f'Missing at least one broadcast handle on rank {dist.get_rank()}'
        _ = [x.wait() for x in self.broadcast_handles]
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        if False:
            return 10
        '\n        Clear the data structures that are modified per-iteration.\n\n        This function should be called at the end of an iteration.\n        '
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()

class ZeroRedundancyOptimizer(Optimizer, Joinable):
    """
    Wrap an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>` and shards its states across ranks in the group.

    The sharing is done as described by ZeRO_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    Example::

        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``. To adjust for this, one option
        is to prepend dummy inputs.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(self, params, optimizer_class: Type[Optimizer], process_group: Optional[Any]=None, parameters_as_bucket_view: bool=False, overlap_with_ddp: bool=False, **defaults: Any):
        if False:
            while True:
                i = 10
        'Init.'
        params = self._verify_and_init_params(params)
        self._verify_same_dense_param_type()
        self.initialized = False
        Optimizer.__init__(self, params, defaults)
        Joinable.__init__(self)
        self._param_to_rank_cache: Dict[torch.Tensor, int] = {}
        self._param_to_index_cache: Dict[torch.Tensor, int] = {}
        self._partition_parameters_cache: List[List[Dict]] = []
        self._index_to_param_cache: List[torch.Tensor] = []
        self._device_to_params_per_rank_cache: Dict[torch.device, List[List[torch.Tensor]]] = {}
        self._bucket_assignments_per_rank_cache: List[Dict[int, _DDPBucketAssignment]] = []
        self._is_trainable_mask = self._get_is_trainable_mask()
        self._default_device = self._all_params[0].device
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size: int = dist.get_world_size(self.process_group)
        self.rank: int = dist.get_rank(self.process_group)
        self.global_rank: int = dist.distributed_c10d.get_global_rank(self.process_group, self.rank)
        self._overlap_with_ddp: bool = overlap_with_ddp
        self._optim_defaults = defaults
        self._optim_constructor = self._get_optimizer_constructor(optimizer_class)
        if not overlap_with_ddp:
            self._init_local_optimizer()
        else:
            self._overlap_info: _OverlapInfo = _OverlapInfo(self.world_size)
            if parameters_as_bucket_view:
                logger.warning('`parameters_as_bucket_view=True` will be ignored since `overlap_with_ddp=True`; instead, a different bucketing strategy will be used')
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._buckets: List[List[torch.Tensor]] = []
        self._build_param_buckets()
        self._all_state_dicts: List[Dict[str, Any]] = []
        self.initialized = True

    def _clear_cache(self) -> None:
        if False:
            return 10
        'Clear the cached data structures giving partition information.'
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()
        self._device_to_params_per_rank_cache.clear()
        self._bucket_assignments_per_rank_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        if False:
            while True:
                i = 10
        "\n        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.\n\n        This can be useful when fine tuning a pre-trained network, as frozen\n        layers can be made trainable and added to the :class:`Optimizer` as\n        training progresses.\n\n        Arguments:\n            param_group (dict): specifies the parameters to be optimized and\n                group-specific optimization options.\n\n        .. warning:: This method handles updating the shards on all partitions\n            but needs to be called on all ranks. Calling this on a subset of\n            the ranks will cause the training to hang because communication\n            primitives are called depending on the managed parameters and\n            expect all the ranks to participate on the same set of parameters.\n        "
        if self.initialized and self._overlap_with_ddp:
            raise RuntimeError('ZeroRedundancyOptimizer with `overlap_with_ddp=True` only supports a single parameter group')
        super().add_param_group(param_group)
        if self.initialized:
            self._clear_cache()
            param_groups = self._partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def consolidate_state_dict(self, to: int=0) -> None:
        if False:
            return 10
        '\n        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.\n\n        Arguments:\n            to (int): the rank that receives the optimizer states (default: 0).\n\n        Raises:\n            RuntimeError: if ``overlap_with_ddp=True`` and this method is\n                called before this :class:`ZeroRedundancyOptimizer` instance\n                has been fully initialized, which happens once\n                :class:`DistributedDataParallel` gradient buckets have been\n                rebuilt.\n\n        .. warning:: This needs to be called on all ranks.\n        '
        self._check_overlap_initialized()
        self._sync_param_groups(self.param_groups, self.optim.param_groups)
        empty_messenger = torch.tensor([0], dtype=torch.uint8, device=self._default_device)
        self._all_state_dicts = []
        for rank in range(self.world_size):
            global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
            if self.rank == to:
                if rank == self.rank:
                    self._all_state_dicts.append(_recursive_copy_to_device(self.optim.state_dict(), non_blocking=True, device=torch.device('cpu')))
                else:
                    local_state_dict = _broadcast_object(empty_messenger, src_rank=global_rank, group=self.process_group, device=self._default_device)
                    self._all_state_dicts.append(_recursive_copy_to_device(local_state_dict, non_blocking=True, device=torch.device('cpu')))
            elif rank == self.rank:
                _ = _broadcast_object(self.optim.state_dict(), src_rank=self.global_rank, group=self.process_group, device=self._default_device)
            elif rank != to:
                _ = _broadcast_object(empty_messenger, src_rank=global_rank, group=self.process_group, device=self._default_device)

    def _verify_params_per_rank(self, params_per_rank: List[List[torch.Tensor]]) -> None:
        if False:
            return 10
        '\n        Verify ``params_per_rank`` for :meth:`_partition_parameters`.\n\n        The verification is done by checking that ``params_per_rank`` has length equal\n        to the world size and that it does not contain any parameters not passed into the\n        :class:`ZeroRedundancyOptimizer` constructor.\n\n        The parameters in ``params_per_rank`` being a strict subset of those\n        passed into the constructor is valid since some parameters may be\n        frozen.\n\n        Raises:\n            ValueError: if ``params_per_rank`` does not have length equal to\n                the world size or if it contains a parameter that was not\n                passed into the :class:`ZeroRedundancyOptimizer` constructor.\n        '
        if len(params_per_rank) != self.world_size:
            raise ValueError('`params_per_rank` must have length equal to the world size')
        all_params_set = set(self._all_params)
        for params in params_per_rank:
            for param in params:
                if param not in all_params_set:
                    raise ValueError('Passing a new parameter in `params_per_rank` that was not passed into the ZeroRedundancyOptimizer constructor')

    def _partition_param_group(self, param_group: Dict[str, Any], params_per_rank: List[List[torch.Tensor]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Partition the parameter group ``param_group`` according to ``params_per_rank``.\n\n        The partition will modify the ``self._partition_parameters_cache``. This method should\n        only be used as a subroutine for :meth:`_partition_parameters`.\n\n        Arguments:\n            param_group (dict[str, Any]): a parameter group as normally defined\n                in an optimizer state.\n            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of\n                length world size containing :class:`list` s of parameters to\n                assign to each rank.\n        '
        for (rank, params) in enumerate(params_per_rank):
            rank_param_group = copy.copy(param_group)
            rank_param_group['params'] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(self, params_per_rank: Optional[List[List[torch.Tensor]]]=None) -> List[List[Dict]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Partitions parameters across distributed data parallel ranks.\n\n        Arguments:\n            params_per_rank (list[list[torch.Tensor]], optional): a\n                :class:`list` of length world size containing :class:`list` s\n                of parameters to assign to each rank; this provides a way to\n                specify a partition manually.\n                If ``None``, the parameters are partitioned according to an\n                internal algorithm.\n                (default: ``None``)\n\n        Returns:\n            A :class:`list` where each element of the list contains the\n            ``param_groups`` for a rank (which itself is a :class:`list` of\n            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank\n            stores the ``param_groups`` for all ranks for the collective\n            communication in :meth:`step`.\n\n        Raises:\n            ValueError: see :meth:`_validate_params_per_rank`.\n            RuntimeError: if ``params_per_rank`` is not ``None`` and this\n                :class:`ZeroRedundancyOptimizer` instance is using more than\n                one parameter group.\n        '
        if params_per_rank is None:
            if len(self._partition_parameters_cache) == 0:
                self._partition_parameters_cache = [[] for _ in range(self.world_size)]
                sizes = [0] * self.world_size
                for param_group in self.param_groups:
                    param_group_params_per_rank: List[List] = [[] for _ in range(self.world_size)]
                    params_sorted = sorted(param_group['params'], key=lambda t: t.numel(), reverse=True)
                    for param in params_sorted:
                        rank = self._get_min_index(sizes)
                        param_group_params_per_rank[rank].append(param)
                        sizes[rank] += param.numel()
                    self._partition_param_group(param_group, param_group_params_per_rank)
            return self._partition_parameters_cache
        assert len(self._partition_parameters_cache) == 0, 'Specifying `params_per_rank` should only be done when the parameters have not been partitioned yet'
        if len(self.param_groups) != 1:
            raise RuntimeError('Specifying `params_per_rank` only supports a single parameter group')
        self._verify_params_per_rank(params_per_rank)
        self._partition_parameters_cache = [[] for _ in range(self.world_size)]
        param_group = self.param_groups[0]
        self._partition_param_group(param_group, params_per_rank)
        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        if False:
            for i in range(10):
                print('nop')
        ':class:`dict` mapping parameters to their assigned data parallel rank in the partition.'
        if len(self._param_to_rank_cache) == 0:
            for (rank, param_groups) in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group['params']:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> Dict[torch.Tensor, int]:
        if False:
            i = 10
            return i + 15
        "\n        :class:`dict` mapping parameters to their indices in the global optimizer state.\n\n        NOTE: This assumes that the global optimizer state's indexing (in\n        ``state_dict``) follows a linear ordering over the parameter groups.\n        "
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {p: i for (i, p) in enumerate(chain(*(g['params'] for g in self.param_groups)))}
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> List[torch.Tensor]:
        if False:
            return 10
        'List mapping parameter indices in the global optimizer scheme to the actual params.'
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(chain(*(g['params'] for g in self.param_groups)))
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        if False:
            print('Hello World!')
        '\n        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.\n\n        Arguments:\n            rank (int): the source rank.\n\n        Returns:\n            A :class:`list` of async work handles for the ``broadcast()`` s\n            performed to synchronize the parameters.\n        '
        assert not self._overlap_with_ddp, '`_broadcast_params_from_rank()` should not be used if `overlap_with_ddp=True`; instead, the broadcasting should happen in the DDP communication hook'
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
                handles.append(dist.broadcast(tensor=bucket, src=global_rank, group=self.process_group, async_op=True))
        else:
            param_groups = self._partition_parameters()[rank]
            global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
            for param_group in param_groups:
                for param in param_group['params']:
                    handles.append(dist.broadcast(tensor=param.data, src=global_rank, group=self.process_group, async_op=True))
        return handles

    def _sync_params(self):
        if False:
            return 10
        '\n        Sync all parameter shards across the ranks.\n\n        This rank sends its shard of the parameters to all other ranks and\n        receives a shard from each other rank. This is done using\n        ``broadcast()``. Parameters are sent bucket-by-bucket if\n        ``parameters_as_bucket_view=True``and sent parameter-by-parameter\n        otherwise.\n        '
        handles = []
        for rank in range(self.world_size):
            handles.extend(self._broadcast_params_from_rank(rank))
        _ = [x.wait() for x in handles]

    @property
    def _device_to_params_per_rank(self) -> Dict[torch.device, List[List[torch.Tensor]]]:
        if False:
            print('Hello World!')
        "\n        Return device parameters assigned per rank.\n\n        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter\n        lists filtered to only include the parameters stored on that device.\n        Each per-rank parameter list gives the parameters assigned to that rank\n        to update.\n\n        This is used for constructing the parameter buckets if\n        ``parameters_as_bucket_view=True``.\n\n        Let ``dev_i`` denote the ``i``th device for this rank. Then:\n        ``dev_0`` maps to a list containing:\n            rank 0's assigned parameters stored on ``dev_0``,\n            rank 1's assigned parameters stored on ``dev_0``,\n            ...\n        ``dev_1`` maps to a list containing:\n            rank 0's assigned parameters stored on ``dev_1``,\n            rank 1's assigned parameters stored on ``dev_1``,\n            ...\n        ...\n        "
        assert self.parameters_as_bucket_view, '`_device_to_params_per_rank` should only be used if `parameters_as_bucket_view=True`'
        if len(self._device_to_params_per_rank_cache) == 0:
            for (rank, param_groups) in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group['params']:
                        device = param.device
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [[] for _ in range(self.world_size)]
                        self._device_to_params_per_rank_cache[device][rank].append(param)
        return self._device_to_params_per_rank_cache

    def _get_min_index(self, values: List[int], disallowed_indices: Optional[Set[int]]=None) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return ``values.index(min(values))``, except only uses one pass.\n\n        It also excludes any indices in ``disallowed_indices`` if provided.\n\n        Arguments:\n            values: (List[int]): :class:`list` of values.\n            disallowed_indices (Optional[Set[int]]): indices that are\n                disallowed from being the returned min index.\n        '
        min_index = -1
        min_value = float('inf')
        for (i, value) in enumerate(values):
            if disallowed_indices and i in disallowed_indices:
                continue
            if value < min_value:
                min_value = value
                min_index = i
        assert min_index >= 0, 'All indices are disallowed'
        return min_index

    def _assign_bucket_subset_to_rank(self, bucket_index: int, bucket_params: List[torch.Tensor], bucket_offset: int, assigned_rank: int, assigned_ranks_per_bucket: List[Set[int]]) -> None:
        if False:
            print('Hello World!')
        "\n        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.\n\n        The model parameters given by ``bucket_params`` represents a (possibly non-strict)\n        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.\n\n        Arguments:\n            bucket_index (int): index of the :class:`DistributedDataParallel`\n                gradient bucket.\n            bucket_params (List[torch.Tensor]): subset of the parameters\n                corresponding to the bucket to assign.\n            bucket_offset (int): offset giving the index of the first element\n                in ``bucket_params`` in the bucket's full parameter list.\n            assigned_rank (int): group rank to assign to.\n            assigned_ranks_per_bucket (List[Set[int]]): :class:`set` of group ranks\n                assigned to each bucket.\n        "
        overlap_info = self._overlap_info
        if len(bucket_params) == 0:
            raise ValueError('Empty bucket assignment')
        params_per_rank = overlap_info.params_per_rank
        offsets = overlap_info.offsets
        self._bucket_assignments_per_rank_cache[assigned_rank][bucket_index] = _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)
        if self.global_rank == assigned_rank:
            offsets[bucket_index] = len(params_per_rank[assigned_rank])
        params_per_rank[assigned_rank].extend(bucket_params)
        assigned_ranks_per_bucket[bucket_index].add(assigned_rank)
        self._overlap_info.num_bucket_assignments += 1

    @property
    def _bucket_assignments_per_rank(self) -> List[Dict[int, _DDPBucketAssignment]]:
        if False:
            return 10
        '\n        Return DDP bucket parameters assigned per rank.\n\n        :class:`list` of length world size consisting of :class:`dict` s\n        mapping bucket indices to :class:`_DDPBucketAssignment` s for each\n        rank.\n        '
        assert self._overlap_with_ddp, '`_bucket_assignments_per_rank` only be used if `overlap_with_ddp=True`'
        if len(self._bucket_assignments_per_rank_cache) > 0:
            return self._bucket_assignments_per_rank_cache
        overlap_info = self._overlap_info
        assert overlap_info.status == _OverlapStatus.INITIALIZED
        self._bucket_assignments_per_rank_cache = [{} for _ in range(self.world_size)]
        params_per_bucket = overlap_info.params_per_bucket
        if overlap_info.shard_buckets:
            assert overlap_info.total_size is not None, '`total_size` was not computed'
            threshold = overlap_info.total_size / self.world_size
            size_per_rank = [0 for _ in range(self.world_size)]
        num_buckets = len(params_per_bucket)
        overlap_info.assigned_ranks_per_bucket = [set() for _ in range(num_buckets)]
        assigned_ranks_per_bucket = overlap_info.assigned_ranks_per_bucket
        if not overlap_info.shard_buckets:
            for (bucket_index, bucket_params) in enumerate(params_per_bucket):
                assert len(bucket_params) > 0, 'Empty bucket'
                assigned_rank = self._get_assigned_rank(bucket_index)
                self._assign_bucket_subset_to_rank(bucket_index, bucket_params, 0, assigned_rank, assigned_ranks_per_bucket)
        else:
            params_per_bucket_enum = sorted(enumerate(params_per_bucket), key=lambda x: sum((p.numel() for p in x[1])))
            for (bucket_index, bucket_params) in params_per_bucket_enum:
                assert len(bucket_params) > 0, 'Empty bucket'
                bucket_offset = 0
                assignment_size = 0
                for (param_index, param) in enumerate(bucket_params):
                    param_numel = param.numel()
                    if assignment_size + param_numel >= threshold and param_index > bucket_offset:
                        assigned_rank = self._get_min_index(size_per_rank, assigned_ranks_per_bucket[bucket_index])
                        self._assign_bucket_subset_to_rank(bucket_index, bucket_params[bucket_offset:param_index], bucket_offset, assigned_rank, assigned_ranks_per_bucket)
                        size_per_rank[assigned_rank] += assignment_size
                        bucket_offset = param_index
                        assignment_size = 0
                    assignment_size += param_numel
                assigned_rank = self._get_min_index(size_per_rank, assigned_ranks_per_bucket[bucket_index])
                self._assign_bucket_subset_to_rank(bucket_index, bucket_params[bucket_offset:], bucket_offset, assigned_rank, assigned_ranks_per_bucket)
                size_per_rank[assigned_rank] += assignment_size
        return self._bucket_assignments_per_rank_cache

    def _local_step(self, gradients: Optional[List[Optional[torch.Tensor]]]=None, closure: Optional[Callable[[], float]]=None, **kwargs: Any) -> Optional[float]:
        if False:
            return 10
        '\n        Perform a single optimizer step without syncing parameters across ranks.\n\n        Arguments:\n            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`\n                of length equal to the number of parameters assigned to this\n                rank containing gradient tensors or ``None`` as its elements;\n                a ``None`` in the :class:`list` indicates that the\n                corresponding parameter should not be updated.\n                If the argument itself is ``None``, then all parameters are\n                updated, and the gradients are assumed to be already populated.\n                (default: ``None``)\n            closure (Callable): a closure that re-evaluates the model and\n                returns the loss; optional for most optimizers and should be\n                ``None`` if ``gradients`` is not ``None``; (default: ``None``)\n        Returns:\n            Optional loss depending on the underlying local optimizer.\n\n        .. warning::\n            The argument ``gradients`` should only be specified (i.e. not\n            ``None``) if ``overlap_with_ddp=True``, in which case\n            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.\n        '
        Join.notify_join_context(self)
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            if self._overlap_with_ddp:
                raise RuntimeError('ZeroRedundancyOptimizer with `overlap_with_ddp=True` does not support changing parameter trainability at run time')
            logger.warning('ZeroRedundancyOptimizer detected that the trainable parameters changed; rebuilding the parameter buckets if enabled')
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask
        self._sync_param_groups(self.param_groups, self.optim.param_groups)
        if gradients is None:
            loss = self.optim.step(**kwargs) if closure is None else self.optim.step(closure=closure, **kwargs)
        else:
            assert self._overlap_with_ddp, 'Specifying `gradients` should not be used when `overlap_with_ddp=False`'
            assert closure is None, '`closure` is not supported when using a local functional optimizer'
            loss = self.optim.step(gradients=gradients)
        self._sync_param_groups(self.optim.param_groups, self.param_groups)
        return loss

    def step(self, closure: Optional[Callable[[], float]]=None, **kwargs: Any) -> Optional[float]:
        if False:
            return 10
        '\n        Perform a single optimizer step and syncs parameters across all ranks.\n\n        Arguments:\n            closure (Callable): a closure that re-evaluates the model and\n                returns the loss; optional for most optimizers.\n        Returns:\n            Optional loss depending on the underlying local optimizer.\n\n        .. note: Any extra parameters are passed to the base optimizer as-is.\n        '
        if self._overlap_with_ddp:
            logger.warning('`step()` should not be included in the training loop when `overlap_with_ddp=True`')
            return None
        loss = self._local_step(closure=closure, **kwargs)
        self._sync_params()
        return loss

    def join_hook(self, **kwargs):
        if False:
            return 10
        '\n        Return the ZeRO join hook.\n\n        It enables training on uneven inputs by\n        shadowing the collective communications in the optimizer step.\n\n        Gradients must be properly set before this hook is called.\n\n        Arguments:\n            kwargs (dict): a :class:`dict` containing any keyword arguments\n                to modify the behavior of the join hook at run time; all\n                :class:`Joinable` instances sharing the same join context\n                manager are forwarded the same value for ``kwargs``.\n\n        This hook does not support any keyword arguments; i.e. ``kwargs`` is\n        unused.\n        '
        return _ZeROJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        if False:
            while True:
                i = 10
        'Return default device.'
        return self._default_device

    @property
    def join_process_group(self) -> Any:
        if False:
            print('Hello World!')
        'Return process group.'
        return self.process_group

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Load the state pertaining to the given rank from the input ``state_dict``, updating the local optimizer as needed.\n\n        Arguments:\n            state_dict (dict): optimizer state; should be an object returned\n                from a call to :meth:`state_dict`.\n\n        Raises:\n            RuntimeError: if ``overlap_with_ddp=True`` and this method is\n                called before this :class:`ZeroRedundancyOptimizer` instance\n                has been fully initialized, which happens once\n                :class:`DistributedDataParallel` gradient buckets have been\n                rebuilt.\n        '
        self._check_overlap_initialized()
        for (index, value) in state_dict['state'].items():
            param = self._index_to_param[index]
            if self._param_to_rank[param] != self.rank:
                state_dict['state'][index] = None
            else:
                self.optim.state[param] = _recursive_copy_to_device(value, non_blocking=True, device=param.device)
                for (state_name, state_value) in self.optim.state[param].items():
                    if torch.is_tensor(state_value) and state_value.dim() == 0:
                        self.optim.state[param][state_name] = state_value.cpu()
        super().load_state_dict(state_dict)
        self._sync_param_groups(state_dict['param_groups'], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    def state_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Return the last global optimizer state known to this rank.\n\n        .. warning:\n            If the state has not been consolidated to this rank, this raises a\n            runtime error, and even if it has, the state may not be up-to-date,\n            depending on when :meth:`consolidate_state_dict` was last called.\n\n        Raises:\n            RuntimeError: if ``overlap_with_ddp=True`` and this method is\n                called before this :class:`ZeroRedundancyOptimizer` instance\n                has been fully initialized, which happens once\n                :class:`DistributedDataParallel` gradient buckets have been\n                rebuilt; or if this method is called without a preceding call\n                to :meth:`consolidate_state_dict`.\n        '
        self._check_overlap_initialized()
        if len(self._all_state_dicts) == 0:
            raise RuntimeError(f'Optimizer state has not been consolidated on this rank. Please call `consolidate_state_dict(to={self.rank})` on all ranks beforehand if you meant to save the global state.')
        state_dict = super().state_dict()
        for (rank, local_state_dict) in enumerate(self._all_state_dicts):
            local_param_groups = local_state_dict['param_groups']
            global_param_groups = self._partition_parameters()[rank]
            assert len(local_param_groups) == len(global_param_groups), 'Mismatch between number of local and global parameter groups'
            for (local_param_group, global_param_group) in zip(local_param_groups, global_param_groups):
                local_param_indices = local_param_group['params']
                global_params = global_param_group['params']
                assert len(local_param_indices) == len(global_params), 'Mismatch between number of local and global parameters in parameter group'
                for (local_param_index, global_param) in zip(local_param_indices, global_params):
                    if local_param_index in local_state_dict['state']:
                        global_param_index = self._param_to_index[global_param]
                        state_dict['state'][global_param_index] = local_state_dict['state'][local_param_index]
        state_dict['state'] = dict(sorted(state_dict['state'].items()))
        return state_dict

    @staticmethod
    def _sync_param_groups(src_param_groups: List[Dict[Any, Any]], dst_param_groups: List[Dict[Any, Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sync the attributes from the source parameter groups to the destination parameter groups.\n\n        Example attributes include learning rate or scheduler attributes. The\n        two parameter groups should have the same length (i.e. same number of\n        parameter groups).\n\n        Arguments:\n            src_param_groups (list[dict]): parameter groups giving the\n                attribute settings to copy.\n            dst_param_groups (list[dict]): parameter groups giving the\n                attribute settings to set.\n        '
        assert len(src_param_groups) == len(dst_param_groups), 'Mismatch between number of source and destination parameter groups'
        for (src_param_group, dst_param_group) in zip(src_param_groups, dst_param_groups):
            for attr in filter(lambda x: x != 'params', src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]

    def _build_param_buckets(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Build parameter buckets if ``parameters_as_bucket_view=True``.\n\n        For each device that stores this rank's parameters, there is a\n        bucket (represented as a tensor) containing all of the parameters on\n        that device that are assigned to a given rank in the parameter update\n        partition.\n\n        This method is called in the constructor and any time parameter\n        trainability is changed.\n\n        .. warning::\n            The current implementation assumes that all of the parameters in a\n            bucket are of the same dense type when allocating the bucket's\n            tensor.\n\n        .. warning::\n            If the model parameters are stored across more than one device,\n            then the storage partitioning must be the same across all\n            processes in order for parameter synchronization to work.\n        "
        if not self.parameters_as_bucket_view or self._overlap_with_ddp:
            return
        num_devices = len(self._device_to_params_per_rank)
        self._buckets = [[] for _ in range(num_devices)]
        for (dev_i, (device, params_per_rank)) in enumerate(self._device_to_params_per_rank.items()):
            for params in params_per_rank:
                bucket_size = 0
                dtype = None
                trainable_params = []
                for param in params:
                    if not _is_trainable(param):
                        param.data = param.data.detach().clone()
                    else:
                        bucket_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype
                if bucket_size == 0:
                    bucket = torch.zeros(1, device=device)
                else:
                    bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next
                self._buckets[dev_i].append(bucket)

    def _build_ddp_param_buckets(self) -> None:
        if False:
            print('Hello World!')
        '\n        Build the DDP bucket with parameters assigned to this rank.\n\n        For each DDP bucket with parameters assigned to this rank, flattens the\n        data of those parameters into a single tensor and saves the tensor to\n        the ``tensor`` attribute in the corresponding\n        :class:`_DDPBucketAssignment` instance stored in\n        ``self._bucket_assignments_per_rank``.\n\n        :class:`DistributedDataParallel` guarantees that the parameters\n        corresponding to a gradient bucket have the same device and the same\n        dtype.\n        '
        for bucket_assignments in self._bucket_assignments_per_rank:
            for bucket_assignment in bucket_assignments.values():
                params = bucket_assignment.parameters
                bucket_size = 0
                dtype = None
                for param in params:
                    assert _is_trainable(param), 'Model parameter corresponding to a gradient in a DDP bucket should require a gradient'
                    bucket_size += param.numel()
                    dtype = param.dtype
                assert bucket_size > 0, 'Empty bucket'
                tensor = torch.empty(bucket_size, dtype=dtype, device=bucket_assignment.device)
                offset = 0
                for param in params:
                    offset_next = offset + param.numel()
                    tensor[offset:offset_next].copy_(param.data.flatten())
                    param.data = tensor[offset:offset_next].view_as(param.data)
                    offset = offset_next
                bucket_assignment.tensor = tensor

    def _verify_and_init_params(self, params: Any) -> Union[List[torch.Tensor], List[dict]]:
        if False:
            while True:
                i = 10
        '\n        Verify the type of ``params`` and initializes ``self._all_params`` as a :class:`list` of all parameters.\n\n        The initializagtion will first make sure that provided ``params`` is valid.\n\n        Arguments:\n            params (Any): Candidate parameter list or parameter groups to verify.\n\n        Raises:\n            TypeError: ``params`` has an invalid type.\n            ValueError: ``params`` is empty.\n\n        Returns:\n            The persistent form of ``params`` to be passed into the parent\n            :class:`Optimizer` constructor -- i.e. returns ``params`` as a\n            :class:`list` to ensure that it can be iterated over again.\n        '
        if isinstance(params, torch.Tensor):
            raise TypeError(f'`params` argument should be an iterable of Tensors, but got {torch.typename(params)}')
        try:
            all_params = list(params)
        except TypeError as e:
            raise TypeError(f'`params` argument should be an iterable of Tensors or dicts, but got {torch.typename(params)}') from e
        if len(all_params) == 0:
            raise ValueError('ZeroRedundancyOptimizer got an empty parameter list')
        all_tensors = True
        all_dicts = True
        for param in all_params:
            all_tensors &= isinstance(param, torch.Tensor)
            all_dicts &= isinstance(param, dict)
        if not all_tensors and (not all_dicts):
            raise TypeError('`params` argument should be an iterable of Tensors or dicts')
        if all_tensors:
            self._all_params = all_params
        elif all_dicts:
            self._all_params = []
            for param_group in all_params:
                if 'params' not in param_group:
                    raise ValueError("Each parameter group passed-in via `params` must have a 'params' key mapping to the parameters in the group")
                self._all_params.extend(param_group['params'])
        return all_params

    def _verify_same_dense_param_type(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Verify that all parameters are of the same dense type.\n\n        The method assumes that ``self._all_params`` has been initialized\n        and is non-empty.\n\n        Raises:\n            ValueError: ``params`` contains sparse parameters or parameters\n            of varying dense types.\n\n        NOTE: This method can be removed once support for sparse parameters\n        and varying parameter types is added.\n        '
        typename = torch.typename(self._all_params[0])
        if self._all_params[0].is_sparse:
            raise ValueError(f'ZeroRedundancyOptimizer only supports using the same dense type for all parameters but got {typename}')
        for param in self._all_params[1:]:
            other_typename = torch.typename(param)
            if other_typename != typename:
                raise ValueError(f'ZeroRedundancyOptimizer only supports using the same dense type for all parameters but got both {typename} and {other_typename}')

    def _get_is_trainable_mask(self) -> List[bool]:
        if False:
            print('Hello World!')
        'Return a boolean mask indicating if each parameter is trainable (``requires_grad``) or not.'
        return list(map(_is_trainable, self._all_params))

    def _init_local_optimizer(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Initialize this rank's local optimizer, responsible for its subset of the parameters.\n\n        The local optimizer is saved in ``self.optim``.\n        "
        assert self._optim_constructor is not None, 'The local optimizer class has not been set'
        param_groups = self._partition_parameters()[self.rank]
        if self._overlap_with_ddp:
            assert len(param_groups) == 1, 'Initializing the local functional optimizer with more than one parameter group'
            params = param_groups[0]['params']
            if '_allow_empty_param_list' in inspect.signature(self._optim_constructor).parameters:
                self.optim: Any = self._optim_constructor(params, **self._optim_defaults, _allow_empty_param_list=True)
            else:
                logger.warning('%s does not support the argument `_allow_empty_param_list`; ZeroRedundancyOptimizer may error due to an empty parameter list', self._optim_constructor)
                self.optim: Any = self._optim_constructor(params, **self._optim_defaults)
            if dist.get_debug_level() != dist.DebugLevel.OFF:
                local_numel = sum((p.numel() for p in params))
                num_assigned_buckets = len(self._bucket_assignments_per_rank[self.global_rank])
                logger.info('rank %s with %s parameters across %s buckets', self.global_rank, local_numel, num_assigned_buckets)
                if self.global_rank == 0:
                    logger.info('%s DDP buckets and %s bucket assignments', len(self._overlap_info.params_per_bucket), self._overlap_info.num_bucket_assignments)
        else:
            self.optim: Optimizer = self._optim_constructor(param_groups, **self._optim_defaults)
        if self._overlap_with_ddp and (not hasattr(self.optim, 'param_groups')):
            assert hasattr(self.optim, 'param_group'), 'The functional optimizer should set at least one of the attributes `param_group` or `param_groups`'
            self.optim.param_groups = [self.optim.param_group]
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

    def _init_zero_for_overlap(self) -> None:
        if False:
            print('Hello World!')
        'Perform a delayed initialization of the local optimizer and the supporting data structures.'
        assert self._overlap_with_ddp, '`_init_zero_for_overlap()` should only be called when `overlap_with_ddp=True`'
        self._overlap_info.status = _OverlapStatus.INITIALIZED
        self._clear_cache()
        self._partition_parameters(self._overlap_info.params_per_rank)
        self._build_ddp_param_buckets()
        self._init_local_optimizer()

    def _get_assigned_rank(self, bucket_index: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Return the single rank assigned to a :class:`DistributedDataParallel` gradient bucket.\n\n        Arguments:\n            bucket_index (int): index of the :class:`DistributedDataParallel`\n                bucket for which to get the assigned rank.\n        '
        assert not self._overlap_info.shard_buckets, 'The bucket assignment requires global bucket information and will be computed later; there should be no need to use this method'
        return bucket_index % self.world_size

    def _check_overlap_initialized(self):
        if False:
            print('Hello World!')
        '\n        Check the delayed initialization depending on the value of ``overlap_with_ddp``.\n\n        The delayed initialization has occurred (see\n        :meth:`_init_zero_for_overlap`) if ``overlap_with_ddp=True``, and\n        raises a ``RuntimeError`` if not. This should preface methods that\n        should not be run before that delayed initialization.\n\n        Raises:\n            RuntimeError: if ``overlap_with_ddp=True`` and\n                :meth:`_init_zero_for_overlap` has not been called.\n        '
        if self._overlap_with_ddp and self._overlap_info.status != _OverlapStatus.INITIALIZED:
            raise RuntimeError('This method should not be called until this ZeroRedundancyOptimizer instance has been fully initialized')

    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        if False:
            return 10
        '\n        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.\n\n        Returns:\n            - ``optimizer_class`` if ``overlap_with_ddp=False`` and\n                ``optimizer_class`` is not a functional optimizer.\n            - ``optimizer_class`` if ``overlap_with_ddp=True`` and\n                ``optimizer_class`` is already a functional optimizer.\n            - The functional equivalent of ``optimizer_class`` if\n                ``overlap_with_ddp=True`` and ``optimizer_class`` is not\n                already a functional optimizer (assuming the equivalent\n                exists).\n\n        Raises:\n            ValueError:\n\n                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is\n                    neither a functional optimizer nor translatable to a\n                    functional optimizer.\n                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a\n                    functional optimizer.\n        '
        functional_optims = functional_optim_map.values()
        if not self._overlap_with_ddp:
            if optimizer_class in functional_optims:
                raise ValueError(f'Passing in a functional optimizer {optimizer_class} when `overlap_with_ddp=False`')
            else:
                return optimizer_class
        elif optimizer_class in functional_optims:
            return optimizer_class
        elif optimizer_class in functional_optim_map:
            optim_constructor = functional_optim_map[optimizer_class]
            logger.info('Using the functional optimizer %s instead of %s since `overlap_with_ddp=True`', optim_constructor, optimizer_class)
            return optim_constructor
        else:
            raise ValueError(f'Using `ddp_with_overlap=True` requires using a functional optimizer, but there is no supported functional optimizer equivalent for {optimizer_class}')