"""Distributed Collective Communication (c10d)."""
import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import AllgatherOptions, AllreduceCoalescedOptions, AllreduceOptions, AllToAllOptions, _DistributedBackendOptions, BarrierOptions, BroadcastOptions, GatherOptions, PrefixStore, ProcessGroup, ReduceOp, ReduceOptions, ReduceScatterOptions, ScatterOptions, Store, DebugLevel, get_debug_level, Work
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous
DistStoreError = torch._C._DistStoreError
__all__ = ['Backend', 'BackendConfig', 'GroupMember', 'P2POp', 'all_gather', 'all_gather_coalesced', 'all_gather_multigpu', 'all_gather_object', 'all_reduce', 'all_reduce_coalesced', 'all_reduce_multigpu', 'all_to_all', 'all_to_all_single', 'barrier', 'batch_isend_irecv', 'broadcast', 'broadcast_multigpu', 'broadcast_object_list', 'destroy_process_group', 'gather', 'gather_object', 'get_backend_config', 'get_backend', 'get_rank', 'get_world_size', 'group', 'init_process_group', 'irecv', 'is_gloo_available', 'is_initialized', 'is_mpi_available', 'is_backend_available', 'is_nccl_available', 'is_torchelastic_launched', 'is_ucc_available', 'isend', 'monitored_barrier', 'new_group', 'new_subgroups', 'new_subgroups_by_enumeration', 'recv', 'reduce', 'reduce_multigpu', 'reduce_scatter', 'reduce_scatter_multigpu', 'scatter', 'scatter_object_list', 'send', 'supports_complex', 'AllreduceCoalescedOptions', 'AllreduceOptions', 'AllToAllOptions', 'BarrierOptions', 'BroadcastOptions', 'GatherOptions', 'PrefixStore', 'ProcessGroup', 'ReduceOp', 'ReduceOptions', 'ReduceScatterOptions', 'ScatterOptions', 'Store', 'DebugLevel', 'get_debug_level', 'Work', 'default_pg_timeout', 'get_group_rank', 'get_global_rank', 'get_process_group_ranks', 'reduce_op', 'all_gather_into_tensor', 'reduce_scatter_tensor']
_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
_UCC_AVAILABLE = True
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler

def _export_c_types():
    if False:
        i = 10
        return i + 15
    _public_types_to_change_module = [AllreduceCoalescedOptions, AllreduceOptions, AllToAllOptions, BarrierOptions, BroadcastOptions, GatherOptions, PrefixStore, ProcessGroup, ReduceOp, ReduceOptions, ReduceScatterOptions, ScatterOptions, Store, DebugLevel, get_debug_level, Work]
    for type in _public_types_to_change_module:
        type.__module__ = 'torch.distributed.distributed_c10d'
_export_c_types()
try:
    from torch._C._distributed_c10d import ProcessGroupMPI
    ProcessGroupMPI.__module__ = 'torch.distributed.distributed_c10d'
    __all__ += ['ProcessGroupMPI']
except ImportError:
    _MPI_AVAILABLE = False
try:
    from torch._C._distributed_c10d import ProcessGroupNCCL
    ProcessGroupNCCL.__module__ = 'torch.distributed.distributed_c10d'
    __all__ += ['ProcessGroupNCCL']
except ImportError:
    _NCCL_AVAILABLE = False
try:
    from torch._C._distributed_c10d import ProcessGroupGloo
    from torch._C._distributed_c10d import _ProcessGroupWrapper
    ProcessGroupGloo.__module__ = 'torch.distributed.distributed_c10d'
    __all__ += ['ProcessGroupGloo']
except ImportError:
    _GLOO_AVAILABLE = False
try:
    from torch._C._distributed_c10d import ProcessGroupUCC
    ProcessGroupUCC.__module__ = 'torch.distributed.distributed_c10d'
    __all__ += ['ProcessGroupUCC']
except ImportError:
    _UCC_AVAILABLE = False
logger = logging.getLogger(__name__)
PG_WRAPPER_STORE_PREFIX = 'pg_wrapper'

def supports_complex(reduceOp: ReduceOp) -> bool:
    if False:
        print('Hello World!')
    'Return true if reduce ops is supported. False otherwise.'
    denyList = [ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PRODUCT, ReduceOp.BAND, ReduceOp.BOR, ReduceOp.BXOR]
    return reduceOp not in denyList

class Backend:
    """
    An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, and other registered backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """
    UNDEFINED = 'undefined'
    GLOO = 'gloo'
    NCCL = 'nccl'
    UCC = 'ucc'
    MPI = 'mpi'
    _BackendPlugin = namedtuple('_BackendPlugin', ['creator_fn', 'extended_api'])
    _plugins: Dict[str, _BackendPlugin] = {}
    backend_list = [UNDEFINED, GLOO, NCCL, UCC, MPI]
    default_device_backend_map: Dict[str, str] = {'cpu': GLOO, 'cuda': NCCL}
    backend_capability: Dict[str, List[str]] = {GLOO: ['cpu', 'cuda'], NCCL: ['cuda'], UCC: ['cpu', 'cuda'], MPI: ['cpu', 'cuda']}
    backend_type_map: Dict[str, ProcessGroup.BackendType] = {UNDEFINED: ProcessGroup.BackendType.UNDEFINED, GLOO: ProcessGroup.BackendType.GLOO, NCCL: ProcessGroup.BackendType.NCCL, UCC: ProcessGroup.BackendType.UCC}

    def __new__(cls, name: str):
        if False:
            while True:
                i = 10
        'Create and return a new instance of the class.'
        if not isinstance(name, str):
            raise ValueError(f'Backend name must be a string, but got: {name}')
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)
        if value == Backend.UNDEFINED:
            value = name.lower()
        return value

    @classmethod
    def register_backend(cls, name, func, extended_api=False, devices: Optional[Union[str, List[str]]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a new backend with the given name and instantiating function.\n\n        This class method is used by 3rd party ``ProcessGroup`` extension to\n        register new backends.\n\n        Args:\n            name (str): Backend name of the ``ProcessGroup`` extension. It\n                        should match the one in ``init_process_group()``.\n            func (function): Function handler that instantiates the backend.\n                             The function should be implemented in the backend\n                             extension and takes four arguments, including\n                             ``store``, ``rank``, ``world_size``, and ``timeout``.\n            extended_api (bool, optional): Whether the backend supports extended argument structure.\n                                           Default: ``False``. If set to ``True``, the backend\n                                           will get an instance of ``c10d::DistributedBackendOptions``, and\n                                           a process group options object as defined by the backend implementation.\n            device (str or list of str, optional): device type this backend\n                            supports, e.g. "cpu", "cuda", etc. If `None`,\n                            assuming both "cpu" and "cuda"\n\n        .. note:: This support of 3rd party backend is experimental and subject to change.\n\n        '
        if name != Backend.UCC or (name == Backend.UCC and is_ucc_available()):
            assert not hasattr(Backend, name.upper()), f'{name.upper()} c10d backend already exist'
        assert name.upper() not in Backend._plugins, f'{name.upper()} c10d backend creator function already exist'
        setattr(Backend, name.upper(), name.lower())
        Backend.backend_list.append(name.lower())
        if devices is not None:
            for device in devices:
                if device != 'cpu' and device != 'cuda':
                    Backend.default_device_backend_map[device] = name.lower()
        Backend.backend_type_map[name.lower()] = ProcessGroup.BackendType.CUSTOM
        if devices is None:
            warnings.warn(f'Device capability of {name} unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.')
            Backend.backend_capability[name.lower()] = ['cpu', 'cuda']
        elif isinstance(devices, str):
            Backend.backend_capability[name.lower()] = [devices]
        else:
            Backend.backend_capability[name.lower()] = devices
        Backend._plugins[name.upper()] = Backend._BackendPlugin(func, extended_api)

class BackendConfig:
    """Backend configuration class."""

    def __init__(self, backend: Union[str, Backend]):
        if False:
            return 10
        'Init.'
        self.device_backend_map: Dict[torch.device, Backend] = {}
        if backend == Backend.UNDEFINED:
            for device in Backend.default_device_backend_map:
                if is_backend_available(Backend.default_device_backend_map[device]):
                    self.device_backend_map[device] = Backend.default_device_backend_map[device]
        elif backend.lower() in Backend.backend_list:
            supported_devices = Backend.backend_capability[backend.lower()]
            backend_val = Backend(backend)
            self.device_backend_map = {device: backend_val for device in supported_devices}
        elif ':' in backend.lower():
            backend_str_error_message = f"""The custom backend string argument is invalid: {backend}.\n                Custom backend string is an experimental feature where the backend string must be in the format:\n                "<device_type1>:<backend1>,<device_type2>:<backend2>...". e.g. 'cpu:gloo,cuda:nccl'"""
            for device_backend_pair_str in backend.lower().split(','):
                device_backend_pair = device_backend_pair_str.split(':')
                if len(device_backend_pair) != 2:
                    raise ValueError(f'Invalid device:backend pairing:                                      {device_backend_pair_str}. {backend_str_error_message}')
                (device, backend) = device_backend_pair
                if device in self.device_backend_map:
                    raise ValueError(f'Duplicate device type {device}                                      in backend string: {backend}. {backend_str_error_message}')
                self.device_backend_map[device] = Backend(backend)
        else:
            warnings.warn(f'Device capability of {backend} unknown, assuming `cpu` and `cuda`. You can specify it in `device:backend` format in `init_process_group` call.')
            backend_val = Backend(backend)
            self.device_backend_map = {'cpu': backend_val, 'cuda': backend_val, 'xpu': backend_val}
        logger.info(f'Using backend config: {self.device_backend_map}')

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return all the device:backend pairs separated by commas.'
        return ','.join((f'{device}:{backend}' for (device, backend) in self.device_backend_map.items()))

    def get_device_backend_map(self):
        if False:
            i = 10
            return i + 15
        'Return backend map of the device.'
        return self.device_backend_map

class _reduce_op:
    """
    Deprecated enum-like class.

    For reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        for (k, v) in ReduceOp.RedOpType.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.RedOpType.__members__

    def __getattribute__(self, key):
        if False:
            return 10
        warnings.warn('torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead')
        return object.__getattribute__(self, key)
reduce_op = _reduce_op()

class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
    """

    def __init__(self, op: Callable, tensor: torch.Tensor, peer: int, group: Optional[ProcessGroup]=None, tag: int=0):
        if False:
            i = 10
            return i + 15
        'Init.'
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag

    def __new__(cls, op: Callable, tensor: torch.Tensor, peer: int, group: Optional[ProcessGroup]=None, tag: int=0):
        if False:
            i = 10
            return i + 15
        'Create and return a new instance of the class.'
        _check_op(op)
        _check_single_tensor(tensor, 'tensor')
        return object.__new__(cls)

class _CollOp:
    """
    A class to capture collective operations.

    Args:
        op (Callable): A collective function, e.g. ``torch.distributed.all_reduce``.
        tensor (Tensor): Tensor to operate on.
        dst_tensor (Tensor, optional): Provided when source and destinaton tensors are not the same.
        redop (ReduceOp, optional): reduce operation.
        root (int, optional): root of broadcast or reduce.
    """

    def __init__(self, op: Callable, tensor: torch.Tensor, dst_tensor: Optional[torch.Tensor]=None, redop: Optional[ReduceOp]=None, root: Optional[int]=None):
        if False:
            print('Hello World!')
        self.op = op
        self.tensor = tensor
        self.dst_tensor = dst_tensor
        self.redop = redop
        self.root = root
_pg_map: Dict[ProcessGroup, Tuple[str, Optional[Store]]] = {}
_pg_names: Dict[ProcessGroup, str] = {}
_pg_group_ranks: Dict[ProcessGroup, Dict[int, int]] = {}
_pg_backend_config: Dict[ProcessGroup, str] = {}
_group_count = 0
_tags_to_pg: Dict[str, List[ProcessGroup]] = {}
_pg_to_tag: Dict[ProcessGroup, str] = {}

class _World:
    """
    Container class for c10d process group state.

    This is used during registration and lookup of PG state.

    .. warning:: This is an experimental API intended to expose the inner workings
       of c10d and is subject to change..
    """

    def __init__(self):
        if False:
            return 10
        self._default_pg = None
        self._pg_coalesce_state: Dict[ProcessGroup, List[Union[_CollOp, P2POp]]] = {}
        self._pg_default_device: Dict[ProcessGroup, torch.device] = {}

    @property
    def default_pg(self):
        if False:
            print('Hello World!')
        '\n        Process group that includes all ranks of the cluster.\n\n        This default ProcessGroup is used by c10d APIs when a ProcessGroup is needed\n        but None is provided.\n        '
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value):
        if False:
            return 10
        self._default_pg = value

    @property
    def pg_map(self) -> Dict[ProcessGroup, Tuple[str, Optional[Store]]]:
        if False:
            i = 10
            return i + 15
        "\n        Provide Mapping from ProcessGroup to backend name and store.\n\n        For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)\n        For MPI pg, it is a map from ProcessGroup to (Backend, None)\n\n        TODO don't expose the map, expose fine grained ops\n        "
        global _pg_map
        return _pg_map

    @property
    def pg_names(self) -> Dict[ProcessGroup, str]:
        if False:
            print('Hello World!')
        "\n        Process group's names, map from ProcessGroup to str.\n\n        TODO don't expose the map, expose fine grained ops\n        "
        global _pg_names
        return _pg_names

    @property
    def pg_group_ranks(self) -> Dict[ProcessGroup, Dict[int, int]]:
        if False:
            return 10
        "\n        Process group's global rank to local rank mapping.\n\n        TODO don't expose the map, expose fine grained ops\n        "
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_backend_config(self) -> Dict[ProcessGroup, str]:
        if False:
            while True:
                i = 10
        "\n        Process group's backend config.\n\n        TODO don't expose the map, expose fine grained ops\n        "
        global _pg_backend_config
        return _pg_backend_config

    @property
    def group_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "\n        Process group count for default naming.\n\n        TODO don't expose group_count, use something else instead\n        "
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value):
        if False:
            return 10
        'Use to compute the name of ProcessGroups when using global synchronization.'
        global _group_count
        _group_count = value

    @property
    def tags_to_pg(self) -> Dict[str, List[ProcessGroup]]:
        if False:
            print('Hello World!')
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> Dict[ProcessGroup, str]:
        if False:
            i = 10
            return i + 15
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_coalesce_state(self) -> Dict[ProcessGroup, List[Union[_CollOp, P2POp]]]:
        if False:
            i = 10
            return i + 15
        return self._pg_coalesce_state

    @property
    def pg_default_device(self) -> Dict[ProcessGroup, torch.device]:
        if False:
            print('Hello World!')
        return self._pg_default_device

    @property
    def pg_config_info(self) -> List[Dict[str, Union[int, str]]]:
        if False:
            print('Hello World!')
        '\n        Return a list of dict with process groups and backends.\n\n        Along with their unique IDs and configurations (types and ranks).\n        '
        config_info = []
        default_pg_size = _get_group_size(None)
        for (pg, backend) in self.pg_map.items():
            backend_type = Backend.backend_type_map[backend[0]]
            ranks = self.pg_group_ranks[pg]
            config_info.append({'pg_name': self.pg_names[pg], 'backend_id': pg._backend_id(backend_type), 'backend_config': self.pg_backend_config[pg], 'ranks': list(ranks.keys()) if len(ranks) != default_pg_size else [], 'group_size': len(ranks), 'group_count': self.group_count})
        return config_info
_world = _World()
'Holds the singleton instance of ``_World`` used by c10. Experimental extension point to override it'

class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """

    @property
    def WORLD(cls) -> Optional[ProcessGroup]:
        if False:
            for i in range(10):
                print('nop')
        return _world.default_pg

    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):
        if False:
            i = 10
            return i + 15
        _world.default_pg = pg

class group(metaclass=_WorldMeta):
    """Group class. Placeholder."""
    pass

class GroupMember(metaclass=_WorldMeta):
    """Group member class."""
    NON_GROUP_MEMBER = -100

def _get_default_timeout(backend: Backend) -> timedelta:
    if False:
        for i in range(10):
            print('nop')
    if backend == Backend.NCCL:
        if not isinstance(default_pg_nccl_timeout, timedelta):
            warnings.warn('Attempted to get default timeout for nccl backend, but NCCL support is not compiled')
            return default_pg_timeout
        return default_pg_nccl_timeout
    else:
        return default_pg_timeout

def _check_valid_timeout(timeout: Any) -> None:
    if False:
        return 10
    if not isinstance(timeout, timedelta):
        raise TypeError(f'Expected timeout argument to be of type datetime.timedelta, got {timeout}')
_default_pg_init_method = None
STORE_BASED_BARRIER_PREFIX = 'store_based_barrier_key'

def _get_pg_default_device(group: Optional[ProcessGroup]=None) -> torch.device:
    if False:
        i = 10
        return i + 15
    '\n    Return the device to use with ``group`` for control flow usage (object collectives, barrier).\n\n    There are selection rules:\n        1. If user specifies exactly one backend in ``init_process_group`` call:\n            use that backend\n        2. Else if user specifies multiple "device:backend" pairs in init_process_group:\n            If "cpu" is among those pairs, use "cpu" (because the object is in cpu memory);\n            Otherwise, use the first backend (sort of a random pick).\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n\n    Returns:\n        torch.device: The device to use with ``group``.\n\n    '
    group = group or _get_default_group()
    if group in _world.pg_default_device:
        return _world.pg_default_device[group]
    if not isinstance(group, ProcessGroup):
        warnings.warn(f'You are using a Backend {type(group)} as a ProcessGroup. This usage is deprecated since PyTorch 2.0. Please use a public API of PyTorch Distributed instead.')
        _world.pg_default_device[group] = torch.device('cpu')
        return _world.pg_default_device[group]
    '\n    ``group._device_types`` is a property pybind that returns the devices\n    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the\n    ``group`` supports multiple devices.\n    '
    devices = group._device_types
    if len(devices) == 1:
        _world.pg_default_device[group] = devices[0]
    elif len(devices) == 0:
        _world.pg_default_device[group] = torch.device('cpu')
    elif torch.device('cpu') in devices:
        _world.pg_default_device[group] = torch.device('cpu')
    else:
        _world.pg_default_device[group] = devices[0]
    logger.info(f'Using device {_world.pg_default_device[group]} for object collectives.')
    return _world.pg_default_device[group]

@_time_logger
def _store_based_barrier(rank, store, group_name, rendezvous_count, timeout, logging_interval=timedelta(seconds=10)):
    if False:
        while True:
            i = 10
    '\n    Store based barrier for synchronizing processes.\n\n    Barrier based on store which is used for synchronizing processes after\n    ``init_process_group`` or ``new_group``. Intended to be used only with\n    those two methods and is not a generic alternative to ``barrier()``.\n    '
    store_key = f'{STORE_BASED_BARRIER_PREFIX}:{group_name}'
    store.add(store_key, 1)
    logger.info('Added key: %s to store for rank: %s', store_key, rank)
    world_size = rendezvous_count
    worker_count = store.add(store_key, 0)
    last_worker_key = f'{store_key}:last_worker'
    if worker_count == world_size:
        store.set(last_worker_key, '1')
    logging_interval = max(logging_interval, timedelta(seconds=10 + world_size / 1000))
    start = time.time()
    while True:
        try:
            store.wait([last_worker_key], logging_interval)
            break
        except RuntimeError as e:
            worker_count = store.add(store_key, 0)
            logger.info('Waiting in store based barrier to initialize process group for rank: %s, key: %s (world_size=%s, num_workers_joined=%s, timeout=%s)', rank, store_key, world_size, worker_count, timeout)
            if timedelta(seconds=time.time() - start) > timeout:
                raise DistStoreError('Timed out initializing process group in store based barrier on rank {}, for key: {} (world_size={}, num_workers_joined={}, timeout={})'.format(rank, store_key, world_size, worker_count, timeout))
    logger.info('Rank %s: Completed store-based barrier for key:%s with %s nodes.', rank, store_key, world_size)

def _rank_not_in_group(group: ProcessGroup):
    if False:
        print('Hello World!')
    "Check if the current process's rank is not in a given group."
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER

def _warn_not_in_group(op_name):
    if False:
        print('Hello World!')
    global_rank = -1 if GroupMember.WORLD is None else GroupMember.WORLD.rank()
    warnings.warn(f'Running {op_name} on global rank {global_rank} which does not belong to the given group.')

def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    if False:
        while True:
            i = 10
    '\n    Translate a global rank into a group rank.\n\n    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.\n\n    Args:\n        group (ProcessGroup): ProcessGroup to find the relative rank.\n        global_rank (int): Global rank to query.\n\n    Returns:\n        Group rank of ``global_rank`` relative to ``group``\n\n    N.B. calling this function on the default process group returns identity\n    '
    if group is GroupMember.WORLD:
        return global_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(f'Group {group} is not registered, please create group with torch.distributed.new_group API')
    group_ranks = _world.pg_group_ranks[group]
    if global_rank not in group_ranks:
        raise ValueError(f'Global rank {global_rank} is not part of group {group}')
    return group_ranks[global_rank]

def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    if False:
        print('Hello World!')
    '\n    Translate a group rank into a global rank.\n\n    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.\n\n    Args:\n        group (ProcessGroup): ProcessGroup to find the global rank from.\n        group_rank (int): Group rank to query.\n\n    Returns:\n        Global rank of ``group_rank`` relative to ``group``\n\n    N.B. calling this function on the default process group returns identity\n    '
    if group is GroupMember.WORLD:
        return group_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(f'Group {group} is not registered, please create group with torch.distributed.new_group API')
    for (rank, grp_rank) in _world.pg_group_ranks[group].items():
        if grp_rank == group_rank:
            return rank
    raise ValueError(f'Group rank {group_rank} is not part of group {group}')

def _get_global_rank(group, rank):
    if False:
        for i in range(10):
            print('nop')
    'Use get_global_rank as this method is deprecated.'
    warnings.warn('torch.distributed.distributed_c10d._get_global_rank is deprecated please use torch.distributed.distributed_c10d.get_global_rank instead')
    return get_global_rank(group, rank)

def get_process_group_ranks(group: ProcessGroup):
    if False:
        i = 10
        return i + 15
    '\n    Get all ranks associated with ``group``.\n\n    Args:\n        group (ProcessGroup): ProcessGroup to get all ranks from.\n\n    Returns:\n        List of global ranks ordered by group rank.\n    '
    return list(_world.pg_group_ranks[group].keys())

def _get_group_size(group):
    if False:
        i = 10
        return i + 15
    "Get a given group's world size."
    if group is GroupMember.WORLD or group is None:
        default_pg = _get_default_group()
        return default_pg.size()
    return group.size()

def _check_single_tensor(param, param_name):
    if False:
        while True:
            i = 10
    'Check that the parameter ``param_name`` is a single tensor.'
    if not isinstance(param, torch.Tensor):
        raise TypeError(f'Invalid function argument. Expected parameter `{param_name}` to be of type torch.Tensor.')

def _check_tensor_list(param, param_name):
    if False:
        print('Hello World!')
    'Check that the parameter ``param_name`` is a list of tensors.'
    if not isinstance(param, list) or not all((isinstance(p, torch.Tensor) for p in param)):
        raise TypeError(f'Invalid function argument. Expected parameter `{param_name}` to be of type List[torch.Tensor].')

def _as_iterable(obj) -> collections.abc.Iterable:
    if False:
        for i in range(10):
            print('nop')
    return obj if isinstance(obj, list) else (obj,)

def _ensure_all_tensors_same_dtype(*tensors) -> None:
    if False:
        while True:
            i = 10
    last_dtype = None
    for tensor in itertools.chain(*map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        if tensor_dtype.is_complex:
            tensor_dtype = torch.float32 if tensor_dtype == torch.complex64 else torch.complex128
        if last_dtype is None:
            last_dtype = tensor_dtype
        elif last_dtype != tensor_dtype:
            raise ValueError(f'Invalid usage of tensors with different dtypesFound {last_dtype} and  {tensor.dtype}')

def _check_op(op):
    if False:
        i = 10
        return i + 15
    'Check that the ``op`` is either isend or irecv.'
    if op not in [isend, irecv]:
        raise ValueError('Invalid ``op``. Expected ``op`` to be of type ``torch.distributed.isend`` or ``torch.distributed.irecv``.')

def _check_p2p_op_list(p2p_op_list):
    if False:
        print('Hello World!')
    '\n    Check that the ``p2p_op_list`` is a list of P2POp instances.\n\n    Also, check that all ops use the same group.\n    '
    if not isinstance(p2p_op_list, list) or not all((isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list)):
        raise ValueError('Invalid ``p2p_op_list``. Each op is expected to to be of type ``torch.distributed.P2POp``.')
    group = p2p_op_list[0].group
    if not all((group == p2p_op.group for p2p_op in p2p_op_list)):
        raise ValueError('All ops need to use the same group.')

def is_mpi_available() -> bool:
    if False:
        print('Hello World!')
    'Check if the MPI backend is available.'
    return _MPI_AVAILABLE

def is_nccl_available() -> bool:
    if False:
        return 10
    'Check if the NCCL backend is available.'
    return _NCCL_AVAILABLE

def is_gloo_available() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if the Gloo backend is available.'
    return _GLOO_AVAILABLE

def is_ucc_available() -> bool:
    if False:
        print('Hello World!')
    'Check if the UCC backend is available.'
    return _UCC_AVAILABLE

def is_backend_available(backend: str) -> bool:
    if False:
        print('Hello World!')
    '\n    Check backend availability.\n\n    Checks if the given backend is available and supports the built-in backends or\n    third-party backends through function ``Backend.register_backend``.\n\n    Args:\n        backend (str): Backend name.\n    Returns:\n        bool: Returns true if the backend is available otherwise false.\n    '
    available_func = getattr(torch.distributed, f'is_{backend.lower()}_available', None)
    if available_func:
        return available_func()
    return backend.lower() in Backend.backend_list

def is_initialized() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if the default process group has been initialized.'
    return GroupMember.WORLD is not None

def is_torchelastic_launched() -> bool:
    if False:
        while True:
            i = 10
    '\n    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).\n\n    The existence of ``TORCHELASTIC_RUN_ID`` environment\n    variable is used as a proxy to determine whether the current process\n    was launched with torchelastic. This is a reasonable proxy since\n    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a\n    non-null value indicating the job id for peer discovery purposes..\n    '
    return os.getenv('TORCHELASTIC_RUN_ID') is not None

def _is_barrier_after_init() -> int:
    if False:
        i = 10
        return i + 15
    return int(os.getenv('TORCH_DIST_INIT_BARRIER', '0'))

def _get_default_group():
    if False:
        return 10
    'Get the default process group created by init_process_group.'
    if not is_initialized():
        raise ValueError('Default process group has not been initialized, please make sure to call init_process_group.')
    return GroupMember.WORLD

def _get_default_store():
    if False:
        print('Hello World!')
    'Get the default store created by init_process_group.'
    if not is_initialized():
        raise ValueError('Default process group has not been initialized, please make sure to call init_process_group.')
    default_pg = _get_default_group()
    (_, default_store) = _world.pg_map[default_pg]
    return default_store

def _update_default_pg(pg):
    if False:
        print('Hello World!')
    _world.default_pg = pg
    rank = pg.rank() if pg is not None and pg != GroupMember.NON_GROUP_MEMBER else -1
    torch._C._distributed_c10d._set_global_rank(rank)

def get_backend_config(group: Optional[ProcessGroup]=None) -> str:
    if False:
        return 10
    '\n    Return the backend configuration of the given process group.\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. The\n            default is the general main process group. If another specific group\n            is specified, the calling process must be part of :attr:`group`.\n\n    Returns:\n        The backend configuration of the given process group as a lower case string.\n\n    '
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    if _rank_not_in_group(pg):
        raise ValueError('Invalid process group specified')
    backend_config = _world.pg_backend_config.get(pg)
    assert backend_config is not None
    return str(backend_config)

def get_backend(group: Optional[ProcessGroup]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Return the backend of the given process group.\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. The\n            default is the general main process group. If another specific group\n            is specified, the calling process must be part of :attr:`group`.\n\n    Returns:\n        The backend of the given process group as a lower case string.\n\n    '
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    if _rank_not_in_group(pg):
        raise ValueError('Invalid process group specified')
    pg_store = _world.pg_map[pg] if pg in _world.pg_map else None
    assert pg_store is not None
    return pg_store[0]
_exception_logger

@_time_logger
def init_process_group(backend: Union[str, Backend]=None, init_method: Optional[str]=None, timeout: Optional[timedelta]=None, world_size: int=-1, rank: int=-1, store: Optional[Store]=None, group_name: str='', pg_options: Optional[Any]=None):
    if False:
        i = 10
        return i + 15
    '\n    Initialize the default distributed process group.\n\n    This will also initialize the distributed package.\n\n    There are 2 main ways to initialize a process group:\n        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.\n        2. Specify ``init_method`` (a URL string) which indicates where/how\n           to discover peers. Optionally specify ``rank`` and ``world_size``,\n           or encode all required parameters in the URL and omit them.\n\n    If neither is specified, ``init_method`` is assumed to be "env://".\n\n\n    Args:\n        backend (str or Backend, optional): The backend to use. Depending on\n            build-time configurations, valid values include ``mpi``, ``gloo``,\n            ``nccl``, and ``ucc``. If the backend is not provided, then both a ``gloo``\n            and ``nccl`` backend will be created, see notes below for how multiple\n            backends are managed. This field can be given as a lowercase string\n            (e.g., ``"gloo"``), which can also be accessed via\n            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using\n            multiple processes per machine with ``nccl`` backend, each process\n            must have exclusive access to every GPU it uses, as sharing GPUs\n            between processes can result in deadlocks. ``ucc`` backend is\n            experimental.\n        init_method (str, optional): URL specifying how to initialize the\n                                     process group. Default is "env://" if no\n                                     ``init_method`` or ``store`` is specified.\n                                     Mutually exclusive with ``store``.\n        world_size (int, optional): Number of processes participating in\n                                    the job. Required if ``store`` is specified.\n        rank (int, optional): Rank of the current process (it should be a\n                              number between 0 and ``world_size``-1).\n                              Required if ``store`` is specified.\n        store(Store, optional): Key/value store accessible to all workers, used\n                                to exchange connection/address information.\n                                Mutually exclusive with ``init_method``.\n        timeout (timedelta, optional): Timeout for operations executed against\n            the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends.\n            This is the duration after which collectives will be aborted asynchronously and the process will crash.\n            This is done since CUDA execution is async and it is no longer safe to continue executing user code since\n            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.\n            When NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.\n\n        group_name (str, optional, deprecated): Group name. This argument is ignored\n        pg_options (ProcessGroupOptions, optional): process group options\n            specifying what additional options need to be passed in during\n            the construction of specific process groups. As of now, the only\n            options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``\n            backend, ``is_high_priority_stream`` can be specified so that\n            the nccl backend can pick up high priority cuda streams when\n            there\'re compute kernels waiting.\n\n    .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source\n        on a system that supports MPI.\n\n    .. note:: Support for multiple backends is experimental. Currently when no backend is\n        specified, both ``gloo`` and ``nccl`` backends will be created. The ``gloo`` backend\n        will be used for collectives with CPU tensors and the ``nccl`` backend will be used\n        for collectives with CUDA tensors. A custom backend can be specified by passing in\n        a string with format "<device_type>:<backend_name>,<device_type>:<backend_name>", e.g.\n        "cpu:gloo,cuda:custom_backend".\n\n    '
    global _world
    global _backend
    global _default_pg_init_method
    if GroupMember.WORLD is not None:
        raise ValueError('trying to initialize the default process group twice!')
    assert store is None or init_method is None, 'Cannot specify both init_method and store.'
    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = 'env://'
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend('undefined')
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)
    '\n    Group name is not visible to users unless they access\n    internals of c10d. This means we can ignore the value\n    they provide as it not exposed in a public way.\n    '
    group_name = _process_group_name([], use_hashed_name=False)
    if backend == Backend.MPI:
        if world_size != -1 or rank != -1:
            warnings.warn(f'For MPI backend, world_size ({world_size}) and rank ({rank}) are ignored since they are assigned by the MPI runtime.')
        (default_pg, _) = _new_process_group_helper(-1, -1, [], backend, None, group_name, timeout=timeout)
        _update_default_pg(default_pg)
    else:
        if store is None:
            rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
            (store, rank, world_size) = next(rendezvous_iterator)
            store.set_timeout(timeout)
            store = PrefixStore('default_pg', store)
        (default_pg, _) = _new_process_group_helper(world_size, rank, [], backend, store, group_name, pg_options=pg_options, timeout=timeout)
        _update_default_pg(default_pg)
    _world.pg_group_ranks[GroupMember.WORLD] = {i: i for i in range(GroupMember.WORLD.size())}
    _backend = _world.pg_map[GroupMember.WORLD][0]
    _default_pg_init_method = init_method
    if _is_barrier_after_init() == 1:
        logger.info('Performing barrier after ProcessGroup initialization since TORCH_DIST_INIT_BARRIER = 1')
        if backend == Backend.MPI:
            barrier()
        else:
            _store_based_barrier(rank, store, group_name, world_size, timeout)

def _new_process_group_helper(group_size, group_rank, global_ranks_in_group, backend, store, group_name, pg_options=None, timeout=None, pg_tag=None):
    if False:
        while True:
            i = 10
    '\n    Create a new distributed process group.\n\n    This function must be called by ALL processes in the global group, even if\n    the calling process is not part of the newly created group. In that case,\n    this function returns GroupMember.NON_GROUP_MEMBER.\n\n    This function is called with ``global_ranks_in_group == []`` for the default group.\n    '
    global _world
    if group_name in _world.pg_names.values():
        raise ValueError('The specified group name has already been created, please use a different group name')
    _check_valid_timeout(timeout)
    if pg_tag not in [None, '']:
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            (_, prefix_store) = _world.pg_map[existing_group]
            return (existing_group, prefix_store)
    is_default_group = len(global_ranks_in_group) == 0
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            return (GroupMember.NON_GROUP_MEMBER, None)
    prefix_store = PrefixStore(f'{group_name}/', store)
    base_pg_options = ProcessGroup.Options(backend=str(backend))
    base_pg_options._timeout = timeout
    pg: ProcessGroup = ProcessGroup(prefix_store, group_rank, group_size, base_pg_options)
    backend_config = BackendConfig(backend)
    for (device, backend_str) in backend_config.get_device_backend_map().items():
        backend_prefix_store = PrefixStore(f'{device}/', prefix_store)
        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError("Distributed package doesn't have MPI built in. MPI is only included if you build PyTorch from source on a host that has MPI installed.")
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(backend_prefix_store, backend_class.rank(), backend_class.size(), base_pg_options)
        elif backend_str == Backend.GLOO:
            backend_class = ProcessGroupGloo(backend_prefix_store, group_rank, group_size, timeout=timeout)
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if pg_options is not None:
                assert isinstance(pg_options, ProcessGroupNCCL.Options), 'Expected pg_options argument to be of type ProcessGroupNCCL.Options'
                if pg_options._timeout != timeout:
                    warnings.warn('pg_options._timeout was specified, but timeout kwarg has a default value that will always override it. ')
            else:
                pg_options = ProcessGroupNCCL.Options()
                pg_options.is_high_priority_stream = False
            pg_options._timeout = timeout
            backend_class = ProcessGroupNCCL(backend_prefix_store, group_rank, group_size, pg_options)
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            backend_class = ProcessGroupUCC(backend_prefix_store, group_rank, group_size, timeout=timeout)
            backend_type = ProcessGroup.BackendType.UCC
        else:
            assert backend_str.upper() in Backend._plugins, f'Unknown c10d backend type {backend_str.upper()}'
            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM
            if not extended_api:
                backend_class = creator_fn(backend_prefix_store, group_rank, group_size, timeout)
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group
                backend_class = creator_fn(dist_backend_opts, pg_options)
        if backend_str in [Backend.GLOO, Backend.NCCL]:
            backend_class._set_sequence_number_for_group()
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class
            break
        if backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]:
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info('TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but\n                                GLOO is not available. Build with Gloo to\n                                create a wrapper process group in debug mode\n                                to aid collective desynchronization debugging.')
                else:
                    backend_class = _create_process_group_wrapper(wrapped_pg=backend_class, store_prefix=group_name, store=backend_prefix_store, rank=group_rank, world_size=group_size, timeout=timeout)
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)
            break
        pg._register_backend(torch.device(device), backend_type, backend_class)
    assert group_name is not None
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    pg._set_group_name(group_name)
    _world.pg_backend_config[pg] = str(backend_config)
    if pg_tag in [None, '']:
        pg_tag = f'ptd:{group_name}'
        _world.tags_to_pg.setdefault('', []).append(pg)
    else:
        pg_tag = f'user:{pg_tag}'
    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return (pg, prefix_store)

def destroy_process_group(group: Optional[ProcessGroup]=None):
    if False:
        print('Hello World!')
    '\n    Destroy a given process group, and deinitialize the distributed package.\n\n    Args:\n        group (ProcessGroup, optional): The process group to be destroyed, if\n                                        group.WORLD is given, all process\n                                        groups including the default one will\n                                        be destroyed.\n    '
    global _world
    if group == GroupMember.NON_GROUP_MEMBER:
        return
    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group
    assert pg is not None
    if _world.pg_map.get(pg, None) is None:
        raise ValueError('Invalid process group specified')
    if pg.name().lower() == 'nccl' and pg._has_hooks():
        pg._wait_for_pending_works()
    if group is None or group == GroupMember.WORLD:
        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _world.pg_default_device.clear()
        _world.group_count = 0
    else:
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_default_device:
            del _world.pg_default_device[pg]
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn("Some coalesced collectives haven't been launched when ProcessGroup is destroyed. They will be cleaned.")
            del _world.pg_coalesce_state[pg]
        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith('ptd:'):
                    _world.tags_to_pg[''].remove(pg)
            except Exception:
                pass

def get_rank(group: Optional[ProcessGroup]=None) -> int:
    if False:
        while True:
            i = 10
    '\n    Return the rank of the current process in the provided ``group``, default otherwise.\n\n    Rank is a unique identifier assigned to each process within a distributed\n    process group. They are always consecutive integers ranging from 0 to\n    ``world_size``.\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n\n    Returns:\n        The rank of the process group\n        -1, if not part of the group\n\n    '
    if _rank_not_in_group(group):
        return -1
    default_pg = _get_default_group()
    if group is None or group is GroupMember.WORLD:
        return default_pg.rank()
    return get_group_rank(group, default_pg.rank())

def get_world_size(group: Optional[ProcessGroup]=None) -> int:
    if False:
        return 10
    '\n    Return the number of processes in the current process group.\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n\n    Returns:\n        The world size of the process group\n        -1, if not part of the group\n\n    '
    if _rank_not_in_group(group):
        return -1
    return _get_group_size(group)

def isend(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup]=None, tag: int=0) -> Work:
    if False:
        while True:
            i = 10
    '\n    Send a tensor asynchronously.\n\n    .. warning::\n        Modifying ``tensor`` before the request completes causes undefined\n        behavior.\n\n    .. warning::\n        ``tag`` is not supported with the NCCL backend.\n\n    Args:\n        tensor (Tensor): Tensor to send.\n        dst (int): Destination rank.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        tag (int, optional): Tag to match send with remote recv\n\n    Returns:\n        A distributed request object.\n        None, if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('isend')
        return
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        return default_pg.send([tensor], dst, tag)
    else:
        group_dst_rank = get_group_rank(group, dst)
        return group.send([tensor], group_dst_rank, tag)

def irecv(tensor: torch.Tensor, src: Optional[int]=None, group: Optional[ProcessGroup]=None, tag: int=0) -> Work:
    if False:
        i = 10
        return i + 15
    '\n    Receives a tensor asynchronously.\n\n    .. warning::\n        ``tag`` is not supported with the NCCL backend.\n\n    Args:\n        tensor (Tensor): Tensor to fill with received data.\n        src (int, optional): Source rank. Will receive from any\n            process if unspecified.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        tag (int, optional): Tag to match recv with remote send\n\n    Returns:\n        A distributed request object.\n        None, if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('irecv')
        return
    if group is None or group is GroupMember.WORLD:
        pg = _get_default_group()
    else:
        pg = group
    if src is None:
        return pg.recv_anysource([tensor], tag)
    elif pg is GroupMember.WORLD:
        return pg.recv([tensor], src, tag)
    else:
        group_src_rank = get_group_rank(pg, src)
        return pg.recv([tensor], group_src_rank, tag)

@_exception_logger
def send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup]=None, tag: int=0) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Send a tensor synchronously.\n\n    Args:\n        tensor (Tensor): Tensor to send.\n        dst (int): Destination rank. Destination rank should not be the same\n            as the rank of the current process.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        tag (int, optional): Tag to match send with remote recv\n\n    '
    if get_rank() == dst:
        raise ValueError('Invalid destination rank: destination rank should not be the same as the rank of the current process.')
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('send')
        return
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        default_pg.send([tensor], dst, tag).wait()
    else:
        group_dst_rank = get_group_rank(group, dst)
        group.send([tensor], group_dst_rank, tag).wait()

@_exception_logger
def recv(tensor: torch.Tensor, src: Optional[int]=None, group: Optional[ProcessGroup]=None, tag: int=0) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Receives a tensor synchronously.\n\n    Args:\n        tensor (Tensor): Tensor to fill with received data.\n        src (int, optional): Source rank. Will receive from any\n            process if unspecified.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        tag (int, optional): Tag to match recv with remote send\n\n    Returns:\n        Sender rank\n        -1, if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('recv')
        return -1
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    if src is None:
        work = pg.recv_anysource([tensor], tag)
        work.wait()
        src_rank = work._source_rank()
        if group is None or group is GroupMember.WORLD:
            return src_rank
        else:
            return get_global_rank(pg, src_rank)
    else:
        if group is None or group is GroupMember.WORLD:
            pg.recv([tensor], src, tag).wait()
        else:
            group_src_rank = get_group_rank(pg, src)
            pg.recv([tensor], group_src_rank, tag).wait()
        return src

class _IllegalWork(Work):

    def __getattribute__(self, name):
        if False:
            print('Hello World!')
        if name in ['is_success', 'exception', 'wait', 'source_rank', '_source_rank', 'result', 'synchronize']:
            raise ValueError(f'Illegal to call {name} on IllegalWork object')

class _CoalescingManager:

    def __init__(self):
        if False:
            return 10
        self.works: List[Work] = []

    def append(self, work: Work):
        if False:
            return 10
        if work:
            self.works.append(work)

    def wait(self):
        if False:
            i = 10
            return i + 15
        for work in self.works:
            work.wait()

@contextlib.contextmanager
def _coalescing_manager(group: Optional[ProcessGroup]=None, device: Optional[torch.device]=None, async_ops: Optional[bool]=False):
    if False:
        i = 10
        return i + 15
    '\n    Context manager used to coalesce collectives or P2P operations when possible.\n\n    Args:\n        group (`ProcessGroup`, optional): The process group to work on. If None,\n            the default process group will be used.\n        device (`torch.device`, optional): Default is None, set to a device if\n            there isn\'t a `**_coalesced` implementation by the backend.\n        async_ops (`bool`, optional): whether the coalesced ops are async ops.\n\n    Examples:\n        >>> # xdoctest: +SKIP("no rank")\n        >>> # Synchronous ops\n        >>> with _coalescing_manager():\n        >>>     for i in range(num_colls):\n        >>>         dist.all_reduce(tensors[i])\n        >>> # Asynchronous ops\n        >>> with _coalescing_manager(async_ops=True) as cm:\n        >>>     for i in range(num_colls):\n        >>>         dist.all_reduce(tensors[i])\n        >>> cm.wait()\n\n    .. warning::\n       :func:`_coalescing_manager` currently do not support coalescing\n       all-reduces with different reduce operators, e.g.  `ReduceOp.SUM` mixed\n       with `ReduceOp.PRODUCT`.\n    '
    group = group or _get_default_group()
    op_list = _world.pg_coalesce_state.setdefault(group, [])
    if op_list:
        raise ValueError('ProcessGroup has non-empty op list at the start of coalescing')
    if device:
        group._start_coalescing(device)
    cm = _CoalescingManager()
    yield cm
    op_list = _world.pg_coalesce_state.pop(group)
    if op_list:
        op0 = op_list[0].op
        if op0 == all_reduce:
            tensors = []
            for op in op_list:
                tensors.append(op.tensor)
            opts = AllreduceCoalescedOptions()
            opts.reduceOp = op_list[0].redop
            work = group.allreduce_coalesced(tensors, opts)
        elif op0 == all_gather_into_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(op.dst_tensor)
            work = group.allgather_into_tensor_coalesced(outputs, inputs)
        elif op0 == reduce_scatter_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(op.dst_tensor)
                opts = ReduceScatterOptions()
                opts.reduceOp = op_list[0].redop
            work = group.reduce_scatter_tensor_coalesced(outputs, inputs, opts)
        else:
            raise AssertionError(f'Coalescing manager does not support fast-path coalescing of {op0}, yet {op0} is still recorded in op list. This is an internal error of c10d.')
    if device:
        work = group._end_coalescing(device)
    if async_ops:
        cm.append(work)
    else:
        work.wait()

def batch_isend_irecv(p2p_op_list):
    if False:
        while True:
            i = 10
    '\n    Send or Receive a batch of tensors asynchronously and return a list of requests.\n\n    Process each of the operations in ``p2p_op_list`` and return the corresponding\n    requests. NCCL, Gloo, and UCC backend are currently supported.\n\n    Args:\n        p2p_op_list: A list of point-to-point operations(type of each operator is\n            ``torch.distributed.P2POp``). The order of the isend/irecv in the list\n            matters and it needs to match with corresponding isend/irecv on the\n            remote end.\n\n    Returns:\n        A list of distributed request objects returned by calling the corresponding\n        op in the op_list.\n\n    Examples:\n        >>> # xdoctest: +SKIP("no rank")\n        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank\n        >>> recv_tensor = torch.randn(2, dtype=torch.float32)\n        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)\n        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)\n        >>> reqs = batch_isend_irecv([send_op, recv_op])\n        >>> for req in reqs:\n        >>>     req.wait()\n        >>> recv_tensor\n        tensor([2, 3])     # Rank 0\n        tensor([0, 1])     # Rank 1\n\n    .. note:: Note that when this API is used with the NCCL PG backend, users must set\n        the current GPU device with `torch.cuda.set_device`, otherwise it will\n        lead to unexpected hang issues.\n\n        In addition, if this API is the first collective call in the ``group``\n        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in\n        this API call; otherwise, the behavior is undefined. If this API call is\n        not the first collective call in the ``group``, batched P2P operations\n        involving only a subset of ranks of the ``group`` are allowed.\n    '
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    if device.type == 'cuda':
        with _coalescing_manager(group, device, async_ops=True) as cm:
            for p2p_op in p2p_op_list:
                p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        return cm.works
    else:
        reqs = []
        for p2p_op in p2p_op_list:
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            if work:
                reqs.append(work)
        return reqs

@_exception_logger
def broadcast_multigpu(tensor_list, src, group=None, async_op=False, src_tensor=0):
    if False:
        i = 10
        return i + 15
    '\n    Broadcasts the tensor to the whole group with multiple GPU tensors per node.\n\n    ``tensor`` must have the same number of elements in all the GPUs from\n    all processes participating in the collective. each tensor in the list must\n    be on a different GPU\n\n    Only nccl and gloo backend are currently supported\n    tensors should only be GPU tensors\n\n    Args:\n        tensor_list (List[Tensor]): Tensors that participate in the collective\n            operation. If ``src`` is the rank, then the specified ``src_tensor``\n            element of ``tensor_list`` (``tensor_list[src_tensor]``) will be\n            broadcast to all other tensors (on different GPUs) in the src process\n            and all tensors in ``tensor_list`` of other non-src processes.\n            You also need to make sure that ``len(tensor_list)`` is the same\n            for all the distributed processes calling this function.\n\n        src (int): Source rank.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n        src_tensor (int, optional): Source tensor rank within ``tensor_list``\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    warnings.warn('torch.distributed.broadcast_multigpu will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions')
    if _rank_not_in_group(group):
        _warn_not_in_group('broadcast_multigpu')
        return
    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = src_tensor
    opts.asyncOp = async_op
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.broadcast(tensor_list, opts)
    else:
        group_src_rank = get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast(tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def broadcast(tensor, src, group=None, async_op=False):
    if False:
        i = 10
        return i + 15
    '\n    Broadcasts the tensor to the whole group.\n\n    ``tensor`` must have the same number of elements in all processes\n    participating in the collective.\n\n    Args:\n        tensor (Tensor): Data to be sent if ``src`` is the rank of current\n            process, and tensor to be used to save received data otherwise.\n        src (int): Source rank.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('broadcast')
        return
    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0
    opts.asyncOp = async_op
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.broadcast([tensor], opts)
    else:
        group_src_rank = get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        while True:
            i = 10
    '\n    Reduces the tensor data across all machines in a way that all get the final result.\n\n    This function reduces a number of tensors on every node,\n    while each tensor resides on different GPUs.\n    Therefore, the input tensor in the tensor list needs to be GPU tensors.\n    Also, each tensor in the tensor list needs to reside on a different GPU.\n\n    After the call, all ``tensor`` in ``tensor_list`` is going to be bitwise\n    identical in all processes.\n\n    Complex tensors are supported.\n\n    Only nccl and gloo backend is currently supported\n    tensors should only be GPU tensors\n\n    Args:\n        tensor_list (List[Tensor]): List of input and output tensors of\n            the collective. The function operates in-place and requires that\n            each tensor to be a GPU tensor on different GPUs.\n            You also need to make sure that ``len(tensor_list)`` is the same for\n            all the distributed processes calling this function.\n        op (optional): One of the values from\n            ``torch.distributed.ReduceOp``\n            enum.  Specifies an operation used for element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If\n            ``None``, the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    warnings.warn('torch.distributed.all_reduce_multigpu will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions')
    if _rank_not_in_group(group):
        return
    tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list]
    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allreduce(tensor_list, opts)
    else:
        work = group.allreduce(tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    Reduces the tensor data across all machines in a way that all get the final result.\n\n    After the call ``tensor`` is going to be bitwise identical in all processes.\n\n    Complex tensors are supported.\n\n    Args:\n        tensor (Tensor): Input and output of the collective. The function\n            operates in-place.\n        op (optional): One of the values from\n            ``torch.distributed.ReduceOp``\n            enum.  Specifies an operation used for element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    Examples:\n        >>> # xdoctest: +SKIP("no rank")\n        >>> # All tensors below are of torch.int64 type.\n        >>> # We have 2 process groups, 2 ranks.\n        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank\n        >>> tensor\n        tensor([1, 2]) # Rank 0\n        tensor([3, 4]) # Rank 1\n        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)\n        >>> tensor\n        tensor([4, 6]) # Rank 0\n        tensor([4, 6]) # Rank 1\n\n        >>> # All tensors below are of torch.cfloat type.\n        >>> # We have 2 process groups, 2 ranks.\n        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)\n        >>> tensor\n        tensor([1.+1.j, 2.+2.j]) # Rank 0\n        tensor([3.+3.j, 4.+4.j]) # Rank 1\n        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)\n        >>> tensor\n        tensor([4.+4.j, 6.+6.j]) # Rank 0\n        tensor([4.+4.j, 6.+6.j]) # Rank 1\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('all_reduce')
        return
    if tensor.is_complex():
        if not supports_complex(op):
            raise ValueError(f'all_reduce does not support {op} on complex tensors')
        tensor = torch.view_as_real(tensor)
    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        group = _get_default_group()
    if group in _world.pg_coalesce_state.keys():
        coll = _CollOp(all_reduce, tensor, None, op, None)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None
    work = group.allreduce([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    WARNING: at this time individual shape checking is not implemented across nodes.\n\n    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the\n    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce\n    operation will proceed without complaint and return erroneous outputs. This lack\n    of shape checking results in significant performance improvements but users of this\n    function should take extra care to ensure that each node passes in tensors whose\n    shapes match across nodes.\n\n    Reduces each tensor in tensors (residing on the same device) across all machines\n    in such a way that all get the final result.\n\n    After the call each tensor in tensors is going to bitwise identical\n    in all processes.\n\n    Complex tensors are supported.\n\n    Args:\n        tensors (List[Tensor]): Input and output of the collective. The function\n            operates in-place.\n        op (Optional[ReduceOp]): One of the values from\n            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for\n            element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (Optional[bool]): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    '
    warnings.warn('torch.distributed.all_reduce_coalesced will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#collective-functions')
    _check_tensor_list(tensors, 'tensor')
    _ensure_all_tensors_same_dtype(tensors)
    if _rank_not_in_group(group):
        _warn_not_in_group('all_reduce_coalesced')
        return
    if any((t.is_complex() for t in tensors)) and (not supports_complex(op)):
        raise ValueError(f'all_reduce does not support {op} on complex tensors')
    tensors = [t if not t.is_complex() else torch.view_as_real(t) for t in tensors]
    opts = AllreduceCoalescedOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allreduce_coalesced(tensors, opts)
    else:
        work = group.allreduce_coalesced(tensors, opts)
    if async_op:
        return work.get_future()
    else:
        work.wait()

@_exception_logger
def reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=None, async_op=False, dst_tensor=0):
    if False:
        i = 10
        return i + 15
    '\n    Reduces the tensor data on multiple GPUs across all machines.\n\n    Each tensor in ``tensor_list`` should reside on a separate GPU.\n\n    Only the GPU of ``tensor_list[dst_tensor]`` on the process with rank ``dst``\n    is going to receive the final result.\n\n    Only nccl backend is currently supported\n    tensors should only be GPU tensors\n\n    Args:\n        tensor_list (List[Tensor]): Input and output GPU tensors of the\n            collective. The function operates in-place.\n            You also need to make sure that ``len(tensor_list)`` is the same for\n            all the distributed processes calling this function.\n        dst (int): Destination rank\n        op (optional): One of the values from\n            ``torch.distributed.ReduceOp``\n            enum.  Specifies an operation used for element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n        dst_tensor (int, optional): Destination tensor rank within\n                                    ``tensor_list``\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, otherwise\n\n    '
    warnings.warn('torch.distributed.reduce_multigpu will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions')
    if _rank_not_in_group(group):
        _warn_not_in_group('reduce_multigpu')
        return
    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    opts.rootTensor = dst_tensor
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.reduce(tensor_list, opts)
    else:
        group_dst_rank = get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce(tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    Reduces the tensor data across all machines.\n\n    Only the process with rank ``dst`` is going to receive the final result.\n\n    Args:\n        tensor (Tensor): Input and output of the collective. The function\n            operates in-place.\n        dst (int): Destination rank\n        op (optional): One of the values from\n            ``torch.distributed.ReduceOp``\n            enum.  Specifies an operation used for element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('reduce')
        return
    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.reduce([tensor], opts)
    else:
        group_dst_rank = get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_gather_multigpu(output_tensor_lists, input_tensor_list, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    Gathers tensors from the whole group in a list.\n\n    Each tensor in ``tensor_list`` should reside on a separate GPU\n\n    Only nccl backend is currently supported\n    tensors should only be GPU tensors\n\n    Complex tensors are supported.\n\n    Args:\n        output_tensor_lists (List[List[Tensor]]): Output lists. It should\n            contain correctly-sized tensors on each GPU to be used for output\n            of the collective, e.g. ``output_tensor_lists[i]`` contains the\n            all_gather result that resides on the GPU of\n            ``input_tensor_list[i]``.\n\n            Note that each element of ``output_tensor_lists`` has the size of\n            ``world_size * len(input_tensor_list)``, since the function all\n            gathers the result from every single GPU in the group. To interpret\n            each element of ``output_tensor_lists[i]``, note that\n            ``input_tensor_list[j]`` of rank k will be appear in\n            ``output_tensor_lists[i][k * world_size + j]``\n\n            Also note that ``len(output_tensor_lists)``, and the size of each\n            element in ``output_tensor_lists`` (each element is a list,\n            therefore ``len(output_tensor_lists[i])``) need to be the same\n            for all the distributed processes calling this function.\n\n        input_tensor_list (List[Tensor]): List of tensors(on different GPUs) to\n            be broadcast from current process.\n            Note that ``len(input_tensor_list)`` needs to be the same for\n            all the distributed processes calling this function.\n\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    warnings.warn('torch.distributed.all_gather_multigpu will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions')
    if _rank_not_in_group(group):
        _warn_not_in_group('all_gather_multigpu')
        return
    output_tensor_lists = [[t if not t.is_complex() else torch.view_as_real(t) for t in l] for l in output_tensor_lists]
    input_tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list]
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allgather(output_tensor_lists, input_tensor_list)
    else:
        work = group.allgather(output_tensor_lists, input_tensor_list)
    if async_op:
        return work
    else:
        work.wait()

def _object_to_tensor(obj, device):
    if False:
        while True:
            i = 10
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return (byte_tensor, local_size)

def _tensor_to_object(tensor, tensor_size):
    if False:
        for i in range(10):
            print('nop')
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()

@_exception_logger
def all_gather_object(object_list, obj, group=None):
    if False:
        return 10
    '\n    Gathers picklable objects from the whole group into a list.\n\n    Similar to :func:`all_gather`, but Python objects can be passed in.\n    Note that the object must be picklable in order to be gathered.\n\n    Args:\n        object_list (list[Any]): Output list. It should be correctly sized as the\n            size of the group for this collective and will contain the output.\n        obj (Any): Pickable Python object to be broadcast from current process.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used. Default is ``None``.\n\n    Returns:\n        None. If the calling rank is part of this group, the output of the\n        collective will be populated into the input ``object_list``. If the\n        calling rank is not part of the group, the passed in ``object_list`` will\n        be unmodified.\n\n    .. note:: Note that this API differs slightly from the :func:`all_gather`\n        collective since it does not provide an ``async_op`` handle and thus\n        will be a blocking call.\n\n    .. note:: For NCCL-based processed groups, internal tensor representations\n        of objects must be moved to the GPU device before communication takes\n        place. In this case, the device used is given by\n        ``torch.cuda.current_device()`` and it is the user\'s responsiblity to\n        ensure that this is set so that each rank has an individual GPU, via\n        ``torch.cuda.set_device()``.\n\n    .. warning::\n        :func:`all_gather_object` uses ``pickle`` module implicitly, which is\n        known to be insecure. It is possible to construct malicious pickle data\n        which will execute arbitrary code during unpickling. Only call this\n        function with data you trust.\n\n    .. warning::\n        Calling :func:`all_gather_object` with GPU tensors is not well supported\n        and inefficient as it incurs GPU -> CPU transfer since tensors would be\n        pickled. Please consider using :func:`all_gather` instead.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> # Assumes world_size of 3.\n        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object\n        >>> output = [None for _ in gather_objects]\n        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])\n        >>> output\n        [\'foo\', 12, {1: 2}]\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('all_gather_object')
        return
    current_device = _get_pg_default_device(group)
    (input_tensor, local_size) = _object_to_tensor(obj, current_device)
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.uint8, device=current_device)
    output_tensors = [coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)] for i in range(group_size)]
    all_gather(output_tensors, input_tensor, group=group)
    for (i, tensor) in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)

@_exception_logger
def gather_object(obj, object_gather_list=None, dst=0, group=None):
    if False:
        i = 10
        return i + 15
    '\n    Gathers picklable objects from the whole group in a single process.\n\n    Similar to :func:`gather`, but Python objects can be passed in. Note that the\n    object must be picklable in order to be gathered.\n\n    Args:\n        obj (Any): Input object. Must be picklable.\n        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it\n            should be correctly sized as the size of the group for this\n            collective and will contain the output. Must be ``None`` on non-dst\n            ranks. (default is ``None``)\n        dst (int, optional): Destination rank. (default is 0)\n        group: (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used. Default is ``None``.\n\n    Returns:\n        None. On the ``dst`` rank, ``object_gather_list`` will contain the\n        output of the collective.\n\n    .. note:: Note that this API differs slightly from the gather collective\n        since it does not provide an async_op handle and thus will be a blocking\n        call.\n\n    .. note:: For NCCL-based processed groups, internal tensor representations\n        of objects must be moved to the GPU device before communication takes\n        place. In this case, the device used is given by\n        ``torch.cuda.current_device()`` and it is the user\'s responsiblity to\n        ensure that this is set so that each rank has an individual GPU, via\n        ``torch.cuda.set_device()``.\n\n    .. warning::\n        :func:`gather_object` uses ``pickle`` module implicitly, which is\n        known to be insecure. It is possible to construct malicious pickle data\n        which will execute arbitrary code during unpickling. Only call this\n        function with data you trust.\n\n    .. warning::\n        Calling :func:`gather_object` with GPU tensors is not well supported\n        and inefficient as it incurs GPU -> CPU transfer since tensors would be\n        pickled. Please consider using :func:`gather` instead.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> # Assumes world_size of 3.\n        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object\n        >>> output = [None for _ in gather_objects]\n        >>> dist.gather_object(\n        ...     gather_objects[dist.get_rank()],\n        ...     output if dist.get_rank() == 0 else None,\n        ...     dst=0\n        ... )\n        >>> # On rank 0\n        >>> output\n        [\'foo\', 12, {1: 2}]\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('gather_object')
        return
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    current_device = _get_pg_default_device(group)
    (input_tensor, local_size) = _object_to_tensor(obj, current_device)
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())
    input_tensor.resize_(max_object_size)
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.uint8, device=current_device)
        output_tensors = [coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)] for i in range(group_size)]
    gather(input_tensor, gather_list=output_tensors if my_rank == dst else None, dst=dst, group=group)
    if my_rank != dst:
        return
    for (i, tensor) in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)

@_exception_logger
def broadcast_object_list(object_list, src=0, group=None, device=None):
    if False:
        while True:
            i = 10
    '\n    Broadcasts picklable objects in ``object_list`` to the whole group.\n\n    Similar to :func:`broadcast`, but Python objects can be passed in.\n    Note that all objects in ``object_list`` must be picklable in order to be\n    broadcasted.\n\n    Args:\n        object_list (List[Any]): List of input objects to broadcast.\n            Each object must be picklable. Only objects on the ``src`` rank will\n            be broadcast, but each rank must provide lists of equal sizes.\n        src (int): Source rank from which to broadcast ``object_list``.\n        group: (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used. Default is ``None``.\n        device (``torch.device``, optional): If not None, the objects are\n            serialized and converted to tensors which are moved to the\n            ``device`` before broadcasting. Default is ``None``.\n\n    Returns:\n        ``None``. If rank is part of the group, ``object_list`` will contain the\n        broadcasted objects from ``src`` rank.\n\n    .. note:: For NCCL-based process groups, internal tensor representations\n        of objects must be moved to the GPU device before communication takes\n        place. In this case, the device used is given by\n        ``torch.cuda.current_device()`` and it is the user\'s responsibility to\n        ensure that this is set so that each rank has an individual GPU, via\n        ``torch.cuda.set_device()``.\n\n    .. note:: Note that this API differs slightly from the :func:`all_gather`\n        collective since it does not provide an ``async_op`` handle and thus\n        will be a blocking call.\n\n    .. warning::\n        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which\n        is known to be insecure. It is possible to construct malicious pickle\n        data which will execute arbitrary code during unpickling. Only call this\n        function with data you trust.\n\n    .. warning::\n        Calling :func:`broadcast_object_list` with GPU tensors is not well supported\n        and inefficient as it incurs GPU -> CPU transfer since tensors would be\n        pickled. Please consider using :func:`broadcast` instead.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> if dist.get_rank() == 0:\n        >>>     # Assumes world_size of 3.\n        >>>     objects = ["foo", 12, {1: 2}] # any picklable object\n        >>> else:\n        >>>     objects = [None, None, None]\n        >>> # Assumes backend is not NCCL\n        >>> device = torch.device("cpu")\n        >>> dist.broadcast_object_list(objects, src=0, device=device)\n        >>> objects\n        [\'foo\', 12, {1: 2}]\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('broadcast_object_list')
        return
    current_device = device or _get_pg_default_device(group)
    my_rank = get_rank()
    if my_rank == src:
        (tensor_list, size_list) = zip(*[_object_to_tensor(obj, current_device) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long, device=current_device)
    broadcast(object_sizes_tensor, src=src, group=group)
    if my_rank == src:
        if len(tensor_list) == 1:
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(torch.sum(object_sizes_tensor).item(), dtype=torch.uint8, device=current_device)
    broadcast(object_tensor, src=src, group=group)
    offset = 0
    if my_rank != src:
        for (i, obj_size) in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device('cpu'):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)

@_exception_logger
def scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Scatters picklable objects in ``scatter_object_input_list`` to the whole group.\n\n    Similar to :func:`scatter`, but Python objects can be passed in. On\n    each rank, the scattered object will be stored as the first element of\n    ``scatter_object_output_list``. Note that all objects in\n    ``scatter_object_input_list`` must be picklable in order to be scattered.\n\n    Args:\n        scatter_object_output_list (List[Any]): Non-empty list whose first\n            element will store the object scattered to this rank.\n        scatter_object_input_list (List[Any]): List of input objects to scatter.\n            Each object must be picklable. Only objects on the ``src`` rank will\n            be scattered, and the argument can be ``None`` for non-src ranks.\n        src (int): Source rank from which to scatter\n            ``scatter_object_input_list``.\n        group: (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used. Default is ``None``.\n\n    Returns:\n        ``None``. If rank is part of the group, ``scatter_object_output_list``\n        will have its first element set to the scattered object for this rank.\n\n    .. note:: Note that this API differs slightly from the scatter collective\n        since it does not provide an ``async_op`` handle and thus will be a\n        blocking call.\n\n    .. warning::\n        :func:`scatter_object_list` uses ``pickle`` module implicitly, which\n        is known to be insecure. It is possible to construct malicious pickle\n        data which will execute arbitrary code during unpickling. Only call this\n        function with data you trust.\n\n    .. warning::\n        Calling :func:`scatter_object_list` with GPU tensors is not well supported\n        and inefficient as it incurs GPU -> CPU transfer since tensors would be\n        pickled. Please consider using :func:`scatter` instead.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> if dist.get_rank() == 0:\n        >>>     # Assumes world_size of 3.\n        >>>     objects = ["foo", 12, {1: 2}] # any picklable object\n        >>> else:\n        >>>     # Can be any list on non-src ranks, elements are not used.\n        >>>     objects = [None, None, None]\n        >>> output_list = [None]\n        >>> dist.scatter_object_list(output_list, objects, src=0)\n        >>> # Rank i gets objects[i]. For example, on rank 2:\n        >>> output_list\n        [{1: 2}]\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('scatter_object_list')
        return
    if not isinstance(scatter_object_output_list, list) or len(scatter_object_output_list) < 1:
        raise ValueError('Expected argument scatter_object_output_list to be a list of size at least 1.')
    my_rank = get_rank()
    pg_device = _get_pg_default_device(group)
    if my_rank == src:
        (tensor_list, tensor_sizes) = zip(*[_object_to_tensor(obj, pg_device) for obj in scatter_object_input_list])
        (tensor_list, tensor_sizes) = (list(tensor_list), list(tensor_sizes))
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    broadcast(max_tensor_size, src=src, group=group)
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8, device=pg_device)
    scatter(output_tensor, scatter_list=None if my_rank != src else tensor_list, src=src, group=group)
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    scatter(obj_tensor_size, scatter_list=None if my_rank != src else tensor_sizes, src=src, group=group)
    scatter_object_output_list[0] = _tensor_to_object(output_tensor, obj_tensor_size)

@_exception_logger
def all_gather(tensor_list, tensor, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    Gathers tensors from the whole group in a list.\n\n    Complex tensors are supported.\n\n    Args:\n        tensor_list (list[Tensor]): Output list. It should contain\n            correctly-sized tensors to be used for output of the collective.\n        tensor (Tensor): Tensor to be broadcast from current process.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    Examples:\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # All tensors below are of torch.int64 dtype.\n        >>> # We have 2 process groups, 2 ranks.\n        >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]\n        >>> tensor_list\n        [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1\n        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank\n        >>> tensor\n        tensor([1, 2]) # Rank 0\n        tensor([3, 4]) # Rank 1\n        >>> dist.all_gather(tensor_list, tensor)\n        >>> tensor_list\n        [tensor([1, 2]), tensor([3, 4])] # Rank 0\n        [tensor([1, 2]), tensor([3, 4])] # Rank 1\n\n        >>> # All tensors below are of torch.cfloat dtype.\n        >>> # We have 2 process groups, 2 ranks.\n        >>> tensor_list = [torch.zeros(2, dtype=torch.cfloat) for _ in range(2)]\n        >>> tensor_list\n        [tensor([0.+0.j, 0.+0.j]), tensor([0.+0.j, 0.+0.j])] # Rank 0 and 1\n        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)\n        >>> tensor\n        tensor([1.+1.j, 2.+2.j]) # Rank 0\n        tensor([3.+3.j, 4.+4.j]) # Rank 1\n        >>> dist.all_gather(tensor_list, tensor)\n        >>> tensor_list\n        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 0\n        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 1\n\n    '
    _check_tensor_list(tensor_list, 'tensor_list')
    _check_single_tensor(tensor, 'tensor')
    _ensure_all_tensors_same_dtype(tensor_list, tensor)
    if _rank_not_in_group(group):
        _warn_not_in_group('all_gather')
        return
    tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allgather([tensor_list], [tensor])
    else:
        work = group.allgather([tensor_list], [tensor])
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if False:
        i = 10
        return i + 15
    '\n    Gather tensors from all ranks and put them in a single output tensor.\n\n    Args:\n        output_tensor (Tensor): Output tensor to accommodate tensor elements\n            from all ranks. It must be correctly sized to have one of the\n            following forms:\n            (i) a concatenation of all the input tensors along the primary\n            dimension; for definition of "concatenation", see ``torch.cat()``;\n            (ii) a stack of all the input tensors along the primary dimension;\n            for definition of "stack", see ``torch.stack()``.\n            Examples below may better explain the supported output forms.\n        input_tensor (Tensor): Tensor to be gathered from current rank.\n            Different from the ``all_gather`` API, the input tensors in this\n            API must have the same size across all ranks.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    Examples:\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.\n        >>> # We have two ranks.\n        >>> device = torch.device(f\'cuda:{rank}\')\n        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank\n        >>> tensor_in\n        tensor([1, 2], device=\'cuda:0\') # Rank 0\n        tensor([3, 4], device=\'cuda:1\') # Rank 1\n        >>> # Output in concatenation form\n        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)\n        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)\n        >>> tensor_out\n        tensor([1, 2, 3, 4], device=\'cuda:0\') # Rank 0\n        tensor([1, 2, 3, 4], device=\'cuda:1\') # Rank 1\n        >>> # Output in stack form\n        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)\n        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)\n        >>> tensor_out2\n        tensor([[1, 2],\n                [3, 4]], device=\'cuda:0\') # Rank 0\n        tensor([[1, 2],\n                [3, 4]], device=\'cuda:1\') # Rank 1\n\n    .. warning::\n        The Gloo backend does not support this API.\n\n    '
    _check_single_tensor(input_tensor, 'input_tensor')
    _check_single_tensor(output_tensor, 'output_tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('all_gather_into_tensor')
        return
    output_tensor = output_tensor if not output_tensor.is_complex() else torch.view_as_real(output_tensor)
    input_tensor = input_tensor if not input_tensor.is_complex() else torch.view_as_real(input_tensor)
    opts = AllgatherOptions()
    opts.asyncOp = async_op
    group = group or _get_default_group()
    if group in _world.pg_coalesce_state.keys():
        coll = _CollOp(all_gather_into_tensor, input_tensor, output_tensor)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None
    work = group._allgather_base(output_tensor, input_tensor, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def _all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.\n\n    Args:\n        output_tensor (Tensor): Output tensor. It should contain\n            correctly-sized tensors to be used for output of the collective.\n        input_tensor (Tensor): Tensor to be broadcast from current process.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    .. warning::\n        `_all_gather_base` is a private function. Users should use\n        `all_gather_into_tensor` instead.\n\n    '
    warnings.warn('torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.')
    return all_gather_into_tensor(output_tensor, input_tensor, group, async_op)

@_exception_logger
def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None, async_op=False):
    if False:
        while True:
            i = 10
    '\n    Gathers input tensors from the whole group in a list in a coalesced manner.\n\n    Complex tensors are supported.\n\n    Args:\n        output_tensor_lists (list[list[Tensor]]): Output list. It should contain\n            correctly-sized tensors to be used for output of the collective.\n        input_tensor_list (list[Tensor]): Tensors to be broadcast from\n            current process. At least one tensor has to be non empty.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    Example:\n        we have 2 process groups, 2 ranks.\n        rank 0 passes:\n            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]\n            output_tensor_lists =\n               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],\n                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]\n        rank 1 passes:\n            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]\n            output_tensor_lists =\n               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],\n                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]\n        both rank 0 and 1 get:\n            output_tensor_lists =\n               [[[1, 1], [1, 1]], [2], [3, 3]],\n                [[3, 3], [3, 3]], [5], [1, 1]]].\n\n    WARNING: at this time individual shape checking is not implemented across nodes.\n    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the\n    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the\n    all_gather_coalesced operation will proceed without complaint and return\n    erroneous outputs. This lack of shape checking results in significant\n    performance improvements but users of this function should take extra care\n    to ensure that each node passes in tensors whose shapes match across nodes.\n    '
    warnings.warn('torch.distributed.all_gather_coalesced will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#collective-functions')
    if _rank_not_in_group(group):
        _warn_not_in_group('all_gather_coalesced')
        return
    _check_tensor_list(input_tensor_list, 'input_tensor_list')
    _ensure_all_tensors_same_dtype(input_tensor_list)
    if not isinstance(output_tensor_lists, list):
        raise TypeError('Invalid function argument: output_tensor_lists should be a list')
    for output_tensor_list in output_tensor_lists:
        _check_tensor_list(output_tensor_list, 'output_tensor_lists')
        _ensure_all_tensors_same_dtype(output_tensor_list)
    output_tensor_lists = [[t if not t.is_complex() else torch.view_as_real(t) for t in l] for l in output_tensor_lists]
    input_tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list]
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allgather_coalesced(output_tensor_lists, input_tensor_list)
    else:
        work = group.allgather_coalesced(output_tensor_lists, input_tensor_list)
    if async_op:
        return work.get_future()
    else:
        work.wait()

def _validate_output_list_for_rank(my_rank, dst, gather_list):
    if False:
        while True:
            i = 10
    if dst == my_rank:
        if not gather_list:
            raise ValueError('Argument ``gather_list`` must be specified on destination rank.')
    elif gather_list:
        raise ValueError('Argument ``gather_list`` must NOT be specified on non-destination ranks.')

@_exception_logger
def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    if False:
        i = 10
        return i + 15
    '\n    Gathers a list of tensors in a single process.\n\n    Args:\n        tensor (Tensor): Input tensor.\n        gather_list (list[Tensor], optional): List of appropriately-sized\n            tensors to use for gathered data (default is None, must be specified\n            on the destination rank)\n        dst (int, optional): Destination rank (default is 0)\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if gather_list:
        _check_tensor_list(gather_list, 'gather_list')
    else:
        gather_list = []
    _ensure_all_tensors_same_dtype(tensor, gather_list)
    if _rank_not_in_group(group):
        _warn_not_in_group('gather')
        return
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, gather_list)
    output_tensors = [gather_list] if dst == my_rank else []
    input_tensors = [tensor]
    opts = GatherOptions()
    opts.rootRank = dst
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.gather(output_tensors, input_tensors, opts)
    else:
        group_dst_rank = get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.gather(output_tensors, input_tensors, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Scatters a list of tensors to all processes in a group.\n\n    Each process will receive exactly one tensor and store its data in the\n    ``tensor`` argument.\n\n    Complex tensors are supported.\n\n    Args:\n        tensor (Tensor): Output tensor.\n        scatter_list (list[Tensor]): List of tensors to scatter (default is\n            None, must be specified on the source rank)\n        src (int): Source rank (default is 0)\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n\n    .. note:: Note that all Tensors in scatter_list must have the same size.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> tensor_size = 2\n        >>> t_ones = torch.ones(tensor_size)\n        >>> t_fives = torch.ones(tensor_size) * 5\n        >>> output_tensor = torch.zeros(tensor_size)\n        >>> if dist.get_rank() == 0:\n        >>>     # Assumes world_size of 2.\n        >>>     # Only tensors, all of which must be the same size.\n        >>>     scatter_list = [t_ones, t_fives]\n        >>> else:\n        >>>     scatter_list = None\n        >>> dist.scatter(output_tensor, scatter_list, src=0)\n        >>> # Rank i gets scatter_list[i]. For example, on rank 1:\n        >>> output_tensor\n        tensor([5., 5.])\n\n    '
    _check_single_tensor(tensor, 'tensor')
    if scatter_list:
        _check_tensor_list(scatter_list, 'scatter_list')
    else:
        scatter_list = []
    _ensure_all_tensors_same_dtype(tensor, scatter_list)
    if _rank_not_in_group(group):
        _warn_not_in_group('scatter')
        return
    scatter_list = [t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)
    my_rank = get_rank()
    if src == my_rank:
        if not scatter_list:
            raise ValueError('Argument ``scatter_list`` must be specified on source rank.')
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError('Argument ``scatter_list`` must NOT be specified on non-source ranks.')
        input_tensors = []
        output_tensors = [tensor]
    opts = ScatterOptions()
    opts.rootRank = src
    opts.asyncOp = async_op
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        work = default_pg.scatter(output_tensors, input_tensors, opts)
    else:
        group_src_rank = get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.scatter(output_tensors, input_tensors, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def reduce_scatter_multigpu(output_tensor_list, input_tensor_lists, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        print('Hello World!')
    '\n    Reduce and scatter a list of tensors to the whole group.\n\n    Only nccl backend is currently supported.\n\n    Each tensor in ``output_tensor_list`` should reside on a separate GPU, as\n    should each list of tensors in ``input_tensor_lists``.\n\n    Args:\n        output_tensor_list (List[Tensor]): Output tensors (on different GPUs)\n            to receive the result of the operation.\n\n            Note that ``len(output_tensor_list)`` needs to be the same for all\n            the distributed processes calling this function.\n\n        input_tensor_lists (List[List[Tensor]]): Input lists.  It should\n            contain correctly-sized tensors on each GPU to be used for input of\n            the collective, e.g. ``input_tensor_lists[i]`` contains the\n            reduce_scatter input that resides on the GPU of\n            ``output_tensor_list[i]``.\n\n            Note that each element of ``input_tensor_lists`` has the size of\n            ``world_size * len(output_tensor_list)``, since the function\n            scatters the result from every single GPU in the group.  To\n            interpret each element of ``input_tensor_lists[i]``, note that\n            ``output_tensor_list[j]`` of rank k receives the reduce-scattered\n            result from ``input_tensor_lists[i][k * world_size + j]``\n\n            Also note that ``len(input_tensor_lists)``, and the size of each\n            element in ``input_tensor_lists`` (each element is a list,\n            therefore ``len(input_tensor_lists[i])``) need to be the same for\n            all the distributed processes calling this function.\n\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    '
    warnings.warn('torch.distributed.reduce_scatter_multigpu will be deprecated. If you must use it, please revisit our documentation later at https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions')
    if _rank_not_in_group(group):
        _warn_not_in_group('reduce_scatter_multigpu')
        return
    opts = ReduceScatterOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.reduce_scatter(output_tensor_list, input_tensor_lists, opts)
    else:
        work = group.reduce_scatter(output_tensor_list, input_tensor_lists, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        while True:
            i = 10
    '\n    Reduces, then scatters a list of tensors to all processes in a group.\n\n    Args:\n        output (Tensor): Output tensor.\n        input_list (list[Tensor]): List of tensors to reduce and scatter.\n        op (optional): One of the values from\n            ``torch.distributed.ReduceOp``\n            enum.  Specifies an operation used for element-wise reductions.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    '
    _check_single_tensor(output, 'output')
    _check_tensor_list(input_list, 'input_list')
    _ensure_all_tensors_same_dtype(output, input_list)
    if _rank_not_in_group(group):
        _warn_not_in_group('reduce_scatter')
        return
    opts = ReduceScatterOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.reduce_scatter([output], [input_list], opts)
    else:
        work = group.reduce_scatter([output], [input_list], opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        while True:
            i = 10
    '\n    Reduces, then scatters a tensor to all ranks in a group.\n\n    Args:\n        output (Tensor): Output tensor. It should have the same size across all\n            ranks.\n        input (Tensor): Input tensor to be reduced and scattered. Its size\n            should be output tensor size times the world size. The input tensor\n            can have one of the following shapes:\n            (i) a concatenation of the output tensors along the primary\n            dimension, or\n            (ii) a stack of the output tensors along the primary dimension.\n            For definition of "concatenation", see ``torch.cat()``.\n            For definition of "stack", see ``torch.stack()``.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    Examples:\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.\n        >>> # We have two ranks.\n        >>> device = torch.device(f\'cuda:{rank}\')\n        >>> tensor_out = torch.zeros(2, dtype=torch.int64, device=device)\n        >>> # Input in concatenation form\n        >>> tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device=device)\n        >>> tensor_in\n        tensor([0, 1, 2, 3], device=\'cuda:0\') # Rank 0\n        tensor([0, 1, 2, 3], device=\'cuda:1\') # Rank 1\n        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)\n        >>> tensor_out\n        tensor([0, 2], device=\'cuda:0\') # Rank 0\n        tensor([4, 6], device=\'cuda:1\') # Rank 1\n        >>> # Input in stack form\n        >>> tensor_in = torch.reshape(tensor_in, (world_size, 2))\n        >>> tensor_in\n        tensor([[0, 1],\n                [2, 3]], device=\'cuda:0\') # Rank 0\n        tensor([[0, 1],\n                [2, 3]], device=\'cuda:1\') # Rank 1\n        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)\n        >>> tensor_out\n        tensor([0, 2], device=\'cuda:0\') # Rank 0\n        tensor([4, 6], device=\'cuda:1\') # Rank 1\n\n    .. warning::\n        The Gloo backend does not support this API.\n\n    '
    _check_single_tensor(output, 'output')
    _check_single_tensor(input, 'input')
    if _rank_not_in_group(group):
        _warn_not_in_group('reduce_scatter_tensor')
        return
    opts = ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    group = group or _get_default_group()
    if group in _world.pg_coalesce_state.keys():
        coll = _CollOp(reduce_scatter_tensor, input, output, op, None)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None
    work = group._reduce_scatter_base(output, input, opts)
    if async_op:
        return work
    else:
        work.wait()

def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    if False:
        return 10
    '\n    Reduces, then scatters a flattened tensor to all processes in a group.\n\n    Args:\n        output (Tensor): Output tensor.\n        input (Tensor): Input tensor that is of size output tensor size times world size\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    .. warning::\n        `_reduce_scatter_base` is a private function. Users should use\n        `reduce_scatter_tensor` instead.\n\n    '
    warnings.warn('torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.')
    return reduce_scatter_tensor(output, input, op, group, async_op)

@_exception_logger
def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    if False:
        return 10
    '\n    Split input tensor and then scatter the split list to all processes in a group.\n\n    Later the received tensors are concatenated from all the processes in the group\n    and returned as a single output tensor.\n\n    Complex tensors are supported.\n\n    Args:\n        output (Tensor): Gathered concatenated output tensor.\n        input (Tensor): Input tensor to scatter.\n        output_split_sizes: (list[Int], optional): Output split sizes for dim 0\n            if specified None or empty, dim 0 of ``output`` tensor must divide\n            equally by ``world_size``.\n        input_split_sizes: (list[Int], optional): Input split sizes for dim 0\n            if specified None or empty, dim 0 of ``input`` tensor must divide\n            equally by ``world_size``.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    .. warning::\n        `all_to_all_single` is experimental and subject to change.\n\n    Examples:\n        >>> # xdoctest: +SKIP("Undefined rank")\n        >>> input = torch.arange(4) + rank * 4\n        >>> input\n        tensor([0, 1, 2, 3])     # Rank 0\n        tensor([4, 5, 6, 7])     # Rank 1\n        tensor([8, 9, 10, 11])   # Rank 2\n        tensor([12, 13, 14, 15]) # Rank 3\n        >>> output = torch.empty([4], dtype=torch.int64)\n        >>> dist.all_to_all_single(output, input)\n        >>> output\n        tensor([0, 4, 8, 12])    # Rank 0\n        tensor([1, 5, 9, 13])    # Rank 1\n        tensor([2, 6, 10, 14])   # Rank 2\n        tensor([3, 7, 11, 15])   # Rank 3\n\n        >>> # Essentially, it is similar to following operation:\n        >>> scatter_list = list(input.chunk(world_size))\n        >>> gather_list  = list(output.chunk(world_size))\n        >>> for i in range(world_size):\n        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)\n\n        >>> # Another example with uneven split\n        >>> input\n        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0\n        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1\n        tensor([20, 21, 22, 23, 24])                                     # Rank 2\n        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3\n        >>> input_splits\n        [2, 2, 1, 1]                                                     # Rank 0\n        [3, 2, 2, 2]                                                     # Rank 1\n        [2, 1, 1, 1]                                                     # Rank 2\n        [2, 2, 2, 1]                                                     # Rank 3\n        >>> output_splits\n        [2, 3, 2, 2]                                                     # Rank 0\n        [2, 2, 1, 2]                                                     # Rank 1\n        [1, 2, 1, 2]                                                     # Rank 2\n        [1, 2, 1, 1]                                                     # Rank 3\n        >>> output = ...\n        >>> dist.all_to_all_single(output, input, output_splits, input_splits)\n        >>> output\n        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0\n        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1\n        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2\n        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3\n\n\n        >>> # Another example with tensors of torch.cfloat type.\n        >>> input = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j], dtype=torch.cfloat) + 4 * rank * (1+1j)\n        >>> input\n        tensor([1+1j, 2+2j, 3+3j, 4+4j])                                # Rank 0\n        tensor([5+5j, 6+6j, 7+7j, 8+8j])                                # Rank 1\n        tensor([9+9j, 10+10j, 11+11j, 12+12j])                          # Rank 2\n        tensor([13+13j, 14+14j, 15+15j, 16+16j])                        # Rank 3\n        >>> output = torch.empty([4], dtype=torch.int64)\n        >>> dist.all_to_all_single(output, input)\n        >>> output\n        tensor([1+1j, 5+5j, 9+9j, 13+13j])                              # Rank 0\n        tensor([2+2j, 6+6j, 10+10j, 14+14j])                            # Rank 1\n        tensor([3+3j, 7+7j, 11+11j, 15+15j])                            # Rank 2\n        tensor([4+4j, 8+8j, 12+12j, 16+16j])                            # Rank 3\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('all_to_all_single')
        return
    opts = AllToAllOptions()
    _check_single_tensor(output, 'output')
    _check_single_tensor(input, 'input')
    _ensure_all_tensors_same_dtype(output, input)
    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)
    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.alltoall_base(output, input, output_split_sizes, input_split_sizes, opts)
    else:
        work = group.alltoall_base(output, input, output_split_sizes, input_split_sizes, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if False:
        return 10
    '\n    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.\n\n    Complex tensors are supported.\n\n    Args:\n        output_tensor_list (list[Tensor]): List of tensors to be gathered one\n            per rank.\n        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group.\n\n    .. warning::\n        `all_to_all` is experimental and subject to change.\n\n    Examples:\n        >>> # xdoctest: +SKIP("Undefined rank")\n        >>> input = torch.arange(4) + rank * 4\n        >>> input = list(input.chunk(4))\n        >>> input\n        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0\n        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1\n        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2\n        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3\n        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))\n        >>> dist.all_to_all(output, input)\n        >>> output\n        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0\n        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1\n        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2\n        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3\n\n        >>> # Essentially, it is similar to following operation:\n        >>> scatter_list = input\n        >>> gather_list  = output\n        >>> for i in range(world_size):\n        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)\n\n        >>> input\n        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0\n        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1\n        tensor([20, 21, 22, 23, 24])                                     # Rank 2\n        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3\n        >>> input_splits\n        [2, 2, 1, 1]                                                     # Rank 0\n        [3, 2, 2, 2]                                                     # Rank 1\n        [2, 1, 1, 1]                                                     # Rank 2\n        [2, 2, 2, 1]                                                     # Rank 3\n        >>> output_splits\n        [2, 3, 2, 2]                                                     # Rank 0\n        [2, 2, 1, 2]                                                     # Rank 1\n        [1, 2, 1, 2]                                                     # Rank 2\n        [1, 2, 1, 1]                                                     # Rank 3\n        >>> input = list(input.split(input_splits))\n        >>> input\n        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0\n        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1\n        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2\n        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3\n        >>> output = ...\n        >>> dist.all_to_all(output, input)\n        >>> output\n        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0\n        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1\n        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2\n        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3\n\n        >>> # Another example with tensors of torch.cfloat type.\n        >>> input = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j], dtype=torch.cfloat) + 4 * rank * (1+1j)\n        >>> input = list(input.chunk(4))\n        >>> input\n        [tensor([1+1j]), tensor([2+2j]), tensor([3+3j]), tensor([4+4j])]            # Rank 0\n        [tensor([5+5j]), tensor([6+6j]), tensor([7+7j]), tensor([8+8j])]            # Rank 1\n        [tensor([9+9j]), tensor([10+10j]), tensor([11+11j]), tensor([12+12j])]      # Rank 2\n        [tensor([13+13j]), tensor([14+14j]), tensor([15+15j]), tensor([16+16j])]    # Rank 3\n        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))\n        >>> dist.all_to_all(output, input)\n        >>> output\n        [tensor([1+1j]), tensor([5+5j]), tensor([9+9j]), tensor([13+13j])]          # Rank 0\n        [tensor([2+2j]), tensor([6+6j]), tensor([10+10j]), tensor([14+14j])]        # Rank 1\n        [tensor([3+3j]), tensor([7+7j]), tensor([11+11j]), tensor([15+15j])]        # Rank 2\n        [tensor([4+4j]), tensor([8+8j]), tensor([12+12j]), tensor([16+16j])]        # Rank 3\n\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('all_to_all')
        return
    opts = AllToAllOptions()
    _check_tensor_list(output_tensor_list, 'output_tensor_list')
    _check_tensor_list(input_tensor_list, 'input_tensor_list')
    _ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)
    input_tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list]
    output_tensor_list = [t if not t.is_complex() else torch.view_as_real(t) for t in output_tensor_list]
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.alltoall(output_tensor_list, input_tensor_list, opts)
    else:
        work = group.alltoall(output_tensor_list, input_tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
    if False:
        print('Hello World!')
    '\n    Synchronize all processes.\n\n    This collective blocks processes until the whole group enters this function,\n    if async_op is False, or if async work handle is called on wait().\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. If None,\n            the default process group will be used.\n        async_op (bool, optional): Whether this op should be an async op\n        device_ids ([int], optional): List of device/GPU ids.\n\n    Returns:\n        Async work handle, if async_op is set to True.\n        None, if not async_op or if not part of the group\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('barrier')
        return
    opts = BarrierOptions()
    opts.device = _get_pg_default_device(group)
    if device_ids is not None:
        if isinstance(device_ids, list):
            opts.device_ids = device_ids
        else:
            raise TypeError('Invalid function argument: device_ids type should be List[int]')
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.barrier(opts=opts)
    else:
        work = group.barrier(opts=opts)
    if async_op:
        return work
    else:
        work.wait()

def monitored_barrier(group=GroupMember.WORLD, timeout=None, wait_all_ranks=False):
    if False:
        while True:
            i = 10
    '\n    Synchronize processes similar to ``torch.distributed.barrier``, but consider a configurable timeout.\n\n    It is able to report ranks that did not pass this barrier within the provided timeout.\n    Specifically, for non-zero ranks, will block until a send/recv is processed from rank 0.\n    Rank 0 will block until all send /recv from other ranks are processed, and will report\n    failures for ranks that failed to respond in time. Note that if one rank does not reach the\n    monitored_barrier (for example due to a hang), all other ranks would fail in monitored_barrier.\n\n    This collective will block all processes/ranks in the group, until the\n    whole group exits the function successfully, making it useful for debugging\n    and synchronizing. However, it can have a performance impact and should only\n    be used for debugging or scenarios that require full synchronization points\n    on the host-side. For debugging purposes, this barrier can be inserted\n    before the application\'s collective calls to check if any ranks are\n    desynchronized.\n\n    .. note:: Note that this collective is only supported with the GLOO backend.\n\n    Args:\n        group (ProcessGroup, optional): The process group to work on. If\n            ``None``, the default process group will be used.\n        timeout (datetime.timedelta, optional): Timeout for monitored_barrier.\n            If ``None``, the default process group timeout will be used.\n        wait_all_ranks (bool, optional): Whether to collect all failed ranks or\n            not. By default, this is ``False`` and ``monitored_barrier`` on rank 0\n            will throw on the first failed rank it encounters in order to fail\n            fast. By setting ``wait_all_ranks=True`` ``monitored_barrier`` will\n            collect all failed ranks and throw an error containing information\n            about all failed ranks.\n\n    Returns:\n        ``None``.\n\n    Example::\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> # Note: Process group initialization omitted on each rank.\n        >>> import torch.distributed as dist\n        >>> if dist.get_rank() != 1:\n        >>>     dist.monitored_barrier() # Raises exception indicating that\n        >>> # rank 1 did not call into monitored_barrier.\n        >>> # Example with wait_all_ranks=True\n        >>> if dist.get_rank() == 0:\n        >>>     dist.monitored_barrier(wait_all_ranks=True) # Raises exception\n        >>> # indicating that ranks 1, 2, ... world_size - 1 did not call into\n        >>> # monitored_barrier.\n    '
    if _rank_not_in_group(group):
        _warn_not_in_group('monitored_barrier')
        return
    if get_backend(group) != Backend.GLOO:
        raise ValueError('monitored_barrier is only implemented for GLOO backend.')
    if timeout is None:
        timeout = _get_default_timeout(get_backend(group))
    elif isinstance(timeout, float):
        warnings.warn(f'Please specify timeout arg as a timedelta. Converting current value of {timeout} assuming it represents seconds')
        timeout = timedelta(seconds=timeout)
    _check_valid_timeout(timeout)
    group_to_use = _get_default_group() if group is None else group
    return group_to_use.monitored_barrier(timeout, wait_all_ranks=wait_all_ranks)

def _create_process_group_wrapper(wrapped_pg: ProcessGroup, store_prefix: str, store: Store, rank: int, world_size: int, timeout: timedelta=default_pg_timeout):
    if False:
        while True:
            i = 10
    prefix = f'{PG_WRAPPER_STORE_PREFIX}:{store_prefix}'
    store = PrefixStore(prefix, store)
    helper_pg = ProcessGroupGloo(store, rank, world_size, timeout=timeout)
    wrapped_pg = _ProcessGroupWrapper(wrapped_pg, helper_pg)
    return wrapped_pg

def _process_group_name(ranks, use_hashed_name):
    if False:
        while True:
            i = 10
    global _world
    if use_hashed_name:
        pg_name = hashlib.sha1(bytes('_'.join(map(str, ranks)), 'utf-8')).hexdigest()
        while pg_name in _world.pg_names.values():
            pg_name = hashlib.sha1(bytes(pg_name + '_', 'utf-8')).hexdigest()
    else:
        pg_name = str(_world.group_count)
        _world.group_count += 1
    return pg_name

def _get_backend_from_str(backend: Optional[str]=None) -> Backend:
    if False:
        while True:
            i = 10
    if not backend:
        backend = get_backend(_get_default_group())
    return Backend(backend)

@_time_logger
def new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False):
    if False:
        while True:
            i = 10
    '\n    Create a new distributed group.\n\n    This function requires that all processes in the main group (i.e. all\n    processes that are part of the distributed job) enter this function, even\n    if they are not going to be members of the group. Additionally, groups\n    should be created in the same order in all processes.\n\n    .. warning::\n        Using multiple process groups with the ``NCCL`` backend concurrently\n        is not safe and the user should perform explicit synchronization in\n        their application to ensure only one process group is used at a time.\n        This means collectives from one process group should have completed\n        execution on the device (not just enqueued since CUDA execution is\n        async) before collectives from another process group are enqueued.\n        See `Using multiple NCCL communicators concurrently <https://docs.nvid\n        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using\n        -multiple-nccl-communicators-concurrently>`_ for more details.\n\n    Args:\n        ranks (list[int]): List of ranks of group members. If ``None``, will be\n            set to all ranks. Default is ``None``.\n        timeout (timedelta, optional): see `init_process_group` for details and default value.\n        backend (str or Backend, optional): The backend to use. Depending on\n            build-time configurations, valid values are ``gloo`` and ``nccl``.\n            By default uses the same backend as the global group. This field\n            should be given as a lowercase string (e.g., ``"gloo"``), which can\n            also be accessed via :class:`Backend` attributes (e.g.,\n            ``Backend.GLOO``). If ``None`` is passed in, the backend\n            corresponding to the default process group will be used. Default is\n            ``None``.\n        pg_options (ProcessGroupOptions, optional): process group options\n            specifying what additional options need to be passed in during\n            the construction of specific process groups. i.e. for the ``nccl``\n            backend, ``is_high_priority_stream`` can be specified so that\n            process group can pick up high priority cuda streams.\n        use_local_synchronization (bool, optional): perform a group-local\n            barrier at the end of the process group creation. This is different\n            in that non-member ranks don\'t need to call into API and don\'t\n            join the barrier.\n\n    Returns:\n        A handle of distributed group that can be given to collective calls or None if the rank is not part of ``ranks``.\n\n    N.B. use_local_synchronization doesn\'t work with MPI.\n\n    N.B. While use_local_synchronization=True can be significantly faster with larger\n    clusters and small process groups, care must be taken since it changes cluster behavior\n    as non-member ranks don\'t join the group barrier().\n\n    N.B. use_local_synchronization=True can lead to deadlocks when each rank creates\n    multiple overlaping process groups. To avoid that, make sure all ranks follow the\n    same global creation order.\n    '
    return _new_group_with_tag(ranks, timeout, backend, pg_options, None, use_local_synchronization=use_local_synchronization)

def _new_group_with_tag(ranks=None, timeout=None, backend=None, pg_options=None, pg_tag=None, use_local_synchronization=False):
    if False:
        i = 10
        return i + 15
    '\n    Variant of ``new_group`` that exposes tag creation.\n\n    :: N.B. The mechanism is experimental and tied to the functional collectives effort, see\n    ``torch.distributed._functional_collectives`` for reference on how to use it.\n    '
    global _world
    default_pg = _get_default_group()
    (default_backend, default_store) = _world.pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()
    if not backend:
        backend = default_backend
    backend = Backend(backend)
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)
    if use_local_synchronization:
        if backend == Backend.MPI:
            raise ValueError("MPI backend doesn't support use_local_synchronization=True")
        if ranks is not None and get_rank() not in ranks:
            return None
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise ValueError("the new group's world size should be less or equal to the world size set by init_process_group")
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise ValueError("The new group's rank should be within the world_size set by init_process_group")
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank
    group_name = _process_group_name(ranks, use_hashed_name=use_local_synchronization)
    (pg, pg_store) = _new_process_group_helper(group_world_size, group_rank, ranks, backend, default_store, group_name, pg_options=pg_options, timeout=timeout, pg_tag=pg_tag)
    _world.pg_group_ranks[pg] = {global_rank: group_rank for (group_rank, global_rank) in enumerate(ranks)}
    if _is_barrier_after_init() == 1:
        logger.info('Performing barrier after ProcessGroup initialization since TORCH_DIST_INIT_BARRIER = 1')
        if backend == Backend.MPI:
            barrier()
        else:
            barrier_store = pg_store if use_local_synchronization else default_store
            world_size = len(ranks) if use_local_synchronization else get_world_size()
            _store_based_barrier(global_rank, barrier_store, group_name, world_size, timeout)
    return pg

def new_subgroups(group_size=None, group=None, timeout=None, backend=None, pg_options=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create subgroups of equal size.\n\n    By default, it creates intra-machine subgroups,\n    where each of which contains all the ranks of a machine, based on the assumption\n    that each machine has the same number of devices.\n\n    This is a convenience API that calls ``new_group`` to generate multiple subgroups.\n    It requires that all processes in the main group (i.e. all\n    processes that are part of the distributed job) enter this function, even\n    if they are not going to be members of the group.\n\n    .. warning::\n        If ``group_size`` is passed in, the world size must be divisible by ``group_size``.\n        If no ``group_size`` is passed in, it believe that you are creating a group based\n        on CUDA and determining the group size by number of CUDA devices, and if not all\n        the machines have the same number of devices, the subgroup division will be\n        different across nodes and can cause unexpected behaviors. Therefore, if you are\n        creating a subgroup that does not depend on CUDA (such as Gloo on CPU), please\n        pass in ``group_size`` correctly.\n\n    .. warning::\n        Using multiple process groups with the ``NCCL`` backend concurrently\n        is not safe and the user should perform explicit synchronization in\n        their application to ensure only one process group is used at a time.\n        This means collectives from one process group should have completed\n        execution on the device (not just enqueued since CUDA execution is\n        async) before collectives from another process group are enqueued.\n        See `Using multiple NCCL communicators concurrently <https://docs.nvid\n        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using\n        -multiple-nccl-communicators-concurrently>`_ for more details.\n\n    Args:\n        group_size (int, optional): The size of each subgroup. If ``None``,\n            the default subgroup size is equal to the number of devices on each machine,\n            based on the assumption that each machine has exactly the same\n            number of devices. Default is ``None``.\n        timeout (timedelta, optional): see `init_process_group` for details and default value.\n        backend (str or Backend, optional): The backend to use. Depending on\n            build-time configurations, valid values are ``gloo`` and ``nccl``.\n            By default uses the same backend as the global group. This field\n            should be given as a lowercase string (e.g., ``"gloo"``), which can\n            also be accessed via :class:`Backend` attributes (e.g.,\n            ``Backend.GLOO``). If ``None`` is passed in, the backend\n            corresponding to the default process group will be used. Default is\n            ``None``.\n        pg_options (ProcessGroupOptions, optional): process group options\n            specifying what additional options need to be passed in during\n            the construction of specific process groups. i.e. for the ``nccl``\n            backend, ``is_high_priority_stream`` can be specified so that\n            process group can pick up high priority cuda streams.\n\n    Returns:\n        The subgroup containing the current rank, and all the subgroups used for cleanup.\n\n    Examples:\n        >>> # Create intra-machine subgroups.\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> cur_subgroup, subgroups = dist.new_subgroups()\n        >>> # Allreduce within the machine.\n        >>> rank = dist.get_rank()\n        >>> tensor = torch.ones(1, device=rank) * rank\n        >>> dist.all_reduce(tensor, group=cur_subgroup)\n        >>> tensor\n        tensor([8])     # Assume 8 is the number of CUDA devices per machine.\n        >>> # Cleanup.\n        >>> for subgroup in subgroups:\n        >>>     dist.destroy_process_group(subgroup)\n    '
    if group_size is None:
        if not torch.cuda.is_available():
            raise ValueError("Default group size only takes effect when CUDA is available.If your subgroup using a backend that does not depend on CUDA,please pass in 'group_size' correctly.")
        group_size = torch.cuda.device_count()
    if group_size <= 0:
        raise ValueError(f"The arg 'group_size' ({group_size}) must be positive")
    world_size = get_world_size()
    if world_size < group_size:
        raise ValueError(f"The arg 'group_size' ({group_size}) must not exceed the world size ({world_size})")
    if world_size % group_size != 0:
        raise ValueError("The world size must be divisible by 'group_size'")
    subgroups = []
    cur_subgroup = None
    for subgroup_id in range(world_size // group_size):
        start_rank = subgroup_id * group_size
        end_rank = start_rank + group_size
        ranks_in_subgroup = list(range(start_rank, end_rank))
        subgroup = new_group(ranks=ranks_in_subgroup, timeout=timeout, backend=backend, pg_options=pg_options)
        subgroups.append(subgroup)
        rank = get_rank()
        if rank in ranks_in_subgroup:
            cur_subgroup = subgroup
            logger.info('Rank %s is assigned to subgroup %s', rank, ranks_in_subgroup)
    return (cur_subgroup, subgroups)

def new_subgroups_by_enumeration(ranks_per_subgroup_list, timeout=None, backend=None, pg_options=None):
    if False:
        print('Hello World!')
    '\n    Create subgroups by dividing the global world.\n\n    The division is specified by a nested list of ranks. The subgroups cannot have\n    overlap, and some ranks may not have to be in any subgroup.\n\n    This is a convenience API that calls ``new_group`` to generate multiple subgroups.\n    It requires that all processes in the main group (i.e. all\n    processes that are part of the distributed job) enter this function, even\n    if they are not going to be members of the group.\n\n    .. warning::\n        Using multiple process groups with the ``NCCL`` backend concurrently\n        is not safe and the user should perform explicit synchronization in\n        their application to ensure only one process group is used at a time.\n        This means collectives from one process group should have completed\n        execution on the device (not just enqueued since CUDA execution is\n        async) before collectives from another process group are enqueued.\n        See `Using multiple NCCL communicators concurrently <https://docs.nvid\n        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using\n        -multiple-nccl-communicators-concurrently>`_ for more details.\n\n    Args:\n        ranks_per_subgroup_list (list[list[int]]): A nested list of ranks of\n            group members.\n        timeout (timedelta, optional): see `init_process_group` for details and default value.\n        backend (str or Backend, optional): The backend to use. Depending on\n             build-time configurations, valid values are ``gloo`` and ``nccl``.\n             By default uses the same backend as the global group. This field\n             should be given as a lowercase string (e.g., ``"gloo"``), which can\n             also be accessed via :class:`Backend` attributes (e.g.,\n             ``Backend.GLOO``). If ``None`` is passed in, the backend\n             corresponding to the default process group will be used. Default is\n             ``None``.\n        pg_options (ProcessGroupOptions, optional): process group options\n            specifying what additional options need to be passed in during\n            the construction of specific process groups. i.e. for the ``nccl``\n            backend, ``is_high_priority_stream`` can be specified so that\n            process group can pick up high priority cuda streams.\n\n    Returns:\n        The subgroup containing the current rank, and all the subgroups used for cleanup.\n\n    Examples:\n        >>> # Create two subgroups, where each has 2 processes.\n        >>> # xdoctest: +SKIP("need process group init")\n        >>> cur_subgroup, subgroups = dist.new_subgroups(ranks=[[0, 2], [1, 3]])\n        >>> rank = dist.get_rank()\n        >>> tensor = torch.ones(1, device=rank) * rank\n        >>> dist.all_reduce(tensor, group=cur_subgroup)\n        >>> tensor\n        tensor([2])     # Subgroup 0: ranks 0 and 2\n        tensor([4])     # Subgroup 1: ranks 1 and 3\n    '
    if ranks_per_subgroup_list is None or len(ranks_per_subgroup_list) == 0:
        raise ValueError("The arg 'ranks_per_subgroup_list' cannot be empty")
    subgroups = []
    cur_subgroup = None
    rank_to_ranks_dict = {}
    for ranks in ranks_per_subgroup_list:
        subgroup = new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)
        subgroups.append(subgroup)
        my_rank = get_rank()
        for rank in ranks:
            if rank in rank_to_ranks_dict:
                raise ValueError(f'Rank {rank} has appeared in both subgroup {rank_to_ranks_dict[rank]} and {ranks}')
            rank_to_ranks_dict[rank] = ranks
            if my_rank == rank:
                cur_subgroup = subgroup
                logger.info('Rank %s is assigned to subgroup %s', rank, ranks)
    return (cur_subgroup, subgroups)

def _find_pg_by_ranks_and_tag(tag: str, ranks: List[int]) -> ProcessGroup:
    if False:
        print('Hello World!')
    if len(tag) > 0 and (not tag.startswith('ptd:')) and (not tag.startswith('user:')):
        tag = f'user:{tag}'
    for group in _world.tags_to_pg.get(tag, []):
        if group.size() != len(ranks):
            continue
        group_ranks = get_process_group_ranks(group)
        good = all((r in group_ranks for r in ranks))
        if good:
            return group
    return None

def _find_or_create_pg_by_ranks_and_tag(tag: str, ranks: List[int], stride: int) -> ProcessGroup:
    if False:
        return 10
    assert len(ranks) % stride == 0, f'Ranks length ({len(ranks)}) must be divisible by stride ({stride})'
    my_rank = get_rank()
    my_ranks = None
    if stride == len(ranks):
        my_ranks = ranks.copy()
        assert my_rank in my_ranks, "rankset doesn't include the current node"
    else:
        for i in range(0, len(ranks), stride):
            rank_set = ranks[i:i + stride]
            if my_rank in rank_set:
                my_ranks = rank_set
        assert my_ranks is not None, "rankset doesn't include the current node"
    my_ranks.sort()
    pg = _find_pg_by_ranks_and_tag(tag, my_ranks)
    if pg is not None:
        return pg
    if tag == '':
        raise ValueError('Cannot automatically create PG with empty tag')
    return _new_group_with_tag(my_ranks, pg_tag=tag)

def _get_group_tag(pg: ProcessGroup) -> str:
    if False:
        i = 10
        return i + 15
    'Return the tag associated with ``pg``.'
    tag = _world.pg_to_tag[pg]
    if tag.startswith('user:'):
        tag = tag[5:]
    return tag

def _get_process_group_name(pg: ProcessGroup) -> str:
    if False:
        while True:
            i = 10
    return _world.pg_names.get(pg, 'None')

def _get_process_group_store(pg: ProcessGroup) -> Store:
    if False:
        for i in range(10):
            print('nop')
    return _world.pg_map[pg][1]
dynamo_unsupported_distributed_c10d_ops = [all_reduce_multigpu, recv, all_gather_object, all_gather_coalesced, all_to_all_single, all_reduce, gather_object, all_to_all, all_reduce_coalesced, gather, broadcast_object_list, barrier, reduce_multigpu, scatter, scatter_object_list, reduce, reduce_scatter_multigpu, all_gather, broadcast_multigpu, all_gather_multigpu, reduce_scatter, all_gather_into_tensor, broadcast, reduce_scatter_tensor, send]