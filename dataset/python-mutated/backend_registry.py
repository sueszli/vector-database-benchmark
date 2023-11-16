__all__ = ['init_backend', 'backend_registered', 'construct_rpc_backend_options', 'register_backend', 'BackendType', 'BackendValue']
import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
__all__ = ['backend_registered', 'register_backend', 'construct_rpc_backend_options', 'init_backend', 'BackendValue', 'BackendType']
BackendValue = collections.namedtuple('BackendValue', ['construct_rpc_backend_options_handler', 'init_backend_handler'])

def _backend_type_repr(self):
    if False:
        while True:
            i = 10
    return 'BackendType.' + self.name
_backend_type_doc = '\n    An enum class of available backends.\n\n    PyTorch ships with a builtin ``BackendType.TENSORPIPE`` backend.\n    Additional ones can be registered using the\n    :func:`~torch.distributed.rpc.backend_registry.register_backend` function.\n'
BackendType = enum.Enum(value='BackendType', names=dict())
BackendType.__repr__ = _backend_type_repr
if BackendType.__doc__:
    BackendType.__doc__ = _backend_type_doc

def backend_registered(backend_name):
    if False:
        print('Hello World!')
    '\n    Checks if backend_name is registered as an RPC backend.\n\n    Args:\n        backend_name (str): string to identify the RPC backend.\n    Returns:\n        True if the backend has been registered with ``register_backend``, else\n        False.\n    '
    return backend_name in BackendType.__members__.keys()

def register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler):
    if False:
        i = 10
        return i + 15
    'Registers a new RPC backend.\n\n    Args:\n        backend_name (str): backend string to identify the handler.\n        construct_rpc_backend_options_handler (function):\n            Handler that is invoked when\n            rpc_backend.construct_rpc_backend_options(**dict) is called.\n        init_backend_handler (function): Handler that is invoked when the\n            `_init_rpc_backend()` function is called with a backend.\n             This returns the agent.\n    '
    global BackendType
    if backend_registered(backend_name):
        raise RuntimeError(f'RPC backend {backend_name}: already registered')
    existing_enum_dict = {member.name: member.value for member in BackendType}
    extended_enum_dict = dict({backend_name: BackendValue(construct_rpc_backend_options_handler=construct_rpc_backend_options_handler, init_backend_handler=init_backend_handler)}, **existing_enum_dict)
    BackendType = enum.Enum(value='BackendType', names=extended_enum_dict)
    BackendType.__repr__ = _backend_type_repr
    if BackendType.__doc__:
        BackendType.__doc__ = _backend_type_doc
    return BackendType[backend_name]

def construct_rpc_backend_options(backend, rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC, init_method=rpc_constants.DEFAULT_INIT_METHOD, **kwargs):
    if False:
        return 10
    return backend.value.construct_rpc_backend_options_handler(rpc_timeout, init_method, **kwargs)

def init_backend(backend, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return backend.value.init_backend_handler(*args, **kwargs)

def _init_process_group(store, rank, world_size):
    if False:
        for i in range(10):
            print('nop')
    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT
    group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)
    assert group is not None, 'Failed to initialize default ProcessGroup.'
    if rank != -1 and rank != group.rank():
        raise RuntimeError(f"rank argument {rank} doesn't match pg rank {group.rank()}")
    if world_size != -1 and world_size != group.size():
        raise RuntimeError(f"world_size argument {world_size} doesn't match pg size {group.size()}")
    return group

def _tensorpipe_construct_rpc_backend_options_handler(rpc_timeout, init_method, num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS, _transports=None, _channels=None, **kwargs):
    if False:
        return 10
    from . import TensorPipeRpcBackendOptions
    return TensorPipeRpcBackendOptions(rpc_timeout=rpc_timeout, init_method=init_method, num_worker_threads=num_worker_threads, _transports=_transports, _channels=_channels)

def _tensorpipe_validate_devices(devices, device_count):
    if False:
        for i in range(10):
            print('nop')
    return all((d.type == 'cpu' or (d.type == 'cuda' and 0 <= d.index < device_count) for d in devices))

def _tensorpipe_exchange_and_check_all_device_maps(my_name, my_device_count, my_device_maps, my_devices, group):
    if False:
        for i in range(10):
            print('nop')
    gathered: List[Tuple[str, int, Dict[str, Dict[torch.device, torch.device]], List[torch.device]]] = [('', 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(gathered, (my_name, my_device_count, my_device_maps, my_devices), group)
    all_names = [name for (name, _, _, _) in gathered]
    all_device_counts = {name: count for (name, count, _, _) in gathered}
    all_device_maps = {name: map_ for (name, _, map_, _) in gathered}
    all_devices = {name: devices for (name, _, _, devices) in gathered}
    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = _create_device_list(my_devices, my_device_maps, reverse_device_maps)
    return (reverse_device_maps, my_devices)

def _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True):
    if False:
        print('Hello World!')
    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(f'Node {node} has duplicated devices\ndevices = {devices}')
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(f'Node {node} has devices with invalid indices\ndevices = {devices}\ndevice count = {all_device_counts[node]}')
    for source_node in all_names:
        if is_static_group and (not set(all_device_maps[source_node].keys()).issubset(all_names)):
            raise ValueError(f'Node {source_node} has invalid target node names in its device maps\ndevice maps = {all_device_maps[source_node].keys()}\nnode names = {all_names}')
        for (target_node, map_) in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                raise ValueError(f'Node {source_node} has duplicated target devices in its device map for {target_node}\ndevice map = {map_}')
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    raise ValueError(f'Node {source_node} has unexpected source devices in its device map for {target_node}\ndevice map = {map_}\ndevices = {all_devices[source_node]}')
            elif not _tensorpipe_validate_devices(map_.keys(), all_device_counts[source_node]):
                raise ValueError(f'Node {source_node} has source devices with invalid indices in its device map for {target_node}\ndevice map = {map_}\ndevice count = {all_device_counts[source_node]}')
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(f'Node {source_node} has unexpected target devices in its device map for {target_node}\ndevice map = {map_}\ndevices = {all_devices[target_node]}')
            elif target_node in all_device_counts and (not _tensorpipe_validate_devices(map_.values(), all_device_counts[target_node])):
                raise ValueError(f'Node {source_node} has target devices with invalid indices in its device map for {target_node}\ndevice map = {map_}\ndevice count = {all_device_counts[target_node]}')

def _create_device_list(my_devices, my_device_maps, reverse_device_maps):
    if False:
        print('Hello World!')
    if not my_devices:
        devices_set: Set[torch.device] = set()
        for map_ in my_device_maps.values():
            devices_set.update(map_.keys())
        for map_ in reverse_device_maps.values():
            devices_set.update(map_.keys())
        devices_set.discard(torch.device('cpu'))
        my_devices = list(devices_set)
    my_devices = sorted(my_devices, key=lambda d: d.index)
    return my_devices

def _create_reverse_mapping(my_name, all_names, all_device_maps):
    if False:
        return 10
    reverse_device_maps: Dict[str, Dict[torch.device, torch.device]] = {}
    for node in all_names:
        if my_name in all_device_maps[node]:
            reverse_device_maps[node] = {v: k for (k, v) in all_device_maps[node][my_name].items()}
    return reverse_device_maps

def _get_device_infos():
    if False:
        print('Hello World!')
    from . import TensorPipeAgent
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    opts = agent._get_backend_options()
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available() and opts.devices:
        torch.cuda.init()
    return (device_count, opts.device_maps, opts.devices)

def _set_devices_and_reverse_device_map(agent):
    if False:
        while True:
            i = 10
    from . import TensorPipeAgent
    agent = cast(TensorPipeAgent, agent)
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    all_worker_infos = agent.get_worker_infos()
    (all_device_counts, all_device_maps, all_devices, all_names) = ({}, {}, {}, [])
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        if worker_name != my_name:
            (device_count, device_map, devices) = api.rpc_sync(worker_name, _get_device_infos)
        else:
            opts = agent._get_backend_options()
            (device_count, device_map, devices) = (torch.cuda.device_count(), opts.device_maps, opts.devices)
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        all_names.append(worker_name)
    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices, is_static_group=False)
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    for worker_name in all_names:
        all_devices[worker_name] = _create_device_list(all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps)
        api.rpc_sync(worker_name, _update_group_membership, args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True))

def _tensorpipe_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    if False:
        for i in range(10):
            print('nop')
    from . import TensorPipeAgent
    from . import TensorPipeRpcBackendOptions
    if not isinstance(store, dist.Store):
        raise TypeError(f'`store` must be a c10d::Store. {store}')
    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError(f'`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {rpc_backend_options}')
    device_count = torch.cuda.device_count()
    is_static_group = True if world_size else False
    if is_static_group:
        group = _init_process_group(store, rank, world_size)
        (reverse_device_maps, devices) = _tensorpipe_exchange_and_check_all_device_maps(name, device_count, rpc_backend_options.device_maps, rpc_backend_options.devices, group)
        if torch.cuda.is_available() and devices:
            torch.cuda.init()
        agent = TensorPipeAgent(store, name, rank, world_size, rpc_backend_options, reverse_device_maps, devices)
        api._init_rpc_states(agent)
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        group.barrier().wait()
        return agent
    else:
        with _group_membership_management(store, name, True):
            agent = TensorPipeAgent(store, name, rank, world_size, rpc_backend_options, {}, [])
            api._init_rpc_states(agent)
            try:
                _set_devices_and_reverse_device_map(agent)
                pass
            except Exception:
                api.shutdown()
                raise
            return agent
register_backend('TENSORPIPE', _tensorpipe_construct_rpc_backend_options_handler, _tensorpipe_init_backend_handler)