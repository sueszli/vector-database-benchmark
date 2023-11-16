"""APIs exposed under the namespace ray.util.collective."""
import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
logger = logging.getLogger(__name__)
try:
    from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
except ImportError:
    _NCCL_AVAILABLE = False
    logger.warning('NCCL seems unavailable. Please install Cupy following the guide at: https://docs.cupy.dev/en/stable/install.html.')
try:
    from ray.util.collective.collective_group.gloo_collective_group import GLOOGroup
except ImportError:
    _GLOO_AVAILABLE = False

def nccl_available():
    if False:
        i = 10
        return i + 15
    return _NCCL_AVAILABLE

def gloo_available():
    if False:
        for i in range(10):
            print('nop')
    return _GLOO_AVAILABLE

class GroupManager(object):
    """Use this class to manage the collective groups we created so far.

    Each process will have an instance of `GroupManager`. Each process
    could belong to multiple collective groups. The membership information
    and other metadata are stored in the global `_group_mgr` object.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._name_group_map = {}
        self._group_name_map = {}

    def create_collective_group(self, backend, world_size, rank, group_name):
        if False:
            print('Hello World!')
        'The entry to create new collective groups in the manager.\n\n        Put the registration and the group information into the manager\n        metadata as well.\n        '
        backend = types.Backend(backend)
        if backend == types.Backend.MPI:
            raise RuntimeError('Ray does not support MPI.')
        elif backend == types.Backend.GLOO:
            logger.debug("Creating GLOO group: '{}'...".format(group_name))
            g = GLOOGroup(world_size, rank, group_name, store_type='ray_internal_kv', device_type='tcp')
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        elif backend == types.Backend.NCCL:
            logger.debug("Creating NCCL group: '{}'...".format(group_name))
            g = NCCLGroup(world_size, rank, group_name)
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        return self._name_group_map[group_name]

    def is_group_exist(self, group_name):
        if False:
            for i in range(10):
                print('nop')
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        if False:
            print('Hello World!')
        'Get the collective group handle by its name.'
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' is not initialized.".format(group_name))
            return None
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        if False:
            while True:
                i = 10
        'Group destructor.'
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' does not exist.".format(group_name))
            return
        g = self._name_group_map[group_name]
        del self._group_name_map[g]
        del self._name_group_map[group_name]
        g.destroy_group()
        name = 'info_' + group_name
        try:
            store = ray.get_actor(name)
            ray.kill(store)
        except ValueError:
            pass
_group_mgr = GroupManager()

def is_group_initialized(group_name):
    if False:
        for i in range(10):
            print('nop')
    'Check if the group is initialized in this process by the group name.'
    return _group_mgr.is_group_exist(group_name)

def init_collective_group(world_size: int, rank: int, backend=types.Backend.NCCL, group_name: str='default'):
    if False:
        print('Hello World!')
    'Initialize a collective group inside an actor process.\n\n    Args:\n        world_size: the total number of processes in the group.\n        rank: the rank of the current process.\n        backend: the CCL backend to use, NCCL or GLOO.\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    _check_inside_actor()
    backend = types.Backend(backend)
    _check_backend_availability(backend)
    global _group_mgr
    if not group_name:
        raise ValueError("group_name '{}' needs to be a string.".format(group_name))
    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError('Trying to initialize a group twice.')
    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)

def create_collective_group(actors, world_size: int, ranks: List[int], backend=types.Backend.NCCL, group_name: str='default'):
    if False:
        return 10
    'Declare a list of actors as a collective group.\n\n    Note: This function should be called in a driver process.\n\n    Args:\n        actors: a list of actors to be set in a collective group.\n        world_size: the total number of processes in the group.\n        ranks (List[int]): the rank of each actor.\n        backend: the CCL backend to use, NCCL or GLOO.\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    backend = types.Backend(backend)
    _check_backend_availability(backend)
    name = 'info_' + group_name
    try:
        ray.get_actor(name)
        raise RuntimeError('Trying to initialize a group twice.')
    except ValueError:
        pass
    if len(ranks) != len(actors):
        raise RuntimeError("Each actor should correspond to one rank. Got '{}' ranks but '{}' actors".format(len(ranks), len(actors)))
    if set(ranks) != set(range(len(ranks))):
        raise RuntimeError("Ranks must be a permutation from 0 to '{}'. Got '{}'.".format(len(ranks), ''.join([str(r) for r in ranks])))
    if world_size <= 0:
        raise RuntimeError("World size must be greater than zero. Got '{}'.".format(world_size))
    if not all(ranks) >= 0:
        raise RuntimeError('Ranks must be non-negative.')
    if not all(ranks) < world_size:
        raise RuntimeError('Ranks cannot be greater than world_size.')
    from ray.util.collective.util import Info
    name = 'info_' + group_name
    actors_id = [a._ray_actor_id for a in actors]
    info = Info.options(name=name, lifetime='detached').remote()
    ray.get([info.set_info.remote(actors_id, world_size, ranks, backend)])

def destroy_collective_group(group_name: str='default') -> None:
    if False:
        print('Hello World!')
    'Destroy a collective group given its group name.'
    _check_inside_actor()
    global _group_mgr
    _group_mgr.destroy_collective_group(group_name)

def get_rank(group_name: str='default') -> int:
    if False:
        print('Hello World!')
    'Return the rank of this process in the given group.\n\n    Args:\n        group_name: the name of the group to query\n\n    Returns:\n        the rank of this process in the named group,\n        -1 if the group does not exist or the process does\n        not belong to the group.\n    '
    _check_inside_actor()
    if not is_group_initialized(group_name):
        return -1
    g = _group_mgr.get_group_by_name(group_name)
    return g.rank

def get_collective_group_size(group_name: str='default') -> int:
    if False:
        while True:
            i = 10
    'Return the size of the collective group with the given name.\n\n    Args:\n        group_name: the name of the group to query\n\n    Returns:\n        The world size of the collective group, -1 if the group does\n            not exist or the process does not belong to the group.\n    '
    _check_inside_actor()
    if not is_group_initialized(group_name):
        return -1
    g = _group_mgr.get_group_by_name(group_name)
    return g.world_size

def allreduce(tensor, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        i = 10
        return i + 15
    'Collective allreduce the tensor across the group.\n\n    Args:\n        tensor: the tensor to be all-reduced on this process.\n        group_name: the collective group name to perform allreduce.\n        op: The reduce operation.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduceOp = op
    g.allreduce([tensor], opts)

def allreduce_multigpu(tensor_list: list, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        print('Hello World!')
    'Collective allreduce a list of tensors across the group.\n\n    Args:\n        tensor_list (List[tensor]): list of tensors to be allreduced,\n            each on a GPU.\n        group_name: the collective group name to perform allreduce.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduceOp = op
    g.allreduce(tensor_list, opts)

def barrier(group_name: str='default'):
    if False:
        i = 10
        return i + 15
    'Barrier all processes in the collective group.\n\n    Args:\n        group_name: the name of the group to barrier.\n\n    Returns:\n        None\n    '
    g = _check_and_get_group(group_name)
    g.barrier()

def reduce(tensor, dst_rank: int=0, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        print('Hello World!')
    'Reduce the tensor across the group to the destination rank.\n\n    Args:\n        tensor: the tensor to be reduced on this process.\n        dst_rank: the rank of the destination process.\n        group_name: the collective group name to perform reduce.\n        op: The reduce operation.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    opts = types.ReduceOptions()
    opts.reduceOp = op
    opts.root_rank = dst_rank
    opts.root_tensor = 0
    g.reduce([tensor], opts)

def reduce_multigpu(tensor_list: list, dst_rank: int=0, dst_tensor: int=0, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        while True:
            i = 10
    'Reduce the tensor across the group to the destination rank\n    and destination tensor.\n\n    Args:\n        tensor_list: the list of tensors to be reduced on this process;\n            each tensor located on a GPU.\n        dst_rank: the rank of the destination process.\n        dst_tensor: the index of GPU at the destination.\n        group_name: the collective group name to perform reduce.\n        op: The reduce operation.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    _check_root_tensor_valid(len(tensor_list), dst_tensor)
    opts = types.ReduceOptions()
    opts.reduceOp = op
    opts.root_rank = dst_rank
    opts.root_tensor = dst_tensor
    g.reduce(tensor_list, opts)

def broadcast(tensor, src_rank: int=0, group_name: str='default'):
    if False:
        i = 10
        return i + 15
    'Broadcast the tensor from a source process to all others.\n\n    Args:\n        tensor: the tensor to be broadcasted (src) or received (destination).\n        src_rank: the rank of the source process.\n        group_name: the collective group name to perform broadcast.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    opts = types.BroadcastOptions()
    opts.root_rank = src_rank
    opts.root_tensor = 0
    g.broadcast([tensor], opts)

def broadcast_multigpu(tensor_list, src_rank: int=0, src_tensor: int=0, group_name: str='default'):
    if False:
        return 10
    'Broadcast the tensor from a source GPU to all other GPUs.\n\n    Args:\n        tensor_list: the tensors to broadcast (src) or receive (dst).\n        src_rank: the rank of the source process.\n        src_tensor: the index of the source GPU on the source process.\n        group_name: the collective group name to perform broadcast.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    _check_root_tensor_valid(len(tensor_list), src_tensor)
    opts = types.BroadcastOptions()
    opts.root_rank = src_rank
    opts.root_tensor = src_tensor
    g.broadcast(tensor_list, opts)

def allgather(tensor_list: list, tensor, group_name: str='default'):
    if False:
        while True:
            i = 10
    'Allgather tensors from each process of the group into a list.\n\n    Args:\n        tensor_list: the results, stored as a list of tensors.\n        tensor: the tensor (to be gathered) in the current process\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    if len(tensor_list) != g.world_size:
        raise RuntimeError('The length of the tensor list operands to allgather must be equal to world_size.')
    opts = types.AllGatherOptions()
    g.allgather([tensor_list], [tensor], opts)

def allgather_multigpu(output_tensor_lists: list, input_tensor_list: list, group_name: str='default'):
    if False:
        for i in range(10):
            print('nop')
    'Allgather tensors from each gpus of the group into lists.\n\n    Args:\n        output_tensor_lists (List[List[tensor]]): gathered results, with shape\n            must be num_gpus * world_size * shape(tensor).\n        input_tensor_list: (List[tensor]): a list of tensors, with shape\n            num_gpus * shape(tensor).\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_lists_input(output_tensor_lists)
    _check_tensor_list_input(input_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllGatherOptions()
    g.allgather(output_tensor_lists, input_tensor_list, opts)

def reducescatter(tensor, tensor_list: list, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        print('Hello World!')
    'Reducescatter a list of tensors across the group.\n\n    Reduce the list of the tensors across each process in the group, then\n    scatter the reduced list of tensors -- one tensor for each process.\n\n    Args:\n        tensor: the resulted tensor on this process.\n        tensor_list: The list of tensors to be reduced and scattered.\n        group_name: the name of the collective group.\n        op: The reduce operation.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    if len(tensor_list) != g.world_size:
        raise RuntimeError('The length of the tensor list operands to reducescatter must not be equal to world_size.')
    opts = types.ReduceScatterOptions()
    opts.reduceOp = op
    g.reducescatter([tensor], [tensor_list], opts)

def reducescatter_multigpu(output_tensor_list, input_tensor_lists, group_name: str='default', op=types.ReduceOp.SUM):
    if False:
        for i in range(10):
            print('nop')
    'Reducescatter a list of tensors across all GPUs.\n\n    Args:\n        output_tensor_list: the resulted list of tensors, with\n            shape: num_gpus * shape(tensor).\n        input_tensor_lists: the original tensors, with shape:\n            num_gpus * world_size * shape(tensor).\n        group_name: the name of the collective group.\n        op: The reduce operation.\n\n    Returns:\n        None.\n    '
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_lists_input(input_tensor_lists)
    _check_tensor_list_input(output_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.ReduceScatterOptions()
    opts.reduceOp = op
    g.reducescatter(output_tensor_list, input_tensor_lists, opts)

def send(tensor, dst_rank: int, group_name: str='default'):
    if False:
        i = 10
        return i + 15
    'Send a tensor to a remote process synchronously.\n\n    Args:\n        tensor: the tensor to send.\n        dst_rank: the rank of the destination process.\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    if dst_rank == g.rank:
        raise RuntimeError("The destination rank '{}' is self.".format(dst_rank))
    opts = types.SendOptions()
    opts.dst_rank = dst_rank
    g.send([tensor], opts)

def send_multigpu(tensor, dst_rank: int, dst_gpu_index: int, group_name: str='default', n_elements: int=0):
    if False:
        i = 10
        return i + 15
    'Send a tensor to a remote GPU synchronously.\n\n    The function asssume each process owns >1 GPUs, and the sender\n    process and receiver process has equal nubmer of GPUs.\n\n    Args:\n        tensor: the tensor to send, located on a GPU.\n        dst_rank: the rank of the destination process.\n        dst_gpu_index: the destination gpu index.\n        group_name: the name of the collective group.\n        n_elements: if specified, send the next n elements\n            from the starting address of tensor.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('send_multigpu call requires NCCL.')
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    if dst_rank == g.rank:
        raise RuntimeError("The dst_rank '{}' is self. Considering doing GPU to GPU memcpy instead?".format(dst_rank))
    if n_elements < 0:
        raise RuntimeError("The n_elements '{}' should >= 0.".format(n_elements))
    opts = types.SendOptions()
    opts.dst_rank = dst_rank
    opts.dst_gpu_index = dst_gpu_index
    opts.n_elements = n_elements
    g.send([tensor], opts)

def recv(tensor, src_rank: int, group_name: str='default'):
    if False:
        while True:
            i = 10
    'Receive a tensor from a remote process synchronously.\n\n    Args:\n        tensor: the received tensor.\n        src_rank: the rank of the source process.\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    if src_rank == g.rank:
        raise RuntimeError("The destination rank '{}' is self.".format(src_rank))
    opts = types.RecvOptions()
    opts.src_rank = src_rank
    g.recv([tensor], opts)

def recv_multigpu(tensor, src_rank: int, src_gpu_index: int, group_name: str='default', n_elements: int=0):
    if False:
        return 10
    'Receive a tensor from a remote GPU synchronously.\n\n    The function asssume each process owns >1 GPUs, and the sender\n    process and receiver process has equal nubmer of GPUs.\n\n    Args:\n        tensor: the received tensor, located on a GPU.\n        src_rank: the rank of the source process.\n        src_gpu_index (int)： the index of the source gpu on the src process.\n        group_name: the name of the collective group.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('recv_multigpu call requires NCCL.')
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    if src_rank == g.rank:
        raise RuntimeError("The dst_rank '{}' is self. Considering doing GPU to GPU memcpy instead?".format(src_rank))
    if n_elements < 0:
        raise RuntimeError("The n_elements '{}' should be >= 0.".format(n_elements))
    opts = types.RecvOptions()
    opts.src_rank = src_rank
    opts.src_gpu_index = src_gpu_index
    opts.n_elements = n_elements
    g.recv([tensor], opts)

def synchronize(gpu_id: int):
    if False:
        print('Hello World!')
    'Synchronize the current process to a give device.\n\n    Args:\n        gpu_id: the GPU device id to synchronize.\n\n    Returns:\n        None\n    '
    if not types.cupy_available():
        raise RuntimeError('synchronize call requires CUDA and NCCL.')
    import cupy as cp
    cp.cuda.Device(gpu_id).synchronize()

def _check_and_get_group(group_name):
    if False:
        print('Hello World!')
    'Check the existence and return the group handle.'
    _check_inside_actor()
    global _group_mgr
    if not is_group_initialized(group_name):
        try:
            name = 'info_' + group_name
            mgr = ray.get_actor(name=name)
            (ids, world_size, rank, backend) = ray.get(mgr.get_info.remote())
            worker = ray._private.worker.global_worker
            id_ = worker.core_worker.get_actor_id()
            r = rank[ids.index(id_)]
            _group_mgr.create_collective_group(backend, world_size, r, group_name)
        except ValueError as exc:
            if 'collective_group_name' in os.environ and os.environ['collective_group_name'] == group_name:
                rank = int(os.environ['collective_rank'])
                world_size = int(os.environ['collective_world_size'])
                backend = os.environ['collective_backend']
                _group_mgr.create_collective_group(backend, world_size, rank, group_name)
            else:
                raise RuntimeError("The collective group '{}' is not initialized in the process.".format(group_name)) from exc
    g = _group_mgr.get_group_by_name(group_name)
    return g

def _check_single_tensor_input(tensor):
    if False:
        while True:
            i = 10
    'Check if the tensor is with a supported type.'
    if isinstance(tensor, np.ndarray):
        return
    if types.cupy_available():
        if isinstance(tensor, types.cp.ndarray):
            return
    if types.torch_available():
        if isinstance(tensor, types.th.Tensor):
            return
    raise RuntimeError("Unrecognized tensor type '{}'. Supported types are: np.ndarray, torch.Tensor, cupy.ndarray.".format(type(tensor)))

def _check_backend_availability(backend: types.Backend):
    if False:
        while True:
            i = 10
    'Check whether the backend is available.'
    if backend == types.Backend.GLOO:
        if not gloo_available():
            raise RuntimeError('GLOO is not available.')
    elif backend == types.Backend.NCCL:
        if not nccl_available():
            raise RuntimeError('NCCL is not available.')

def _check_inside_actor():
    if False:
        print('Hello World!')
    'Check if currently it is inside a Ray actor/task.'
    worker = ray._private.worker.global_worker
    if worker.mode == ray.WORKER_MODE:
        return
    else:
        raise RuntimeError('The collective APIs shall be only used inside a Ray actor or task.')

def _check_rank_valid(g, rank: int):
    if False:
        while True:
            i = 10
    'Check the rank: 0 <= rank < world_size.'
    if rank < 0:
        raise ValueError("rank '{}' is negative.".format(rank))
    if rank >= g.world_size:
        raise ValueError("rank '{}' must be less than world size '{}'".format(rank, g.world_size))

def _check_tensor_list_input(tensor_list):
    if False:
        return 10
    'Check if the input is a list of supported tensor types.'
    if not isinstance(tensor_list, list):
        raise RuntimeError("The input must be a list of tensors. Got '{}'.".format(type(tensor_list)))
    if not tensor_list:
        raise RuntimeError('Got an empty list of tensors.')
    for t in tensor_list:
        _check_single_tensor_input(t)

def _check_tensor_lists_input(tensor_lists):
    if False:
        return 10
    'Check if the input is a list of lists of supported tensor types.'
    if not isinstance(tensor_lists, list):
        raise RuntimeError("The input must be a list of lists of tensors. Got '{}'.".format(type(tensor_lists)))
    if not tensor_lists:
        raise RuntimeError(f'Did not receive tensors. Got: {tensor_lists}')
    for t in tensor_lists:
        _check_tensor_list_input(t)

def _check_root_tensor_valid(length, root_tensor):
    if False:
        print('Hello World!')
    'Check the root_tensor device is 0 <= root_tensor < length'
    if root_tensor < 0:
        raise ValueError("root_tensor '{}' is negative.".format(root_tensor))
    if root_tensor >= length:
        raise ValueError("root_tensor '{}' is greater than the number of GPUs: '{}'".format(root_tensor, length))