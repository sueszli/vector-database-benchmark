from typing import Callable, Tuple, List, Any, Union
from easydict import EasyDict
import os
import numpy as np
import torch
import torch.distributed as dist
from .default_helper import error_wrapper

def get_rank() -> int:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Get the rank of current process in total world_size\n    '
    return error_wrapper(dist.get_rank, 0)()

def get_world_size() -> int:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Get the world_size(total process number in data parallel training)\n    '
    return error_wrapper(dist.get_world_size, 1)()
broadcast = dist.broadcast
allgather = dist.all_gather
broadcast_object_list = dist.broadcast_object_list

def allreduce(x: torch.Tensor) -> None:
    if False:
        i = 10
        return i + 15
    dist.all_reduce(x)
    x.div_(get_world_size())

def allreduce_async(name: str, x: torch.Tensor) -> None:
    if False:
        while True:
            i = 10
    x.div_(get_world_size())
    dist.all_reduce(x, async_op=True)

def reduce_data(x: Union[int, float, torch.Tensor], dst: int) -> Union[int, float, torch.Tensor]:
    if False:
        return 10
    if np.isscalar(x):
        x_tensor = torch.as_tensor([x]).cuda()
        dist.reduce(x_tensor, dst)
        return x_tensor.item()
    elif isinstance(x, torch.Tensor):
        dist.reduce(x, dst)
        return x
    else:
        raise TypeError('not supported type: {}'.format(type(x)))

def allreduce_data(x: Union[int, float, torch.Tensor], op: str) -> Union[int, float, torch.Tensor]:
    if False:
        i = 10
        return i + 15
    assert op in ['sum', 'avg'], op
    if np.isscalar(x):
        x_tensor = torch.as_tensor([x]).cuda()
        dist.all_reduce(x_tensor)
        if op == 'avg':
            x_tensor.div_(get_world_size())
        return x_tensor.item()
    elif isinstance(x, torch.Tensor):
        dist.all_reduce(x)
        if op == 'avg':
            x.div_(get_world_size())
        return x
    else:
        raise TypeError('not supported type: {}'.format(type(x)))
synchronize = torch.cuda.synchronize

def get_group(group_size: int) -> List:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Get the group segmentation of ``group_size`` each group\n    Arguments:\n        - group_size (:obj:`int`) the ``group_size``\n    '
    rank = get_rank()
    world_size = get_world_size()
    if group_size is None:
        group_size = world_size
    assert world_size % group_size == 0
    return simple_group_split(world_size, rank, world_size // group_size)

def dist_mode(func: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Wrap the function so that in can init and finalize automatically before each call\n    '

    def wrapper(*args, **kwargs):
        if False:
            return 10
        dist_init()
        func(*args, **kwargs)
        dist_finalize()
    return wrapper

def dist_init(backend: str='nccl', addr: str=None, port: str=None, rank: int=None, world_size: int=None) -> Tuple[int, int]:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Init the distributed training setting\n    '
    assert backend in ['nccl', 'gloo'], backend
    os.environ['MASTER_ADDR'] = addr or os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = port or os.environ.get('MASTER_PORT', '10314')
    if rank is None:
        local_id = os.environ.get('SLURM_LOCALID', os.environ.get('RANK', None))
        if local_id is None:
            raise RuntimeError('please indicate rank explicitly in dist_init method')
        else:
            rank = int(local_id)
    if world_size is None:
        ntasks = os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', None))
        if ntasks is None:
            raise RuntimeError('please indicate world_size explicitly in dist_init method')
        else:
            world_size = int(ntasks)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    world_size = get_world_size()
    rank = get_rank()
    return (rank, world_size)

def dist_finalize() -> None:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Finalize distributed training resources\n    '
    pass

class DDPContext:

    def __init__(self) -> None:
        if False:
            return 10
        pass

    def __enter__(self) -> None:
        if False:
            print('Hello World!')
        dist_init()

    def __exit__(self, *args, **kwargs) -> Any:
        if False:
            return 10
        dist_finalize()

def simple_group_split(world_size: int, rank: int, num_groups: int) -> List:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Split the group according to ``worldsize``, ``rank`` and ``num_groups``\n\n    .. note::\n        With faulty input, raise ``array split does not result in an equal division``\n    '
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]

def to_ddp_config(cfg: EasyDict) -> EasyDict:
    if False:
        i = 10
        return i + 15
    w = get_world_size()
    if 'batch_size' in cfg.policy:
        cfg.policy.batch_size = int(np.ceil(cfg.policy.batch_size / w))
    if 'batch_size' in cfg.policy.learn:
        cfg.policy.learn.batch_size = int(np.ceil(cfg.policy.learn.batch_size / w))
    if 'n_sample' in cfg.policy.collect:
        cfg.policy.collect.n_sample = int(np.ceil(cfg.policy.collect.n_sample / w))
    if 'n_episode' in cfg.policy.collect:
        cfg.policy.collect.n_episode = int(np.ceil(cfg.policy.collect.n_episode / w))
    return cfg