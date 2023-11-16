import builtins
import torch
from torch.distributed._shard.sharding_spec import ChunkShardingSpec, EnumerableShardingSpec, ShardMetadata
from torch.distributed._shard.sharding_spec._internals import get_chunked_dim_size, get_split_size

def generate_chunk_sharding_specs_for_test(sharding_dim):
    if False:
        print('Hello World!')
    return [ChunkShardingSpec(dim=sharding_dim, placements=['rank:0/cuda:0', 'rank:1/cuda:1', 'rank:2/cuda:2', 'rank:3/cuda:3']), ChunkShardingSpec(dim=sharding_dim, placements=['rank:2/cuda:2', 'rank:3/cuda:3', 'rank:0/cuda:0', 'rank:1/cuda:1']), ChunkShardingSpec(dim=sharding_dim, placements=['rank:3/cuda:3', 'rank:0/cuda:0', 'rank:1/cuda:1', 'rank:2/cuda:2'])]

def generate_enumerable_sharding_specs_for_test():
    if False:
        print('Hello World!')
    return [EnumerableShardingSpec([ShardMetadata(shard_offsets=[0, 0], shard_sizes=[5, 5], placement='rank:0/cuda:0'), ShardMetadata(shard_offsets=[5, 0], shard_sizes=[5, 5], placement='rank:1/cuda:1'), ShardMetadata(shard_offsets=[0, 5], shard_sizes=[5, 5], placement='rank:2/cuda:2'), ShardMetadata(shard_offsets=[5, 5], shard_sizes=[5, 5], placement='rank:3/cuda:3')])]

def generate_local_weight_sharding_params_for_test(local_weight, sharded_dim, gpu_num, spec, rank):
    if False:
        i = 10
        return i + 15
    '\n    Shard the local weight based the given spec, so we can compare against\n    the one from sharded tensor.\n\n    Args:\n        local_weight: weight matrix to be sharded.\n        sharded_dim: The dimension which we shard on.\n        gpu_num: number of ranks.\n        spec: sharding spec.\n        rank: # of cuda process.\n\n    Returns:\n        start_pos: start position of sharded weight on the given rank.\n        chunk_size: chunk size of sharded weight on the given rank.\n    '
    sharding_dim_size = local_weight.size(sharded_dim)
    split_size = get_split_size(sharding_dim_size, gpu_num)
    current_offsets = 0
    start_pos = current_offsets
    for (idx, placement) in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return (start_pos, chunk_size)

def clone_module_parameter(module, param_name):
    if False:
        print('Hello World!')
    '\n    Clone a parameter from a given existing module.\n\n    Args:\n        module (:class:`torch.nn.Module`): Module whose parameter needs to be cloned.\n        param_name (str): Name of the parameter of ``module`` that needs to be cloned.\n\n    Returns: cloned tensor as :class:`torch.nn.Parameter`.\n    '
    tensor = getattr(module, param_name)
    return torch.nn.Parameter(tensor.detach().clone())

def gen_binary_op_func(python_op, inplace=False):
    if False:
        print('Hello World!')
    src_lines = ['def f(lhs, rhs):']
    if 'torch' in python_op:
        src_lines.append(f'  return {python_op}(lhs, rhs)\n')
    elif inplace:
        src_lines.append(f'  lhs {python_op}= rhs\n  return lhs\n')
    else:
        src_lines.append(f'  return lhs {python_op} rhs\n')
    code_str = '\n'.join(src_lines)
    g = {'torch': torch}
    builtins.exec(code_str, g)
    return g['f']