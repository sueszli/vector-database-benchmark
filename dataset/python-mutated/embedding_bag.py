from typing import cast, List
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import _all_gather_base_input, _handle_col_wise_sharding_base, _handle_max_norm_col_wise, _handle_row_wise_mask

@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    if False:
        return 10
    '\n    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.\n    This method computes a sharded embedding bag aggregation and has the following limitations:\n\n    1. Supports only sharding of ``weight``.\n    2. Supports only ``ChunkShardingSpec``.\n    3. Supports only a single local shard per rank.\n    4. Supports all specs except for scale_grad_by_freq, sparse, etc.\n\n    Based on the dimension that the weight is sharded on, there are two\n    algorithms:\n\n    ROWWISE SHARDING\n    ================\n    For row-wise sharding the weight is sharded on dimension 0.\n\n    The overall algorithm can be best explained with an example. Let\'s assume\n    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across\n    4 GPUs creating 4 shard of (4 x 17).\n    The algorithm is as follows:\n\n    1. First the input is all gathered to all ranks, since this is SPMD and\n       input is actually sharded across all ranks. The inputs then become a\n       4 (4 x 6) tensor on each rank. For example if the given input is\n       tensor([[6, 5, 2, 9, 6, 3],\n               [3, 1, 2, 4, 7, 6],\n               [4, 0, 4, 9, 8, 9],\n               [8, 6, 6, 4, 6, 1]])\n       on rank 0.\n       Then on every rank, we will have this tensor.\n       If input itself is already replicated, no all-gather will be done.\n    2. Next, we mask the ID which are not stored on that rank.\n       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID\n       inside the set of numbers. The rest of them will be masked to an extra row.\n       The masked matrix will be used for embedding look up and is like:\n       tensor([[4, 4, 2, 4, 4, 4],\n               [4, 1, 2, 4, 4, 4],\n               [4, 0, 4, 4, 4, 4],\n               [4, 4, 4, 4, 4, 1]])\n    3. If ``max_norm`` is specified, the extra row guarantees that the mask ID will\n       not affect the behavior of weigh re-norm.\n    4. The example above only happens in one rank and each rank does a very similar thing.\n       For "Mean" mode we need to divide by either column size (2D) or the interval length\n       defined by the offset (excluding the row specified in ``padding_idx``).\n       We also need to mask the unexisting row to neg Inf so that negative value does not\n       gets wiped out in the "Max" mode.\n\n    COLWISE SHARDING\n    ================\n    For col-wise sharding the weight is sharded on dimension 1.\n\n    The overall algorithm can be best explained with an example. Let\'s assume\n    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across\n    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).\n    The algorithm is as follows:\n\n    1. First the input is broadcasted to all ranks, since this is SPMD we\n       actually do an all_gather for all the inputs resulting in 4 (4 x 6)\n       inputs on each rank.\n    2. Next we perform local embedding bag operation under the given mode by\n       apply each input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).\n       This results in 4 (5 x 4) ((2 x 4) for the last) matrices on each rank.\n       We transpose the aggregation result.\n    3. Next, we concatenate these 4 matrices and perform an all2all to share the\n       appropriate (5 x 4) or (2 x 4) matrices to each rank.\n    4. Now, each rank receives a (17 x 4) matrix which is basically the\n       size of the result we need.\n    5. If placements are not in order any appropriate rearrangement of columns\n       are done for the (17 x 4) matrix and finally we transpose the output again.\n    6. If max_norm is specified, we manually sum up the norm and renorm. Because\n       the renorm must be in place, we need to override the local_shard to mimic\n       this behavior.\n    '
    _validate_embedding_bag_param(args, kwargs)
    input = args[0]
    weight = args[1]
    offsets = kwargs.get('offsets')
    per_sample_weights = kwargs.get('per_sample_weights')
    mode = kwargs.get('mode')
    max_norm = kwargs.get('max_norm')
    norm_type = kwargs.get('norm_type')
    include_last_offset = kwargs.get('include_last_offset')
    padding_idx = kwargs.get('padding_idx')
    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if include_last_offset:
        offsets = offsets[:-1]
    if sharding_dim == 1:
        (output, local_shard) = _handle_col_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, pg)
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, rank, pg)
    else:
        raise RuntimeError(f'nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!')

def _validate_embedding_bag_param(args, kwargs):
    if False:
        while True:
            i = 10
    '\n    Validate input params of sharded embeddingBag op.\n\n    Args:\n        input: list of ID used for lookup and aggregation.\n        weight: sharded weight tensor.\n        kwargs: same as normal EmbeddingBag.\n\n    Return: None.\n    '
    input = args[0]
    weight = args[1]
    offsets = kwargs.get('offsets')
    per_sample_weights = kwargs.get('per_sample_weights')
    mode = kwargs.get('mode')
    max_norm = kwargs.get('max_norm')
    scale_grad_by_freq = kwargs.get('scale_grad_by_freq')
    sparse = kwargs.get('sparse')
    include_last_offset = kwargs.get('include_last_offset')
    if not isinstance(input, torch.Tensor):
        raise TypeError('input need to be torch.Tensor')
    if offsets is not None and (not isinstance(offsets, torch.Tensor)):
        raise TypeError('offsets need to be torch.Tensor')
    if per_sample_weights is not None and (not isinstance(per_sample_weights, torch.Tensor)):
        raise TypeError('per_sample_weights need to be torch.Tensor')
    if not isinstance(weight, ShardedTensor):
        raise TypeError('weight needs to be ShardedTensor')
    if len(input.size()) > 2:
        raise ValueError('Input more than 2 dims not supported')
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError('Weight needs to have exactly 2 dims')
    if int(torch.min(input).item()) < 0:
        raise ValueError('Index out of range in Input %d %d', int(torch.min(input).item()), weight_size[1])
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError('Index out of range in Input %d %d', int(torch.max(input).item()), weight_size[1])
    if offsets is not None and len(input.size()) != 1:
        raise ValueError('Input dimension needs to be exactly 1 dim')
    if len(input.size()) == 1 and offsets is None:
        raise ValueError('offsets is required for 1D input')
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(f'per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}')
    if mode is None:
        mode = 'mean'
    if mode not in ['sum', 'mean', 'max']:
        raise ValueError(f"mode '{mode}' is not supported")
    if scale_grad_by_freq:
        raise RuntimeError('nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!')
    if sparse:
        raise RuntimeError('nn.Embedding weight sharded with flag on "sparse" not supported!')
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    if include_last_offset and cast(List[int], offsets)[-1] != input.size(0):
        raise ValueError('offsets need to have the input size in the end when the flag "include_last_offset" is on!')
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError('Only ChunkShardingSpec supported for ShardedTensor ops!')
    if len(weight.local_shards()) != 1:
        raise ValueError('Only one local shard supported!')

def _handle_col_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, pg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Entry-point function to handle the logic of col-wise sharding of weight\n    for embeddingBag. (Detailed explanations of the logic can be found in\n    the comment for sharded_embedding_bag.)\n\n    Args:\n        input: list of ID used for lookup and aggregation.\n        world_size: number of ranks.\n        weight: sharded weight tensor.\n        local_shard: col-wise shared local weight used for lookup.\n        offsets: list of start positions of each bag for 1D input.\n        per_sample_weights: weights for weighted sum mode.\n        mode: aggregation method of each bag.\n        max_norm: If given, each embedding vector with norm larger\n            than max_norm is renormalized to have norm max_norm.\n            Note: this will modify weight in-place.\n        norm_type: The p in the p-norm to compute for the max_norm option.\n        padding_idx: If specified, the entries at padding_idx do\n            not contribute to the gradient; therefore, the embedding\n            vector at padding_idx is not updated during training,\n            i.e. it remains as a fixed “pad”.\n            Note that the embedding vector at padding_idx is\n            excluded from the reduction.\n        pg: process group.\n\n    Return:\n        output: final result of lookup and aggregation.\n        local_shard: col-wise shared local weight used for lookup.\n            If max_norm, this will be the renormed weight.\n    '
    (gathered_inputs, gathered_per_sample_weights, gathered_offsets) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)
    if max_norm is not None:
        local_shard = _handle_max_norm_col_wise(max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg)
    output = _handle_col_wise_sharding_base(torch.nn.functional.embedding_bag, 1, input, world_size, weight, local_shard, pg, gathered_inputs, mode=mode, gathered_per_sample_weights=gathered_per_sample_weights, gathered_offsets=gathered_offsets, padding_idx=padding_idx)
    return (output, local_shard)

def _handle_row_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, rank, pg):
    if False:
        i = 10
        return i + 15
    '\n    Entry-point function to handle the logic of row-wise sharding of weight\n    for embeddingBag. (Detailed explanations of the logic can be found in\n    the comment for sharded_embedding_bag.)\n\n    Args:\n        input: list of ID used for lookup and aggregation.\n        world_size: number of ranks.\n        weight: sharded weight tensor.\n        local_shard: row-wise shared local weight used for lookup.\n        offsets: list of start positions of each bag for 1D input.\n        per_sample_weights: weights for weighted sum mode.\n        mode: aggregation method of each bag.\n        max_norm: If given, each embedding vector with norm larger\n            than max_norm is renormalized to have norm max_norm.\n            Note: this will modify weight in-place.\n        norm_type: The p in the p-norm to compute for the max_norm option.\n        padding_idx: If specified, the entries at padding_idx do\n            not contribute to the gradient; therefore, the embedding\n            vector at padding_idx is not updated during training,\n            i.e. it remains as a fixed “pad”.\n            Note that the embedding vector at padding_idx is\n            excluded from the reduction.\n        rank: # of cuda process.\n        pg: process group.\n\n    Returns:\n        gathered_output: final result of lookup and aggregation.\n    '
    if input.dim() > 1 and per_sample_weights is None:
        gather_inp = _all_gather_base_input(input, pg)
    else:
        (gathered_inputs, gathered_per_sample_weights, gathered_offsets) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)
        cat_dim = 0 if input.dim() != 1 else -1
        gather_inp = torch.cat(gathered_inputs, dim=cat_dim)
        if per_sample_weights is not None:
            per_sample_weights = torch.cat(gathered_per_sample_weights, dim=cat_dim)
        offset_add = 0 if input.dim() > 1 else input.size(0)
        if offsets is not None:
            offsets_list = torch.cat([gathered_offsets[i] + offset_add * i for i in range(pg.size())], dim=cat_dim)
    (lookup_input, padding_local, padding_row) = _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank)
    if mode == 'max':
        padding_row[:] = -float('Inf')
    if max_norm is not None:
        torch.nn.functional.embedding_bag(torch.unique(lookup_input)[:-1], local_shard, offsets=torch.tensor([0], device=local_shard.device, dtype=torch.long), mode=mode, per_sample_weights=None, max_norm=max_norm, norm_type=norm_type, padding_idx=padding_local)
        max_norm = None
    result = torch.nn.functional.embedding_bag(lookup_input, torch.cat([local_shard, padding_row]), offsets=offsets_list if offsets is not None else offsets, mode=mode if mode != 'mean' else 'sum', per_sample_weights=per_sample_weights, max_norm=max_norm, norm_type=norm_type, padding_idx=padding_local)
    op = ReduceOp.SUM if mode != 'max' else ReduceOp.MAX
    local_shards = result.chunk(pg.size())
    result = reduce_scatter(torch.empty_like(local_shards[0]), list(local_shards), op=op, group=pg)
    if mode == 'mean':
        if input.dim() > 1:
            padding_idx = padding_idx if padding_idx is not None else -1
            split_sizes = torch.sum(torch.ne(input, padding_idx), dim=-1, dtype=local_shard.dtype)
        else:
            split_sizes = torch.cat((offsets[1:offsets.size(0)] - offsets[0:-1], (input.size(0) - offsets[-1]).unsqueeze(0)), dim=-1)
        return torch.div(result, split_sizes.unsqueeze(1))
    return result

def _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg):
    if False:
        i = 10
        return i + 15
    '\n    In case we need to gather input and all other parameters of embeddingBag\n    ops, we need to stack all input together to perform ``all_gather``\n    collective communication just once.\n\n    Note that since offsets does not share the same size as input and\n    is always smaller than input, we resize it during the communication.\n\n    Args:\n        input: tensor to be applied op on.\n        per_sample_weights: weights for weighted sum mode.\n        offsets: when input is 1D. offsets determines the starting\n            index position of each bag (sequence) in input.\n        pg: process group.\n\n    Returns:\n        gathered_inputs: list of input tensor gathered from each rank.\n        gathered_per_sample_weights: list of per_sample_weights from each rank.\n        gathered_offsets: list of offsets from each rank.\n    '
    input_to_gather = [input]
    if per_sample_weights is not None:
        input_to_gather.append(per_sample_weights)
    if offsets is not None:
        input_to_gather.append(offsets.clone().resize_(input.size()))
    gathered_inputs = all_gather(torch.stack(input_to_gather), group=pg)
    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        gathered_per_sample_weights = [t[1] for t in gathered_inputs]
    gathered_offsets = None
    if offsets is not None:
        idx = 2 if per_sample_weights is not None else 1
        gathered_offsets = [t[idx].resize_(offsets.size()).to(offsets.dtype) for t in gathered_inputs]
    gathered_inputs = [t[0].to(input.dtype) for t in gathered_inputs]
    return (gathered_inputs, gathered_per_sample_weights, gathered_offsets)