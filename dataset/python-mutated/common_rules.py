from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import _is_inplace_op, _is_out_variant_op, OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    if False:
        while True:
            i = 10
    return string[:idx] + new_char + string[idx + 1:]

def _gen_reshard_suggestions(op_schema: OpSchema, input_dims: List[str], input_specs: Tuple[DTensorSpec, ...], dim_to_sharding: Dict[str, int], pending_sum: List[int]) -> OutputSharding:
    if False:
        return 10
    suggested_arg_specs: List[DTensorSpec] = []
    for (input_dim, input_spec) in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(DTensorSpec.from_dim_map(mesh=input_spec.mesh, dim_map=dim_map, sums=pending_sum, tensor_meta=input_spec.tensor_meta))
    suggested_schema = OpSchema(op_schema.op, tuple(suggested_arg_specs), {})
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    return OutputSharding(None, schema_suggestions=[suggested_schema], failed_reason='Input placements op sharding propagation failed, need to reshard!')

def einop_rule(equation: str, op_schema: OpSchema, *, linearity: bool=False, enforce_sharding: Optional[Dict[str, int]]=None) -> OutputSharding:
    if False:
        print('Hello World!')
    "\n    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.\n\n    This is mostly borrowed from @zdevito's sharding simulator. Examples:\n        mk,kn->mn - einsum\n        ij,ij->ij - addition\n        ij,j->ij - broadcasted addition\n        ij->i - reduction\n    Other ops could use this propagation algorithm when applied, note\n    that einsum propagation only deal with list of specs (DTensor specs)\n    as it only works on list of tensors!\n\n    linearity in einop_rule means that the calling op `f` follows this rule:\n        f(a + b) = f(a) + f(b)\n\n    In this case we can propagate the partial sum, note that linearity in einop\n    only applies to partial sum, not other operations like min/max (which are\n    associative but not linear).\n    "
    (inputs, outputs) = equation.split('->')
    (input_dims, output_dims) = (inputs.split(','), outputs.split(','))
    input_specs = op_schema.args_spec
    output_dim = output_dims[0]
    dim_to_sharding: Dict[str, int] = {}
    dim_to_size: Dict[str, int] = {}
    pending_sums_counter: Dict[int, int] = {}
    seen_shardings: Dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        if False:
            return 10
        if a != b:
            if a == -1 or b == -1:
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                raise RuntimeError(f'{equation}: dim {dim} sharded two different ways: {a} and {b}')
        else:
            return a
    for (input_dim, input_spec) in zip(input_dims, input_specs):
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = '+'
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1
        for (idx, (dim, mesh_dim)) in enumerate(zip(input_dim, input_spec.dim_map)):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                dim_to_sharding[dim] = merge_sharding(dim, dim_to_sharding[dim], mesh_dim)
                assert dim_to_size[dim] == input_spec.shape[idx]
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if merged_sharding_for_dim in seen_shardings and dim != seen_shardings[merged_sharding_for_dim]:
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim
    if pending_sums_counter and (not linearity):
        return _gen_reshard_suggestions(op_schema, input_dims, input_specs, dim_to_sharding, [])
    else:
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True
    for (mesh_dim, dims) in seen_shardings.items():
        if len(dims) > 1:
            costs = []
            for d in dims:
                cost = 0
                for (input_dim, input_spec) in zip(input_dims, input_specs):
                    if d in input_dim and input_spec.dim_map[input_dim.index(d)] == mesh_dim:
                        assert input_spec.tensor_meta is not None
                        global_shape = input_spec.tensor_meta.shape
                        local_shape = compute_local_shape(global_shape, input_spec.mesh, input_spec.placements)
                        cost += prod(local_shape) * input_spec.mesh.size(mesh_dim)
                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1
    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        return _gen_reshard_suggestions(op_schema, input_dims, input_specs, dim_to_sharding, pending_sums)
    for (dim, shard_on_mesh) in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)
    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == '1':
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])
    assert input_specs[0].tensor_meta is not None
    tensor_meta = TensorMeta(torch.Size(output_shape), input_specs[0].tensor_meta.stride, input_specs[0].tensor_meta.dtype)
    return OutputSharding(DTensorSpec.from_dim_map(input_specs[0].mesh, output_dim_map, pending_sums, tensor_meta=tensor_meta))

def pointwise_rule(op_schema: OpSchema, linearity: bool=False) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    '\n    Propagate the sharding for pointwise operations.\n\n    Examples:\n        ij,ij->ij - addition/mul\n        ij,j->ij - broadcasted addition\n    '
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    input_specs = op_schema.args_spec
    max_dim = max((input.ndim for input in input_specs))
    dimchars = []
    singleton_counter: List[int] = [0] * max_dim
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        if len(input_specs) > 1:
            for i in range(max_dim):
                if i < start_dim:
                    singleton_counter[i] += 1
                elif input.shape[i - start_dim] == 1:
                    singleton_counter[i] += 1
                    p = _replace_char_in_str(p, '1', i - start_dim)
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    for output_dim_idx in range(len(out_dimchars)):
        out_dimchar = out_dimchars[output_dim_idx]
        if singleton_counter[output_dim_idx] == len(input_specs):
            out_dimchars = _replace_char_in_str(out_dimchars, '1', output_dim_idx)
    fmt = f"{','.join((p for p in dimchars))}->{out_dimchars}"
    enforce_sharding: Dict[str, int] = {}
    if _is_inplace_op(op_schema.op):
        for (out_dimchar, mesh_dim) in zip(out_dimchars, input_specs[0].dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    elif _is_out_variant_op(op_schema.op):
        out_spec = cast(DTensorSpec, op_schema.kwargs_schema['out'])
        for (out_dimchar, mesh_dim) in zip(out_dimchars, out_spec.dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    return einop_rule(fmt, op_schema, linearity=linearity, enforce_sharding=enforce_sharding)