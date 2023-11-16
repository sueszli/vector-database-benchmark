import itertools
from dataclasses import dataclass
from typing import List, Tuple
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement, Replicate, Shard

@dataclass
class EinsumDims:
    contracting_dims: List[str]
    batch_dims: List[str]
    lhs_out_only_dims: List[str]
    rhs_out_only_dims: List[str]

    @classmethod
    def parse_equation(cls, equation: str) -> Tuple[List[str], str]:
        if False:
            while True:
                i = 10
        '\n        Parse the einsum equation str to input dim chars and output dim char\n        '
        (inputs, outputs) = equation.split('->')
        (input_dims, output_dims) = (inputs.split(','), outputs.split(','))
        assert len(input_dims) <= 2, 'Only support at most two inputs'
        assert len(output_dims) == 1, 'Only support single output'
        output_dim = output_dims[0]
        return (input_dims, output_dim)

    @classmethod
    def parse_dims(cls, input_dims: List[str], output_dim: str) -> 'EinsumDims':
        if False:
            return 10
        '\n        Parse the dims and extract the contracting, batch, and free dimensions\n        for the left and right hand sides.\n        '
        dim_char_set = set()
        for input_dim in input_dims:
            for input_char in list(input_dim):
                dim_char_set.add(input_char)
        all_dim_chars = sorted(dim_char_set)
        (lhs_out_only_dims, rhs_out_only_dims) = ([], [])
        (batch_dims, contracting_dims) = ([], [])
        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim
                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    assert len(input_dims) == 2, 'free dimension only supported for two inputs!'
                    (lhs, rhs) = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError('Invalid dimension character')
        return cls(contracting_dims=contracting_dims, batch_dims=batch_dims, lhs_out_only_dims=lhs_out_only_dims, rhs_out_only_dims=rhs_out_only_dims)

def gen_einsum_strategies(equation: str, mesh: DeviceMesh, *, linearity: bool=False) -> OpStrategy:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a strategy list for the ops that follow einsum style notation.\n    '
    (input_dims, output_dim) = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)
    all_mesh_dim_strategies = []
    for mesh_dim in range(mesh.ndim):
        mesh_dim_strategies = []
        placement_list: List[Placement] = [Replicate()] * (len(input_dims) + 1)
        mesh_dim_strategies.append(placement_list)
        if mesh.size(mesh_dim) <= 1:
            continue
        for batch_dim in edims.batch_dims:
            output_batch_dim = output_dim.index(batch_dim)
            placement_list = [Shard(output_batch_dim)]
            for input_dim in input_dims:
                input_batch_dim = input_dim.index(batch_dim)
                placement_list.append(Shard(input_batch_dim))
            mesh_dim_strategies.append(placement_list)
        for contracting_dim in edims.contracting_dims:
            placement_list = [_Partial()]
            for input_dim in input_dims:
                input_contracting_dim = input_dim.index(contracting_dim)
                placement_list.append(Shard(input_contracting_dim))
            mesh_dim_strategies.append(placement_list)
        for lhs_dim in edims.lhs_out_only_dims:
            lhs_free_dim = output_dim.index(lhs_dim)
            lhs_placement_list: List[Placement] = [Shard(lhs_free_dim), Shard(lhs_free_dim), Replicate()]
            mesh_dim_strategies.append(lhs_placement_list)
        for rhs_dim in edims.rhs_out_only_dims:
            rhs_free_dim = output_dim.index(rhs_dim)
            rhs_placement_list: List[Placement] = [Shard(rhs_free_dim), Replicate(), Shard(rhs_free_dim)]
            mesh_dim_strategies.append(rhs_placement_list)
        if linearity:
            linearity_placement_list: List[Placement] = [_Partial()]
            for input_dim in input_dims:
                linearity_placement_list.append(_Partial())
            mesh_dim_strategies.append(linearity_placement_list)
        all_mesh_dim_strategies.append(mesh_dim_strategies)
    strategy_combs = itertools.product(*all_mesh_dim_strategies)
    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))
        strat = PlacementStrategy(output_spec=spec_list[0], input_specs=spec_list[1:])
        all_strategies.append(strat)
    return OpStrategy(all_strategies)