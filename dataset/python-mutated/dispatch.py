import functools
import operator
from typing import cast, Dict, List, Optional, Sequence, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import _is_inplace_op, _is_out_variant_op, OpInfo, OpSchema, OutputSpecType
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed._tensor.sharding_prop import ShardingPropagator
try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree
aten = torch.ops.aten

def decompose_handler(op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> object:
    if False:
        return 10
    '\n    Decomposes a op to core ATen op, this handler is mostly here\n    for inference mode usage where the ops are not core aten ops.\n    '
    r = op_call.decompose(*args, **kwargs)
    if r is not NotImplemented:
        return r
    else:
        raise RuntimeError('Decomposition failed')

def is_same_size_handler(op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    lhs = cast(torch.Tensor, args[0])
    rhs = cast(torch.Tensor, args[1])
    return lhs.shape == rhs.shape

class OpDispatcher:
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.sharding_propagator = ShardingPropagator()
        self._random_ops = {aten.native_dropout.default, aten.normal_.default, aten.rand_like.default, aten.randn_like.default, aten.randint_like.default, aten.randint_like.low_dtype, aten.randint_like.low_dtype_out, aten.uniform_.default}
        self._custom_op_handlers = {aten.linear.default: decompose_handler, aten.is_same_size.default: is_same_size_handler}

    def dispatch(self, op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> object:
        if False:
            return 10
        '\n        Main dispatching logic\n        '
        if op_call in self._custom_op_handlers:
            return self._custom_op_handlers[op_call](op_call, args, kwargs)
        runtime_schema_info = self.sharding_propagator.op_to_schema_info.get(op_call, None)
        if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
            (tree_args, args_spec) = pytree.tree_flatten(args)
            args_list: Sequence[object] = tree_args
        else:
            (args_list, args_spec) = (args, None)
        args_schema: List[object] = []
        kwargs_schema: Dict[str, object] = {}
        local_args: List[object] = []
        local_kwargs: Dict[str, object] = {}
        mesh: Optional[DeviceMesh] = None
        for arg in args_list:
            if isinstance(arg, dtensor.DTensor):
                args_schema.append(arg._spec)
                local_args.append(arg._local_tensor)
                if mesh is not None:
                    if mesh != arg.device_mesh:
                        raise NotImplementedError(f'{op_call}: DTensor does not support cross-mesh operation yet!')
                else:
                    mesh = arg.device_mesh
            elif isinstance(arg, torch.Tensor):
                if arg.ndim == 0 and mesh is not None:
                    args_schema.append(DTensorSpec(mesh, (Replicate(),) * mesh.ndim, tensor_meta=TensorMeta(shape=arg.shape, stride=arg.stride(), dtype=arg.dtype)))
                    local_args.append(arg)
                else:
                    raise RuntimeError(f'{op_call}: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!')
            else:
                args_schema.append(arg)
                local_args.append(arg)
        for (k, v) in kwargs.items():
            if isinstance(v, dtensor.DTensor):
                kwargs_schema[k] = v._spec
                local_kwargs[k] = v._local_tensor
                if mesh is not None:
                    if mesh != v.device_mesh:
                        raise NotImplementedError(f'{op_call}: DTensor does not support cross-mesh operation yet!')
                else:
                    mesh = v.device_mesh
            elif isinstance(v, torch.Tensor):
                raise RuntimeError(f'{op_call}: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!')
            else:
                kwargs_schema[k] = v
                local_kwargs[k] = v
        assert mesh is not None, 'found no DeviceMesh from dtensor args!'
        op_info = OpInfo(mesh, OpSchema(op_call, pytree.tree_unflatten(args_schema, args_spec) if args_spec else tuple(args_schema), kwargs_schema, schema_info=runtime_schema_info), args_schema, tuple(local_args), local_kwargs, args_spec)
        self.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        assert output_sharding is not None, 'output sharding should not be None'
        if mesh.get_coordinate() is None:
            spec = output_sharding.output_spec
            ret_list = op_info.schema.op._schema.returns
            if spec is None:
                local_results: object = None
            else:

                def default_tensor(spec: DTensorSpec) -> torch.Tensor:
                    if False:
                        while True:
                            i = 10
                    if spec.tensor_meta is not None:
                        shape = spec.tensor_meta.shape
                        dtype = spec.tensor_meta.dtype
                        if len(shape) == 0:
                            return torch.zeros((), dtype=dtype)
                        else:
                            return torch.tensor([], dtype=dtype)
                    else:
                        raise RuntimeError(f'{spec} has no tensor metadata.')
                if isinstance(spec, DTensorSpec):
                    local_results = default_tensor(spec)
                elif isinstance(spec, Sequence):
                    local_results = [default_tensor(s) if s is not None else None for s in spec]
                    assert isinstance(local_results, List)
                    if None in local_results:
                        ret_type = str(ret_list[0].type)
                        raise NotImplementedError(f'return type {ret_type} in DTensor op is not supported')
        else:
            if output_sharding.needs_redistribute:
                assert output_sharding.schema_suggestions is not None
                suggested_input_schema = output_sharding.schema_suggestions[0]
                self.redistribute_local_args(op_info, suggested_input_schema)
            local_tensor_args = pytree.tree_unflatten(cast(List[object], op_info.local_args), args_spec) if args_spec else op_info.local_args
            local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
            if op_call in self._random_ops and is_rng_supported_mesh(mesh):
                if not random._rng_tracker:
                    raise RuntimeError('A CudaRNGStateTracker instance must be instantiated before executing a random op over a DTensor. Try calling random.manual_seed() or distribute_tensor() before executing a DTensor random op.')
                with random._rng_tracker._distribute_region(cast(DTensorSpec, args_schema[0])):
                    local_results = op_call(*local_tensor_args, **local_kwargs)
            else:
                local_results = op_call(*local_tensor_args, **local_kwargs)
        if output_sharding.output_spec is None:
            if op_call == aten.equal.default:
                obj_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, local_results)
                obj_list = list(filter(lambda x: x is not None, obj_list))
                local_results = functools.reduce(operator.and_, obj_list, True)
        if _is_inplace_op(op_call):
            if output_sharding.output_spec is not None:
                return args[0]
            else:
                return None
        elif _is_out_variant_op(op_call):
            output_specs = (output_sharding.output_spec,) if not isinstance(output_sharding.output_spec, tuple) else output_sharding.output_spec
            out_dts = []
            spec_idx = 0
            for argument in op_call._schema.arguments:
                if argument.is_out:
                    out_dt = cast(dtensor.DTensor, kwargs[argument.name])
                    out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                    out_dts.append(out_dt)
                    spec_idx += 1
            assert len(out_dts) >= 1, 'out variant should have at least one out arg'
            return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
        else:
            return self.wrap(local_results, output_sharding.output_spec)

    @staticmethod
    def redistribute_local_args(op_info: OpInfo, suggested_input_schema: OpSchema) -> None:
        if False:
            i = 10
            return i + 15
        if op_info.args_tree_spec is not None:
            flatten_args_schema_to_reshard = tuple(pytree.tree_leaves(suggested_input_schema.args_schema))
        else:
            flatten_args_schema_to_reshard = suggested_input_schema.args_schema
        new_local_args: List[object] = []
        for (i, arg_spec) in enumerate(op_info.flat_args_schema):
            reshard_arg_spec = flatten_args_schema_to_reshard[i]
            if isinstance(arg_spec, DTensorSpec):
                local_tensor = cast(torch.Tensor, op_info.local_args[i])
                if arg_spec != reshard_arg_spec:
                    resharded_local_tensor = redistribute_local_tensor(local_tensor, arg_spec, reshard_arg_spec)
                    new_local_args.append(resharded_local_tensor)
                else:
                    new_local_args.append(local_tensor)
            else:
                new_local_args.append(reshard_arg_spec)
        op_info.local_args = tuple(new_local_args)

    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object:
        if False:
            for i in range(10):
                print('nop')

        def to_dt(res, spec):
            if False:
                for i in range(10):
                    print('nop')
            assert spec is not None and isinstance(spec, DTensorSpec), f'output spec does not match with output! Expected DTensorSpec, got {spec}.'
            assert spec.tensor_meta is not None
            return dtensor.DTensor(res, spec.mesh, spec.placements, shape=spec.tensor_meta.shape, dtype=spec.tensor_meta.dtype, requires_grad=res.requires_grad, stride=spec.tensor_meta.stride)
        if isinstance(res, torch.Tensor):
            return to_dt(res, spec)
        elif isinstance(res, (list, tuple)):
            assert spec is not None and isinstance(spec, (list, tuple)), f'output spec does not match with output! Expected list/tuple, got {spec}.'
            res_list = []
            for (e, s) in zip(res, spec):
                if isinstance(e, (list, tuple)) and isinstance(s, (list, tuple)):
                    res_list.append(type(e)([to_dt(ee, ss) for (ee, ss) in zip(e, s)]))
                elif e is not None and s is not None:
                    res_list.append(to_dt(e, s))
                else:
                    res_list.append(None)
            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            return res