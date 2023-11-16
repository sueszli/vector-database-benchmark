from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement, Replicate, Shard, TensorMeta
aten = torch.ops.aten

@register_prop_rule([aten._foreach_neg.default, aten._foreach_reciprocal.default, aten._foreach_sqrt.default])
def _prop__foreach_unaop(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    self = op_schema.args_schema[0]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self))
    return OutputSharding(output_spec=self)

@register_prop_rule([aten._foreach_add.List, aten._foreach_div.List, aten._foreach_mul.List])
def _prop__foreach_binop_list(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    (self, other) = op_schema.args_schema[:2]
    scalar = None if len(op_schema.args_schema) < 3 else op_schema.args_schema[2]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self)), f'Expect a List[DTensorSpec] but got {self}'
    assert isinstance(other, list) and all((isinstance(o, DTensorSpec) for o in other)), f'Expect a List[DTensorSpec] but got {other}'
    assert len(self) == len(other), f'Two tensor lists must match in length, but got {len(self)} and {len(other)}'
    if any((s != o for (s, o) in zip(self, other))):
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(self, self, scalar) if scalar else (self, self), kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=self)

@register_prop_rule([aten._foreach_add.Scalar, aten._foreach_div.Scalar, aten._foreach_mul.Scalar, aten._foreach_sub.Scalar])
def _prop__foreach_binop_scalar(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    (self, scalar) = op_schema.args_schema
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self))
    assert not isinstance(scalar, list)
    return OutputSharding(output_spec=self)

@register_prop_rule([aten._foreach_addcdiv.Scalar, aten._foreach_addcmul.Scalar])
def _prop__foreach_addcop_scalar(op_schema: OpSchema):
    if False:
        for i in range(10):
            print('nop')
    (self, tensor1, tensor2) = op_schema.args_schema[:3]
    scalar = None if len(op_schema.args_schema) < 4 else op_schema.args_schema[3]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self))
    assert isinstance(tensor1, list) and all((isinstance(s, DTensorSpec) for s in self))
    assert isinstance(tensor2, list) and all((isinstance(s, DTensorSpec) for s in self))
    if any((s != t1 or s != t2 for (s, t1, t2) in zip(self, tensor1, tensor2))):
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(self, self, self, scalar) if scalar else (self, self, self), kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=self)

@register_prop_rule([aten._foreach_pow.ScalarAndTensor])
def _prop__foreach_pow_scalar_and_tensor(op_schema: OpSchema):
    if False:
        for i in range(10):
            print('nop')
    (scala, exponent) = op_schema.args_schema
    assert isinstance(exponent, list) and all((isinstance(s, DTensorSpec) for s in exponent))
    return OutputSharding(output_spec=exponent)

@register_prop_rule([aten._fused_adam.default])
def _prop__fused_adam(op_schema: OpSchema):
    if False:
        while True:
            i = 10
    NT = 5
    tesnor_list_args: Tuple[List[DTensorSpec]] = op_schema.args_schema[:NT]
    assert all((isinstance(schema, list) for schema in tesnor_list_args))
    assert all((isinstance(s, DTensorSpec) for schema in tesnor_list_args for s in schema))
    tensor_schemas: Tuple[List[DTensorSpec]] = [schema for schema in tesnor_list_args if len(schema)]
    assert all((len(s) == len(tensor_schemas[0]) for s in tensor_schemas)), f'expect the same number of gradients and states, but got {[len(s) for s in tensor_schemas]}.'
    if any((any((t != ts[0] for t in ts)) for ts in zip(*tensor_schemas))):
        new_schemas: Tuple[List[DTensorSpec]] = tuple((op_schema.args_schema[0] if len(s) else s for s in tesnor_list_args))
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=new_schemas + op_schema.args_schema[NT:], kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=(op_schema.args_schema[0],) * NT)

@register_prop_rule(aten.nll_loss_forward.default)
def _prop_nll_loss_forward(op_schema: OpSchema) -> OutputSharding:
    if False:
        print('Hello World!')
    (self, target) = op_schema.args_schema[:2]
    assert isinstance(self, DTensorSpec)
    assert isinstance(target, DTensorSpec)
    if self.placements != target.placements:
        new_self = DTensorSpec(mesh=self.mesh, placements=target.placements, tensor_meta=self.tensor_meta)
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(new_self, target) + op_schema.args_schema[2:], kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=(DTensorSpec(mesh=self.mesh, placements=(_Partial(),)), DTensorSpec(mesh=self.mesh, placements=(Replicate(),))))

@register_prop_rule(aten.nll_loss_backward.default)
def _prop_nll_loss_backward(op_schema: OpSchema) -> OutputSharding:
    if False:
        print('Hello World!')
    (grad_output, self) = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(self, DTensorSpec)
    return OutputSharding(output_spec=self)

@register_prop_rule(aten.stack.default)
def _prop_stack(op_schema: OpSchema) -> OutputSharding:
    if False:
        return 10
    tensors = op_schema.args_schema[0]
    dim = 0 if len(op_schema.args_schema) == 1 else cast(int, op_schema.args_schema[1])
    assert isinstance(tensors, list) and len(tensors) > 0, 'expect at least one tensor to stack'
    assert all((isinstance(t, DTensorSpec) for t in tensors)), f'expect a list of DTensorSpecs, but got {tensors}'
    assert all((t.shape == tensors[0].shape for t in tensors)), f'expect all tensors to have the same shape, but got {tensors}.'
    assert all((t.placements == tensors[0].placements for t in tensors)), f'expect all tensors to have the same placements, but got {tensors}.'
    assert all((not p.is_shard(dim) for p in tensors[0].placements)), 'DTensor does not support stack on sharded dimension.'
    return OutputSharding(output_spec=DTensorSpec(mesh=tensors[0].mesh, placements=tensors[0].placements))

@register_prop_rule(aten.select.int)
def _prop_select(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    (tensor, dim) = op_schema.args_schema[:2]
    assert isinstance(tensor, DTensorSpec)
    assert isinstance(dim, int)
    placements: Sequence[Placement] = tensor.placements
    assert all((not p.is_shard(dim) for p in placements)), 'DTensor does not support select on sharded dimension.'
    new_placements: List[Placement] = []
    for p in placements:
        if isinstance(p, Shard) and p.dim > dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return OutputSharding(output_spec=DTensorSpec(mesh=tensor.mesh, placements=tuple(new_placements)))

@register_prop_rule(aten.native_layer_norm.default)
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    (input, normalized_shape, weight, bias, eps) = op_schema.args_schema
    assert isinstance(input, DTensorSpec)
    assert isinstance(normalized_shape, (tuple, list))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all((isinstance(p, Replicate) for p in weight.placements))
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all((isinstance(p, Replicate) for p in bias.placements))
    batch_ndim = len(input.shape) - len(normalized_shape)
    assert all((isinstance(p, Replicate) or (isinstance(p, Shard) and p.dim < batch_ndim,) for p in input.placements))
    stats_spec = DTensorSpec(mesh=input.mesh, placements=input.placements)
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))

@register_prop_rule(aten.native_layer_norm_backward.default)
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    if False:
        for i in range(10):
            print('nop')
    (grad, input, normalized_shape, result1, result2, weight, bias, grad_input_mask) = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all((isinstance(s, Replicate) for s in weight.placements))
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all((isinstance(s, Replicate) for s in bias.placements))
    assert any((isinstance(s, Shard) and s.dim == 0 for s in grad.placements)), f'Got {grad.placements}'
    weight_grad = DTensorSpec(mesh=weight.mesh, placements=tuple([_Partial()] * weight.mesh.ndim)) if weight else None
    bias_grad = DTensorSpec(mesh=bias.mesh, placements=tuple([_Partial()] * bias.mesh.ndim)) if bias else None
    return OutputSharding(output_spec=(grad if grad_input_mask[0] else None, weight_grad if grad_input_mask[1] else None, bias_grad if grad_input_mask[2] else None))

def _refine_sharding(op_schema: OpSchema, active_dim: Optional[int]) -> Sequence[Placement]:
    if False:
        print('Hello World!')
    'Considers 2 first inputs of op_schema as having same shape, and returns suggested placement for a pointwise operation.'
    args_schema = []
    for s in op_schema.args_schema[:2]:
        assert isinstance(s, DTensorSpec) and s.tensor_meta is not None
        args_schema.append(DTensorSpec(mesh=s.mesh, placements=s.placements, tensor_meta=TensorMeta(shape=torch.Size(s.shape[0:active_dim] + (1,) + s.shape[active_dim + 1:]) if active_dim is not None else s.shape, stride=s.tensor_meta.stride, dtype=s.tensor_meta.dtype)))
    op_schema = OpSchema(op=op_schema.op, args_schema=args_schema, kwargs_schema={})
    output_sharding = pointwise_rule(op_schema, linearity=False)
    if output_sharding.output_spec:
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        return output_sharding.output_spec.placements
    else:
        assert output_sharding.schema_suggestions is not None
        out_schema = output_sharding.schema_suggestions[0].args_schema[0]
        assert isinstance(out_schema, DTensorSpec)
        return tuple(out_schema.placements)

@register_prop_rule(aten.slice_scatter.default)
def prop_slice_scatter(op_schema: OpSchema) -> OutputSharding:
    if False:
        print('Hello World!')
    defaults = (None, None, 0, None, None, 1)
    (input, src, dim, start, end, step) = op_schema.args_schema + defaults[len(op_schema.args_schema):]
    assert isinstance(input, DTensorSpec)
    assert isinstance(src, DTensorSpec)
    assert isinstance(dim, int)
    if dim < 0:
        dim += input.ndim
    if input.shape[dim] == src.shape[dim]:
        assert start == 0
        assert end >= src.shape[dim]
        dim = None
    input_suggestion = list(_refine_sharding(op_schema, dim))
    for (i, p) in enumerate(input_suggestion):
        if isinstance(p, Shard) and p.dim == dim:
            input_suggestion[i] = Replicate()
    input_suggestion = tuple(input_suggestion)
    if input_suggestion == tuple(input.placements) and src.placements == tuple(input.placements):
        return OutputSharding(output_spec=DTensorSpec(mesh=input.mesh, placements=input.placements))
    else:
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(DTensorSpec(mesh=input.mesh, placements=input_suggestion, tensor_meta=input.tensor_meta), DTensorSpec(mesh=src.mesh, placements=input_suggestion, tensor_meta=src.tensor_meta)) + op_schema.args_schema[2:], kwargs_schema=op_schema.kwargs_schema)])