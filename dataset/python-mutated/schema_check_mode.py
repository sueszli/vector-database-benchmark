from collections import namedtuple
from copy import deepcopy
from itertools import combinations
import torch
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
Mutation = namedtuple('Mutation', ['op_name', 'arg_name'])
Aliasing = namedtuple('Aliasing', ['op_name', 'arg_name', 'output_number'])
SchemaArgument = torch._C._SchemaArgument
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

class SchemaCheckMode(TorchDispatchMode):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ops = []
        self.mutated = []
        self.aliasing = []

    def reset_cache(self):
        if False:
            i = 10
            return i + 15
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self):
        if False:
            for i in range(10):
                print('nop')
        print(*self.ops, sep=',')

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10

        def bitwise_equal(lhs, rhs):
            if False:
                for i in range(10):
                    print('nop')
            if lhs.is_quantized:
                return torch.equal(lhs, rhs)
            else:
                return torch.allclose(lhs, rhs, equal_nan=True)

        def has_mutated(before, after, md):
            if False:
                while True:
                    i = 10
            are_tensors = type(before) == torch.Tensor and type(after) == torch.Tensor
            if are_tensors and before.layout != torch.sparse_csr and (after.layout != torch.sparse_csr):
                return not (before.size() == after.size() and bitwise_equal(before, after) and (md[0] == after.stride()) and (md[1] == after._typed_storage()._cdata))
            return False

        def has_aliased(lhs, rhs):
            if False:
                print('Hello World!')
            try:
                return torch._C._overlaps(lhs, rhs)
            except Exception as exception:
                if str(exception).startswith('Cannot inspect value of type '):
                    return False
                else:
                    raise exception

        def standardize_name(name):
            if False:
                for i in range(10):
                    print('nop')
            return name if name != 'self' else 'input'

        def unwrap(e):
            if False:
                while True:
                    i = 10
            if isinstance(e, torch.Tensor) and (not type(e) == torch.Tensor):
                try:
                    return e.elem
                except AttributeError as t:
                    return e
            return e

        def parse_metadata(e):
            if False:
                print('Hello World!')
            if isinstance(e, torch.Tensor):
                if not type(e) == torch.Tensor:
                    try:
                        current = e.elem
                        return (deepcopy(current.stride()), current._typed_storage()._cdata)
                    except AttributeError as t:
                        return None
                elif e.layout != torch.sparse_csr:
                    return (deepcopy(e.stride()), e._typed_storage()._cdata)
            return None
        self.ops.append(func._schema.name)
        pre_arguments = normalize_function(func, args, kwargs, normalize_to_only_use_kwargs=True).kwargs
        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {name: tree_map(unwrap, c_p_args.get(name)) for name in c_p_args}
        cloned_metadata = {name: [parse_metadata(a) for a in pytree.tree_leaves(pre_arguments.get(name))] for name in pre_arguments}
        out = func(*args, **kwargs)
        arguments = {name: tree_map(unwrap, pre_arguments.get(name)) for name in pre_arguments}
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)
        schema_info = SchemaInfo(func._schema)
        schema_info.add_argument_values(pre_arguments)
        for i in range(len(func._schema.arguments)):
            arg = func._schema.arguments[i]
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                before = cloned_arguments.get(name)
                md = cloned_metadata.get(name)
                after = arguments.get(name)
                for j in range(len(tuple_out)):
                    unsafe_ops = ('aten::_unsafe_view', 'aten::unsafe_split')
                    if has_aliased(tuple_out[j], after) and func._schema.name not in unsafe_ops:
                        if not schema_info.may_contain_alias(SchemaArgument(SchemaArgType.output, j), SchemaArgument(SchemaArgType.input, i)):
                            raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')
                        else:
                            self.aliasing.append(Aliasing(func._schema.name, name, f'output_{j}'))
                    if after is tuple_out[j] and isinstance(after, torch.Tensor):
                        if not schema_info.is_mutable(SchemaArgument(SchemaArgType.input, i)) and func not in [torch.ops.aten.lift.default, torch.ops.aten.lift_fresh.default]:
                            raise RuntimeError(f'Dispatcher operators below autograd are not allowed to directly return inputs.\nHowever, we found that `outputs[{str(j)}] is {name}')
                if any((has_mutated(a, b, c) for (a, b, c) in zip(pytree.tree_leaves(before), pytree.tree_leaves(after), md))):
                    if not schema_info.is_mutable(SchemaArgument(SchemaArgType.input, i)):
                        raise RuntimeError(f'Argument {name} is not defined as mutable but was mutated')
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))
        for (i, j) in combinations(range(len(func._schema.returns)), 2):
            if has_aliased(tuple_out[i], tuple_out[j]):
                if not schema_info.may_contain_alias(SchemaArgument(SchemaArgType.output, i), SchemaArgument(SchemaArgType.output, j)):
                    raise RuntimeError(f'Outputs {i} and {j} alias unexpectedly')
        return out