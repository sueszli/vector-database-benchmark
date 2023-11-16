import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, Transpose
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List
_TRANSFORMATION_RULES: Dict[Constraint, Callable] = {}

def register_transformation_rule(call_target):
    if False:
        for i in range(10):
            print('nop')

    def register(fn):
        if False:
            i = 10
            return i + 15
        if call_target in _TRANSFORMATION_RULES:
            raise RuntimeError(f'Transformation rule already registered for {call_target}!')
        _TRANSFORMATION_RULES[call_target] = fn
        return fn
    return register

def valid_index(index, dims):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a list of dimensions, checks if an index is valid in the list\n    '
    try:
        dims[index]
        return T()
    except IndexError:
        return F()

@register_transformation_rule(Transpose)
def transform_transpose(constraint, counter):
    if False:
        while True:
            i = 10
    '\n    Similar to a sequence of two index-selects\n    '
    (dims, counter) = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index1 = valid_index(constraint.index1, dims)
    is_valid_index2 = valid_index(constraint.index2, dims)
    new_dims = copy.deepcopy(dims)
    nat_constraints = gen_nat_constraints(dims)
    if is_valid_index1 == T() and is_valid_index2 == T():
        new_dims[constraint.index1] = dims[constraint.index2]
        new_dims[constraint.index2] = dims[constraint.index1]
    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq), *nat_constraints, is_valid_index1, is_valid_index2, BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])
    return (transformed_constraint, counter)

@register_transformation_rule(IndexSelect)
def transform_index_select(constraint, counter):
    if False:
        for i in range(10):
            print('nop')
    '\n    The constraints consider the given tensor size, checks if the index is valid\n    and if so, generates a constraint for replacing the input dimension\n    with the required dimension\n    '
    (dims, counter) = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index = valid_index(constraint.index, dims)
    nat_constraints = gen_nat_constraints(dims)
    if is_valid_index == T():
        new_dims = copy.deepcopy(dims)
        new_dims[constraint.index] = constraint.dim_replace
    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq), *nat_constraints, is_valid_index, BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])
    return (transformed_constraint, counter)

@register_transformation_rule(GetItem)
def transform_get_item(constraint, counter):
    if False:
        for i in range(10):
            print('nop')
    '\n    generate an equality of the form:\n    t = [a1, ..., an]\n    then generate constraints that check if the given index is valid\n    given this particular tensor size.\n    If the index is valid, generate a constraint to get the item\n    Note that we already handled the Dyn input case in the previous\n    step.\n    Args:\n        constraint: GetItem which assumes we are getting an item from a tensor (not Dyn)\n        counter: variable tracking\n    Returns: simplified constraints for GetItem\n\n    '
    (dims, counter) = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)
    is_valid_index = valid_index(constraint.index, dims)
    all_constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq), *nat_constraints, is_valid_index]
    if is_valid_index == T():
        all_constraints.append(BinConstraintD(constraint.res, dims[constraint.index], op_eq))
    return (Conj(all_constraints), counter)

def valid_index_tensor(index, dims):
    if False:
        return 10
    '\n    if the slice instances exceed the length of the dimensions\n    then this is a type error so we return False\n    '
    slice_count = 0
    for s in index:
        if isinstance(s, slice):
            slice_count += 1
    if slice_count > len(dims):
        return F()
    else:
        return T()

@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(constraint, counter):
    if False:
        for i in range(10):
            print('nop')
    "\n    When the index is a tuple, then the output will be a tensor\n    TODO: we have to check if this is the case for all HF models\n\n    The cases we are covering here are a tuple with one of:\n     - slice with default argument\n     - None\n\n     None appends 1 to the input tensor dimensions\n     so each occurrence of 'None' increases the rank by 1\n\n     slice with default arguments does not change the rank\n    "
    assert isinstance(constraint.index_tuple, tuple)
    (dims, counter) = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)
    none_c = constraint.index_tuple.count(None)
    resulting_tensor_dims = (none_c + len(dims)) * [None]
    dim_index = 0
    for i in range(len(constraint.index_tuple)):
        if constraint.index_tuple[i] is None:
            resulting_tensor_dims[i] = 1
        elif constraint.index_tuple[i] == slice(None, None, None):
            pass
        else:
            raise NotImplementedError('Method not yet implemented')
    dim_index = 0
    for i in range(len(resulting_tensor_dims)):
        if resulting_tensor_dims[i] is None:
            resulting_tensor_dims[i] = dims[dim_index]
            dim_index += 1
    is_valid_index = valid_index_tensor(constraint.index_tuple, dims)
    if len(resulting_tensor_dims) > 4:
        return (F(), counter)
    else:
        constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq), BinConstraintT(constraint.res, TensorType(resulting_tensor_dims), op_eq), *nat_constraints, is_valid_index]
        return (Conj(constraints), counter)

@register_transformation_rule(BinConstraintT)
def generate_binconstraint_t(constraint, counter):
    if False:
        return 10
    '\n    Transform binary constraints for tensors\n    '
    if constraint.op == op_precision:
        if constraint.lhs == Dyn:
            return (T(), counter)
        elif isinstance(constraint.lhs, TensorType):
            is_fully_static = all((d != Dyn for d in constraint.lhs.__args__))
            if is_fully_static:
                return (BinConstraintT(constraint.lhs, constraint.rhs, op_eq), counter)
            else:
                new_dims = []
                for _ in range(len(constraint.lhs.__args__)):
                    (dim, counter) = gen_dvar(counter)
                    new_dims.append(dim)
                new_dim_constraints = [BinConstraintD(old_dim, new_dim, op_precision) for (new_dim, old_dim) in zip(new_dims, constraint.lhs.__args__)] + [BinConstraintT(constraint.rhs, TensorType(new_dims), op_eq)] + [BinConstraintD(1, new_dim, op_leq) for new_dim in new_dims]
                return (Conj(new_dim_constraints), counter)
    elif constraint.op == op_matching:
        assert isinstance(constraint.rhs, TensorType)
        d1 = constraint.rhs.__args__[0]
        d2 = constraint.rhs.__args__[1]
        d3 = constraint.rhs.__args__[2]
        d4 = constraint.rhs.__args__[3]
        conj = [BinConstraintT(constraint.lhs, Dyn, op_eq), BinConstraintD(d1, Dyn, op_eq), BinConstraintD(d2, Dyn, op_eq), BinConstraintD(d3, Dyn, op_eq), BinConstraintD(d4, Dyn, op_eq)]
        return (Disj([Conj(conj), BinConstraintT(constraint.lhs, TensorType([d1, d2, d3, d4]), op_eq)]), counter)
    elif constraint.op == op_consistency:
        c_dyn = Disj([BinConstraintT(constraint.lhs, Dyn, op_eq), BinConstraintT(constraint.rhs, Dyn, op_eq)])
        ([c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4], counter) = gen_consistency_constraints(constraint, counter)
        return (Disj([c_dyn, c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4]), counter)
    elif constraint.op == op_leq:
        assert isinstance(constraint.rhs, int)
        disj = [BinConstraintT(constraint.lhs, Dyn, op_eq)]
        for i in range(1, constraint.rhs + 1):
            dims = []
            for j in range(1, i + 1):
                (dim_var, counter) = gen_dvar(counter)
                dims.append(dim_var)
            disj.append(BinConstraintT(constraint.lhs, TensorType(dims), op_eq))
        return (Disj(disj), counter)
    else:
        return (constraint, counter)

@register_transformation_rule(BinConstraintD)
def generate_binconstraint_d(constraint, counter):
    if False:
        while True:
            i = 10
    '\n    Transform binary constraints for dimensions\n    '
    if constraint.op == op_precision:
        if isinstance(constraint.lhs, int):
            return (BinConstraintD(constraint.lhs, constraint.rhs, op_eq), counter)
        elif constraint.lhs == Dyn:
            return (T(), counter)
    elif constraint.op == op_consistency:
        return (Disj([BinConstraintD(constraint.lhs, constraint.rhs, op_eq), BinConstraintD(constraint.rhs, Dyn, op_eq), BinConstraintD(constraint.lhs, Dyn, op_eq)]), counter)
    else:
        return (constraint, counter)

@register_transformation_rule(Conj)
def generate_conj(constraint, counter):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform conjunctions\n    '
    new = []
    for c in constraint.conjucts:
        (new_c, counter) = transform_constraint(c, counter)
        new.append(new_c)
    return (Conj(new), counter)

@register_transformation_rule(Disj)
def generate_disj(constraint, counter):
    if False:
        print('Hello World!')
    '\n    Transform disjunctions\n    '
    new = []
    for c in constraint.disjuncts:
        (new_c, counter) = transform_constraint(c, counter)
        new.append(new_c)
    return (Disj(new), counter)

@register_transformation_rule(TGreatestUpperBound)
def generate_gub(constraint, counter):
    if False:
        while True:
            i = 10
    '\n    Transform greatest upper bound for tensors. Results in equality and Greatest Upper Bound\n    on dimensions\n    '
    c1 = Conj([Disj([BinConstraintT(constraint.rhs1, Dyn, op_eq), BinConstraintT(constraint.rhs2, Dyn, op_eq)]), BinConstraintT(constraint.res, Dyn, op_eq)])
    ([c2, c3, c4, c5], counter) = gen_greatest_upper_bound(constraint, counter)
    return (Disj([c1, c2, c3, c4, c5]), counter)

@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint, counter):
    if False:
        i = 10
        return i + 15
    '\n    Transform greatest upper bound for dimensions into equality constraints\n    '
    c1 = Conj([BinConstraintD(constraint.rhs1, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs2, op_eq)])
    c2 = Conj([BinConstraintD(constraint.rhs2, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    c3 = Conj([BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    return (Disj([c1, c2, c3]), counter)

@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint, counter):
    if False:
        return 10
    (d, counter) = gen_tensor_dims(4, counter)
    conv_result = TensorType([d[0], d[1], d[2], d[3]])
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)
    c2 = Conj([BinConstraintD(d[1], constraint.c_out, op_eq), BinConstraintD(d[1], Dyn, op_neq)])
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    (c4, c5) = calc_last_two_dims(constraint, d)
    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq), BinConstraintD(0, d[1], op_leq), BinConstraintD(0, d[2], op_leq), BinConstraintD(0, d[3], op_leq)])
    return (Conj([c1, c2, c3, c4, c5, leq_constraints]), counter)

@register_transformation_rule(CalcMaxPool)
def generate_calc_maxpool(constraint, counter):
    if False:
        print('Hello World!')
    '\n    Transform maxpool constraints\n    '
    (d, counter) = gen_tensor_dims(4, counter)
    maxpool_result = TensorType([d[0], d[1], d[2], d[3]])
    c1 = BinConstraintT(constraint.maxpool_result, maxpool_result, op_eq)
    c2 = BinConstraintD(constraint.matching_constraint[1], d[1], op_eq)
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    (c4, c5) = calc_last_two_dims(constraint, d)
    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq), BinConstraintD(0, d[1], op_leq), BinConstraintD(0, d[2], op_leq), BinConstraintD(0, d[3], op_leq)])
    return (Conj([c1, c2, c3, c4, c5, leq_constraints]), counter)

@register_transformation_rule(CalcProduct)
def generate_calc_product(constraint, counter):
    if False:
        while True:
            i = 10
    '\n    Transform flatten constraints\n    '
    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)
    boundary_check = 0 <= start and start < end and (end <= n)
    c_boundary = T() if boundary_check else F()
    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]
    all_possibilities = generate_all_int_dyn_dim_possibilities(mid)
    all_constraints = []
    for p in all_possibilities:
        p = list(p)
        contains_dyn = not all((constraint.op == op_neq for constraint in p))
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq)] + p))
        else:
            (new_var, counter) = gen_dvar(counter)
            mid_eq_prod = Conj([BinConstraintD(new_var, Prod(mid), op_eq), BinConstraintD(new_var, Dyn, op_neq)])
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq), mid_eq_prod] + p))
    return (Conj([Disj(all_constraints), c_boundary]), counter)

@register_transformation_rule(CanReshape)
def generate_reshape(constraint, counter):
    if False:
        return 10
    '\n    Transform reshape constraints\n    '
    (d, counter) = gen_tensor_dims(4, counter)
    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]
    target = constraint.target.__args__
    is_fully_static = all((d != Dyn for d in target))
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)
    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)
    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)
    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)
    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)
    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)
    if is_fully_static:
        c3_tensor1 = Disj([d1_eq_dyn, Conj([d1_neq_dyn, BinConstraintD(d1, Prod(target), op_eq)])])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])
        all_tensor_2 = Conj([c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)])
        all_tensor_3 = Conj([c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)])
        all_tensor_4 = Conj([c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)])
        return (Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]), nat_d1, nat_d2, nat_d3, nat_d4]), counter)
    else:
        new_target = []
        for n in target:
            if n != Dyn:
                new_target.append(n)
        c3_tensor1 = Disj([d1_eq_dyn, Conj([d1_neq_dyn, is_dim_div_by_target(new_target, d1)])])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj([d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))])
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3]))])
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, d4_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4]))])
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])
        return (Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]), nat_d1, nat_d2, nat_d3, nat_d4]), counter)

@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(constraint, counter):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform broadcasting constraints\n    '
    (e11, e12) = (constraint.res1, constraint.res2)
    (e1, e2) = (constraint.input1, constraint.input2)
    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)
    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])
    (final_tensor_1_constraint, _, _, nat_dims_1, counter) = gen_broadcasting_constraints(e1, e2, e11, e12, 1, counter)
    (final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, final_tensor_2_constraint_padding_arg2, nat_dims_2, counter) = gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)
    (final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, final_tensor_3_constraint_padding_arg2, nat_dims_3, counter) = gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)
    (final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, final_tensor_4_constraint_padding_arg2, nat_dims_4, counter) = gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)
    final_result = Disj([e1_dyn_constraint, e2_dyn_constraint, final_tensor_1_constraint, final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, final_tensor_2_constraint_padding_arg2, final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, final_tensor_3_constraint_padding_arg2, final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, final_tensor_4_constraint_padding_arg2])
    return (Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]), counter)

def transform_constraint(constraint: Constraint, counter: int):
    if False:
        print('Hello World!')
    '\n    Transforms a constraint into a simpler constraint.\n    Ex: precision and consistency are transformed to equality\n    Args:\n        constraint: constraint to be transformed\n        counter: for variable tracking\n\n    Returns: Constraint\n\n    '
    if type(constraint) in _TRANSFORMATION_RULES:
        return _TRANSFORMATION_RULES[type(constraint)](constraint, counter)
    else:
        return (constraint, counter)

def calc_last_two_dims(constraint, d: List[DVar]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates constraints for the last two dimensions of a convolution or a maxpool output\n    Args:\n        constraint: CalcConv or CalcMaxPool\n        d: The list of output dimensions\n\n    Returns: Constraints for calculating the last two dimensions of the output\n\n    '
    assert isinstance(constraint, (CalcConv, CalcMaxPool))
    b3 = constraint.matching_constraint[2]
    b4 = constraint.matching_constraint[3]
    b3_dyn = Conj([BinConstraintD(d[2], Dyn, op_eq), BinConstraintD(b3, Dyn, op_eq)])
    b4_dyn = Conj([BinConstraintD(d[3], Dyn, op_eq), BinConstraintD(b4, Dyn, op_eq)])
    d3_not_dyn = Conj([BinConstraintD(d[2], Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq)])
    d4_not_dyn = Conj([BinConstraintD(d[3], Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq)])
    padding = (constraint.padding, constraint.padding) if isinstance(constraint.padding, int) else constraint.padding
    kernel = (constraint.kernel, constraint.kernel) if isinstance(constraint.kernel, int) else constraint.kernel
    stride = (constraint.stride, constraint.stride) if isinstance(constraint.stride, int) else constraint.stride
    dilation = (constraint.dilation, constraint.dilation) if isinstance(constraint.dilation, int) else constraint.dilation
    f1 = BinConstraintD(b3, BinConstraintD(2, padding[0], op_mul), op_add)
    f2 = BinConstraintD(dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul)
    f3 = BinConstraintD(BinConstraintD(BinConstraintD(f1, f2, op_sub), 1, op_sub), stride[0], op_div)
    f4 = BinConstraintD(f3, 1, op_add)
    c4 = Disj([b3_dyn, Conj([d3_not_dyn, BinConstraintD(d[2], f4, op_eq)])])
    f11 = BinConstraintD(b4, BinConstraintD(2, padding[1], op_mul), op_add)
    f22 = BinConstraintD(dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul)
    f33 = BinConstraintD(BinConstraintD(BinConstraintD(f11, f22, op_sub), 1, op_sub), stride[1], op_div)
    f44 = BinConstraintD(f33, 1, op_add)
    c5 = Disj([b4_dyn, Conj([d4_not_dyn, BinConstraintD(d[3], f44, op_eq)])])
    return (c4, c5)

def generate_all_int_dyn_dim_possibilities(my_list: List[DVar]):
    if False:
        print('Hello World!')
    '\n    Generate all possibilities of being equal or not equal to dyn for my_list\n    Args:\n        my_list: List of tensor dimensions\n\n    Returns: A list of a list of constraints. Each list of constraints corresponds to\n    one possibility about the values of the dimension variables\n    '
    eq_possibilities = [BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))]
    neq_possibilities = [BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))]
    d_possibilities = []
    for i in zip(eq_possibilities, neq_possibilities):
        d_possibilities.append(list(i))
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities

def is_target_div_by_dim(target: List[int], dim: List[DVar]):
    if False:
        return 10
    '\n    Generate constraints to check if the target dimensions are divisible by the input dimensions\n    Args:\n        target: Target dimensions\n        dim: Input dimensions\n\n    Returns: Constraints to check divisibility\n\n    '
    return BinConstraintD(BinConstraintD(Prod(target), dim, op_mod), 0, op_eq)

def is_dim_div_by_target(target: List[int], dim: List[DVar]):
    if False:
        while True:
            i = 10
    '\n    Generate constraints to check if the input dimensions is divisible by the target dimensions\n    Args:\n        target: Target dimensions\n        dim:  Input dimensions\n\n    Returns: Constraints to check divisibility\n\n    '
    return BinConstraintD(BinConstraintD(dim, Prod(target), op_mod), 0, op_eq)

def gen_all_reshape_possibilities(list_of_dims, target):
    if False:
        i = 10
        return i + 15
    '\n    Consider all possibilities what the input dimensions could be (number or dynamic)\n    Then generate the appropriate constraints using multiplication or mod depending on the possibility\n    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn\n    for the input. Target is fixed because at most one dimension could be dyn.\n    We have different cases for this.\n\n    Args:\n        list_of_dims: The input list of dimensions\n        target: The tensor we want to reshape to\n\n    Returns: A disjunction of transformed reshape constraints\n\n    '
    all_possibilities = generate_all_int_dyn_dim_possibilities(list_of_dims)
    all_constraints = []
    for p in all_possibilities:
        to_multiply = []
        p = list(p)
        for constraint in p:
            assert isinstance(constraint, BinConstraintD)
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)
        if not to_multiply:
            all_constraints.append(Conj(p))
        elif len(to_multiply) < len(list_of_dims):
            all_constraints.append(Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))]))
        else:
            all_constraints.append(Conj(p + [BinConstraintD(Prod(list_of_dims), Prod(target), op_eq)]))
    return Disj(all_constraints)

def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding=False):
    if False:
        return 10
    "\n    Apply broadcasting to the 'index' dimension of tensor_input1.\n    Args:\n        tensor_input1: should represent [d1, ..., d_index, ...] where d_index = 1\n        tensor_input2: represents the second input\n        res1: broadcasted result 1\n        res2: broadcasted result 2\n        index: the index to broadcast\n        padding: If padding was used, then tensor_input1[index] does not exist\n\n    Returns:\n\n    "
    if tensor_input1[index] is None:
        assert padding
    if not padding:
        return Conj([BinConstraintD(tensor_input1[index], 1, op_eq), BinConstraintD(res1[index], res2[index], op_eq), BinConstraintD(res2[index], tensor_input2[index], op_eq)])
    else:
        return Conj([BinConstraintD(res1[index], res2[index], op_eq), BinConstraintD(res2[index], tensor_input2[index], op_eq)])

def apply_padding(e1_var: TVar, e11: BinConstraintT, e2: BinConstraintT, e12: BinConstraintT, d2: List[DVar], d11: List[DVar], d12: List[DVar], counter: int):
    if False:
        i = 10
        return i + 15
    '\n    We are considering the possibility where one input has less dimensions than\n    another input, so we apply padding to the broadcasted results\n\n    Args:\n        e1_var: Variable representing the first input where padding will be\n        e11: constraint of the form e11 = Tensortype[d1, ..., dn]\n        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]\n        e12: constraint of the form e11 = Tensortype[d1, ..., dn]\n        d2: Tensor variables for the second input\n        d11: Tensor variables for the broadcasted first input\n        d12: Tensor variables for the broadcasted second input\n        counter: variable tracking\n\n    Returns: A new constraint whose goal is to apply padding to the broadcasted result\n\n    '
    res = []
    for i in range(1, len(d2)):
        (d1, counter) = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(d1 + d2 + d11 + d12)
        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)
        simulate_padding = [None] * (len(d2) - i)
        assert len(simulate_padding + d1) == len(d2)
        broadcast_padding = []
        for j in range(len(d2) - i):
            broadcast_padding.append(broadcast_dim(simulate_padding, d2, d11, d12, j, True))
        all_broadcasting_possibilities = generate_all_broadcasting_possibilities_no_padding(d1, d2[len(d2) - i:], d11[len(d2) - i:], d12[len(d2) - i:])
        c = Conj([e1, e11, e2, e12, *broadcast_padding, all_broadcasting_possibilities, *nat_constraints])
        res.append(c)
    return (Disj(res), counter)

def no_broadcast_dim_with_index(d1: List[DVar], d2: List[DVar], d3: List[DVar], d4: List[DVar], i: int):
    if False:
        while True:
            i = 10
    '\n    Args:\n        d1: input 1\n        d2: input 2\n        d3: simulated broadcasting for input 1\n        d4: simulated broadcasting for input 2\n        i: the rank of the resulting tensor addition\n\n    Returns: Constraints for when no broadcasting occurs\n    '
    return Conj([Disj([Conj([BinConstraintD(d1[i], 1, op_eq), BinConstraintD(d2[i], 1, op_eq)]), Conj([BinConstraintD(d1[i], 1, op_neq), BinConstraintD(d2[i], 1, op_neq)])]), BinConstraintD(d1[i], d3[i], op_eq), BinConstraintD(d2[i], d4[i], op_eq)])

def gen_lists_of_dims(num_tensors: int, dim_size: int, counter: int):
    if False:
        return 10
    '\n    Generate lists of DVar to represent tensor dimensions\n    Args:\n        num_tensors: the required number of tensors\n        dim_size: the number of dimensions for each tensor\n        counter: variable tracking\n\n    Returns: A list of a list of tensor dimensions\n\n    '
    res = []
    for _ in range(num_tensors):
        (dims, counter) = gen_tensor_dims(dim_size, counter)
        res.append(dims)
    return (res, counter)

def create_equality_constraints_for_broadcasting(e1: TVar, e2: TVar, e11: TVar, e12: TVar, d1: List[DVar], d2: List[DVar], d11: List[DVar], d12: List[DVar]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create equality constraints for when no broadcasting occurs\n    Args:\n        e1: Input 1\n        e2: Input 2\n        e11: Broadcasted input 1\n        e12: Broadcasted input 2\n        d1: Variables that store dimensions for e1\n        d2: Variables that store dimensions for e2\n        d11: Variables that store dimensions for e11\n        d12: Variables that store dimensions for e22\n\n    Returns: Four equality constraints\n\n    '
    e1_tensor = BinConstraintT(e1, TensorType(d1), op_eq)
    e11_tensor = BinConstraintT(e11, TensorType(d11), op_eq)
    e2_tensor = BinConstraintT(e2, TensorType(d2), op_eq)
    e12_tensor = BinConstraintT(e12, TensorType(d12), op_eq)
    return [e1_tensor, e11_tensor, e2_tensor, e12_tensor]

def gen_consistency_constraints(constraint: Constraint, counter: int):
    if False:
        for i in range(10):
            print('nop')
    '\n    Args:\n        constraint: Consistency constraint on tensors\n        counter: for variable tracking\n\n    Returns: Equality and consistency constraints on dimensions\n\n    '
    all_constraints = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        (new_dims_rhs_1, counter) = gen_tensor_dims(i, counter)
        (new_dims_rhs_2, counter) = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)
        c_tensor_i = Conj([BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq), BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq)] + [BinConstraintD(d1, d2, op_consistency) for (d1, d2) in zip(new_dims_rhs_1, new_dims_rhs_2)] + nat_constraints)
        all_constraints.append(c_tensor_i)
    return (all_constraints, counter)

def gen_greatest_upper_bound(constraint: TGreatestUpperBound, counter: int):
    if False:
        return 10
    '\n    Args:\n        constraint: Greatest upper bound on tensors\n        counter: variable tracking\n\n    Returns: A set of equality constraints and DGreatestUpperBound constraints\n\n    '
    all_constraints = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c = []
        (dims1, counter) = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)
        (dims2, counter) = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)
        (dims3, counter) = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)
        c += [BinConstraintT(constraint.rhs1, c1tensor, op_eq), BinConstraintT(constraint.rhs2, c2tensor, op_eq), BinConstraintT(constraint.res, c3tensor, op_eq)] + gen_nat_constraints(dims1 + dims2 + dims3)
        assert len(c3tensor.__args__) == len(c1tensor.__args__) == len(c2tensor.__args__)
        for i in range(len(c3tensor.__args__)):
            c.append(DGreatestUpperBound(c3tensor.__args__[i], c1tensor.__args__[i], c2tensor.__args__[i]))
        all_constraints.append(Conj(c))
    return (all_constraints, counter)

def generate_all_broadcasting_possibilities_no_padding(d1: List[DVar], d2: List[DVar], d11: List[DVar], d12: List[DVar]):
    if False:
        return 10
    '\n    Generate broadcasting constraints assuming no padding. Broadcasting can happen at any dimension.\n    We look at all combinations for all dimensions in d1 and d2\n    Args:\n        d1: input1 dimensions\n        d2: input2 dimensions\n        d11: broadcasted input1 dimensions\n        d12: broadcasted input2 dimensions\n\n    Returns: broadcasting constraints relating the input dimensions to the broadcasted dimensions\n\n    '
    size = len(d1)
    res2 = []
    for i in range(size):
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)
        res2.append(Disj([t1, t2, t3]))
    return Conj(res2)

def gen_broadcasting_constraints(e1: TVar, e2: TVar, e11: TVar, e12: TVar, i: int, counter: int):
    if False:
        while True:
            i = 10
    '\n    Simulates broadcasting on e1 and e2 and returns the results\n    respectively in e11 and e12. Because of gradual types,\n    e1 and e2 may not be equal. Similarly, e11 and e12 may not\n    be equal. e11 and e12 should be guaranteed to be consistent\n    as they represent the shapes of the tensors to be added after\n    broadcasting.\n    Args:\n        e1: TVar representing the type of input 1\n        e2: TVar representing the type of input 2\n        e11: TVar representing the representing broadcasted input 1\n        e12: TVar representing the representing broadcasted input 2\n        i: The rank of the resulting type of addition\n        counter: for variable tracking\n\n    Returns: Simplified broadcasting constraints\n\n    '
    (dims, counter) = gen_lists_of_dims(4, i, counter)
    [d1, d2, d3, d4] = dims
    nat_dims_i = gen_nat_constraints(list(itertools.chain(*dims)))
    initialize_tensors_constraints = create_equality_constraints_for_broadcasting(e1, e2, e11, e12, d1, d2, d3, d4)
    [e1_tensor, e11_tensor, e2_tensor, e12_tensor] = initialize_tensors_constraints
    final_tensor_constraint_no_padding = Conj([*initialize_tensors_constraints, generate_all_broadcasting_possibilities_no_padding(d1, d2, d3, d4)])
    (final_tensor_constraint_padding_arg1, counter) = apply_padding(e1, e11_tensor, e2_tensor, e12_tensor, d2, d3, d4, counter)
    (final_tensor_constraint_padding_arg2, counter) = apply_padding(e2, e12_tensor, e1_tensor, e11_tensor, d1, d4, d3, counter)
    return (final_tensor_constraint_no_padding, final_tensor_constraint_padding_arg1, final_tensor_constraint_padding_arg2, nat_dims_i, counter)