"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo

def mul(lin_op, val_dict, is_abs: bool=False):
    if False:
        print('Hello World!')
    'Multiply the expression tree by a vector.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        The root of an expression tree.\n    val_dict : dict\n        A map of variable id to value.\n    is_abs : bool, optional\n        Multiply by the absolute value of the matrix?\n\n    Returns\n    -------\n    NumPy matrix\n        The result of the multiplication.\n    '
    if lin_op.type is lo.VARIABLE:
        if lin_op.data in val_dict:
            if is_abs:
                return np.abs(val_dict[lin_op.data])
            else:
                return val_dict[lin_op.data]
        else:
            return np.mat(np.zeros(lin_op.shape))
    elif lin_op.type is lo.NO_OP:
        return np.mat(np.zeros(lin_op.shape))
    else:
        eval_args = []
        for arg in lin_op.args:
            eval_args.append(mul(arg, val_dict, is_abs))
        if is_abs:
            return op_abs_mul(lin_op, eval_args)
        else:
            return op_mul(lin_op, eval_args)

def tmul(lin_op, value, is_abs: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Multiply the transpose of the expression tree by a vector.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        The root of an expression tree.\n    value : NumPy matrix\n        The vector to multiply by.\n    is_abs : bool, optional\n        Multiply by the absolute value of the matrix?\n\n    Returns\n    -------\n    dict\n        A map of variable id to value.\n    '
    if lin_op.type is lo.VARIABLE:
        return {lin_op.data: value}
    elif lin_op.type is lo.NO_OP:
        return {}
    else:
        if is_abs:
            result = op_abs_tmul(lin_op, value)
        else:
            result = op_tmul(lin_op, value)
        result_dicts = []
        for arg in lin_op.args:
            result_dicts.append(tmul(arg, result, is_abs))
        return sum_dicts(result_dicts)

def sum_dicts(dicts):
    if False:
        print('Hello World!')
    'Sums the dictionaries entrywise.\n\n    Parameters\n    ----------\n    dicts : list\n        A list of dictionaries with numeric entries.\n\n    Returns\n    -------\n    dict\n        A dict with the sum.\n    '
    sum_dict = {}
    for val_dict in dicts:
        for (id_, value) in val_dict.items():
            if id_ in sum_dict:
                sum_dict[id_] = sum_dict[id_] + value
            else:
                sum_dict[id_] = value
    return sum_dict

def op_mul(lin_op, args):
    if False:
        return 10
    'Applies the linear operator to the arguments.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        A linear operator.\n    args : list\n        The arguments to the operator.\n\n    Returns\n    -------\n    NumPy matrix or SciPy sparse matrix.\n        The result of applying the linear operator.\n    '
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = lin_op.data
    elif lin_op.type is lo.NO_OP:
        return None
    elif lin_op.type is lo.SUM:
        result = sum(args)
    elif lin_op.type is lo.NEG:
        result = -args[0]
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {})
        result = coeff * args[0]
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {})
        result = args[0] / divisor
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.sum(args[0])
    elif lin_op.type is lo.INDEX:
        (row_slc, col_slc) = lin_op.data
        result = args[0][row_slc, col_slc]
    elif lin_op.type is lo.TRANSPOSE:
        result = args[0].T
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, args[0])
    elif lin_op.type is lo.PROMOTE:
        result = np.ones(lin_op.shape) * args[0]
    elif lin_op.type is lo.DIAG_VEC:
        val = intf.from_2D_to_1D(args[0])
        result = np.diag(val)
    else:
        raise Exception('Unknown linear operator.')
    return result

def op_abs_mul(lin_op, args):
    if False:
        print('Hello World!')
    'Applies the absolute value of the linear operator to the arguments.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        A linear operator.\n    args : list\n        The arguments to the operator.\n\n    Returns\n    -------\n    NumPy matrix or SciPy sparse matrix.\n        The result of applying the linear operator.\n    '
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = np.abs(lin_op.data)
    elif lin_op.type is lo.NEG:
        result = args[0]
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {}, True)
        result = coeff * args[0]
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {}, True)
        result = args[0] / divisor
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, args[0], is_abs=True)
    else:
        result = op_mul(lin_op, args)
    return result

def op_tmul(lin_op, value):
    if False:
        print('Hello World!')
    "Applies the transpose of the linear operator to the arguments.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        A linear operator.\n    value : NumPy matrix\n        A numeric value to apply the operator's transpose to.\n\n    Returns\n    -------\n    NumPy matrix or SciPy sparse matrix.\n        The result of applying the linear operator.\n    "
    if lin_op.type is lo.SUM:
        result = value
    elif lin_op.type is lo.NEG:
        result = -value
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {})
        if np.isscalar(coeff):
            result = coeff * value
        else:
            result = coeff.T * value
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {})
        result = value / divisor
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.mat(np.ones(lin_op.args[0].shape)) * value
    elif lin_op.type is lo.INDEX:
        (row_slc, col_slc) = lin_op.data
        result = np.mat(np.zeros(lin_op.args[0].shape))
        result[row_slc, col_slc] = value
    elif lin_op.type is lo.TRANSPOSE:
        result = value.T
    elif lin_op.type is lo.PROMOTE:
        result = np.ones(lin_op.shape[0]).dot(value)
    elif lin_op.type is lo.DIAG_VEC:
        result = np.diag(value)
        if isinstance(result, np.matrix):
            result = result.A[0]
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, value, transpose=True)
    else:
        raise Exception('Unknown linear operator.')
    return result

def op_abs_tmul(lin_op, value):
    if False:
        while True:
            i = 10
    "Applies the linear operator |A.T| to the arguments.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        A linear operator.\n    value : NumPy matrix\n        A numeric value to apply the operator's transpose to.\n\n    Returns\n    -------\n    NumPy matrix or SciPy sparse matrix.\n        The result of applying the linear operator.\n    "
    if lin_op.type is lo.NEG:
        result = value
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {}, True)
        if np.isscalar(coeff):
            result = coeff * value
        else:
            result = coeff.T * value
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {}, True)
        result = value / divisor
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, value, True, True)
    else:
        result = op_tmul(lin_op, value)
    return result

def conv_mul(lin_op, rh_val, transpose: bool=False, is_abs: bool=False):
    if False:
        i = 10
        return i + 15
    'Multiply by a convolution operator.\n\n    arameters\n    ----------\n    lin_op : LinOp\n        The root linear operator.\n    rh_val : NDArray\n        The vector being convolved.\n    transpose : bool\n        Is the transpose of convolution being applied?\n    is_abs : bool\n        Is the absolute value of convolution being applied?\n\n    Returns\n    -------\n    NumPy NDArray\n        The convolution.\n    '
    constant = mul(lin_op.data, {}, is_abs)
    (constant, rh_val) = map(intf.from_1D_to_2D, [constant, rh_val])
    if transpose:
        constant = np.flipud(constant)
        return fftconvolve(rh_val, constant, mode='valid')
    elif constant.size >= rh_val.size:
        return fftconvolve(constant, rh_val, mode='full')
    else:
        return fftconvolve(rh_val, constant, mode='full')

def get_constant(lin_op):
    if False:
        return 10
    'Returns the constant term in the expression.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        The root linear operator.\n\n    Returns\n    -------\n    NumPy NDArray\n        The constant term as a flattened vector.\n    '
    constant = mul(lin_op, {})
    const_size = constant.shape[0] * constant.shape[1]
    return np.reshape(constant, const_size, 'F')

def get_constr_constant(constraints):
    if False:
        while True:
            i = 10
    'Returns the constant term for the constraints matrix.\n\n    Parameters\n    ----------\n    constraints : list\n        The constraints that form the matrix.\n\n    Returns\n    -------\n    NumPy NDArray\n        The constant term as a flattened vector.\n    '
    constants = [get_constant(c.expr) for c in constraints]
    return np.hstack(constants)

def prune_constants(constraints):
    if False:
        while True:
            i = 10
    'Returns a new list of constraints with constant terms removed.\n\n    Parameters\n    ----------\n    constraints : list\n        The constraints that form the matrix.\n\n    Returns\n    -------\n    list\n        The pruned constraints.\n    '
    pruned_constraints = []
    for constr in constraints:
        constr_type = type(constr)
        expr = copy.deepcopy(constr.expr)
        is_constant = prune_expr(expr)
        if is_constant:
            expr = lo.LinOp(lo.NO_OP, expr.shape, [], None)
        pruned = constr_type(expr, constr.constr_id, constr.shape)
        pruned_constraints.append(pruned)
    return pruned_constraints

def prune_expr(lin_op) -> bool:
    if False:
        i = 10
        return i + 15
    "Prunes constant branches from the expression.\n\n    Parameters\n    ----------\n    lin_op : LinOp\n        The root linear operator.\n\n    Returns\n    -------\n    bool\n        Were all the expression's arguments pruned?\n    "
    if lin_op.type is lo.VARIABLE:
        return False
    elif lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST, lo.PARAM]:
        return True
    pruned_args = []
    is_constant = True
    for arg in lin_op.args:
        arg_constant = prune_expr(arg)
        if not arg_constant:
            is_constant = False
            pruned_args.append(arg)
    lin_op.args[:] = pruned_args[:]
    return is_constant