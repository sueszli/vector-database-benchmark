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
from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr

class Counter:
    """A counter for ids.

    Attributes
    ----------
    count : int
        The current count.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self.count = 1
ID_COUNTER = Counter()

def get_id() -> int:
    if False:
        for i in range(10):
            print('nop')
    'Returns a new id and updates the id counter.\n\n    Returns\n    -------\n    int\n        A new id.\n    '
    new_id = ID_COUNTER.count
    ID_COUNTER.count += 1
    return new_id

def create_var(shape: Tuple[int, ...], var_id=None):
    if False:
        i = 10
        return i + 15
    'Creates a new internal variable.\n\n    Parameters\n    ----------\n    shape : tuple\n        The (rows, cols) dimensions of the variable.\n    var_id : int\n        The id of the variable.\n\n    Returns\n    -------\n    LinOP\n        A LinOp representing the new variable.\n    '
    if var_id is None:
        var_id = get_id()
    return lo.LinOp(lo.VARIABLE, shape, [], var_id)

def create_param(shape: Tuple[int, ...], param_id=None):
    if False:
        print('Hello World!')
    'Wraps a parameter.\n\n    Parameters\n    ----------\n    shape : tuple\n        The (rows, cols) dimensions of the operator.\n\n    Returns\n    -------\n    LinOP\n        A LinOp wrapping the parameter.\n    '
    if param_id is None:
        param_id = get_id()
    return lo.LinOp(lo.PARAM, shape, [], param_id)

def create_const(value, shape: Tuple[int, ...], sparse: bool=False):
    if False:
        return 10
    'Wraps a constant.\n\n    Parameters\n    ----------\n    value : scalar, NumPy matrix, or SciPy sparse matrix.\n        The numeric constant to wrap.\n    shape : tuple\n        The (rows, cols) dimensions of the constant.\n    sparse : bool\n        Is the constant a SciPy sparse matrix?\n\n    Returns\n    -------\n    LinOP\n        A LinOp wrapping the constant.\n    '
    if shape == (1, 1):
        op_type = lo.SCALAR_CONST
        if not np.isscalar(value):
            value = value[0, 0]
    elif sparse:
        op_type = lo.SPARSE_CONST
    else:
        op_type = lo.DENSE_CONST
    return lo.LinOp(op_type, shape, [], value)

def is_scalar(operator) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether a LinOp is a scalar.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The LinOp to test.\n\n    Returns\n    -------\n        True if the LinOp is a scalar, False otherwise.\n    '
    return len(operator.shape) == 0 or np.prod(operator.shape, dtype=int) == 1

def is_const(operator) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether a LinOp is constant.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The LinOp to test.\n\n    Returns\n    -------\n        True if the LinOp is a constant, False otherwise.\n    '
    return operator.type in [lo.SCALAR_CONST, lo.SPARSE_CONST, lo.DENSE_CONST, lo.PARAM]

def sum_expr(operators):
    if False:
        while True:
            i = 10
    'Add linear operators.\n\n    Parameters\n    ----------\n    operators : list\n        A list of linear operators.\n\n    Returns\n    -------\n    LinOp\n        A LinOp representing the sum of the operators.\n    '
    return lo.LinOp(lo.SUM, operators[0].shape, operators, None)

def neg_expr(operator):
    if False:
        return 10
    'Negate an operator.\n\n    Parameters\n    ----------\n    expr : LinOp\n        The operator to be negated.\n\n    Returns\n    -------\n    LinOp\n        The negated operator.\n    '
    return lo.LinOp(lo.NEG, operator.shape, [operator], None)

def sub_expr(lh_op, rh_op):
    if False:
        for i in range(10):
            print('nop')
    'Difference of linear operators.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the difference.\n    rh_op : LinOp\n        The right-hand operator in the difference.\n\n    Returns\n    -------\n    LinOp\n        A LinOp representing the difference of the operators.\n    '
    return sum_expr([lh_op, neg_expr(rh_op)])

def promote_lin_ops_for_mul(lh_op, rh_op):
    if False:
        while True:
            i = 10
    'Promote arguments for multiplication.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the multiplication.\n    rh_op : LinOp\n        The right-hand operator in the multiplication.\n\n    Returns\n    -------\n    LinOp\n       Promoted left-hand operator.\n    LinOp\n       Promoted right-hand operator.\n    tuple\n       Shape of the product\n    '
    (lh_shape, rh_shape, shape) = u.shape.mul_shapes_promote(lh_op.shape, rh_op.shape)
    lh_op = lo.LinOp(lh_op.type, lh_shape, lh_op.args, lh_op.data)
    rh_op = lo.LinOp(rh_op.type, rh_shape, rh_op.args, rh_op.data)
    return (lh_op, rh_op, shape)

def mul_expr(lh_op, rh_op, shape: Tuple[int, ...]):
    if False:
        while True:
            i = 10
    'Multiply two linear operators, with the constant on the left.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the product.\n    rh_op : LinOp\n        The right-hand operator in the product.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the product.\n    '
    return lo.LinOp(lo.MUL, shape, [rh_op], lh_op)

def rmul_expr(lh_op, rh_op, shape: Tuple[int, ...]):
    if False:
        while True:
            i = 10
    'Multiply two linear operators, with the constant on the right.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the product.\n    rh_op : LinOp\n        The right-hand operator in the product.\n    shape : tuple\n        The shape of the product.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the product.\n    '
    return lo.LinOp(lo.RMUL, shape, [lh_op], rh_op)

def multiply(lh_op, rh_op):
    if False:
        while True:
            i = 10
    'Multiply two linear operators elementwise.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the product.\n    rh_op : LinOp\n        The right-hand operator in the product.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the product.\n    '
    shape = max(lh_op.shape, rh_op.shape)
    return lo.LinOp(lo.MUL_ELEM, shape, [rh_op], lh_op)

def kron_r(lh_op, rh_op, shape: Tuple[int, ...]):
    if False:
        print('Hello World!')
    'Kronecker product of two matrices, where the right operand is a Variable\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the product.\n    rh_op : LinOp\n        The right-hand operator in the product.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the Kronecker product.\n    '
    return lo.LinOp(lo.KRON_R, shape, [rh_op], lh_op)

def kron_l(lh_op, rh_op, shape: Tuple[int, ...]):
    if False:
        for i in range(10):
            print('nop')
    'Kronecker product of two matrices, where the left operand is a Variable\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the product.\n    rh_op : LinOp\n        The right-hand operator in the product.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the Kronecker product.\n    '
    return lo.LinOp(lo.KRON_L, shape, [lh_op], rh_op)

def div_expr(lh_op, rh_op):
    if False:
        print('Hello World!')
    'Divide one linear operator by another.\n\n    Assumes rh_op is a scalar constant.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the quotient.\n    rh_op : LinOp\n        The right-hand operator in the quotient.\n    shape : tuple\n        The shape of the quotient.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the quotient.\n    '
    return lo.LinOp(lo.DIV, lh_op.shape, [lh_op], rh_op)

def promote(operator, shape: Tuple[int, ...]):
    if False:
        i = 10
        return i + 15
    'Promotes a scalar operator to the given shape.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to promote.\n    shape : tuple\n        The dimensions to promote to.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the promotion.\n    '
    return lo.LinOp(lo.PROMOTE, shape, [operator], None)

def sum_entries(operator, shape: Tuple[int, ...]):
    if False:
        print('Hello World!')
    'Sum the entries of an operator.\n\n    Parameters\n    ----------\n    expr : LinOp\n        The operator to sum the entries of.\n    shape : tuple\n        The shape of the sum.\n\n    Returns\n    -------\n    LinOp\n        An operator representing the sum.\n    '
    return lo.LinOp(lo.SUM_ENTRIES, shape, [operator], None)

def trace(operator):
    if False:
        return 10
    'Sum the diagonal entries of an operator.\n\n    Parameters\n    ----------\n    expr : LinOp\n        The operator to sum the diagonal entries of.\n\n    Returns\n    -------\n    LinOp\n        An operator representing the sum of the diagonal entries.\n    '
    return lo.LinOp(lo.TRACE, (1, 1), [operator], None)

def index(operator, shape: Tuple[int, ...], keys):
    if False:
        while True:
            i = 10
    'Indexes/slices an operator.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The expression to index.\n    keys : tuple\n        (row slice, column slice)\n    shape : tuple\n        The shape of the expression after indexing.\n\n    Returns\n    -------\n    LinOp\n        An operator representing the indexing.\n    '
    return lo.LinOp(lo.INDEX, shape, [operator], keys)

def conv(lh_op, rh_op, shape: Tuple[int, ...]):
    if False:
        for i in range(10):
            print('nop')
    '1D discrete convolution of two vectors.\n\n    Parameters\n    ----------\n    lh_op : LinOp\n        The left-hand operator in the convolution.\n    rh_op : LinOp\n        The right-hand operator in the convolution.\n    shape : tuple\n        The shape of the convolution.\n\n    Returns\n    -------\n    LinOp\n        A linear operator representing the convolution.\n    '
    return lo.LinOp(lo.CONV, shape, [rh_op], lh_op)

def transpose(operator):
    if False:
        return 10
    'Transposes an operator.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to transpose.\n\n    Returns\n    -------\n    LinOp\n       A linear operator representing the transpose.\n    '
    if len(operator.shape) < 2:
        return operator
    elif len(operator.shape) > 2:
        raise NotImplementedError()
    else:
        shape = (operator.shape[1], operator.shape[0])
        return lo.LinOp(lo.TRANSPOSE, shape, [operator], None)

def reshape(operator, shape: Tuple[int, ...]):
    if False:
        i = 10
        return i + 15
    'Reshapes an operator.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to reshape.\n    shape : tuple\n        The (rows, cols) of the reshaped operator.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the reshaped expression.\n    '
    return lo.LinOp(lo.RESHAPE, shape, [operator], None)

def diag_vec(operator, k: int=0):
    if False:
        for i in range(10):
            print('nop')
    'Converts a vector to a diagonal matrix.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to convert to a diagonal matrix.\n    k : int\n        The offset of the diagonal.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the diagonal matrix.\n    '
    rows = operator.shape[0] + abs(k)
    shape = (rows, rows)
    return lo.LinOp(lo.DIAG_VEC, shape, [operator], k)

def diag_mat(operator, k: int=0):
    if False:
        return 10
    'Converts the diagonal of a matrix to a vector.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to convert to a vector.\n    k : int\n        The offset of the diagonal.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the matrix diagonal.\n    '
    shape = (operator.shape[0] - abs(k), 1)
    return lo.LinOp(lo.DIAG_MAT, shape, [operator], k)

def upper_tri(operator):
    if False:
        i = 10
        return i + 15
    'Vectorized upper triangular portion of a square matrix.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The matrix operator.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the vectorized upper triangle.\n    '
    entries = operator.shape[0] * operator.shape[1]
    shape = ((entries - operator.shape[0]) // 2, 1)
    return lo.LinOp(lo.UPPER_TRI, shape, [operator], None)

def hstack(operators, shape: Tuple[int, ...]):
    if False:
        for i in range(10):
            print('nop')
    'Concatenates operators horizontally.\n\n    Parameters\n    ----------\n    operator : list\n        The operators to stack.\n    shape : tuple\n        The (rows, cols) of the stacked operators.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the stacked expression.\n    '
    return lo.LinOp(lo.HSTACK, shape, operators, None)

def vstack(operators, shape: Tuple[int, ...]):
    if False:
        return 10
    'Concatenates operators vertically.\n\n    Parameters\n    ----------\n    operator : list\n        The operators to stack.\n    shape : tuple\n        The (rows, cols) of the stacked operators.\n\n    Returns\n    -------\n    LinOp\n       LinOp representing the stacked expression.\n    '
    return lo.LinOp(lo.VSTACK, shape, operators, None)

def get_constr_expr(lh_op, rh_op):
    if False:
        for i in range(10):
            print('nop')
    'Returns the operator in the constraint.\n    '
    if rh_op is None:
        return lh_op
    else:
        return sum_expr([lh_op, neg_expr(rh_op)])

def create_eq(lh_op, rh_op=None, constr_id=None):
    if False:
        for i in range(10):
            print('nop')
    'Creates an internal equality constraint.\n\n    Parameters\n    ----------\n    lh_term : LinOp\n        The left-hand operator in the equality constraint.\n    rh_term : LinOp\n        The right-hand operator in the equality constraint.\n    constr_id : int\n        The id of the CVXPY equality constraint creating the constraint.\n\n    Returns\n    -------\n    LinEqConstr\n    '
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_op, rh_op)
    return LinEqConstr(expr, constr_id, lh_op.shape)

def create_leq(lh_op, rh_op=None, constr_id=None):
    if False:
        for i in range(10):
            print('nop')
    'Creates an internal less than or equal constraint.\n\n    Parameters\n    ----------\n    lh_term : LinOp\n        The left-hand operator in the <= constraint.\n    rh_term : LinOp\n        The right-hand operator in the <= constraint.\n    constr_id : int\n        The id of the CVXPY equality constraint creating the constraint.\n\n    Returns\n    -------\n    LinLeqConstr\n    '
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_op, rh_op)
    return LinLeqConstr(expr, constr_id, lh_op.shape)

def create_geq(lh_op, rh_op=None, constr_id=None):
    if False:
        return 10
    'Creates an internal greater than or equal constraint.\n\n    Parameters\n    ----------\n    lh_term : LinOp\n        The left-hand operator in the >= constraint.\n    rh_term : LinOp\n        The right-hand operator in the >= constraint.\n    constr_id : int\n        The id of the CVXPY equality constraint creating the constraint.\n\n    Returns\n    -------\n    LinLeqConstr\n    '
    if rh_op is not None:
        rh_op = neg_expr(rh_op)
    return create_leq(neg_expr(lh_op), rh_op, constr_id)

def get_expr_vars(operator):
    if False:
        while True:
            i = 10
    'Get a list of the variables in the operator and their shapes.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to extract the variables from.\n\n    Returns\n    -------\n    list\n        A list of (var id, var shape) pairs.\n    '
    if operator.type == lo.VARIABLE:
        return [(operator.data, operator.shape)]
    else:
        vars_ = []
        for arg in operator.args:
            vars_ += get_expr_vars(arg)
        return vars_

def get_expr_params(operator):
    if False:
        for i in range(10):
            print('nop')
    'Get a list of the parameters in the operator.\n\n    Parameters\n    ----------\n    operator : LinOp\n        The operator to extract the parameters from.\n\n    Returns\n    -------\n    list\n        A list of parameter objects.\n    '
    if operator.type == lo.PARAM:
        return operator.data.parameters()
    else:
        params = []
        for arg in operator.args:
            params += get_expr_params(arg)
        if isinstance(operator.data, lo.LinOp):
            params += get_expr_params(operator.data)
        return params

def copy_constr(constr, func):
    if False:
        return 10
    'Creates a copy of the constraint modified according to func.\n\n    Parameters\n    ----------\n    constr : LinConstraint\n        The constraint to modify.\n    func : function\n        Function to modify the constraint expression.\n\n    Returns\n    -------\n    LinConstraint\n        A copy of the constraint with the specified changes.\n    '
    expr = func(constr.expr)
    return type(constr)(expr, constr.constr_id, constr.shape)

def replace_new_vars(expr, id_to_new_var):
    if False:
        while True:
            i = 10
    'Replaces the given variables in the expression.\n\n    Parameters\n    ----------\n    expr : LinOp\n        The expression to replace variables in.\n    id_to_new_var : dict\n        A map of id to new variable.\n\n    Returns\n    -------\n    LinOp\n        An LinOp identical to expr, but with the given variables replaced.\n    '
    if expr.type == lo.VARIABLE and expr.data in id_to_new_var:
        return id_to_new_var[expr.data]
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_new_vars(arg, id_to_new_var))
        return lo.LinOp(expr.type, expr.shape, new_args, expr.data)

def check_param_val(param):
    if False:
        return 10
    'Wrapper on accessing a parameter.\n\n    Parameters\n    ----------\n    param : Parameter\n        The parameter whose value is being accessed.\n\n    Returns\n    -------\n    The numerical value of the parameter.\n\n    Raises\n    ------\n    ValueError\n        Raises error if parameter value is None.\n    '
    val = param.value
    if val is None:
        raise ValueError('Problem has missing parameter value.')
    else:
        return val

def replace_params_with_consts(expr):
    if False:
        print('Hello World!')
    'Replaces parameters with constant nodes.\n\n    Parameters\n    ----------\n    expr : LinOp\n        The expression to replace parameters in.\n\n    Returns\n    -------\n    LinOp\n        An LinOp identical to expr, but with the parameters replaced.\n    '
    if expr.type == lo.PARAM:
        return create_const(check_param_val(expr.data), expr.shape)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        if isinstance(expr.data, lo.LinOp) and expr.data.type == lo.PARAM:
            data_lin_op = expr.data
            assert isinstance(data_lin_op.shape, tuple)
            val = check_param_val(data_lin_op.data)
            data = create_const(val, data_lin_op.shape)
        else:
            data = expr.data
        return lo.LinOp(expr.type, expr.shape, new_args, data)