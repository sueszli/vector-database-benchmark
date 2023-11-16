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
import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul

def get_mul_funcs(sym_data):
    if False:
        print('Hello World!')

    def accAmul(x, y, is_abs: bool=False):
        if False:
            return 10
        rows = y.shape[0]
        var_dict = vec_to_dict(x, sym_data.var_offsets, sym_data.var_sizes)
        y += constr_mul(sym_data.constraints, var_dict, rows, is_abs)

    def accATmul(x, y, is_abs: bool=False):
        if False:
            return 10
        terms = constr_unpack(sym_data.constraints, x)
        val_dict = constr_tmul(sym_data.constraints, terms, is_abs)
        y += dict_to_vec(val_dict, sym_data.var_offsets, sym_data.var_sizes, sym_data.x_length)
    return (accAmul, accATmul)

def constr_unpack(constraints, vector):
    if False:
        for i in range(10):
            print('nop')
    'Unpacks a vector into a list of values for constraints.\n    '
    values = []
    offset = 0
    for constr in constraints:
        (rows, cols) = constr.size
        val = np.zeros((rows, cols))
        for col in range(cols):
            val[:, col] = vector[offset:offset + rows]
            offset += rows
        values.append(val)
    return values

def vec_to_dict(vector, var_offsets, var_sizes):
    if False:
        for i in range(10):
            print('nop')
    'Converts a vector to a map of variable id to value.\n\n    Parameters\n    ----------\n    vector : NumPy matrix\n        The vector of values.\n    var_offsets : dict\n        A map of variable id to offset in the vector.\n    var_sizes : dict\n        A map of variable id to variable size.\n\n    Returns\n    -------\n    dict\n        A map of variable id to variable value.\n    '
    val_dict = {}
    for (id_, offset) in var_offsets.items():
        size = var_sizes[id_]
        value = np.zeros(size)
        offset = var_offsets[id_]
        for col in range(size[1]):
            value[:, col] = vector[offset:size[0] + offset]
            offset += size[0]
        val_dict[id_] = value
    return val_dict

def dict_to_vec(val_dict, var_offsets, var_sizes, vec_len):
    if False:
        for i in range(10):
            print('nop')
    'Converts a map of variable id to value to a vector.\n\n    Parameters\n    ----------\n    val_dict : dict\n        A map of variable id to value.\n    var_offsets : dict\n        A map of variable id to offset in the vector.\n    var_sizes : dict\n        A map of variable id to variable size.\n    vector : NumPy matrix\n        The vector to store the values in.\n    '
    vector = np.zeros(vec_len)
    for (id_, value) in val_dict.items():
        size = var_sizes[id_]
        offset = var_offsets[id_]
        for col in range(size[1]):
            if np.isscalar(value):
                vector[offset:size[0] + offset] = value
            else:
                vector[offset:size[0] + offset] = np.squeeze(value[:, col])
            offset += size[0]
    return vector

def constr_mul(constraints, var_dict, vec_size, is_abs):
    if False:
        print('Hello World!')
    'Multiplies a vector by the matrix implied by the constraints.\n\n    Parameters\n    ----------\n    constraints : list\n        A list of linear constraints.\n    var_dict : dict\n        A dictionary mapping variable id to value.\n    vec_size : int\n        The length of the product vector.\n    is_abs : bool\n        Multiply by the absolute value of the matrix?\n    '
    product = np.zeros(vec_size)
    offset = 0
    for constr in constraints:
        result = mul(constr.expr, var_dict, is_abs)
        (rows, cols) = constr.size
        for col in range(cols):
            if np.isscalar(result):
                product[offset:offset + rows] = result
            else:
                product[offset:offset + rows] = np.squeeze(result[:, col])
            offset += rows
    return product

def constr_tmul(constraints, values, is_abs):
    if False:
        print('Hello World!')
    'Multiplies a vector by the transpose of the constraints matrix.\n\n    Parameters\n    ----------\n    constraints : list\n        A list of linear constraints.\n    values : list\n        A list of NumPy matrices.\n    is_abs : bool\n        Multiply by the absolute value of the matrix?\n\n    Returns\n    -------\n    dict\n        A mapping of variable id to value.\n    '
    products = []
    for (constr, val) in zip(constraints, values):
        products.append(tmul(constr.expr, val, is_abs))
    return sum_dicts(products)